# ---------------------------------------------------------------------
# main_torch.py – PyTorch training loop for Neural Diffusion Processes
# ---------------------------------------------------------------------
# make imports work when run directly
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # project root …/ndp_pytorch
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, datetime, math, pprint, random, string
import os
from dataclasses import asdict
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt
import torch, tqdm
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import neural_diffusion_processes as ndp
from neural_diffusion_processes.model import BiDimensionalAttentionModel
from neural_diffusion_processes.process import GaussianDiffusion, cosine_schedule
from neural_diffusion_processes.types import Batch
from data import get_batch
from config import Config

from torch.utils.tensorboard import SummaryWriter
import json

# ------------------------------------------------------------------ #
#  Helpers                                                           #
# ------------------------------------------------------------------ #
def _experiment_name() -> str:
    now = datetime.datetime.now().strftime("%b%d_%H%M%S")
    tag = "".join(random.choice(string.ascii_lowercase) for _ in range(4))
    return f"{now}_{tag}"

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _save_fig(fig, path: Path, show_live: bool = False, dpi: int = 200):
    fig.tight_layout()
    fig.savefig(path.as_posix(), dpi=dpi)
    if show_live:
        import matplotlib.pyplot as plt
        plt.show(block=False)
        plt.pause(0.001)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Dataset wrapper                                                   #
# ------------------------------------------------------------------ #
class InfiniteDataset(IterableDataset):
    def __init__(self, cfg: Config, train: bool):
        self.cfg, self.train = cfg, train
        #self.gen = torch.Generator().manual_seed(cfg.seed)
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        # Give each worker a distinct, persistent generator
        base_seed = self.cfg.seed + (info.id if info else 0)
        print(base_seed)
        g = torch.Generator().manual_seed(base_seed)
        while True:
            raw = get_batch(
                g,
                batch_size=self.cfg.batch_size,
                name=self.cfg.dataset,
                task="training" if self.train else "interpolation",
                input_dim=self.cfg.input_dim,
                gp_conditional_targets=True,  # ← use GP conditional targets
                p_drop_ctx=0.2,  # ← classifier-free style (tweak as you like)
            )

            # Build the ndp.types.Batch expected by ndp.process.loss using POSITIONAL args
            # (avoid keywords; your error came from unexpected keyword 'x')
            tb = Batch(x_target=raw.x_target, y_target=raw.y_target, mask_target=raw.mask_target)

            # Attach context so the loss-closure can forward it into the model
            tb.x_context = raw.x_context
            tb.y_context = raw.y_context
            tb.mask_context = raw.mask_context

            yield tb


# ------------------------------------------------------------------ #
def build_network(cfg: Config) -> nn.Module:
    return BiDimensionalAttentionModel(
        n_layers   = cfg.network.n_layers,
        hidden_dim = cfg.network.hidden_dim,
        num_heads  = cfg.network.num_heads,
    )

def build_process(cfg: Config) -> GaussianDiffusion:
    device = _device()
    betas = cosine_schedule(cfg.diffusion.beta_start,
                            cfg.diffusion.beta_end,
                            cfg.diffusion.timesteps).to(device)
    return GaussianDiffusion(betas)

@torch.no_grad()
def _ema_update(ema: nn.Module, online: nn.Module, decay: float):
    for p_ema, p in zip(ema.parameters(), online.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

def make_loss_fn(process, cfg):
    def _loss_fn(model, batch, key):
        def eps_model(t, yt, x, mask, *, key):
            # Ensure yt is [B,N,1]
            if yt.ndim == 3 and yt.shape[1] == 1 and yt.shape[2] == x.shape[1]:
                yt = yt.transpose(1, 2)

            return model(
                x, yt, t, mask,
                x_context=getattr(batch, "x_context", None),
                y_context=getattr(batch, "y_context", None),
                mask_context=getattr(batch, "mask_context", None),
            )
        return ndp.process.loss(process, eps_model, batch, key,
                                num_timesteps=cfg.diffusion.timesteps,
                                loss_type=cfg.loss_type)
    return _loss_fn




# ------------------------------------------------------------------ #
def train(cfg: Config):
    device  = _device()
    log_dir = Path("logs") / "regression" / _experiment_name()
    log_dir.mkdir(parents=True, exist_ok=True)
    print("Logging to:", log_dir)


    # # ── TensorBoard writer ─────────────────────────────────────────────
    # tb_dir = log_dir / "tb"
    # writer = SummaryWriter(tb_dir.as_posix())
    # try:
    #     writer.add_text("config/json", f"```json\n{json.dumps(asdict(cfg), indent=2)}\n```", global_step=0)
    # except Exception:
    #     writer.add_text("config/str", str(asdict(cfg)), global_step=0)
    # ───────────────────────────────────────────────────────────────────
    plots_dir = log_dir / "plots"

    data = DataLoader(
        InfiniteDataset(cfg, train=True),
        batch_size=None,
        num_workers=4,  # 🔁 Try 4–8, or more if you have CPU cores
        prefetch_factor=2,  # Optional: prefetch batches to reduce wait time
        pin_memory=True  # Useful for GPU transfers
    )

    model     = build_network(cfg).to(device)
    model_ema = build_network(cfg).to(device)
    model_ema.load_state_dict(model.state_dict()) # make the initial paras of model_ema identical to model
    process   = build_process(cfg)
    loss_fn   = make_loss_fn(process, cfg)

    print("model.hidden_dim =", model.hidden_dim)
    # print("mha_crossD K weight shape =",
    #       tuple(model.layers[0].mha_crossD.k_proj.weight.shape))

    # ---- optimiser & LR schedule (warm-up + cosine) ---------------------
    optimiser = AdamW(model.parameters(),
                lr=cfg.optimizer.peak_lr,
                betas=(0.9, 0.999))                # weight_decay left at default 0

    warmup_steps = 0.05 * cfg.steps_per_epoch * cfg.optimizer.num_warmup_epochs
    total_steps  = cfg.total_steps

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))

    lr_sched = LambdaLR(optimiser, lr_lambda)

    device = _device()  # already cuda:0
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    # ---- training loop --------------------------------------------------
    model.train()
    pbar = tqdm.tqdm(range(1, total_steps + 1))

    for step, batch in zip(pbar, data):
        # ✅ Keep the same object; move any tensors in-place
        for k, v in list(batch.__dict__.items()):
            if torch.is_tensor(v):
                batch.__dict__[k] = v.to(device, non_blocking=True)

        # if step == 1:
        #     debug_batch(batch, dataset_name=cfg.dataset, active_dims=list(range(cfg.input_dim)),
        #                 title=f"Step_{step}_input_data")

        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch, gen)
        loss.backward()
        # clip returns total grad norm — log it
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step(); lr_sched.step()

        _ema_update(model_ema, model, cfg.optimizer.ema_rate)

        # ── TensorBoard scalars ────────────────────────────────────────
        # writer.add_scalar("train/loss", float(loss.item()), step)
        # writer.add_scalar("train/lr", lr_sched.get_last_lr()[0], step)
        # writer.add_scalar("train/grad_norm", grad_norm, step)
        # ───────────────────────────────────────────────────────────────

        if step % 100 == 0 or step == 1:
            pbar.set_description(f"loss {loss.item():.3f} • lr {lr_sched.get_last_lr()[0]:.2e}")

        if step == 1 or step % (total_steps // 8) == 0: # TODO: 4 set the num of plots
            _plot_samples(
                model_ema, process, cfg, device,
                title=f"step_{step:07d}",
                out_dir=plots_dir,
                show_live=False,  # set True if you want pop-up windows
            )

        if step >= total_steps:
            break

    torch.save(model_ema.state_dict(), log_dir / "model_ema.pt")
    print("Training complete – weights saved.")

# ------------------------------------------------------------------ #
@torch.no_grad()
def _plot_samples(model, process, cfg, device, title, out_dir: Path, show_live: bool = False):
    """Save unconditional samples as a PNG via matplotlib (no TensorBoard)."""
    if cfg.input_dim != 1:
        return

    _ensure_dir(out_dir)

    import matplotlib.pyplot as plt
    x = torch.linspace(-2, 2, 60, device=device).unsqueeze(-1)  # [N,1]

    # model closure for the diffusion sampler
    def net_fn(t, yt, xx, m, *, key):
        out = model(
            xx.unsqueeze(0),   # [N,1] -> [1,N,1]
            yt.unsqueeze(0),   # [N,1] -> [1,N,1]
            t.view(1),         # []    -> [1]
            m.unsqueeze(0) if m is not None else m
        )
        return out.squeeze(0)  # [N,1]

    gen = torch.Generator(device=device).manual_seed(0)
    ys = torch.stack([process.sample(gen, x, None, model_fn=net_fn) for _ in range(8)])  # [8,N,1]
    ys = ys.squeeze(-1).cpu()   # [8,N]
    xx = x[:, 0].detach().cpu() # [N]

    fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=150)
    for i in range(ys.shape[0]):
        ax.plot(xx, ys[i], alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    _save_fig(fig, out_dir / f"{title}.png", show_live=show_live)


# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default=None)
    cfg = Config() if (ns := parser.parse_args()).cfg_file is None \
        else Config.from_file(ns.cfg_file)

    pprint.pprint(asdict(cfg))
    train(cfg)



if __name__ == "__main__":
    main()

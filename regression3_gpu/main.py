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
        g = torch.Generator().manual_seed(base_seed)
        while True:
            #gen = torch.Generator().manual_seed(self.cfg.seed)  # ✅ move here
            yield get_batch(
                g,                                   # positional
                batch_size=self.cfg.batch_size,
                name=self.cfg.dataset,
                task="training" if self.train else "interpolation",
                input_dim=self.cfg.input_dim,
            )

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

def make_loss_fn(process: GaussianDiffusion, cfg: Config):
    def _loss_fn(
        model: torch.nn.Module, # BidimentionalAttentionModel
        batch: Batch,
        key: torch.Generator,          # positional, not a kwarg
    ) -> torch.Tensor:

        # --- adapter: reorder args & ignore `key` -----------------
        def eps_model(t, yt, x, mask, *, key):
            # t: [B], yt: [B,N,1], x: [B,N,D], mask: [B,N]
            return model(x, yt, t, mask)  # returns [B,N,1]

        # pos. args: process, network, batch, key
        # kw-only: num_timesteps, loss_type
        return ndp.process.loss(
            process,                    # GaussianDiffusion
            eps_model,                      # your BiDim model
            batch,                      # Batch instance
            key,                        # <— here!
            num_timesteps=cfg.diffusion.timesteps,
            loss_type=cfg.loss_type,
        )
    return _loss_fn


# ------------------------------------------------------------------ #
def train(cfg: Config):
    device  = _device()
    log_dir = Path("logs") / "regression" / _experiment_name()
    log_dir.mkdir(parents=True, exist_ok=True)
    print("Logging to:", log_dir)


    # ── TensorBoard writer ─────────────────────────────────────────────
    tb_dir = log_dir / "tb"
    writer = SummaryWriter(tb_dir.as_posix())
    try:
        writer.add_text("config/json", f"```json\n{json.dumps(asdict(cfg), indent=2)}\n```", global_step=0)
    except Exception:
        writer.add_text("config/str", str(asdict(cfg)), global_step=0)
    # ───────────────────────────────────────────────────────────────────

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
        batch = Batch(**{k: v.to(device, non_blocking=True) for k, v in batch.__dict__.items()})
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
        writer.add_scalar("train/loss", float(loss.item()), step)
        writer.add_scalar("train/lr", lr_sched.get_last_lr()[0], step)
        writer.add_scalar("train/grad_norm", grad_norm, step)
        # ───────────────────────────────────────────────────────────────

        if step % 100 == 0 or step == 1:
            pbar.set_description(f"loss {loss.item():.3f} • lr {lr_sched.get_last_lr()[0]:.2e}")

        if step == 1 or step % (total_steps // 8) == 0: # TODO: 4 set the num of plots
            _plot_samples(model_ema, process, cfg, device,
                          title=f"step_{step:07d}",
                          out_dir=log_dir / "plots",
                          writer=writer, step=step)

        if step >= total_steps:
            break

    torch.save(model_ema.state_dict(), log_dir / "model_ema.pt")
    print("Training complete – weights saved.")

# ------------------------------------------------------------------ #
@torch.no_grad()
def _plot_samples(model, process, cfg, device, title, out_dir: Path, writer: SummaryWriter | None = None, step: int | None = None):
    if cfg.input_dim != 1:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    x = torch.linspace(-2, 2, 60, device=device).unsqueeze(-1)
    #x = torch.linspace(-2, 2, 60, device=device).unsqueeze(0).unsqueeze(-1)  # [1, 60, 1]

    net_fn = lambda t, yt, xx, m, *, key: (
        model(  # BiDimensionalAttentionModel
            xx.unsqueeze(0),  # [N,D] ➜ [1,N,D]
            yt.unsqueeze(0),  # [N,1] ➜ [1,N,1]
            t.view(1),  # []    ➜ [1]
            m.unsqueeze(0) if m is not None else m
        ).squeeze(0)  # back to [N,1] for the diffusion code
    )

    gen = torch.Generator(device=device).manual_seed(0)
    ys = torch.stack([process.sample(gen, x, None, model_fn=net_fn) for _ in range(8)]
                     ).squeeze(-1)  # [8,N]

    fig = plt.figure(figsize=(4, 3))
    plt.plot(x.cpu(), ys.cpu().T, color="C0", alpha=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / f"{title}.png", dpi=200)

    # Log to TensorBoard
    if writer is not None and step is not None:
        writer.add_figure("samples/unconditional_grid", fig, global_step=step)

    plt.close()

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

# def plot_gp_batch(batch, title="Step 1 GP samples"):
#     import matplotlib.pyplot as plt
#
#     x_t = batch.x_target.detach().cpu()   # [B, N, 1]
#     y_t = batch.y_target.detach().cpu()   # [B, N, 1]
#     B, N, _ = x_t.shape
#
#     plt.figure(figsize=(6,4))
#     for i in range(min(8, B)):
#         x = x_t[i, :, 0]
#         y = y_t[i, :, 0]
#         idx = torch.argsort(x)            # sort indices by x
#         plt.plot(x[idx], y[idx], alpha=0.7)
#     plt.xlabel("x"); plt.ylabel("y"); plt.title(title)
#     plt.tight_layout();
#     plt.savefig(f"test_gp.png", dpi=200)
#     plt.close()
#     print(f"Saved test_gp.png")
#
#
# def print_tensor_stats(name, value):
#     if value.numel() == 0:
#         print(f"  ⚠ {name} is empty: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
#         return
#
#     print(f"  🔹 {name}: {{"
#           f"'shape': {tuple(value.shape)}, "
#           f"'dtype': '{value.dtype}', "
#           f"'device': '{value.device}', "
#           f"'mean': {value.float().mean().item():.6f}, "
#           f"'var': {value.float().var(unbiased=False).item():.6f}, "
#           f"'min': {value.min().item():.6f}, "
#           f"'max': {value.max().item():.6f}}}")
#
# def debug_batch(batch, dataset_name=None, active_dims=None, title="Batch Debug Plot", out_dir="debug_plots"):
#     import os
#     import matplotlib.pyplot as plt
#     os.makedirs(out_dir, exist_ok=True)
#
#     # Tensor stats
#     for k, v in batch.__dict__.items():
#         print_tensor_stats(k, v)
#
#     # Plot if 1D input
#     if batch.x_context.shape[-1] == 1 and batch.x_target.shape[-1] == 1:
#         if batch.x_context.numel() > 0 and batch.y_context.numel() > 0:
#             plt.scatter(batch.x_context.cpu(), batch.y_context.cpu(), color='blue', label="Context", s=20)
#         if batch.x_target.numel() > 0 and batch.y_target.numel() > 0:
#             plt.scatter(batch.x_target.cpu(), batch.y_target.cpu(), color='red', label="Target", s=20, alpha=0.6)
#
#         plt.title(title)
#         plt.legend()
#         plt.tight_layout()
#         file_path = os.path.join(out_dir, f"{title.replace(' ', '_')}.png")
#         plt.savefig(file_path, dpi=150)
#         plt.close()
#         print(f"Saved plot to {file_path}")
#
#     plot_gp_batch(batch)



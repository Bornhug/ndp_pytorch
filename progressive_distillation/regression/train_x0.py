from __future__ import annotations

import sys
import math
import pprint
import datetime
import random
import string
from dataclasses import asdict
from pathlib import Path

import torch
import tqdm
from torch import nn
from torch.optim import AdamW
from torch.utils.data import IterableDataset, DataLoader

# Make local package imports work when run directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_diffusion_processes.model import BiDimensionalAttentionModel
from neural_diffusion_processes.process_pd import (
    GaussianDiffusionPD,
    cosine_schedule,
    loss_x0,
)
from regression.data import get_batch
from regression.config import Config
from neural_diffusion_processes.types import Batch


def _experiment_name() -> str:
    now = datetime.datetime.now().strftime("%b%d_%H%M%S")
    tag = "".join(random.choice(string.ascii_lowercase) for _ in range(4))
    return f"{now}_{tag}"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InfiniteDataset(IterableDataset):
    def __init__(self, cfg: Config, train: bool):
        self.cfg, self.train = cfg, train

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        base_seed = self.cfg.seed + (info.id if info else 0)
        g = torch.Generator().manual_seed(base_seed)
        while True:
            raw = get_batch(
                g,
                batch_size=self.cfg.batch_size,
                name=self.cfg.dataset,
                task="training" if self.train else "interpolation",
                input_dim=self.cfg.input_dim,
                gp_conditional_targets=True,
                p_drop_ctx=0.0,
            )

            tb = Batch(x_target=raw.x_target, y_target=raw.y_target, mask_target=raw.mask_target)
            tb.x_context = raw.x_context
            tb.y_context = raw.y_context
            tb.mask_context = raw.mask_context
            yield tb


def build_network(cfg: Config) -> nn.Module:
    return BiDimensionalAttentionModel(
        n_layers=cfg.network.n_layers,
        hidden_dim=cfg.network.hidden_dim,
        num_heads=cfg.network.num_heads,
    )


def build_process(cfg: Config) -> GaussianDiffusionPD:
    device = _device()
    betas = cosine_schedule(
        cfg.diffusion.beta_start,
        cfg.diffusion.beta_end,
        cfg.diffusion.timesteps,
    ).to(device)
    return GaussianDiffusionPD(betas)


@torch.no_grad()
def _ema_update(ema: nn.Module, online: nn.Module, decay: float):
    for p_ema, p in zip(ema.parameters(), online.parameters()):
        p_ema.data.mul_((decay)).add_(p.data, alpha=1.0 - decay)


def make_loss_fn(process: GaussianDiffusionPD, cfg: Config):
    def _loss_fn(model, batch, key):
        def x0_model(t, yt, x, mask, *, key):
            return model(
                batch.x_target,
                yt,
                t,
                None,
                x_context=getattr(batch, "x_context", None),
                y_context=getattr(batch, "y_context", None),
                mask_context=getattr(batch, "mask_context", None),
            )

        return loss_x0(
            process,
            x0_model,
            batch,
            key,
            num_timesteps=cfg.diffusion.timesteps,
            loss_weighting=getattr(cfg, "loss_weighting", "lambda"),
            loss_type=cfg.loss_type,
        )

    return _loss_fn


@torch.no_grad()
def _plot_samples(model, process: GaussianDiffusionPD, cfg: Config, device, title: str, out_dir: Path):
    if cfg.input_dim != 1:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    x = torch.linspace(-2, 2, 60, device=device).unsqueeze(-1)

    def net_fn(t, yt, xx, m, *, key):
        out = model(
            xx.unsqueeze(0),
            yt.unsqueeze(0),
            t.view(1),
            None,
        )
        return out.squeeze(0)

    gen = torch.Generator(device=device).manual_seed(0)
    ys = torch.stack([process.sample(gen, x, model_fn=net_fn) for _ in range(8)])
    ys = ys.squeeze(-1).cpu()
    xx = x[:, 0].detach().cpu()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=150)
    for i in range(ys.shape[0]):
        ax.plot(xx, ys[i], alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    (out_dir / f"{title}.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig((out_dir / f"{title}.png").as_posix(), dpi=200)
    plt.close(fig)


def train(cfg: Config):
    device = _device()
    # Save under repo_root/progressive_distillation/logs/<exp>
    log_dir = Path("progressive_distillation") / "logs" / _experiment_name()
    log_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = log_dir / "plots"

    data = DataLoader(
        InfiniteDataset(cfg, train=True),
        batch_size=None,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )

    model = build_network(cfg).to(device)
    model_ema = build_network(cfg).to(device)
    model_ema.load_state_dict(model.state_dict())
    process = build_process(cfg)
    loss_fn = make_loss_fn(process, cfg)

    opt = AdamW(model.parameters(), lr=cfg.optimizer.peak_lr, betas=(0.9, 0.999))

    warmup_steps = int(0.05 * cfg.steps_per_epoch * cfg.optimizer.num_warmup_epochs)
    total_steps = cfg.total_steps

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    model.train()
    pbar = tqdm.tqdm(range(1, total_steps + 1))
    for step, batch in zip(pbar, data):
        for k, v in list(batch.__dict__.items()):
            if torch.is_tensor(v):
                batch.__dict__[k] = v.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch, gen)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        _ema_update(model_ema, model, cfg.optimizer.ema_rate)

        if step % 100 == 0 or step == 1:
            pbar.set_description(f"loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.2e}")

        if step == 1 or step % max(1, (total_steps // 8)) == 0:
            _plot_samples(model_ema, process, cfg, device,
                          title=f"step_{step:07d}", out_dir=plots_dir)

        if step >= total_steps:
            break

    torch.save(model_ema.state_dict(), log_dir / "model_ema.pt")


def main():
    cfg = Config()
    pprint.pprint(asdict(cfg))
    train(cfg)


if __name__ == "__main__":
    main()

from __future__ import annotations

import datetime
import json
import math
from dataclasses import dataclass
import argparse
from typing import Optional
from pathlib import Path
from typing import List, Optional, Sequence
import sys

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import tqdm

# Make package imports work when run directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neural_diffusion_processes.model import BiDimensionalAttentionModel
from neural_diffusion_processes.process_pd import GaussianDiffusionPD as GaussianDiffusion, cosine_schedule

from regression.config import Config
from regression.train_x0 import InfiniteDataset, build_network


@dataclass
class StageConfig:
    src_steps: int
    dst_steps: int


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ema_update(ema: nn.Module, online: nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema.parameters(), online.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def _coarse_grain_betas(betas: torch.Tensor, ratio: int) -> torch.Tensor:
    if ratio <= 0:
        raise ValueError(f"Coarsening ratio must be positive, got {ratio}.")
    if betas.numel() % ratio != 0:
        raise ValueError(
            f"Cannot evenly coarse-grain {betas.numel()} steps by ratio {ratio}. "
            "Provide a schedule where each stage divides the previous one."
        )
    alphas = 1.0 - betas
    segments: List[torch.Tensor] = []
    for start in range(0, betas.numel(), ratio):
        end = start + ratio
        alpha_segment = torch.prod(alphas[start:end])
        segments.append(1.0 - alpha_segment)
    return torch.stack(segments)


def _build_process(cfg: Config, timesteps: int, device: torch.device) -> GaussianDiffusion:
    betas = cosine_schedule(
        cfg.diffusion.beta_start,
        cfg.diffusion.beta_end,
        timesteps,
    ).to(device)
    return GaussianDiffusion(betas)


def _prepare_batch(batch, device: torch.device) -> None:
    for name in ["x_target", "y_target", "mask_target", "x_context", "y_context", "mask_context"]:
        tensor = getattr(batch, name, None)
        if torch.is_tensor(tensor):
            setattr(batch, name, tensor.to(device, non_blocking=True))


def _ddim_step_x0(process: GaussianDiffusion,
                  y: torch.Tensor,
                  x: torch.Tensor,
                  t_i: int,
                  t_j: int,
                  model_fn) -> torch.Tensor:
    """
    Deterministic DDIM step from t_i -> t_j using an x0-prediction model.
    """
    device = y.device
    abar_i = process.alpha_bars[t_i].to(device)
    abar_j = process.alpha_bars[t_j].to(device)
    t_tensor = torch.full((y.size(0),), float(t_i), device=device)
    x0_hat = model_fn(x, y, t_tensor)
    eps_hat = (y - torch.sqrt(abar_i) * x0_hat) / torch.sqrt(torch.clamp(1.0 - abar_i, min=1e-12))
    y_next = torch.sqrt(abar_j) * x0_hat + torch.sqrt(torch.clamp(1.0 - abar_j, min=1e-12)) * eps_hat
    return y_next


def _compute_eps_target(
    y_t: torch.Tensor,
    y_prev: torch.Tensor,
    alpha_bar_t: torch.Tensor,
    alpha_bar_prev: torch.Tensor,
) -> torch.Tensor:
    """
    Given two consecutive states in the forward process (times t and s),
    recover the shared noise ε using closed-form linear algebra.
    """
    eps = 1e-12
    A = torch.sqrt(alpha_bar_t).to(y_t.dtype)
    B = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=eps)).to(y_t.dtype)
    C = torch.sqrt(alpha_bar_prev).to(y_t.dtype)
    D = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev, min=eps)).to(y_t.dtype)

    denom = D - (C * B) / (A + eps)
    return (y_prev - (C / (A + eps)) * y_t) / (denom + eps)


def _build_stage_sequence(
    n_big: int,
    n_small: int,
    explicit: Optional[Sequence[int]] = None,
) -> List[StageConfig]:
    if n_small <= 0 or n_big <= 0:
        raise ValueError("Number of steps must be positive.")
    if n_small > n_big:
        raise ValueError(f"Target steps ({n_small}) must not exceed teacher steps ({n_big}).")

    if explicit is not None:
        if explicit[0] != n_big or explicit[-1] != n_small:
            raise ValueError("Explicit stage schedule must start at N_big and end at N_small.")
        for i in range(1, len(explicit)):
            if explicit[i] >= explicit[i - 1]:
                raise ValueError("Stage schedule must be strictly decreasing.")
        steps = list(explicit)
    else:
        steps = [n_big]
        current = n_big
        while current > n_small:
            next_steps = max(n_small, current // 2)
            if current % next_steps != 0:
                next_steps = n_small
            if current % next_steps != 0:
                raise ValueError(
                    f"Cannot halve from {current} to {next_steps}; supply an explicit schedule."
                )
            steps.append(next_steps)
            current = next_steps

    stages: List[StageConfig] = []
    for src, dst in zip(steps[:-1], steps[1:]):
        if src % dst != 0:
            raise ValueError(f"Stage transition {src}->{dst} is not divisible.")
        stages.append(StageConfig(src_steps=src, dst_steps=dst))
    return stages


def train_pd(
    cfg: Config,
    *,
    teacher_dir: str | Path,
    N_big: int,
    N_small: int,
    stage_steps: Optional[Sequence[int]] = None,
    steps_per_stage: Optional[Sequence[int] | int] = None,
    output_root: str | Path | None = None,
    ema_rate: Optional[float] = None,
    learning_rate: Optional[float] = None,
    num_workers: int = 4,
) -> List[Path]:
    """
    Progressive distillation following Song et al. (2021).

    Args:
        cfg: Training configuration (dataset + optimiser hyperparameters).
        teacher_dir: Directory (or direct checkpoint path) containing the initial teacher weights.
        N_big: Number of diffusion steps for the teacher.
        N_small: Target number of steps for the final student.
        stage_steps: Optional explicit step schedule (must start at N_big, end at N_small).
        steps_per_stage: Either a single int or a list specifying optimisation steps per stage.
        output_root: Optional output directory root for saving distilled weights.
        ema_rate: Optional override for EMA decay (defaults to cfg.optimizer.ema_rate).
        learning_rate: Optional override for AdamW learning rate (defaults to cfg.optimizer.peak_lr).
        num_workers: Number of DataLoader workers per stage.

    Returns:
        List of directories containing the saved checkpoints for each stage.
    """
    device = _default_device()
    ema_decay = ema_rate if ema_rate is not None else cfg.optimizer.ema_rate
    lr = learning_rate if learning_rate is not None else cfg.optimizer.peak_lr

    teacher_path = Path(teacher_dir)
    if teacher_path.is_dir():
        teacher_path = teacher_path / "model_ema.pt"
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_path}")

    stages = _build_stage_sequence(N_big, N_small, explicit=stage_steps)
    if not stages:
        raise ValueError("No distillation stages were generated.")

    timestamp = datetime.datetime.now().strftime("%b%d_%H%M%S")
    run_root = Path(output_root or Path("regression3_gpu") / "logs" / "progressive_distill")
    run_dir = run_root / f"pd_{timestamp}_from{N_big}_to{N_small}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Prepare teacher network and diffusion process
    teacher_model: BiDimensionalAttentionModel = build_network(cfg).to(device)
    state_dict = torch.load(teacher_path, map_location=device)
    teacher_model.load_state_dict(state_dict)
    teacher_model.eval()

    teacher_process = _build_process(cfg, N_big, device)

    # Determine per-stage optimisation steps
    if isinstance(steps_per_stage, Sequence):
        if len(steps_per_stage) != len(stages):
            raise ValueError("steps_per_stage sequence must match number of stages.")
        stage_iters = list(steps_per_stage)
    elif isinstance(steps_per_stage, int):
        stage_iters = [steps_per_stage] * len(stages)
    else:
        default_steps = max(1, cfg.total_steps // max(1, len(stages)))
        stage_iters = [default_steps] * len(stages)

    saved_dirs: List[Path] = []

    for stage_idx, stage in enumerate(stages):
        src_steps, dst_steps = stage.src_steps, stage.dst_steps
        ratio = src_steps // dst_steps
        # Each loop implements one "N <- N/2" iteration from Alg. 2 of progressive distillation.

        stage_dir = run_dir / f"{stage_idx:02d}_{src_steps}to{dst_steps}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Initialise student parameters from the current teacher (Alg. 2: theta <- eta).
        student_model = build_network(cfg).to(device)
        student_model.load_state_dict(teacher_model.state_dict())
        student_model.train()

        student_ema = build_network(cfg).to(device)
        student_ema.load_state_dict(teacher_model.state_dict())
        student_ema.eval()

        # Shrink the schedule (Alg. 2: halve N) by multiplying the teacher alphas across each block.
        student_betas = _coarse_grain_betas(teacher_process.betas, ratio).to(device)
        student_process = GaussianDiffusion(student_betas)

        loader_kwargs = {
            "batch_size": None,
            "num_workers": num_workers,
            "pin_memory": True,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        # Re-use the training data stream so each distillation step draws x ~ D.
        data_loader = DataLoader(
            InfiniteDataset(cfg, train=True),
            **loader_kwargs,
        )
        data_iter = iter(data_loader)

        optimiser = AdamW(
            student_model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=getattr(cfg.optimizer, "weight_decay", 0.0),
        )

        total_updates = stage_iters[stage_idx] # 128000
        warmup_steps = max(1, int(0.1 * total_updates))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

        alpha_bars = teacher_process.alpha_bars
        # Pick teacher timesteps that align with the coarser student grid (Alg. 2: choose i/N).
        block_ends = torch.arange(ratio - 1, src_steps, ratio, device=device, dtype=torch.long)
        noise_gen = torch.Generator(device=device).manual_seed(cfg.seed)
        index_gen = torch.Generator().manual_seed(cfg.seed + stage_idx + 17)

        running_loss = 0.0
        pbar = tqdm.trange(
            1,
            total_updates + 1,
            desc=f"[PD] {src_steps}->{dst_steps}",
            leave=False,
        )

        for step in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            _prepare_batch(batch, device)

            mask_target = getattr(batch, "mask_target", None)
            if mask_target is None:
                mask_target = torch.zeros(batch.y_target.shape[:2], device=device)
            mask_target = mask_target.to(batch.y_target.dtype)
            keep_mask = 1.0 - mask_target

            with torch.no_grad():
                idx = torch.randint(0, block_ends.numel(), (1,), generator=index_gen).item()
                t_idx = int(block_ends[idx].item())
                alpha_bar_t = alpha_bars[t_idx]

                noise = torch.randn(batch.y_target.shape, generator=noise_gen, device=device, dtype=batch.y_target.dtype)
                y_t = (
                    torch.sqrt(alpha_bar_t) * batch.y_target +
                    torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-12)) * noise
                )

                t_prime = max(0, t_idx - max(1, ratio // 2))
                t_dblprime = max(0, t_idx - ratio)

                def teacher_x0(x, y, t_tensor):
                    return teacher_model(
                        x,
                        y,
                        t_tensor,
                        None,
                        x_context=getattr(batch, "x_context", None),
                        y_context=getattr(batch, "y_context", None),
                        mask_context=getattr(batch, "mask_context", None),
                    )

                if t_prime < t_idx:
                    y_tprime = _ddim_step_x0(teacher_process, y_t, batch.x_target, t_idx, t_prime, teacher_x0)
                else:
                    y_tprime = y_t
                if t_dblprime < t_prime:
                    y_tdblprime = _ddim_step_x0(teacher_process, y_tprime, batch.x_target, t_prime, t_dblprime, teacher_x0)
                elif t_dblprime < t_idx:
                    y_tdblprime = _ddim_step_x0(teacher_process, y_t, batch.x_target, t_idx, t_dblprime, teacher_x0)
                else:
                    y_tdblprime = y_t

                a_t = torch.sqrt(alpha_bar_t)
                s_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-12))
                alpha_bar_dbl = alpha_bars[t_dblprime]
                a_d = torch.sqrt(alpha_bar_dbl)
                s_d = torch.sqrt(torch.clamp(1.0 - alpha_bar_dbl, min=1e-12))

                x_tilde = (y_tdblprime - (s_d / (s_t + 1e-12)) * y_t) / (a_d - (s_d / (s_t + 1e-12)) * a_t + 1e-12)

            optimiser.zero_grad(set_to_none=True)

            student_step_idx = t_idx // ratio
            t_student = torch.full(
                (batch.y_target.size(0),),
                float(student_step_idx),
                device=device,
                dtype=torch.float32,
            )

            x0_hat_student = student_model(
                batch.x_target,
                y_t,
                t_student,
                None,
                x_context=getattr(batch, "x_context", None),
                y_context=getattr(batch, "y_context", None),
                mask_context=getattr(batch, "mask_context", None),
            )

            if cfg.loss_type == "l1":
                loss_point = (x0_hat_student - x_tilde).abs().sum(-1)
            else:
                loss_point = ((x0_hat_student - x_tilde) ** 2).sum(-1)

            w = (alpha_bar_t / torch.clamp(1.0 - alpha_bar_t, min=1e-12)).to(loss_point.dtype)
            loss = (loss_point * w).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            _ema_update(student_ema, student_model, ema_decay)

            running_loss = 0.9 * running_loss + 0.1 * float(loss.item()) if step > 1 else float(loss.item())
            pbar.set_postfix(loss=f"{running_loss:.4f}", lr=scheduler.get_last_lr()[0])

        # Save checkpoints and simple metadata
        torch.save(student_model.state_dict(), stage_dir / "model.pt")
        torch.save(student_ema.state_dict(), stage_dir / "model_ema.pt")
        torch.save(student_process.betas.cpu(), stage_dir / "betas.pt")

        metadata = {
            "stage_index": stage_idx,
            "src_steps": src_steps,
            "dst_steps": dst_steps,
            "updates": total_updates,
            "ema_decay": ema_decay,
            "learning_rate": lr,
            "final_loss": running_loss,
        }
        with open(stage_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        saved_dirs.append(stage_dir)

        # Promote the EMA student to become the next teacher (Alg. 2: eta <- theta).
        teacher_model.load_state_dict(student_ema.state_dict())
        teacher_model.eval()
        teacher_process = student_process

    return saved_dirs


__all__ = ["train_pd"]


def _parse_int_list(arg: str | None) -> Optional[list[int]]:
    if not arg:
        return None
    # Accept comma-separated values, optionally wrapped in brackets/quotes.
    s = str(arg).strip().strip("[](){}'\"")
    parts = [p.strip() for p in s.split(
        ","
    ) if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


def main(
    *,
    teacher: str | Path,
    n_small: int,
    n_big: Optional[int] = None,
    schedule: Optional[list[int]] = None,
    steps_per_stage: Optional[list[int] | int] = None,
    output_root: str | Path = Path("progressive_distillation") / "logs" / "progressive_distill",
    ema_rate: Optional[float] = None,
    lr: Optional[float] = None,
    num_workers: int = 4,
):
    cfg = Config()
    N_big = n_big if n_big is not None else cfg.diffusion.timesteps
    N_small = int(n_small)

    saved = train_pd(
        cfg,
        teacher_dir=teacher,
        N_big=N_big,
        N_small=N_small,
        stage_steps=schedule,
        steps_per_stage=steps_per_stage,
        output_root=output_root,
        ema_rate=ema_rate,
        learning_rate=lr,
        num_workers=num_workers,
    )
    print("Saved stages under:")
    for d in saved:
        print(" -", d)



# NOTE: Avoid duplicate definitions; keep a single robust parser above.


def main():
    cfg = Config()

    parser = argparse.ArgumentParser(description="Progressive distillation (x0 targets)")
    parser.add_argument("--teacher", default="progressive_distillation/logs/Oct20_214857_brkx", help="Directory containing model_ema.pt or path to that file")
    parser.add_argument("--n-big", type=int, default=None, help="Teacher diffusion steps (default: cfg.diffusion.timesteps)")
    parser.add_argument("--n-small", type=int, default=1, help="Target student steps")
    parser.add_argument(
        "--schedule",
        type=str,
        default="[512, 256, 128, 64, 32, 16, 8, 1]",
        help="Explicit step schedule starting at N_big and ending at n-small. Accepts comma-separated list or bracketed form",
    )
    parser.add_argument("--steps-per-stage", type=str, default=cfg.total_steps,
                        help="Optim steps per stage: single int or comma-separated list per stage")
    parser.add_argument("--output-root", type=str, default=str(Path("progressive_distillation") / "logs" / "progressive_distill"),
                        help="Output root directory for distilled checkpoints")
    parser.add_argument("--ema-rate", type=float, default=None, help="EMA decay override (defaults to cfg.optimizer.ema_rate)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override (defaults to cfg.optimizer.peak_lr)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per stage")

    args = parser.parse_args()

    N_big = args.n_big if args.n_big is not None else cfg.diffusion.timesteps
    N_small = int(args.n_small)

    stage_steps = _parse_int_list(args.schedule)
    sps_list = _parse_int_list(args.steps_per_stage)
    steps_per_stage = sps_list[0] if (sps_list and len(sps_list) == 1) else sps_list

    saved = train_pd(
        cfg,
        teacher_dir=args.teacher,
        N_big=N_big,
        N_small=N_small,
        stage_steps=stage_steps,
        steps_per_stage=steps_per_stage,
        output_root=args.output_root,
        ema_rate=args.ema_rate,
        learning_rate=args.lr,
        num_workers=args.num_workers,
    )
    print("Saved stages under:")
    for d in saved:
        print(" -", d)


if __name__ == "__main__":
    main()

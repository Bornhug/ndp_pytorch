from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm.auto import tqdm

from .backends import get_backend
from .checkpoints import restore_training_checkpoint, save_training_checkpoint, torch_load
from .config import Config, MODELS
from .gp import sample_training_batch
from .runtime import ema_update, make_generator, move_batch, resolve_device, seed_everything


EXPECTED_PARAMETERS = 549_441


class InfiniteGPBatches(IterableDataset):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        generator = torch.Generator().manual_seed(self.config.training.seed + worker_id)
        while True:
            yield sample_training_batch(generator, self.config.data)


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    step: int
    checkpoint: Path
    ema_checkpoint: Path


def learning_rate(step: int, *, total_steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * step / float(max(1, warmup))
    fraction = min(1.0, (step - warmup) / float(max(1, total_steps - warmup)))
    return peak * 0.5 * (1.0 + math.cos(math.pi * fraction))


def build_optimizer(model: nn.Module, config: Config):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.peak_lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate(
            step,
            total_steps=config.total_steps,
            peak=config.optimizer.peak_lr,
            warmup=config.optimizer.warmup_steps,
        )
        / config.optimizer.peak_lr,
    )
    return optimizer, scheduler


def _loader(config: Config) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": InfiniteGPBatches(config),
        "batch_size": None,
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
    }
    if config.training.num_workers:
        kwargs["prefetch_factor"] = config.training.prefetch_factor
    return DataLoader(**kwargs)


def _finite(name: str, value: torch.Tensor, step: int) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError(f"Non-finite {name} at step {step}; update aborted")


def _trim_metrics(path: Path, checkpoint_step: int) -> None:
    if not path.exists():
        return
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        step = int(record["step"])
        if step <= checkpoint_step:
            records[step] = record
    path.write_text(
        "".join(json.dumps(records[step]) + "\n" for step in sorted(records)),
        encoding="utf-8",
    )


def train_model(
    model_name: str,
    *,
    run_dir: str | Path,
    device: str = "auto",
    max_steps: int | None = None,
) -> TrainResult:
    config = Config.for_model(model_name)
    target_device = resolve_device(device)
    seed_everything(config.training.seed)
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "config.json")

    backend = get_backend(model_name)
    model = backend.build_model(config, target_device)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError(f"{model_name} is not parameter matched")
    ema_model = copy.deepcopy(model).eval()
    process = backend.build_process(config, target_device)
    optimizer, scheduler = build_optimizer(model, config)
    training_generator = make_generator(target_device, config.training.seed)
    checkpoint = output / "checkpoint.pt"
    ema_checkpoint = output / "model_ema.pt"
    start_step = 0
    if checkpoint.exists():
        start_step = restore_training_checkpoint(
            checkpoint,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=target_device,
            training_generator=training_generator,
            expected_backend=model_name,
        )
    metrics_path = output / "metrics.jsonl"
    _trim_metrics(metrics_path, start_step)
    total_steps = config.total_steps if max_steps is None else min(max_steps, config.total_steps)
    if start_step >= total_steps:
        return TrainResult(output, start_step, checkpoint, ema_checkpoint)

    loader = iter(_loader(config))
    model.train()
    with metrics_path.open("a", encoding="utf-8") as metrics:
        progress = tqdm(range(start_step, total_steps), initial=start_step, total=total_steps)
        for zero_step in progress:
            step = zero_step + 1
            batch = move_batch(next(loader), target_device)
            optimizer.zero_grad(set_to_none=True)
            loss = backend.training_loss(
                model, process, batch, training_generator, config
            )
            _finite("loss", loss, step)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.grad_clip_norm
            )
            _finite("pre-clipping gradient norm", gradient_norm, step)
            optimizer.step()
            scheduler.step()
            ema_update(ema_model, model, config.optimizer.ema_rate)

            if step % config.training.log_every == 0 or step == total_steps:
                record = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gradient_norm": float(gradient_norm),
                }
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                progress.set_postfix(loss=f"{record['loss']:.4f}")

            if step % config.steps_per_epoch == 0 or step == total_steps:
                torch.save(ema_model.state_dict(), ema_checkpoint)
                save_training_checkpoint(
                    checkpoint,
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    config=config,
                    training_generator=training_generator,
                )
    return TrainResult(output, total_steps, checkpoint, ema_checkpoint)


def _complete_run(path: Path, model_name: str) -> bool:
    checkpoint = path / "checkpoint.pt"
    ema = path / "model_ema.pt"
    if not checkpoint.exists() or not ema.exists():
        return False
    payload = torch_load(checkpoint, map_location="cpu")
    return (
        isinstance(payload, dict)
        and payload.get("backend") == model_name
        and int(payload.get("step", -1)) == Config.for_model(model_name).total_steps
    )


def train_all(*, runs_root: str | Path, device: str = "auto", max_steps: int | None = None) -> list[TrainResult]:
    root = Path(runs_root)
    results = []
    for model_name in MODELS:
        run_dir = root / model_name
        if max_steps is None and _complete_run(run_dir, model_name):
            results.append(
                TrainResult(
                    run_dir,
                    Config.for_model(model_name).total_steps,
                    run_dir / "checkpoint.pt",
                    run_dir / "model_ema.pt",
                )
            )
            continue
        results.append(
            train_model(
                model_name,
                run_dir=run_dir,
                device=device,
                max_steps=max_steps,
            )
        )
    return results


from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import Config
from .models.ndp import convert_original_attention_state_dict


def torch_load(path: str | Path, *, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.4
        return torch.load(path, map_location=map_location)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_backend(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("backend"), str):
        return payload["backend"]
    config = payload.get("config")
    if isinstance(config, dict) and isinstance(config.get("backend"), str):
        return config["backend"]
    return None


def extract_state(payload: Any, *, prefer_ema: bool) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a mapping")
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    keys = ("ema_model", "model_ema", "model") if prefer_ema else ("model", "ema_model", "model_ema")
    for key in keys:
        state = payload.get(key)
        if isinstance(state, dict):
            return state
    raise ValueError("Checkpoint has no model state")


def load_model_state(
    model: nn.Module,
    payload: Any,
    *,
    expected_backend: str,
    prefer_ema: bool,
) -> None:
    actual = checkpoint_backend(payload)
    if actual is not None and actual != expected_backend:
        raise ValueError(
            f"Checkpoint backend '{actual}' cannot load as '{expected_backend}'"
        )
    state = extract_state(payload, prefer_ema=prefer_ema)
    if expected_backend in {"ndp_cond", "ndp_uncond"}:
        state = convert_original_attention_state_dict(state)
    model.load_state_dict(state, strict=True)


def config_from_checkpoint(payload: Any, *, expected_backend: str) -> Config:
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        return Config.for_model(expected_backend)
    return Config.from_dict(payload["config"], backend=expected_backend)


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    config: Config,
    training_generator: torch.Generator,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "backend": config.backend,
            "step": int(step),
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.to_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "training_generator_state": training_generator.get_state(),
        },
        path,
    )
    return path


def restore_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    training_generator: torch.Generator,
    expected_backend: str,
) -> int:
    payload = torch_load(path, map_location=device)
    if not isinstance(payload, dict) or "optimizer" not in payload:
        raise ValueError("Resume requires a complete training checkpoint")
    load_model_state(
        model, payload, expected_backend=expected_backend, prefer_ema=False
    )
    load_model_state(
        ema_model, payload, expected_backend=expected_backend, prefer_ema=True
    )
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if payload.get("cpu_rng_state") is not None:
        torch.set_rng_state(payload["cpu_rng_state"].cpu())
    if device.type == "cuda" and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    if payload.get("training_generator_state") is not None:
        training_generator.set_state(payload["training_generator_state"])
    return int(payload.get("step", 0))


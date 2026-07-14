from __future__ import annotations

import hashlib
import random

import numpy as np
import torch
from torch import nn

from .types import RegressionBatch


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if requested and requested != "auto":
        device = torch.device(requested)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(device: torch.device | str, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


class SampleMajorGenerator:
    """Counter-based batched noise invariant to vectorized chunk size."""

    def __init__(
        self,
        device: torch.device | str,
        base_seed: int,
        total: int,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.base_seed = int(base_seed)
        self.total = int(total)
        self.start = int(start)
        self.end = self.total if end is None else int(end)
        self.call_index = 0

    @classmethod
    def from_base_seed(
        cls,
        device: torch.device | str,
        base_seed: int,
        count: int,
    ) -> "SampleMajorGenerator":
        return cls(device, base_seed, count)

    def subset(self, start: int, end: int) -> "SampleMajorGenerator":
        return SampleMajorGenerator(
            self.device,
            self.base_seed,
            self.total,
            start=start,
            end=end,
        )

    def randn_like(self, reference: torch.Tensor) -> torch.Tensor:
        if reference.shape[0] != self.end - self.start:
            raise ValueError("Sample-major RNG range does not match tensor batch size")
        generator = make_generator(
            self.device,
            derived_seed(self.base_seed, f"sampling_draw:{self.call_index}"),
        )
        self.call_index += 1
        full = torch.randn(
            (self.total, *reference.shape[1:]),
            generator=generator,
            device=reference.device,
            dtype=reference.dtype,
        )
        return full[self.start : self.end]

    def get_state(self) -> int:
        return self.call_index

    def set_state(self, state: int) -> None:
        self.call_index = int(state)


def derived_seed(base_seed: int, stream: str) -> int:
    raw = f"{base_seed}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


@torch.no_grad()
def ema_update(ema: nn.Module, online: nn.Module, decay: float) -> None:
    for ema_parameter, online_parameter in zip(ema.parameters(), online.parameters()):
        ema_parameter.mul_(decay).add_(online_parameter, alpha=1.0 - decay)
    for ema_buffer, online_buffer in zip(ema.buffers(), online.buffers()):
        ema_buffer.copy_(online_buffer)


def move_batch(batch: RegressionBatch, device: torch.device) -> RegressionBatch:
    return batch.to(device, non_blocking=True)

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RegressionBatch:
    x_target: Tensor
    y_target: Tensor
    x_context: Tensor
    y_context: Tensor
    mask_target: Tensor
    mask_context: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.x_target.shape[0])

    def validate(self) -> None:
        batch, targets = self.x_target.shape[:2]
        if self.y_target.shape[:2] != (batch, targets):
            raise ValueError("Target input/value shapes differ")
        if self.mask_target.shape != (batch, targets):
            raise ValueError("Target mask shape is invalid")
        if self.x_context.shape[:2] != self.y_context.shape[:2]:
            raise ValueError("Context input/value shapes differ")
        if self.mask_context.shape != self.x_context.shape[:2]:
            raise ValueError("Context mask shape is invalid")

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "RegressionBatch":
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device, non_blocking=non_blocking))
        return self


@dataclass(frozen=True)
class ConditioningSet:
    x_context: Tensor
    y_context: Tensor
    mask_context: Tensor | None = None


@dataclass(frozen=True)
class SamplingRequest:
    sampler: str
    num_steps: int
    num_samples: int
    seed: int
    y_dim: int = 1
    batch_size: int | None = None
    rng_semantics: str = "stable"

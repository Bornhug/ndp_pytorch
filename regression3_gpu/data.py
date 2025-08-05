# ===== data.py =====
"""PyTorch re-implementation of the JAX data utilities used in the original
neural-diffusion-processes repository.  The public API is unchanged so that the
rest of the code-base can keep importing ``data.get_batch`` etc.

Key bug‑fixes 2025‑08‑03
-----------------------
* _Uniform.sample no longer calls ``torch.distributions.Uniform.sample`` with a
  ``generator`` kwarg (not supported on some PyTorch versions).  Sampling is
  done explicitly via ``torch.rand`` so it works on any ≥1.9 install.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import math
import torch
from torch import Tensor
import numpy as np

__all__ = [
    "DATASETS",
    "TASKS",
    "get_batch",
    "_DATASET_CONFIGS",
    "_TASK_CONFIGS",
]

# ------------------------------------------------------------------
#  Compatibility helper – honours Generator on all PyTorch builds
# ------------------------------------------------------------------
def _randn_like(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Draw N(0,1) with the same shape/device/dtype as `x`, using `g`."""
    try:                                 # preferred path (newer wheels)
        return torch.randn_like(x, generator=g)
    except TypeError:                    # fallback (e.g. Win + 2.3.0)
        return torch.randn(
            x.shape, dtype=x.dtype, device=x.device, generator=g
        )


# -----------------------------------------------------------------------------
#                           Helper distributions
# -----------------------------------------------------------------------------

@dataclass
class UniformDiscrete:
    """Inclusive discrete uniform distribution ``[lower, upper]``."""

    lower: int
    upper: int

    def sample(self, g: torch.Generator, shape: Tuple[int, ...]) -> Tensor:
        if self.lower == self.upper:
            return torch.full(shape, self.lower, dtype=torch.int32)
        return torch.randint(self.lower, self.upper + 1, shape, generator=g)


class _Uniform:
    """Continuous uniform helper (works irrespective of PyTorch version)."""

    def __init__(self, low: float, high: float):
        self.low, self.high = float(low), float(high)

    def sample(
        self, *, generator: torch.Generator, sample_shape: Tuple[int, ...]
    ) -> Tensor:
        # Explicit sampling because torch.distributions.Uniform.sample(*, generator)
        # is only available in recent PyTorch versions.
        return self.low + (self.high - self.low) * torch.rand(*sample_shape, generator=generator)


# -----------------------------------------------------------------------------
#                               Public constants
# -----------------------------------------------------------------------------

DATASETS = ["se", "matern", "sawtooth", "step"]
TASKS = ["training", "interpolation"]

# -----------------------------------------------------------------------------
#                                 Configs
# -----------------------------------------------------------------------------

@dataclass
class TaskConfig:
    x_context_dist: _Uniform
    x_target_dist: _Uniform


@dataclass
class DatasetConfig:
    max_input_dim: int  # (inclusive)
    is_gp: bool
    eval_num_target: UniformDiscrete = UniformDiscrete(50, 50)
    eval_num_context: UniformDiscrete = UniformDiscrete(1, 10)


_NOISE_VAR = 0.05 ** 2
_KERNEL_VAR = 1.0
_LENGTHSCALE = 0.25
_JITTER = 1e-6

_DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "se": DatasetConfig(max_input_dim=3, is_gp=True),
    "matern": DatasetConfig(max_input_dim=3, is_gp=True),
    "sawtooth": DatasetConfig(max_input_dim=1, is_gp=False),
    "step": DatasetConfig(max_input_dim=1, is_gp=False),
}

_TASK_CONFIGS: Dict[str, TaskConfig] = {
    "training": TaskConfig(_Uniform(-2.0, 2.0), _Uniform(-2.0, 2.0)),
    "interpolation": TaskConfig(_Uniform(-2.0, 2.0), _Uniform(-2.0, 2.0)),
}

# -----------------------------------------------------------------------------
#                         Functional distribution base
# -----------------------------------------------------------------------------

class FunctionalDistribution:
    """Base class – each dataset must implement ``sample``."""

    def sample(self, g: torch.Generator, x: Tensor) -> Tensor:  # (N, D) -> (N, 1)
        raise NotImplementedError


# =========== Gaussian‑process helpers ===========

def _rbf_kernel(x1: Tensor, x2: Tensor, lengthscale: float, variance: float) -> Tensor:
    diff = x1[:, None, :] - x2[None, :, :]
    sqdist = (diff ** 2).sum(-1)
    return variance * torch.exp(-0.5 * sqdist / (lengthscale ** 2))


def _matern52_kernel(x1: Tensor, x2: Tensor, lengthscale: float, variance: float) -> Tensor:
    sqrt5 = math.sqrt(5.0)
    diff = ((x1[:, None, :] - x2[None, :, :]) ** 2).sum(-1).sqrt()
    r = diff / lengthscale
    return variance * (1 + sqrt5 * r + 5.0 / 3.0 * r ** 2) * torch.exp(-sqrt5 * r)


import inspect

class GPFunctionalDistribution(FunctionalDistribution):
    """Gaussian-process sampler that works on every PyTorch version."""

    def __init__(self, kernel_fn):
        self.kernel_fn = kernel_fn

    # -------- internal helper ------------------------------------------------
    @staticmethod
    def _rsample(mvn: torch.distributions.MultivariateNormal,
                 g: torch.Generator) -> Tensor:
        # Fast path for builds whose rsample already takes generator=
        if "generator" in inspect.signature(mvn.rsample).parameters:
            return mvn.rsample(generator=g)

        # ---------- manual fallback (all older / Windows builds) --------------
        L = torch.linalg.cholesky(mvn.covariance_matrix)  # (N, N)

        # --- inside GPFunctionalDistribution._rsample ---------------------
        try:
            z = torch.randn_like(mvn.mean, generator=g)
        except TypeError:
            z = torch.randn(
                mvn.mean.shape, dtype=mvn.mean.dtype, device=mvn.mean.device, generator=g
            )

        #  (replace the block above with)  z = _randn_like(mvn.mean, g)

        return mvn.mean + L @ z

    # -------- public API -----------------------------------------------------
    def sample(self, g: torch.Generator, x: Tensor) -> Tensor:
        n = x.size(0)
        K = self.kernel_fn(x, x) + _JITTER * torch.eye(n, device=x.device, dtype=x.dtype)
        mvn = torch.distributions.MultivariateNormal(
            torch.zeros(n, dtype=x.dtype, device=x.device), K
        )
        # --- inside GPFunctionalDistribution.sample -----------------------
        f = self._rsample(mvn, g)
        y = f + math.sqrt(_NOISE_VAR) * _randn_like(f, g)

        return y.unsqueeze(-1)



# -----------------------------------------------------------------------------
#                         Dataset factory registry
# -----------------------------------------------------------------------------

DatasetFactory = Callable[[List[int]], FunctionalDistribution]
_DATASET_FACTORIES: Dict[str, DatasetFactory] = {}


def register_dataset_factory(name: str):
    def decorator(fn: DatasetFactory):
        _DATASET_FACTORIES[name] = fn
        return fn

    return decorator


@register_dataset_factory("se")
def _se_dataset_factory(active_dims: List[int]):
    factor = math.sqrt(len(active_dims))

    def k(a: Tensor, b: Tensor):
        return _rbf_kernel(a[:, active_dims], b[:, active_dims], _LENGTHSCALE * factor, _KERNEL_VAR)

    return GPFunctionalDistribution(k)


@register_dataset_factory("matern")
def _matern_dataset_factory(active_dims: List[int]):
    factor = math.sqrt(len(active_dims))

    def k(a: Tensor, b: Tensor):
        return _matern52_kernel(a[:, active_dims], b[:, active_dims], _LENGTHSCALE * factor, _KERNEL_VAR)

    return GPFunctionalDistribution(k)


class Sawtooth(FunctionalDistribution):
    A = 1.0
    K_max = 20
    mean = 0.5
    variance = 0.07965

    def sample(self, g: torch.Generator, x: Tensor) -> Tensor:
        f = 3.0 + 2.0 * torch.rand((), generator=g, device=x.device, dtype=x.dtype)
        s = -5.0 + 10.0 * torch.rand((), generator=g, device=x.device, dtype=x.dtype)
        ks = torch.arange(1, self.K_max + 1, dtype=x.dtype, device=x.device)[None, :]
        vals = (-1.0) ** ks * torch.sin(2 * math.pi * ks * f * (x - s)) / ks
        k = torch.randint(10, self.K_max + 1, (), generator=g)
        mask = (ks < k).float()
        fs = self.A / 2 + self.A / math.pi * (vals * mask).sum(dim=1, keepdim=True)
        fs = fs - self.mean
        return fs


@register_dataset_factory("sawtooth")
def _sawtooth_dataset_factory(*args):
    return Sawtooth()


class Step(FunctionalDistribution):
    def sample(self, g: torch.Generator, x: Tensor) -> Tensor:
        s = -2.0 + 4.0 * torch.rand((), generator=g, device=x.device, dtype=x.dtype)
        return torch.where(x < s, torch.zeros_like(x), torch.ones_like(x))


@register_dataset_factory("step")
def _step_dataset_factory(*args):
    return Step()


# -----------------------------------------------------------------------------
#                               Batch structure
# -----------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class Batch:
    x_target: Tensor
    y_target: Tensor
    x_context: Tensor
    y_context: Tensor
    mask_target: Tensor
    mask_context: Tensor


# -----------------------------------------------------------------------------
#                            Main batch‑sampling API
# -----------------------------------------------------------------------------

def get_batch(
    g: torch.Generator,
    *,
    batch_size: int,
    name: str,
    task: str,
    input_dim: int,
    device: torch.device | str | None = None,
) -> Batch:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")

    cfg = _DATASET_CONFIGS[name]
    if input_dim > cfg.max_input_dim:
        raise ValueError(f"input_dim {input_dim} > max_input_dim {cfg.max_input_dim} for {name}")

    # Numbers of points --------------------------------------------------------
    if task == "training":
        n_target = cfg.eval_num_target.sample(g, (1,)).item()
        n_context = 0
    else:  # interpolation
        n_target = cfg.eval_num_target.upper
        n_context = cfg.eval_num_context.sample(g, (1,)).item()

    # Inputs -------------------------------------------------------------------
    x_context = _TASK_CONFIGS[task].x_context_dist.sample(
        generator=g, sample_shape=(batch_size, n_context, input_dim)
    )
    x_target = _TASK_CONFIGS[task].x_target_dist.sample(
        generator=g, sample_shape=(batch_size, n_target, input_dim)
    )
    x_all = torch.cat([x_context, x_target], dim=1)

    # Masks --------------------------------------------------------------------
    mask_context = torch.zeros(batch_size, n_context, dtype=torch.float32)
    mask_target = torch.zeros(batch_size, n_target, dtype=torch.float32)

    # Outputs ------------------------------------------------------------------
    active_dims = list(range(input_dim))
    sample_fn = _DATASET_FACTORIES[name](active_dims).sample
    y_all = torch.stack([sample_fn(g, x_all[b]) for b in range(batch_size)], dim=0)
    y_context, y_target = y_all[:, :n_context], y_all[:, n_context:]

    if device is not None:
        x_context, y_context, x_target, y_target = (
            x_context.to(device),
            y_context.to(device),
            x_target.to(device),
            y_target.to(device),
        )
        mask_context = mask_context.to(device)
        mask_target = mask_target.to(device)

    return Batch(
        x_target=x_target,
        y_target=y_target,
        x_context=x_context,
        y_context=y_context,
        mask_target=mask_target,
        mask_context=mask_context,
    )



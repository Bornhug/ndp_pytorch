# ===== data.py =====
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import math
import torch
from torch import Tensor
import inspect

__all__ = [
    "DATASETS",
    "TASKS",
    "get_batch",
    "_DATASET_CONFIGS",
    "_TASK_CONFIGS",
]

# ---------------- Compatibility helpers ----------------

def _randn_like(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    try:
        return torch.randn_like(x, generator=g)
    except TypeError:
        return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=g)

def _rand_like(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    try:
        return torch.rand_like(x, generator=g)
    except TypeError:
        return torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=g)

# ---------------- Small distributions ------------------

@dataclass
class UniformDiscrete:
    low: int
    high: int
    def sample(self, shape: Tuple[int, ...], *, g: torch.Generator) -> Tensor:
        if self.low == self.high:
            return torch.full(shape, self.low, dtype=torch.int64)
        return torch.randint(self.low, self.high + 1, shape, generator=g)

@dataclass
class _Uniform:
    low: float
    high: float
    def sample(self, sample_shape: Tuple[int, ...], *, generator: torch.Generator) -> Tensor:
        return self.low + (self.high - self.low) * torch.rand(*sample_shape, generator=generator)

# ---------------- Public constants ---------------------

DATASETS = ["se", "matern", "sawtooth", "step"]
TASKS = ["training", "interpolation"]

# ---------------- Configs ------------------------------

@dataclass
class TaskConfig:
    x_context_dist: _Uniform
    x_target_dist: _Uniform

@dataclass
class DatasetConfig:
    max_input_dim: int
    is_gp: bool
    eval_num_target: UniformDiscrete = UniformDiscrete(50, 50)
    eval_num_context: UniformDiscrete = UniformDiscrete(1, 10)

_NOISE_VAR   = 0.05 ** 2
_KERNEL_VAR  = 1.0
_LENGTHSCALE = 0.25
_JITTER      = 1e-6

_DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "se":     DatasetConfig(max_input_dim=3, is_gp=True),
    "matern": DatasetConfig(max_input_dim=3, is_gp=True),
    "sawtooth": DatasetConfig(max_input_dim=1, is_gp=False),
    "step":     DatasetConfig(max_input_dim=1, is_gp=False),
}

_TASK_CONFIGS: Dict[str, TaskConfig] = {
    "training":      TaskConfig(_Uniform(-2.0, 2.0), _Uniform(-2.0, 2.0)),
    "interpolation": TaskConfig(_Uniform(-2.0, 2.0), _Uniform(-2.0, 2.0)),
}

# Light defaults for training sizes (kept local to avoid changing config API)
TRAIN_NUM_TARGET  = UniformDiscrete(32, 64)
TRAIN_NUM_CONTEXT = UniformDiscrete(0, 32)

# ---------------- GP helpers ---------------------------

class FunctionalDistribution:
    def sample(self, x: Tensor, g: torch.Generator) -> Tensor:  # (N,D) -> (N,1)
        raise NotImplementedError

def _rbf_kernel(x1: Tensor, x2: Tensor, lengthscale: float, variance: float) -> Tensor:
    diff = x1[:, None, :] - x2[None, :, :]
    sqdist = (diff ** 2).sum(-1)
    return variance * torch.exp(-0.5 * sqdist / (lengthscale ** 2))

def _matern52_kernel(x1: Tensor, x2: Tensor, lengthscale: float, variance: float) -> Tensor:
    sqrt5 = math.sqrt(5.0)
    r = ((x1[:, None, :] - x2[None, :, :]) ** 2).sum(-1).sqrt() / lengthscale
    return variance * (1 + sqrt5 * r + 5.0/3.0 * r**2) * torch.exp(-sqrt5 * r)

class GPFunctionalDistribution(FunctionalDistribution):
    def __init__(self, kernel_fn):
        self.kernel_fn = kernel_fn

    @staticmethod
    def _rsample(mvn: torch.distributions.MultivariateNormal, g: torch.Generator) -> Tensor:
        if "generator" in inspect.signature(mvn.rsample).parameters:
            return mvn.rsample(generator=g)
        L = torch.linalg.cholesky(mvn.covariance_matrix)  # (N,N), lower
        z = _randn_like(mvn.mean, g=g)                    # (N,)
        return mvn.mean + L @ z

    def sample(self, x: Tensor, g: torch.Generator) -> Tensor:
        n = x.size(0)
        K = self.kernel_fn(x, x) + _JITTER * torch.eye(n, device=x.device, dtype=x.dtype)
        mvn = torch.distributions.MultivariateNormal(
            torch.zeros(n, dtype=x.dtype, device=x.device), K
        )
        f = self._rsample(mvn, g)                                   # (N,)
        y = f + math.sqrt(_NOISE_VAR) * _randn_like(f, g)           # (N,)
        return y.unsqueeze(-1)                                      # (N,1)

# --------------- Dataset factory registry ---------------

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
    def sample(self, x: Tensor, g: torch.Generator) -> Tensor:
        # Use first dim only; return (N,1)
        x1 = x[..., 0:1]
        f = 3.0 + 2.0 * torch.rand((), generator=g, device=x.device, dtype=x.dtype)
        s = -5.0 + 10.0 * torch.rand((), generator=g, device=x.device, dtype=x.dtype)
        ks = torch.arange(1, self.K_max + 1, dtype=x.dtype, device=x.device)[None, :]
        vals = (-1.0) ** ks * torch.sin(2 * math.pi * ks * f * (x1 - s)) / ks
        k = torch.randint(10, self.K_max + 1, (), generator=g)
        mask = (ks < k).float()
        fs = self.A / 2 + self.A / math.pi * (vals * mask).sum(dim=1, keepdim=True)
        fs = fs - self.mean
        return fs  # (N,1)

@register_dataset_factory("sawtooth")
def _sawtooth_dataset_factory(*args):
    return Sawtooth()

class Step(FunctionalDistribution):
    def sample(self, x: Tensor, g: torch.Generator) -> Tensor:
        # Use first dim only; return (N,1)
        x1 = x[..., 0:1]
        s = -2.0 + 4.0 * torch.rand((), generator=g, device=x.device, dtype=x.dtype)
        return torch.where(x1 < s, torch.zeros_like(x1), torch.ones_like(x1))  # (N,1)

@register_dataset_factory("step")
def _step_dataset_factory(*args):
    return Step()

# ---------------- Batch structure -----------------------

@dataclass
class Batch:
    x_target: Tensor
    y_target: Tensor
    x_context: Tensor
    y_context: Tensor
    mask_target: Tensor
    mask_context: Tensor

# ---------------- Main API ------------------------------

def get_batch(
    g: torch.Generator,
    *,
    batch_size: int,
    name: str,  # "se" | "matern" | "sawtooth" | "step"
    task: str,  # "training" | "interpolation"
    input_dim: int,
    device: torch.device | str | None = None,
    gp_conditional_targets: bool = False,
    p_drop_ctx: float = 0.0,
) -> Batch:
    if name not in DATASETS: raise ValueError(f"Unknown dataset: {name}")
    if task not in TASKS:    raise ValueError(f"Unknown task: {task}")

    cfg = _DATASET_CONFIGS[name]
    if input_dim > cfg.max_input_dim:
        raise ValueError(f"input_dim {input_dim} > max_input_dim {cfg.max_input_dim} for {name}")

    # Sizes
    if task == "training":
        # Training: N ~ Uniform{32…64}, M ~ Uniform{0…32}
        n_target  = int(TRAIN_NUM_TARGET.sample((1,), g=g).item())
        n_context = int(TRAIN_NUM_CONTEXT.sample((1,), g=g).item())
    else:  # interpolation
        # N fixed to eval_num_target.high = 50, M~ Uniform{1…10}.
        n_target  = int(cfg.eval_num_target.high)
        n_context = int(cfg.eval_num_context.sample((1,), g=g).item())

    # Inputs
    x_context = _TASK_CONFIGS[task].x_context_dist.sample(
        sample_shape=(batch_size, n_context, input_dim), generator=g
    )
    x_target = _TASK_CONFIGS[task].x_target_dist.sample(
        sample_shape=(batch_size, n_target, input_dim), generator=g
    )
    x_all = torch.cat([x_context, x_target], dim=1)

    # Build y_all, then split
    active_dims = list(range(input_dim))
    dataset_factory = _DATASET_FACTORIES[name]
    function_distribution = dataset_factory(active_dims)
    y_all = torch.stack([function_distribution.sample(x_all[b], g) for b in range(batch_size)], dim=0)
    y_context, y_target = y_all[:, :n_context], y_all[:, n_context:]

    # Optional GP conditional resampling of targets
    if gp_conditional_targets and _DATASET_CONFIGS[name].is_gp and hasattr(function_distribution, "kernel_fn"):
        K_fn = function_distribution.kernel_fn
        new_targets = []
        for b in range(batch_size):
            x_c = x_context[b]  # [M,D]
            y_c = y_context[b].squeeze(-1)  # [M]
            x_t = x_target[b]   # [N,D]
            M = x_c.size(0); N = x_t.size(0)
            if M == 0:
                Kxx = K_fn(x_t, x_t) + (_JITTER) * torch.eye(N, device=x_t.device, dtype=x_t.dtype)
                L = torch.linalg.cholesky(Kxx)
                z = torch.randn(N, 1, device=x_t.device, dtype=x_t.dtype, generator=g)
                y_samp = L @ z
                new_targets.append(y_samp)
                continue
            Kcc = K_fn(x_c, x_c) + (_NOISE_VAR + _JITTER) * torch.eye(M, device=x_c.device, dtype=x_c.dtype)
            Kxc = K_fn(x_t, x_c)
            Kxx = K_fn(x_t, x_t) + _JITTER * torch.eye(N, device=x_t.device, dtype=x_t.dtype)
            Lc = torch.linalg.cholesky(Kcc)
            alpha = torch.cholesky_solve(y_c.unsqueeze(-1), Lc)  # (M,1)
            mu = Kxc @ alpha                                     # (N,1)
            v = torch.cholesky_solve(Kxc.T, Lc)                  # (M,N)
            cov = Kxx - Kxc @ v                                  # (N,N)
            L = torch.linalg.cholesky(cov + _JITTER * torch.eye(N, device=cov.device, dtype=cov.dtype))
            z = torch.randn(N, 1, device=x_t.device, dtype=x_t.dtype, generator=g)
            y_samp = mu + L @ z
            new_targets.append(y_samp)
        y_target = torch.stack(new_targets, dim=0) # Posterior y_target

    # Masks (1 = missing/padded)
    mask_context = torch.zeros(batch_size, n_context, dtype=torch.float32)
    mask_target  = torch.zeros(batch_size, n_target,  dtype=torch.float32)

    # Classifier-free style drop AFTER targets are formed
    if p_drop_ctx > 0.0 and n_context > 0:
        drop = torch.rand(batch_size, generator=g) < p_drop_ctx
        for b in range(batch_size):
            if drop[b]:
                mask_context[b].fill_(1.0)

        # zero-out masked positions with proper broadcast
        mc = mask_context.bool()[..., None]         # [B,M,1]
        x_context = x_context.masked_fill(mc, 0.0)
        y_context = y_context.masked_fill(mc, 0.0)

    if device is not None:
        x_context, y_context, x_target, y_target = (
            x_context.to(device),
            y_context.to(device),
            x_target.to(device),
            y_target.to(device),
        )
        mask_context = mask_context.to(device)
        mask_target  = mask_target.to(device)

    return Batch(
        x_target=x_target,
        y_target=y_target,
        x_context=x_context,
        y_context=y_context,
        mask_target=mask_target,
        mask_context=mask_context,
    )

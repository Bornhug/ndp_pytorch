# gp_likelihood_core.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

# Import the EXACT constants & kernels used by your dataset
from data import (
    _LENGTHSCALE, _KERNEL_VAR, _NOISE_VAR, _JITTER,
    _rbf_kernel, _matern52_kernel,
)

# --------------------- numerics ---------------------

def _stable_cholesky(K: Tensor, jitter: float = _JITTER, max_tries: int = 5) -> Tensor:
    eye = torch.eye(K.size(0), device=K.device, dtype=K.dtype)
    base = jitter * K.diag().median().clamp_min(1.0).item()
    err = None
    for i in range(max_tries):
        try:
            return torch.linalg.cholesky(K + (10.0**i) * base * eye)
        except RuntimeError as e:
            err = e
    raise err

@dataclass
class GPHypersFixed:
    ell_eff: float   # effective lengthscale after √D scaling
    var: float       # kernel variance (signal)
    noise: float     # observation noise variance


# ----------------- fixed-kernel builders -----------------

def _kernel_closure(dataset: str, active_dims: List[int]):
    """Return k(a,b) using the *same* kernel & scaling as data.py."""
    D = len(active_dims)
    scale = math.sqrt(D) if D > 0 else 1.0
    ell_eff = float(_LENGTHSCALE * scale)   # matches data.py scaling
    var = float(_KERNEL_VAR)
    noise = float(_NOISE_VAR)

    if dataset == "se":
        def k(a: Tensor, b: Tensor) -> Tensor:
            return _rbf_kernel(a[:, active_dims], b[:, active_dims], ell_eff, var)
    elif dataset == "matern":
        def k(a: Tensor, b: Tensor) -> Tensor:
            return _matern52_kernel(a[:, active_dims], b[:, active_dims], ell_eff, var)
    else:
        raise ValueError(f"GP evaluator only supports datasets 'se' or 'matern', got '{dataset}'.")

    return k, GPHypersFixed(ell_eff=ell_eff, var=var, noise=noise)


# ----------------- posterior & scoring -----------------

@torch.no_grad()
def gp_posterior_fixed(
    dataset: str,
    x_ctx: Tensor, y_ctx: Tensor,
    x_tgt: Tensor,
    active_dims: List[int],
    include_obs_noise: bool = True,
) -> Tuple[Tensor, Tensor, GPHypersFixed]:
    """
    Posterior of y(x_tgt) | (x_ctx,y_ctx) under the *fixed* GP used by data.py.
    Returns (mean:[N], cov:[N,N], hypers).
    """
    # float64 for stability
    xC, yC, xT = x_ctx.double(), y_ctx.double(), x_tgt.double()
    k, θ = _kernel_closure(dataset, active_dims)

    Kdd = k(xC, xC) + θ.noise * torch.eye(xC.size(0), device=xC.device, dtype=xC.dtype)
    KdT = k(xC, xT)
    KTd = KdT.transpose(0, 1)
    KTT = k(xT, xT)

    L = _stable_cholesky(Kdd)
    alpha = torch.cholesky_solve(yC.view(-1, 1), L)     # [M,1]
    m = (KTd @ alpha).view(-1)                          # [N]

    v = torch.cholesky_solve(KdT, L)                    # [M,N]
    S = KTT - (KTd @ v)                                 # [N,N]
    if include_obs_noise:
        S = S + θ.noise * torch.eye(S.size(0), device=S.device, dtype=S.dtype)
    return m, S, θ

def mvn_loglik(y: Tensor, m: Tensor, S: Tensor) -> Tensor:
    """Log N(y; m, S) via Cholesky. Returns scalar tensor (float64)."""
    y, m, S = y.double(), m.double(), S.double()
    L = _stable_cholesky(S)
    diff = (y - m).view(-1, 1)
    alpha = torch.cholesky_solve(diff, L)
    quad = (diff.transpose(0, 1) @ alpha).squeeze()
    logdet = 2.0 * torch.log(torch.diag(L)).sum()
    N = y.numel()
    return -0.5 * (quad + logdet + N * math.log(2 * math.pi))

def mahalanobis2(y: Tensor, m: Tensor, S: Tensor) -> Tensor:
    """Mahalanobis^2 = ||S^{-1/2}(y-m)||^2. χ^2_N under the correct GP."""
    y, m, S = y.double(), m.double(), S.double()
    L = _stable_cholesky(S)
    v = torch.linalg.solve(L, (y - m).view(-1, 1))
    return (v.squeeze()**2).sum()

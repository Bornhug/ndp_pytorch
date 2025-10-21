from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch import Tensor
from scipy.integrate import solve_ivp

# Import the EXACT constants & kernels used by your dataset (from PD package copy)
from regression.data import (
    _LENGTHSCALE, _KERNEL_VAR, _NOISE_VAR, _JITTER,
    _rbf_kernel, _matern52_kernel,
)


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


def _kernel_closure(dataset: str, active_dims: List[int]):
    """Return k(a,b) using the same kernel & scaling as data.py."""
    D = len(active_dims)
    scale = math.sqrt(D) if D > 0 else 1.0
    ell_eff = float(_LENGTHSCALE * scale)
    var = float(_KERNEL_VAR)
    noise = float(_NOISE_VAR)

    if dataset == "se":
        def k(a: Tensor, b: Tensor) -> Tensor:
            return _rbf_kernel(a[:, active_dims], b[:, active_dims], ell_eff, var)
    elif dataset == "matern":
        def k(a: Tensor, b: Tensor) -> Tensor:
            return _matern52_kernel(a[:, active_dims], b[:, active_dims], ell_eff, var)
    else:
        raise ValueError(f"GP evaluator supports only 'se' or 'matern', got '{dataset}'.")

    return k, GPHypersFixed(ell_eff=ell_eff, var=var, noise=noise)


@torch.no_grad()
def gp_prior_fixed(
    dataset: str,
    x_tgt: Tensor,
    active_dims: List[int],
    include_obs_noise: bool = True,
) -> Tuple[Tensor, Tensor, GPHypersFixed]:
    xT = x_tgt.double()
    k, θ = _kernel_closure(dataset, active_dims)
    S = k(xT, xT)
    if include_obs_noise:
        S = S + θ.noise * torch.eye(S.size(0), dtype=S.dtype, device=S.device)
    m = torch.zeros(xT.size(0), dtype=S.dtype, device=S.device)
    return m, S, θ


@torch.no_grad()
def gp_posterior_fixed(
    dataset: str,
    x_ctx: Tensor, y_ctx: Tensor,
    x_tgt: Tensor,
    active_dims: List[int],
    include_obs_noise: bool = True,
) -> Tuple[Tensor, Tensor, GPHypersFixed]:
    xC, yC, xT = x_ctx.double(), y_ctx.double(), x_tgt.double()
    k, θ = _kernel_closure(dataset, active_dims)

    Kdd = k(xC, xC) + θ.noise * torch.eye(xC.size(0), device=xC.device, dtype=xC.dtype)
    KdT = k(xC, xT)
    KTd = KdT.transpose(0, 1)
    KTT = k(xT, xT)

    L = _stable_cholesky(Kdd)
    alpha = torch.cholesky_solve(yC.view(-1, 1), L)
    m = (KTd @ alpha).view(-1)

    v = torch.cholesky_solve(KdT, L)
    S = KTT - (KTd @ v)
    if include_obs_noise:
        S = S + θ.noise * torch.eye(S.size(0), device=S.device, dtype=S.dtype)
    return m, S, θ


def mvn_loglik(y: Tensor, m: Tensor, S: Tensor) -> Tensor:
    y, m, S = y.double(), m.double(), S.double()
    L = _stable_cholesky(S)
    diff = (y - m).view(-1, 1)
    alpha = torch.cholesky_solve(diff, L)
    quad = (diff.transpose(0, 1) @ alpha).squeeze()
    logdet = 2.0 * torch.log(torch.diag(L)).sum()
    N = y.numel()
    return -0.5 * (quad + logdet + N * math.log(2 * math.pi))


def mahalanobis2(y: Tensor, m: Tensor, S: Tensor) -> Tensor:
    y, m, S = y.double(), m.double(), S.double()
    L = _stable_cholesky(S)
    v = torch.linalg.solve(L, (y - m).view(-1, 1))
    return (v.squeeze()**2).sum()


@torch.no_grad()
def sample_gp_conditional_fixed(
    *,
    dataset: str,
    x_target: Tensor,
    x_context: Optional[Tensor] = None,
    y_context: Optional[Tensor] = None,
    active_dims: Optional[List[int]] = None,
    include_obs_noise: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, GPHypersFixed]:
    target_dtype = x_target.dtype
    target_device = x_target.device
    if x_target.ndim == 1:
        x_target = x_target.unsqueeze(-1)
    if active_dims is None:
        active_dims = list(range(x_target.size(-1)))

    has_context = x_context is not None and y_context is not None and x_context.numel() > 0
    if has_context:
        m, S, θ = gp_posterior_fixed(dataset, x_context, y_context, x_target, active_dims, include_obs_noise)
    else:
        m, S, θ = gp_prior_fixed(dataset, x_target, active_dims, include_obs_noise)

    N = m.shape[0]
    L = _stable_cholesky(S)
    eps = torch.randn((N,), dtype=m.dtype, device=m.device, generator=generator)
    sample = m + L @ eps
    return sample.view(-1, 1).to(dtype=target_dtype, device=target_device), θ


@torch.no_grad()
def _time_index_from_unit(t_unit: torch.Tensor, T: int) -> torch.Tensor:
    idx = torch.round(t_unit * (T - 1)).long()
    return idx.clamp_(0, T - 1)


def _hutchinson_divergence(f_of_x, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        fx = f_of_x(x)
        y = (fx * v).sum()
        (JTv,) = torch.autograd.grad(y, x, create_graph=True)
        return (JTv * v).sum()


def _jacobian_trace(f_of_x, x: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        jac = torch.autograd.functional.jacobian(f_of_x, x, vectorize=True)
        jac_flat = jac.reshape(x.numel(), x.numel())
        return jac_flat.diagonal().sum()


@torch.no_grad()
def vp_probflow_loglik_conditional(
    *,
    model,                    # x0-theta network (PD)
    process,                  # GaussianDiffusionPD (provides betas, alpha_bars)
    x_target: torch.Tensor,   # [N, D]
    y_target0: torch.Tensor,  # [N, 1]  -- evaluate log p(y_target0 | context)
    x_context: Optional[torch.Tensor] = None,  # [M, D] or None
    y_context: Optional[torch.Tensor] = None,  # [M, 1] or None
    mask_context: Optional[torch.Tensor] = None,  # [M] with 1=masked
    steps: int = 500,
    hutch: str = "rademacher",
    divergence: str = "exact",
    integrator: str = "solve_ivp",
    rtol: float = 1e-5,
    atol: float = 1e-5,
    eps: float = 1e-5,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    device = y_target0.device
    dtype  = y_target0.dtype

    if x_target.ndim == 1:
        x_target = x_target.unsqueeze(-1)
    if y_target0.ndim == 1:
        y_target0 = y_target0.unsqueeze(-1)
    elif y_target0.ndim == 2 and y_target0.shape[0] == 1:
        y_target0 = y_target0.squeeze(0).unsqueeze(-1)

    y_target0 = y_target0.to(dtype=x_target.dtype, device=x_target.device)
    N, y_dim = y_target0.shape[0], y_target0.shape[1]
    assert y_dim == 1, "Assumes scalar outputs per point (y_dim=1)."

    T = int(process.betas.numel())
    betas = process.betas.to(device=device, dtype=dtype)
    alpha_bars = process.alpha_bars.to(device=device, dtype=dtype)

    mask_tgt = torch.zeros(1, N, device=device, dtype=dtype)
    if x_context is not None and y_context is not None:
        M = x_context.shape[0]
        if mask_context is None:
            mask_context = torch.zeros(M, device=device, dtype=dtype)
    else:
        M = 0
        x_context = y_context = mask_context = None

    y = y_target0.clone().to(device=device, dtype=dtype)
    delta_logp = torch.zeros((), device=device, dtype=dtype)

    use_exact_div = divergence.lower() == "exact"
    if not use_exact_div:
        if rng is None:
            rng = torch.Generator(device=device)
            rng.manual_seed(torch.randint(0, 2**31 - 1, (1,), device=device).item())
        if hutch == "gaussian":
            v = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=rng)
        else:
            v = torch.empty_like(y)
            v.bernoulli_(0.5).mul_(2.0).sub_(1.0)
    else:
        v = None

    def f_pf(y_t: torch.Tensor, t_unit: torch.Tensor) -> torch.Tensor:
        t_unit_clamped = t_unit.clamp(eps, 1.0 - eps)
        t_idx = _time_index_from_unit(t_unit_clamped, T)
        beta_t = betas[t_idx] * float(T)
        abar_t = alpha_bars[t_idx]

        y_in = y_t.unsqueeze(0)
        x_in = x_target.unsqueeze(0)
        t_in = t_idx.to(dtype=torch.float32, device=device).view(1)
        m_tgt = mask_tgt
        if M > 0:
            x_ctx_b = x_context.unsqueeze(0)
            y_ctx_b = y_context.unsqueeze(0)
            m_ctx_b = mask_context.unsqueeze(0)
        else:
            x_ctx_b = y_ctx_b = m_ctx_b = None

        # x0-theta prediction on targets only (PD setting)
        x0_hat = model(
            x_in, y_in, t_in, m_tgt,
            x_context=x_ctx_b, y_context=y_ctx_b, mask_context=m_ctx_b
        ).squeeze(0)

        # Convert x0 to score: s = -(y - sqrt(abar) x0)/(1 - abar)
        denom = (1.0 - abar_t).clamp_min(1e-12)
        score = -(y_t - torch.sqrt(abar_t) * x0_hat) / denom

        return -0.5 * beta_t * (y_t + score)

    if integrator.lower() == "solve_ivp":
        def div_at(y_state: torch.Tensor, t_scalar: float) -> torch.Tensor:
            t_scalar = float(min(max(t_scalar, eps), 1.0 - eps))
            t_tensor = torch.tensor(t_scalar, dtype=dtype, device=device)
            if use_exact_div:
                return _jacobian_trace(lambda z: f_pf(z, t_tensor), y_state)
            else:
                return _hutchinson_divergence(lambda z: f_pf(z, t_tensor), y_state, v)

        def aug_rhs(t_scalar: float, state_np):
            state_tensor = torch.from_numpy(state_np).to(device=device, dtype=dtype)
            y_state = state_tensor[:-1].view_as(y)
            t_scalar_clamped = float(min(max(t_scalar, eps), 1.0 - eps))
            drift = f_pf(y_state, torch.tensor(t_scalar_clamped, dtype=dtype, device=device)).view(-1)
            div = div_at(y_state, t_scalar).view(1)
            rhs = torch.cat([drift, div], dim=0)
            return rhs.detach().cpu().numpy()

        state0 = torch.cat([y.view(-1), delta_logp.view(1)], dim=0).detach().cpu().numpy()
        sol = solve_ivp(
            fun=aug_rhs,
            t_span=(1.0, 0.0),
            y0=state0,
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        stateT = torch.from_numpy(sol.y[:, -1]).to(device=device, dtype=dtype)
        y = stateT[:-1].view_as(y)
        delta_logp = stateT[-1]
    else:
        raise ValueError("Only 'solve_ivp' integrator is supported here.")

    D = y.numel()
    log_pT = -0.5 * (D * math.log(2.0 * math.pi) + (y * y).sum())
    return (log_pT + delta_logp).to(dtype=torch.float32)


@torch.no_grad()
def gp_sample_and_diffusion_loglik(
    dataset: str,
    *,
    model,
    process,
    x_target: Tensor,
    x_context: Optional[Tensor] = None,
    y_context: Optional[Tensor] = None,
    mask_context: Optional[Tensor] = None,
    active_dims: Optional[List[int]] = None,
    include_obs_noise: bool = True,
    steps: int = 512,
    hutch: str = "rademacher",
    divergence: str = "exact",
    integrator: str = "solve_ivp",
    rtol: float = 1e-5,
    atol: float = 1e-5,
    eps: float = 1e-5,
    gp_generator: Optional[torch.Generator] = None,
    ode_rng: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, GPHypersFixed]:
    y_sample, θ = sample_gp_conditional_fixed(
        dataset=dataset,
        x_target=x_target,
        x_context=x_context,
        y_context=y_context,
        active_dims=active_dims,
        include_obs_noise=include_obs_noise,
        generator=gp_generator,
    )

    loglik = vp_probflow_loglik_conditional(
        model=model,
        process=process,
        x_target=x_target,
        y_target0=y_sample,
        x_context=x_context,
        y_context=y_context,
        mask_context=mask_context,
        steps=steps,
        hutch=hutch,
        divergence=divergence,
        integrator=integrator,
        rtol=rtol,
        atol=atol,
        eps=eps,
        rng=ode_rng,
    )

    return y_sample, loglik, θ


@torch.no_grad()
def gaussian_entropy_from_cov(S: torch.Tensor) -> torch.Tensor:
    S = S.double()
    L = _stable_cholesky(S)
    logdet = 2.0 * torch.log(torch.diag(L)).sum()
    N = S.size(0)
    return 0.5 * (N * math.log(2.0 * math.pi * math.e) + logdet)


@torch.no_grad()
def estimate_kl_from_pf_lls(pf_lls: torch.Tensor, S: torch.Tensor) -> dict:
    if pf_lls.ndim != 1:
        pf_lls = pf_lls.view(-1)
    H = gaussian_entropy_from_cov(S).detach().cpu().item()
    E_logq = float(pf_lls.double().mean().item())
    E_logq_std = float(pf_lls.double().std(unbiased=False).item()) if pf_lls.numel() > 1 else 0.0
    K = int(pf_lls.numel())
    KL = -H - E_logq
    return {
        'H': H,
        'E_logq': E_logq,
        'E_logq_std': E_logq_std,
        'K': K,
        'KL': float(KL),
    }


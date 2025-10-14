# gp_likelihood_core.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor
from scipy.integrate import solve_ivp

# Import the EXACT constants & kernels used by your dataset
from data import (
    _LENGTHSCALE, _KERNEL_VAR, _NOISE_VAR, _JITTER,
    _rbf_kernel, _matern52_kernel,
)
from typing import Optional
from config import Config
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


@torch.no_grad()
def gp_prior_fixed(
    dataset: str,
    x_tgt: Tensor,
    active_dims: List[int],
    include_obs_noise: bool = True,
) -> Tuple[Tensor, Tensor, GPHypersFixed]:
    """
    Prior of y(x_tgt) under the *fixed* GP used by data.py.
    Returns (mean:[N], cov:[N,N], hypers).
    If include_obs_noise=True, scores the observed y; otherwise latent f.
    """
    xT = x_tgt.double()
    k, θ = _kernel_closure(dataset, active_dims)  # same kernel & scaling as your data
    S = k(xT, xT)
    if include_obs_noise:
        S = S + θ.noise * torch.eye(S.size(0), dtype=S.dtype, device=S.device)
    m = torch.zeros(xT.size(0), dtype=S.dtype, device=S.device)
    return m, S, θ
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





# likelihood.py
# -----------------------------------------------------------------------------
# Probability-flow ODE likelihood for VP/DDPM with conditional context
# -----------------------------------------------------------------------------


@torch.no_grad()
def sample_gp_conditional_fixed(
    dataset: str,
    x_target: Tensor,
    *,
    x_context: Optional[Tensor] = None,
    y_context: Optional[Tensor] = None,
    active_dims: Optional[List[int]] = None,
    include_obs_noise: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, GPHypersFixed]:
    """Draw a single sample y(x_target) | (x_context, y_context) from the fixed GP.

    Parameters
    ----------
    dataset: str
        Either ``"se"`` or ``"matern"``; must match the kernels used during training.
    x_target: Tensor
        Query inputs [N, D] (or [N]) at which to draw the GP sample.
    x_context / y_context: Tensor, optional
        Conditioning inputs and observations with shapes [M, D] and [M, 1]. If ``None`` or
        empty, a prior sample is drawn instead.
    active_dims: list[int], optional
        Subset of input dimensions to feed to the kernel. Defaults to ``range(D)``.
    include_obs_noise: bool
        Whether to include observation noise (as in the training data) when sampling.
    generator: torch.Generator, optional
        Randomness source. When ``None`` we fall back to ``torch.randn``'s global RNG.

    Returns
    -------
    sample: Tensor
        Drawn GP sample with shape [N, 1] on the same device as ``x_target``.
    hypers: GPHypersFixed
        The fixed kernel hyper-parameters used for the draw (useful for logging).
    """

    target_dtype = x_target.dtype
    target_device = x_target.device

    if x_target.ndim == 1:
        x_target = x_target.unsqueeze(-1)

    if active_dims is None:
        active_dims = list(range(x_target.size(-1)))

    has_context = x_context is not None and y_context is not None and x_context.numel() > 0

    if has_context:
        m, S, θ = gp_posterior_fixed(
            dataset,
            x_ctx=x_context,
            y_ctx=y_context,
            x_tgt=x_target,
            active_dims=active_dims,
            include_obs_noise=include_obs_noise,
        )
    else:
        m, S, θ = gp_prior_fixed(
            dataset,
            x_tgt=x_target,
            active_dims=active_dims,
            include_obs_noise=include_obs_noise,
        )

    N = m.shape[0]
    L = _stable_cholesky(S)

    if generator is None:
        eps = torch.randn(N, dtype=m.dtype, device=m.device)
    else:
        gen_device = getattr(generator, "device", torch.device("cpu"))
        if gen_device != target_device:
            raise ValueError(
                f"Generator device {gen_device} does not match target device {target_device}."
            )
        eps = torch.randn((N,), dtype=m.dtype, device=m.device, generator=generator)

    sample = m + L @ eps
    return sample.view(-1, 1).to(dtype=target_dtype, device=target_device), θ


@torch.no_grad()
def _time_index_from_unit(t_unit: torch.Tensor, T: int) -> torch.Tensor:
    """
    Map a float time in [0,1] to a (clamped) integer index in [0, T-1].
    Returns Long tensor shaped like t_unit (usually scalar).
    """
    idx = torch.round(t_unit * (T - 1)).long()
    return idx.clamp_(0, T - 1)


def _hutchinson_divergence(f_of_x, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Hutchinson trace estimator: v^T J_f(x) v.
    f_of_x: a function that takes a tensor 'x' (requires_grad=True) and returns f(x) with same shape.
    x: [N, y_dim] or [N, 1]
    v: same shape as x, sampled from N(0,I) or Rademacher
    Returns a scalar divergence estimate (tensor with shape []).
    """
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        fx = f_of_x(x)
        y = (fx * v).sum()
        (JTv,) = torch.autograd.grad(y, x, create_graph=True)
        return (JTv * v).sum()


def _jacobian_trace(f_of_x, x: torch.Tensor) -> torch.Tensor:
    """Compute trace(J_f(x)) exactly via autograd jacobian."""
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        jac = torch.autograd.functional.jacobian(f_of_x, x, vectorize=True)
        jac_flat = jac.reshape(x.numel(), x.numel())
        return jac_flat.diagonal().sum()

def vp_probflow_loglik_conditional(
    *,
    model,                    # your eps-theta network
    process,                  # GaussianDiffusion (provides betas, alpha_bars)
    x_target: torch.Tensor,   # [N, D]
    y_target0: torch.Tensor,  # [N, 1]  -- evaluate log p(y_target0 | context)
    x_context: Optional[torch.Tensor] = None,  # [M, D] or None
    y_context: Optional[torch.Tensor] = None,  # [M, 1] or None
    mask_context: Optional[torch.Tensor] = None,  # [M] with 1=masked
    steps: int = 500,         # ODE steps (Heun / RK2) or initial guess for adaptive solver
    hutch: str = "rademacher",
    divergence: str = "exact",
    integrator: str = "solve_ivp",
    rtol: float = 1e-5,
    atol: float = 1e-5,
    eps: float = 1e-5,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Compute log p_theta(y_target0 | context) via the VP probability-flow ODE.

    Assumptions
    ----------
    - VP / DDPM family (terminal p_T is standard Normal).
    - model predicts epsilon; we convert to score by s = -eps / sqrt(1 - alpha_bar_t).
    - Context is fixed and provided clean; only targets are in the ODE state.

    Returns
    -------
    loglik: scalar tensor (float32) on the same device as y_target0.

    Notes
    -----
    divergence : {"hutchinson", "exact"}
        Selects between the stochastic Hutchinson estimator and the exact
        Jacobian trace computation for div f_pf.
    """
    device = y_target0.device
    dtype  = y_target0.dtype

    # Coerce x_target to [N, 1]
    if x_target.ndim == 1:
        x_target = x_target.unsqueeze(-1)  # [N] -> [N,1]

    # Coerce y_target0 to [N, 1]
    if y_target0.ndim == 1:
        y_target0 = y_target0.unsqueeze(-1)  # [N] -> [N,1]
    elif y_target0.ndim == 2 and y_target0.shape[0] == 1:
        # row vector [1,N] -> column [N,1]
        y_target0 = y_target0.squeeze(0).unsqueeze(-1)  # or: y_target0 = y_target0.T.contiguous()

    # (optional) ensure dtype/device match
    y_target0 = y_target0.to(dtype=x_target.dtype, device=x_target.device)

    # Shapes + batchify to [1, N, ...] for the model call
    N, y_dim = y_target0.shape[0], y_target0.shape[1]
    assert y_dim == 1, "This implementation assumes scalar outputs per point (y_dim=1)."

    T = int(process.betas.numel())
    betas = process.betas.to(device=device, dtype=dtype)              # [T]
    alpha_bars = process.alpha_bars.to(device=device, dtype=dtype)    # [T]

    # Masks
    mask_tgt = torch.zeros(1, N, device=device, dtype=dtype)          # [1,N], 0=keep
    if x_context is not None and y_context is not None:
        M = x_context.shape[0]
        if mask_context is None:
            mask_context = torch.zeros(M, device=device, dtype=dtype)
    else:
        M = 0
        x_context = y_context = mask_context = None

    # State: we integrate the augmented ODE (y_t, DeltaLogP)
    y = y_target0.clone().to(device=device, dtype=dtype)              # [N,1]
    delta_logp = torch.zeros((), device=device, dtype=dtype)          # scalar

    use_exact_div = divergence.lower() == "exact"

    if not use_exact_div:
        if rng is None:
            rng = torch.Generator(device=device)
            rng.manual_seed(torch.randint(0, 2**31 - 1, (1,), device=device).item())

        if hutch == "gaussian":
            v = torch.randn_like(y, generator=rng)                    # [N,1]
        else:  # rademacher
            v = torch.empty_like(y)
            v.bernoulli_(0.5).mul_(2.0).sub_(1.0)                     # ±1 with p=0.5
    else:
        v = None

    def f_pf(y_t: torch.Tensor, t_unit: torch.Tensor) -> torch.Tensor:
        """
        Probability-flow ODE drift for VP: f_pf(x,t) = -0.5 * beta(t) * ( x + s(x,t) ).
        Here x == y_t (targets only). Context is injected via the network call.
        """
        # map t∈[0,1] → discrete idx ∈ {0,…,T-1}
        t_unit_clamped = t_unit.clamp(eps, 1.0 - eps)
        t_idx = _time_index_from_unit(t_unit_clamped, T)              # Long []
        # TODO: add * float(T)
        beta_t = betas[t_idx] * float(T)                                    # []
        #print("t_unit:", t_unit)
       # print("beta_t:",beta_t)
        abar_t = alpha_bars[t_idx]                                    # []
        #print("abar_t:",abar_t)

        # model expects batched tensors
        y_in  = y_t.unsqueeze(0)                                      # [1,N,1]
        x_in  = x_target.unsqueeze(0)                                 # [1,N,D]
        t_in  = t_idx.to(dtype=torch.float32, device=device).view(1)  # [1] float
        m_tgt = mask_tgt                                              # [1,N]
        if M > 0:
            x_ctx_b = x_context.unsqueeze(0)                          # [1,M,D]
            y_ctx_b = y_context.unsqueeze(0)                          # [1,M,1]
            m_ctx_b = mask_context.unsqueeze(0)                       # [1,M]
        else:
            x_ctx_b = y_ctx_b = m_ctx_b = None

        # eps-theta prediction on targets only
        eps_hat = model(
            x_in, y_in, t_in, m_tgt,
            x_context=x_ctx_b, y_context=y_ctx_b, mask_context=m_ctx_b
        ).squeeze(0)                                                  # [N,1]

        # Convert to score: s = -eps / sqrt(1 - alpha_bar_t)
        sqrt_one_minus_abar = torch.sqrt((1.0 - abar_t).clamp_min(1e-12))
        score = -eps_hat / sqrt_one_minus_abar                        # [N,1]

        # f_pf for VP
        return -0.5 * beta_t * (y_t + score)                          # [N,1]

    if integrator.lower() == "solve_ivp":
        # Adaptive solver using SciPy's solve_ivp (Dormand–Prince RK45).
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
            #TODO: sign of div
            rhs = torch.cat([drift, div], dim=0)
            return rhs.detach().cpu().numpy()

        state0 = torch.cat([y.view(-1), delta_logp.view(1)], dim=0).detach().cpu().numpy()
        # Let solve_ivp choose adaptive step sizes entirely (no explicit max_step)
        sol = solve_ivp(
            aug_rhs,
            t_span=(eps, 1.0),
            y0=state0,
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        y = torch.from_numpy(sol.y[:-1, -1]).to(device=device, dtype=dtype).view_as(y)
        delta_logp = torch.tensor(sol.y[-1, -1], device=device, dtype=dtype)
    else:
        # Heun (RK2) integrator on the augmented system
        t = torch.tensor(0.0, device=device, dtype=dtype)
        steps_heun = steps if steps is not None else Config.schedule.timesteps
        if steps_heun <= 0:
            steps_heun = Config.schedule.timesteps
        dt = 1.0 / float(steps_heun)
        for _ in range(steps_heun):
            t1 = t
            t2 = (t + dt).clamp(0.0, 1.0)

            # k1 and div at (y, t1)
            def _f1(x): return f_pf(x, t1)
            f1 = _f1(y)
            if use_exact_div:
                div1 = _jacobian_trace(_f1, y)
            else:
                div1 = _hutchinson_divergence(_f1, y, v)

            # Euler proposal
            y_euler = y + dt * f1

            # k2 and div2 at (y_euler, t2)
            def _f2(x): return f_pf(x, t2)
            f2 = _f2(y_euler)
            if use_exact_div:
                div2 = _jacobian_trace(_f2, y_euler)
            else:
                div2 = _hutchinson_divergence(_f2, y_euler, v)

            # Heun updates
            y = y + 0.5 * dt * (f1 + f2)
            delta_logp = delta_logp + 0.5 * dt * ( div1 + div2 )

            t = t2

    # Terminal prior for VP: standard Normal N(0, I)
    D = y.numel()   # N * 1
    log_pT = -0.5 * (D * math.log(2.0 * math.pi) + (y * y).sum())
    print(f"log_pT: {log_pT:.3f}")
    print(f"delta_logp: {delta_logp:.3f}")
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
    """Sample a GP function value at ``x_target`` and score it with the diffusion model.

    This utility mirrors the evaluation flow used in the interpolation benchmarks:
    first draw a sample from the fixed GP used in ``data.py`` (optionally
    conditioned on ``(x_context, y_context)``), then compute the probability-flow
    ODE log-likelihood under the trained diffusion model.

    Returns
    -------
    y_sample: Tensor
        Sampled targets with shape [N, 1] and dtype/device matching ``x_target``.
    loglik: Tensor
        Scalar log-likelihood ``log p_theta(y_sample | context)``.
    hypers: GPHypersFixed
        Kernel hyper-parameters used for the GP draw (for analysis/logging).
    """

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
    # H = 1/2 * ( N * log(2πe) + log |S| ) for Gaussian N(m,S)
    S = S.double()
    L = _stable_cholesky(S)
    logdet = 2.0 * torch.log(torch.diag(L)).sum()
    N = S.size(0)
    return 0.5 * (N * math.log(2.0 * math.pi * math.e) + logdet)

@torch.no_grad()
def estimate_kl_from_pf_lls(pf_lls: torch.Tensor, S: torch.Tensor) -> dict:
    # KL( N(m,S) || q_theta ) = - H(N(m,S)) - E_p[log q_theta(y) ]
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

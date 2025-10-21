from __future__ import annotations

import math
from typing import Protocol, Tuple

import torch


def _expand_to(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return a.reshape(a.shape + (1,) * (ref.ndim - a.ndim))


def cosine_schedule(beta_start: float,
                    beta_end: float,
                    timesteps: int,
                    s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, timesteps, timesteps + 1)
    f = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = f / f[0]
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = betas.clamp(1e-4, 0.9999)
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    return betas * (beta_end - beta_start) + beta_start


class X0Model(Protocol):
    def __call__(self,
                 t: torch.Tensor,
                 yt: torch.Tensor,
                 x: torch.Tensor,
                 mask: torch.Tensor | None,
                 *,
                 key: torch.Generator) -> torch.Tensor:
        ...


class GaussianDiffusionPD:
    """
    Gaussian diffusion utilities for x0-prediction training and sampling.
    """

    def __init__(self, betas: torch.Tensor) -> None:
        self.device = betas.device
        self.dtype = betas.dtype
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    # ---- forward moments -------------------------------------------------
    def pt0(self,
            y0: torch.Tensor,
            t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        abar_t = self.alpha_bars[t].to(y0.device)
        mean = torch.sqrt(abar_t) * y0
        var = (1.0 - abar_t)
        return mean, var

    # ---- forward sample --------------------------------------------------
    def forward(self,
                key: torch.Generator,
                y0: torch.Tensor,
                t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m, v = self.pt0(y0, t)
        noise = torch.randn(y0.shape, dtype=y0.dtype, device=y0.device)
        yt = m + torch.sqrt(v) * noise
        return yt, noise

    # ---- reverse DDPM deterministic mean (requires eps_hat) -------------
    def ddpm_backward_step(self,
                           key: torch.Generator,
                           eps_hat: torch.Tensor,
                           yt: torch.Tensor,
                           t: torch.Tensor) -> torch.Tensor:
        beta_t = _expand_to(self.betas[t], yt)
        alpha_t = _expand_to(self.alphas[t], yt)
        abar_t = _expand_to(self.alpha_bars[t], yt)

        inv_sqrt_alpha = torch.rsqrt(alpha_t)
        coeff = beta_t / torch.sqrt(1.0 - abar_t + 1e-12)
        mean = inv_sqrt_alpha * (yt - coeff * eps_hat)

        z = torch.zeros_like(yt)
        if t.dim() == 0:
            t_item = int(t.item())
        else:
            t_item = int(t.view(-1)[0].item())
        if t_item > 0:
            z = torch.randn(yt.shape, dtype=yt.dtype, device=yt.device)
        return mean + torch.sqrt(beta_t) * z

    # ---- unified sampler: unconditional OR conditional; DDPM or DDIM -----
    @torch.no_grad()
    def sample(self,
               key: torch.Generator,
               x: torch.Tensor,
               *,
               model_fn: X0Model,
               y_dim: int = 1,
               x_context: torch.Tensor | None = None,
               y_context: torch.Tensor | None = None,
               num_sample_steps: int | None = None,
               eta: float = 0.0) -> torch.Tensor:
        """
        Unified DDIM-style sampler (DDPM is recovered by consecutive steps and eta≈1).
        - Unconditional: call without contexts
        - Conditional (unified): pass x_context and y_context; contexts stay clean
        """
        device, dtype = x.device, x.dtype
        T = len(self.betas)

        # timestep schedule (include t=0)
        steps = num_sample_steps if (num_sample_steps is not None and num_sample_steps > 1) else T
        grid = torch.linspace(T - 1, 0, steps=steps, device=device)
        ts = torch.unique_consecutive(grid.round().to(torch.long))
        if ts[-1].item() != 0:
            ts = torch.cat([ts, torch.tensor([0], device=device, dtype=torch.long)])

        # determine context usage and build aug inputs once
        use_ctx = (x_context is not None) and (y_context is not None) and (x_context.numel() > 0)
        num_ctx = (x_context.size(0) if use_ctx else 0)
        if use_ctx:
            x_aug = torch.cat([x_context, x], dim=0)
        else:
            x_aug = x

        # init targets at t_max
        y_t = torch.randn((x.size(0), y_dim), dtype=dtype, device=device, generator=key)

        # main loop over coarse steps
        for i in range(len(ts) - 1):
            t_i = int(ts[i].item()); t_j = int(ts[i + 1].item())
            abar_i = self.alpha_bars[t_i].to(device)
            abar_j = self.alpha_bars[t_j].to(device)

            # model on clean context + current targets
            if use_ctx:
                y_aug_in = torch.cat([y_context, y_t], dim=0)
            else:
                y_aug_in = y_t
            t_f = torch.tensor(float(t_i), device=device)
            x0_hat_aug = model_fn(t_f, y_aug_in, x_aug, None, key=key)
            x0_hat_tgt = x0_hat_aug[num_ctx:] if use_ctx else x0_hat_aug

            eps_hat_tgt = (y_t - torch.sqrt(abar_i) * x0_hat_tgt) / torch.sqrt(torch.clamp(1.0 - abar_i, min=1e-12))

            # DDIM update (eta=0 deterministic; eta≈1 ~ DDPM-like noise)
            if eta > 0:
                term1 = torch.clamp((1.0 - abar_j) / (1.0 - abar_i + 1e-12), min=0.0)
                term2 = torch.clamp(1.0 - (abar_i / (abar_j + 1e-12)), min=0.0)
                sigma_eta = eta * torch.sqrt(term1 * term2)
            else:
                sigma_eta = torch.tensor(0.0, device=device, dtype=y_t.dtype)
            coeff_eps = torch.sqrt(torch.clamp(1.0 - abar_j - sigma_eta**2, min=0.0))
            z = torch.randn(y_t.shape, dtype=y_t.dtype, device=y_t.device, generator=key) if float(sigma_eta) > 0.0 else None
            y_t = torch.sqrt(abar_j) * x0_hat_tgt + coeff_eps * eps_hat_tgt + (sigma_eta * z if z is not None else 0.0)

        return y_t


def loss_x0(process: GaussianDiffusionPD,
            model: X0Model,
            batch,
            key: torch.Generator,
            *,
            num_timesteps: int,
            loss_weighting: str = "lambda",
            loss_type: str = "l2") -> torch.Tensor:
    """
    x0-prediction objective with optional SNR (lambda) weighting.
    Drops masks; treats all targets equally.
    """
    B, N, _ = batch.y_target.shape
    device = batch.y_target.device

    t = torch.randint(0, num_timesteps, (B,), device=device)
    t_ = t.view(B, 1, 1)
    abar_t = process.alpha_bars[t_].to(device)

    noise = torch.randn(batch.y_target.shape, dtype=batch.y_target.dtype, device=batch.y_target.device)
    yt = torch.sqrt(abar_t) * batch.y_target + torch.sqrt(1.0 - abar_t) * noise

    x0_hat = model(
        t.to(dtype=torch.float32),
        yt,
        batch.x_target,
        None,
        key=key,
    )

    if loss_type == "l1":
        per = (x0_hat - batch.y_target).abs().sum(-1)
    else:
        per = ((x0_hat - batch.y_target) ** 2).sum(-1)

    if loss_weighting == "lambda":
        # SNR weighting: abar / (1 - abar)
        w = (abar_t / torch.clamp(1.0 - abar_t, min=1e-12)).squeeze(-1).squeeze(-1)  # [B]
        per = per * w[:, None]

    denom = torch.tensor(float(B * N), device=device)
    return per.sum() / denom

# samplers_ode.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import torch
import math

EpsModel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]

def _ddpm_sigmas(alpha_bars: torch.Tensor) -> torch.Tensor:
    # σ_t = sqrt((1 - ᾱ_t) / ᾱ_t)
    return torch.sqrt(torch.clamp((1.0 - alpha_bars) / (alpha_bars + 1e-12), min=0))

@dataclass
class EulerHeunSampler:
    process: any
    num_sample_steps: int = 25
    rho: float = 7.0           # Karras schedule power
    ddpm: bool = True          # we wrap a DDPM-trained ε-network
    s_churn: float = 0.0       # set >0 to enable stochasticity (Algorithm 2)
    s_tmin: float = 0.05
    s_tmax: float = 50.0
    s_noise: float = 1.003

    def __post_init__(self):
        device = self.process.alpha_bars.device
        self.alpha_bars = self.process.alpha_bars.to(device)

        self.ddpm_sigmas = _ddpm_sigmas(self.alpha_bars)  # [T]
        sigma_min = float(self.ddpm_sigmas[0].item())  # <-- smallest at t=0
        sigma_max = float(self.ddpm_sigmas[-1].item())  # <-- largest at t=T-1

        i = torch.arange(self.num_sample_steps, device=device)
        # Karras σ-grid (high → low)
        s = (sigma_max**(1/self.rho) + (i / (self.num_sample_steps - 1)) * (sigma_min**(1/self.rho) - sigma_max**(1/self.rho))) ** self.rho
        self.sigmas = s

    # ---- Karras "coeff" utilities for DDPM ε-networks ----------------------
    def c_skip(self, sigma: torch.Tensor): return torch.tensor(1.0, device=sigma.device)
    def c_out (self, sigma: torch.Tensor): return -sigma
    def c_in  (self, sigma: torch.Tensor): return torch.rsqrt(1.0 + sigma**2)
    def c_noise(self, sigma: torch.Tensor):
        # map σ → nearest discrete DDPM timestep index
        idx = torch.argmin(torch.abs(self.ddpm_sigmas - sigma))
        return idx.to(torch.long)

    # ---- wrap your ε-network into a data prediction d_θ(x,σ) ----------------

    def d_theta(self, y, sigma, x, mask, model_fn, key):
        t_idx = self.c_noise(sigma)  # 0-D Long idx
        # => DO NOT convert to float; keep it Long
        eps_hat = model_fn(t_idx, self.c_in(sigma) * y, x, mask, key=key)
        return self.c_skip(sigma) * y + self.c_out(sigma) * eps_hat

    # ---- Euler predictor (Algorithm 2 lines 1–8 w/o correction) -------------
    def _euler_predictor(self, y, i, x, mask, model_fn: EpsModel, key: torch.Generator):
        t     = self.sigmas[i]
        t_next= self.sigmas[i+1] if i+1 < len(self.sigmas) else torch.tensor(0., device=y.device)

        # stochastic "churn" (optional)
        gamma = (min(self.s_churn / self.num_sample_steps, math.sqrt(2) - 1.0)
                 if (self.s_tmin <= float(t) <= self.s_tmax) else 0.0)
        t_hat = t + gamma * t
        if gamma > 0:
            # eps = torch.randn_like(y, generator=key) * self.s_noise
            eps = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=key ) * self.s_noise
            y   = y + torch.sqrt(torch.clamp(t_hat**2 - t**2, min=0.0)) * eps

        d = (y - self.d_theta(y, t_hat, x, mask, model_fn, key)) / (t_hat + 1e-12)
        y_next = y + (t_next - t_hat) * d
        return y_next, y, t_next, t_hat, d

    # ---- Heun corrector -----------------------------------------------------
    def _heun_corrector(self, y_next, y_hat, t_next, t_hat, d, x, mask, model_fn: EpsModel, key: torch.Generator):
        if float(t_next) != 0.0:
            d_prime = (y_next - self.d_theta(y_next, t_next, x, mask, model_fn, key)) / (t_next + 1e-12)
            y_next  = y_hat + (t_next - t_hat) * (0.5 * d + 0.5 * d_prime)
        return y_next

    # ---- Unconditional ------------------------------------------------------
    @torch.no_grad()
    def sample_uncond(self, key: torch.Generator, x: torch.Tensor, model_fn: EpsModel,
                      mask: Optional[torch.Tensor] = None, y_dim: int = 1, method: str = "euler"):
        device = x.device
        N = x.size(0)
        y = torch.randn(N, y_dim, device=device, generator=key) * self.sigmas[0]  # init at σ_max
        if mask is None:
            mask = torch.zeros(N, device=device)

        for i in range(len(self.sigmas) - 1):
            y_next, y_hat, t_next, t_hat, d = self._euler_predictor(y, i, x, mask, model_fn, key)
            if method == "heun":
                y_next = self._heun_corrector(y_next, y_hat, t_next, t_hat, d, x, mask, model_fn, key)
            y = y_next
        return y

    # ---- Conditional (context kept fixed) ----------------------------------
    @torch.no_grad()
    def sample_cond(self, key: torch.Generator,
                    x_query: torch.Tensor, mask_tgt: Optional[torch.Tensor],
                    *, x_context: torch.Tensor, y_context: torch.Tensor, mask_context: Optional[torch.Tensor],
                    model_fn: EpsModel, y_dim: int = 1, method: str = "heun",
                    num_inner_steps: int = 5):
        """
        Conditional ODE sampling with RePaint-style inner loop.
        Reuses _euler_predictor and _heun_corrector (no re-implementation).

        For each outer σ_i → σ_{i+1}:
          - Do U inner refinements:
             (A) Backward ODE step (σ_i → σ_{i+1}) on the joint (ctx+qry),
                 with contexts clamped to EDM form y_ctx(σ)=y_context+σ*z_ctx.
             (B) If u < U-1: FORWARD *EDM* step (σ_{i+1} → σ_i) for QUERIES ONLY:
                 y_q ← y_q + sqrt(σ_i^2 - σ_{i+1}^2) * ξ, ξ~N(0,I)
          - After the inner loop we are already at σ_{i+1}.
        """
        device = x_query.device
        if mask_tgt is None:     mask_tgt = torch.zeros(x_query.size(0), device=device)
        if mask_context is None: mask_context = torch.zeros(x_context.size(0), device=device)
        num_ctx = x_context.size(0)

        # init query at σ_max; fixed latent for contexts (kept for whole trajectory)
        y_q = torch.randn(x_query.size(0), y_dim, device=device, generator=key) * self.sigmas[0]
        z_ctx = torch.randn(y_context.shape,dtype=y_context.dtype,device=y_context.device,generator=key)

        # concat once (used for all model calls)
        x_aug = torch.cat([x_context, x_query], dim=0)
        mask_aug = torch.cat([mask_context, mask_tgt], dim=0)

        # wrapper keeps your model signature
        def joint_model_fn(t_scalar, yt, xx, mm, *, key):
            return model_fn(t_scalar, yt, xx, mm, key=key)

        # IMPORTANT: keep s_churn = 0.0 for parity; otherwise contexts should use t_hat.
        for i in range(len(self.sigmas) - 1):
            sigma_i = self.sigmas[i]
            sigma_next = self.sigmas[i + 1]  # smaller

            for u in range(num_inner_steps):
                # (A) BACKWARD ODE step σ_i → σ_{i+1}
                # clamp contexts at current σ_i (EDM coordinates)
                y_ctx_i = y_context + sigma_i * z_ctx
                y_aug = torch.cat([y_ctx_i, y_q], dim=0)

                # Euler predictor
                y_next, y_hat, t_next, t_hat, d = self._euler_predictor(
                    y_aug, i, x_aug, mask_aug, joint_model_fn, key
                )

                # Heun corrector: 2nd slope must see contexts at σ_{i+1}
                if method == "heun" and float(t_next) != 0.0:
                    y_ctx_next = y_context + t_next * z_ctx
                    y_next_aligned = torch.cat([y_ctx_next, y_next[num_ctx:]], dim=0)
                    y_next = self._heun_corrector(
                        y_next_aligned, y_hat, t_next, t_hat, d, x_aug, mask_aug, joint_model_fn, key
                    )

                # keep only queries after the backward step
                y_q = y_next[num_ctx:]

                # (B) FORWARD *EDM* step σ_{i+1} → σ_i for QUERIES ONLY (RePaint refinement)
                # do this only if there is another inner iteration to go
                if u < num_inner_steps - 1:
                    # Add the exact extra noise needed to go from σ_{i+1} back to σ_i in EDM:
                    # y(σ_i) = y(σ_{i+1}) + sqrt(σ_i^2 - σ_{i+1}^2) * ξ
                    delta_var = torch.clamp(sigma_i ** 2 - sigma_next ** 2, min=0.0)
                    if float(delta_var) > 0.0:
                        z = torch.randn(y_q.shape,device=y_q.device, dtype=y_q.dtype, generator=key)
                        y_q = y_q + torch.sqrt(delta_var) * z

            # After U inner loops, y_q is already at σ_{i+1}; continue to next i.

        return y_q



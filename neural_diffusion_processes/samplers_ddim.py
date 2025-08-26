# samplers_ddim.py (refined parts only)
from __future__ import annotations
from dataclasses import dataclass
import torch
from typing import Callable, Optional

EpsModel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]

@dataclass
class DDIMSampler:
    process: any               # expects .alpha_bars [T]
    num_sample_steps: int = 50
    eta: float = 0.0           # 0 = deterministic DDIM, 1 ≈ DDPM variance

    def __post_init__(self):
        T  = int(self.process.alpha_bars.numel())
        ts = torch.linspace(T - 1, 0, steps=self.num_sample_steps)
        idx = ts.round().to(torch.long)                  # coarse, descending indices
        self.timesteps = torch.unique_consecutive(idx)   # remove consecutive dupes
        if self.timesteps[-1].item() != 0:
            self.timesteps = torch.cat(
                [self.timesteps, torch.tensor([0], device=idx.device, dtype=torch.long)]
            )

    # ---- General DDIM step (supports any eta) ------------------------------
    def _ddim_step(self, y, x, mask, t_i: int, t_j: int, model_fn: EpsModel, key: torch.Generator):
        """
        y  : current x_{t_i}
        return: x_{t_j} using DDIM Eq. (12) with σ_eta from Eq. (16)
        """
        device = y.device
        abars = self.process.alpha_bars.to(device)

        t_i_t = torch.tensor(t_i, device=device, dtype=torch.long)
        t_j_t = torch.tensor(t_j, device=device, dtype=torch.long)

        abar_i = abars[t_i_t]            # \bar{α}_{t_i}
        abar_j = abars[t_j_t]            # \bar{α}_{t_j}

        a_i  = torch.sqrt(abar_i)        # √\bar{α}_{t_i}
        ai1  = torch.sqrt(torch.clamp(1.0 - abar_i, min=0.0))
        a_j  = torch.sqrt(abar_j)        # √\bar{α}_{t_j}

        # ε̂(x_{t_i}, t_i)
        eps_hat = model_fn(t_i_t, y, x, mask, key=key)

        # x̂_0 from the forward relation
        x0_hat = (y - ai1 * eps_hat) / (a_i + 1e-12)

        # σ_eta from DDIM Eq. (16)
        if self.eta > 0:
            # sigma_eta = eta * sqrt((1 - abar_j)/(1 - abar_i)) * sqrt(1 - abar_i/abar_j)
            term1 = torch.clamp((1.0 - abar_j) / (1.0 - abar_i + 1e-12), min=0.0)
            term2 = torch.clamp(1.0 - (abar_i / (abar_j + 1e-12)), min=0.0)
            sigma_eta = self.eta * torch.sqrt(term1 * term2)
        else:
            sigma_eta = torch.tensor(0.0, device=device, dtype=y.dtype)

        # coefficient for ε̂ at t_j
        coeff_eps = torch.sqrt(torch.clamp(1.0 - abar_j - sigma_eta**2, min=0.0))

        # optional stochastic kick
        if float(sigma_eta) > 0.0:
            z = torch.randn(y.shape, dtype=y.dtype, device=device, generator=key)
        else:
            z = None

        # DDIM Eq. (12): x_{t_j} = √\bar{α}_{t_j} x̂_0 + √(1-\bar{α}_{t_j}-σ^2) ε̂ + σ z
        y_next = a_j * x0_hat + coeff_eps * eps_hat + (sigma_eta * z if z is not None else 0.0)
        return y_next

    # ---- Unconditional -----------------------------------------------------
    @torch.no_grad()
    def sample_uncond(self, key: torch.Generator, x: torch.Tensor, model_fn: EpsModel,
                      mask: Optional[torch.Tensor] = None, y_dim: int = 1):
        device = x.device
        N = x.size(0)
        y = torch.randn((N, y_dim), dtype=x.dtype, device=device, generator=key)  # x_{t_max} ~ N(0,I)
        if mask is None:
            mask = torch.zeros(N, device=device, dtype=x.dtype)

        for i in range(len(self.timesteps) - 1):
            t_i = int(self.timesteps[i].item())
            t_j = int(self.timesteps[i + 1].item())
            y   = self._ddim_step(y, x, mask, t_i, t_j, model_fn, key)
        return y

    # ---- Conditional (contexts fixed to forward distribution at each t) ----
    @torch.no_grad()
    def sample_cond(self, key,
                    x_query, mask_tgt, *,
                    x_context, y_context, mask_context,
                    model_fn, y_dim=1, num_inner_steps: int = 5):
        """
        Conditional DDIM with RePaint-style inner loop.
        Reuses _ddim_step (reverse jump t_i -> t_j),
        and uses the analytic VP forward step (targets only) t_j -> t_i:
            x_{t_i} = sqrt(abar_i/abar_j) * x_{t_j} + sqrt(1 - abar_i/abar_j) * z
        Contexts are always clamped to the correct forward value at the time
        used in each model call, with a single fixed z_ctx per trajectory.
        """
        device, dtype = x_query.device, x_query.dtype
        if mask_tgt is None:     mask_tgt = torch.zeros(x_query.size(0), device=device, dtype=dtype)
        if mask_context is None: mask_context = torch.ones(x_context.size(0), device=device, dtype=dtype)

        abars = self.process.alpha_bars.to(device)
        num_ctx = x_context.size(0)

        # init targets at t_max; fixed context latent for whole trajectory
        y_t = torch.randn((x_query.size(0), y_dim), dtype=dtype, device=device, generator=key)
        z_ctx = torch.randn(y_context.shape, dtype=y_context.dtype, device=y_context.device, generator=key)

        # concat (x, mask) once — reused for all model calls
        x_aug = torch.cat([x_context, x_query], dim=0)
        mask_aug = torch.cat([mask_context, mask_tgt], dim=0)

        for i in range(len(self.timesteps) - 1):
            t_i = int(self.timesteps[i].item())
            t_j = int(self.timesteps[i + 1].item())

            # precompute scalars/coefs we’ll use repeatedly this outer step
            abar_i = abars[t_i]
            abar_j = abars[t_j]
            a_i = torch.sqrt(abar_i)
            ai1 = torch.sqrt(torch.clamp(1.0 - abar_i, min=0.0))
            a_j = torch.sqrt(abar_j)
            aj1 = torch.sqrt(torch.clamp(1.0 - abar_j, min=0.0))

            # forward VP “return” (targets only) from t_j -> t_i
            # x_{t_i} = sqrt(abar_i/abar_j)*x_{t_j} + sqrt(1 - abar_i/abar_j)*z
            ratio = torch.sqrt(torch.clamp(abar_i / (abar_j + 1e-12), min=0.0))
            extra = torch.sqrt(torch.clamp(1.0 - (abar_i / (abar_j + 1e-12)), min=0.0))

            # ----------------------- inner refinements -----------------------
            # Do U-1 cycles of [reverse t_i->t_j ; forward t_j->t_i (targets only)]
            for u in range(max(0, num_inner_steps - 1)):
                # (A) reverse DDIM step t_i -> t_j on joint (ctx+targets)
                y_ctx_t = a_i * y_context + ai1 * z_ctx  # clamp ctx at t_i
                y_aug = torch.cat([y_ctx_t.to(dtype), y_t], dim=0)
                y_aug_n = self._ddim_step(y_aug, x_aug, mask_aug, t_i, t_j, model_fn, key)
                y_t = y_aug_n[num_ctx:]  # keep updated targets at t_j

                # (B) forward VP step t_j -> t_i for targets only (repaint)
                z = torch.randn(y_t.shape, dtype=y_t.dtype, device=y_t.device, generator=key)
                y_t = ratio * y_t + extra * z  # back to t_i (targets only)

            # ----------------------- final reverse step ----------------------
            # one last reverse DDIM step t_i -> t_j (no forward after)
            y_ctx_t = a_i * y_context + ai1 * z_ctx
            y_aug = torch.cat([y_ctx_t.to(dtype), y_t], dim=0)
            y_aug_n = self._ddim_step(y_aug, x_aug, mask_aug, t_i, t_j, model_fn, key)

            # re-clamp ctx at t_j; carry targets forward
            y_ctx_t_next = a_j * y_context + aj1 * z_ctx
            y_t = torch.cat([y_ctx_t_next.to(dtype), y_aug_n[num_ctx:]], dim=0)[num_ctx:]

        return y_t


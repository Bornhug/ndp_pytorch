# samplers_ddim.py
from __future__ import annotations
from dataclasses import dataclass
import torch
from typing import Callable, Optional

EpsModel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]

@dataclass
class DDIMSampler:
    process: any               # expects .alpha_bars [T]
    num_sample_steps: int = 50 # e.g., 50
    eta: float = 0.0           # 0 = deterministic DDIM

    def __post_init__(self):
        T = int(self.process.alpha_bars.numel())
        # indices from T-1 → 0 (coarse schedule, unique & sorted)
        ts = torch.linspace(T - 1, 0, steps=self.num_sample_steps)
        self.timesteps = torch.unique(ts.round().to(torch.long), sorted=True, return_inverse=False)
        if self.timesteps[-1].item() != 0:
            self.timesteps[-1] = torch.tensor(0)  # ensure last is 0

    # ---- core deterministic DDIM step (eta=0) -----------------------------
    def _ddim_step(self, y, x, mask, t_i: int, t_j: int, model_fn: EpsModel, key: torch.Generator):
        device = y.device
        abars = self.process.alpha_bars.to(device)

        t_i_t = torch.tensor(t_i, device=device, dtype=torch.long)
        t_j_t = torch.tensor(t_j, device=device, dtype=torch.long)

        a_i  = torch.sqrt(abars[t_i_t])
        ai1  = torch.sqrt(1.0 - abars[t_i_t])
        a_j  = torch.sqrt(abars[t_j_t])
        aj1  = torch.sqrt(1.0 - abars[t_j_t])

        eps_hat = model_fn(t_i_t, y, x, mask, key=key) # [N, y_dim]
        x0_hat  = (y - ai1 * eps_hat) / (a_i + 1e-12)
        # η=0 deterministic update:
        y_next  = a_j * x0_hat + aj1 * eps_hat
        return y_next

    # ---- Unconditional ----------------------------------------------------
    @torch.no_grad()
    def sample_uncond(self, key: torch.Generator, x: torch.Tensor, model_fn: EpsModel,
                      mask: Optional[torch.Tensor] = None, y_dim: int = 1):
        device = x.device
        N = x.size(0)
        y = torch.randn(N, y_dim, device=device, generator=key)
        if mask is None:
            mask = torch.zeros(N, device=device)

        for i in range(len(self.timesteps) - 1):
            t_i = int(self.timesteps[i].item())
            t_j = int(self.timesteps[i + 1].item())
            y   = self._ddim_step(y, x, mask, t_i, t_j, model_fn, key)
        return y

    # ---- Conditional (context points fixed) --------------------------------
    @torch.no_grad()
    def sample_cond(self, key: torch.Generator,
                    x_query: torch.Tensor, mask_tgt: Optional[torch.Tensor],
                    *, x_context: torch.Tensor, y_context: torch.Tensor, mask_context: Optional[torch.Tensor],
                    model_fn: EpsModel, y_dim: int = 1):
        device = x_query.device
        if mask_tgt is None:         mask_tgt = torch.zeros(x_query.size(0), device=device)
        if mask_context is None:     mask_context = torch.zeros(x_context.size(0), device=device)

        # Precompute DDPM forward to time t for context
        abars = self.process.alpha_bars.to(device)
        num_ctx = x_context.size(0)
        y_t = torch.randn(x_query.size(0), y_dim, device=device, generator=key)  # init at t_max

        for i in range(len(self.timesteps) - 1):
            t_i = int(self.timesteps[i].item())
            t_j = int(self.timesteps[i + 1].item())
            # simulate y_context at t_i
            a_i = torch.sqrt(abars[t_i])
            ai1 = torch.sqrt(1.0 - abars[t_i])
            eps_ctx = torch.randn_like(y_context, generator=key)
            y_ctx_t = a_i * y_context + ai1 * eps_ctx

            # joint pass
            x_aug    = torch.cat([x_context, x_query], dim=0)
            mask_aug = torch.cat([mask_context, mask_tgt], dim=0)
            y_aug    = torch.cat([y_ctx_t, y_t], dim=0)

            def joint_model_fn(t_scalar, yt, xx, mm, *, key):
                return model_fn(t_scalar, yt, xx, mm, key=key)

            y_aug_next = self._ddim_step(y_aug, x_aug, mask_aug, t_i, t_j, joint_model_fn, key)
            y_t = y_aug_next[num_ctx:]  # drop context portion
        return y_t

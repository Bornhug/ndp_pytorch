# ndp_diffusion_torch.py
# PyTorch port of the original JAX GaussianDiffusion utilities
# ============================================================

from __future__ import annotations
from typing import Protocol, Tuple
import math
import torch
import torch.nn.functional as F


# ---------- helpers ---------------------------------------------------------
def _expand_to(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Make `a` broadcast along the trailing axes of `ref`
    so that shapes line up for element-wise arithmetic.
    """
    return a.reshape(a.shape + (1,) * (ref.ndim - a.ndim))


def cosine_schedule(beta_start: float,
                    beta_end: float,
                    timesteps: int,
                    s: float = 0.008) -> torch.Tensor:
    """
    DDPM-style cosine β-schedule (Nichol & Dhariwal, 2021).

    Returns: 1-D tensor of length `timesteps`
    """
    x  = torch.linspace(0, timesteps, timesteps + 1)
    f  = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = f / f[0]                       # cumulative ᾱ_t
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = betas.clamp(1e-4, 0.9999)           # numerical safety
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    return betas * (beta_end - beta_start) + beta_start


# ---------- protocol for the ε-network -------------------------------------
class EpsModel(Protocol):
    def __call__(self,
                 t: torch.Tensor,        # scalar int64        []
                 yt: torch.Tensor,       # noisy targets       [N, y_dim]
                 x: torch.Tensor,        # inputs              [N, x_dim]
                 mask: torch.Tensor,     # 1 if "missing"      [N]
                 *,
                 key: torch.Generator   # for dropout etc.
                 ) -> torch.Tensor:      # predicted noise     [N, y_dim]
        ...


# ---------- main diffusion engine ------------------------------------------
class GaussianDiffusion:
    """
    Implements q(x_t|x_0) and p_θ(x_{t-1}|x_t) for a fixed β-schedule.
    """

    def __init__(self, betas: torch.Tensor) -> None:
        """
        betas : shape [T]  — variance schedule (0 < β_t < 1)
        """
        self.device = betas.device
        self.dtype  = betas.dtype
        self.betas        = betas                                    # [T]
        self.alphas       = 1.0 - betas                              # [T]
        self.alpha_bars   = torch.cumprod(self.alphas, dim=0)        # [T]

    # ------------------------------------------------------------------ pt0
    def pt0(self,
            y0: torch.Tensor,        # clean targets [N, y_dim]
            t:  torch.Tensor         # scalar int64  []
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Closed-form p(y_t | y_0) moments.
        Returns (mean, var) each of shape [N, y_dim].
        """
        ᾱ_t = _expand_to(self.alpha_bars[t], y0)
        m    = torch.sqrt(ᾱ_t) * y0
        v    = (1.0 - ᾱ_t)
        return m, v   # [N, y_dim]

    # ---------------------------------------------------------------- forward
    def forward(self,
                key: torch.Generator,  # a random seed class
                y0: torch.Tensor,
                t:  torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample y_t and return the exact noise ε so that
            y_t = √ᾱ_t y0 + √(1-ᾱ_t) ε
        """
        m, v   = self.pt0(y0, t)
        noise  = torch.randn_like(y0, generator=key) # We can't use a single scalar noise as noises have to be iid
        yt     = m + torch.sqrt(v) * noise
        return yt, noise # [N, y_dim]

    # -------------------------------------------------- single reverse step
    def ddpm_backward_step(self,
                           key: torch.Generator,
                           noise: torch.Tensor,  # ε̂_θ [N, y_dim]
                           yt:   torch.Tensor,   # [N, y_dim]
                           t:    torch.Tensor    # Scalar []
                           ) -> torch.Tensor:    # [N, y_dim]
        """
        Deterministic DDPM μ_t plus stochastic σ_t z  (Alg. 1 in paper).
        """
        β_t      = _expand_to(self.betas[t], yt)
        α_t      = _expand_to(self.alphas[t], yt)
        ᾱ_t     = _expand_to(self.alpha_bars[t], yt)

        z = torch.zeros_like(yt)  # when t=0, we don't add Gaussian noise
        if t.item() > 0:                           # no noise at t=0
            z = torch.randn_like(yt, generator=key)

        a = 1.0 / torch.sqrt(α_t)
        b = β_t / torch.sqrt(1.0 - ᾱ_t)
        mean = a * (yt - b * noise)
        return mean + torch.sqrt(β_t) * z

    # TODO: check again, seems ddpm_backward_mean_var never used
    # ---------------------------------------------- mean / var diagnostics
    def ddpm_backward_mean_var(self,
                               noise: torch.Tensor, # ε̂_θ [N, y_dim]
                               yt:    torch.Tensor, # [N, y_dim]
                               t:     torch.Tensor  # Scalar []
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        β_t  = _expand_to(self.betas[t], yt)
        α_t  = _expand_to(self.alphas[t], yt)
        ᾱ_t = _expand_to(self.alpha_bars[t], yt)
        mean = (yt - β_t / torch.sqrt(1 - ᾱ_t) * noise) / torch.sqrt(α_t)
        var  = torch.clamp(β_t * (t > 0), min=1e-3)
        return mean, var

    # ----------------------------------------------------------- full sample
    def sample(self,
               key: torch.Generator,
               x:   torch.Tensor,           # [N, x_dim]
               mask: torch.Tensor | None,
               *,
               model_fn: EpsModel,
               output_dim: int = 1
               ) -> torch.Tensor:
        """
        Draw *unconditional* sample y(x) from learned reverse process.
        """
        device = self.device
        B      = x.size(0)
        y = torch.randn(B, output_dim, device=device, dtype=self.dtype,
                        generator=key)             # initial y_T

        if mask is None:
            mask = torch.zeros(B, device=device, dtype=self.dtype)

        for t in reversed(range(len(self.betas))):
            g1, g2 = torch.Generator(device), torch.Generator(device)
            g1.manual_seed(torch.randint(0, 2**63-1, (1,)).item())
            g2.manual_seed(torch.randint(0, 2**63-1, (1,)).item())

            eps_hat = model_fn(torch.tensor(t, device=device),
                               y, x, mask, key=g1)
            y = self.ddpm_backward_step(g2, eps_hat, y,
                                        torch.tensor(t, device=device))
        return y

    # ------------------------------------------------------- repaint sampler
    # (kept identical in spirit; see paper Appendix C)
    # ----------------------------------------------------------------------

    # -------------------------------------------------- conditional sampler
    @torch.no_grad()
    def conditional_sample(
        self,
        key: torch.Generator,
        x: torch.Tensor,                 # [N, x_dim]      – target inputs
        mask: torch.Tensor | None,
        *,
        x_context: torch.Tensor,         # [M, x_dim]
        y_context: torch.Tensor,         # [M, y_dim]
        mask_context: torch.Tensor | None,
        model_fn: EpsModel,
        num_inner_steps: int = 5,
        method: str = "repaint",
    ) -> torch.Tensor:
        """
        RePaint sampling (Lugmayr et al., 2022) for conditional generation.

        The *context* points (x_c, y_c) stay fixed; the algorithm keeps
        resampling the *target* points until they agree with the context
        under the learned reverse process.
        """
        assert method == "repaint", "only the ‘repaint’ method is implemented"

        device, dtype = self.device, self.dtype
        B_tgt         = x.size(0)
        y_dim         = y_context.size(-1)

        # ---- default masks ------------------------------------------------
        if mask         is None: mask         = torch.zeros(B_tgt, device=device, dtype=dtype)
        if mask_context is None: mask_context = torch.zeros(len(x_context), device=device, dtype=dtype)

        # ---- concatenate arrays for one network call ----------------------
        x_aug    = torch.cat([x_context, x], dim=0)          # [(M+N), x_dim]
        mask_aug = torch.cat([mask_context, mask], dim=0)    # [(M+N)]
        M_ctx    = len(x_context)

        # ---- initial noisy target at time T -------------------------------
        y_t = torch.randn(B_tgt, y_dim, device=device, dtype=dtype, generator=key)

        # ---- main outer loop: t = T‑1 … 1  (we skip t=0 like the JAX code)-
        for t in range(len(self.betas) - 1, 0, -1):
            t_tensor = torch.tensor(t, device=device)

            # ---------- inner RePaint loop  (forward‑back‑forward‑…)
            for _ in range(num_inner_steps):
                # (1) simulate y_c at time t
                g_fwd = torch.Generator(device); g_fwd.manual_seed(torch.randint(0, 2**63-1, (1,)).item())
                y_ctx_t, _ = self.forward(g_fwd, y_context, t_tensor)  # [M, y_dim]

                y_aug = torch.cat([y_ctx_t, y_t], dim=0)               # [(M+N), y_dim]

                # (2) reverse step t → t‑1 on both context+target
                g_model = torch.Generator(device); g_model.manual_seed(torch.randint(0, 2**63-1, (1,)).item())
                g_rev   = torch.Generator(device); g_rev.manual_seed(torch.randint(0, 2**63-1, (1,)).item())

                eps_hat = model_fn(t_tensor, y_aug, x_aug, mask_aug, key=g_model)
                y_prev  = self.ddpm_backward_step(g_rev, eps_hat, y_aug, t_tensor)  # [(M+N), y_dim]
                y_t     = y_prev[M_ctx:]                                           # keep only targets

                # (3) *forward* step t‑1 → t **only for targets**
                beta_tm1 = _expand_to(self.betas[t-1], y_t)
                z        = torch.randn_like(y_t, generator=g_fwd)
                y_t      = torch.sqrt(1.0 - beta_tm1) * y_t + torch.sqrt(beta_tm1) * z

            # ---------- final reverse step of outer loop  (t → t‑1)
            # re‑compute context at time t
            g_fwd = torch.Generator(device); g_fwd.manual_seed(torch.randint(0, 2**63-1, (1,)).item())
            y_ctx_t, _ = self.forward(g_fwd, y_context, t_tensor)

            y_aug  = torch.cat([y_ctx_t, y_t], dim=0)
            g_model = torch.Generator(device); g_model.manual_seed(torch.randint(0, 2**63-1, (1,)).item())
            g_rev   = torch.Generator(device); g_rev.manual_seed(torch.randint(0, 2**63-1, (1,)).item())

            eps_hat = model_fn(t_tensor, y_aug, x_aug, mask_aug, key=g_model)
            y_prev  = self.ddpm_backward_step(g_rev, eps_hat, y_aug, t_tensor)
            y_t     = y_prev[M_ctx:]                    # drop context portion

        # after the loop y_t is at time 0 → final sample
        return y_t



# ---------- training loss --------------------------------------------------
def diffusion_loss(process: GaussianDiffusion,
                   network: EpsModel,
                   batch,                       # Batch object with .x_target …
                   key: torch.Generator,
                   *,
                   num_timesteps: int,
                   loss_type: str = "l1") -> torch.Tensor:
    """
    Monte-Carlo estimate of E_t |ε – ε̂| or E_t (ε – ε̂)²   (per DDPM paper).
    """

    metric = (lambda a, b: (a - b).abs()) if loss_type == "l1" \
             else (lambda a, b: (a - b) ** 2)

    def single_point_loss(k: torch.Generator,
                          t: int,
                          y: torch.Tensor,
                          x: torch.Tensor,
                          mask: torch.Tensor):
        yt, noise = process.forward(k, y, torch.tensor(t, device=y.device))
        noise_hat = network(torch.tensor(t, device=y.device),
                            yt, x, mask, key=k)
        per_point = metric(noise, noise_hat).sum(-1)      # [N]
        per_point = per_point * (1.0 - mask)              # ignore masked
        denom = len(mask) - mask.sum()
        return per_point.sum() / denom

    B = batch.x_target.size(0)

    # (i) Strided low-discrepancy draw of t  ∈ {0…T-1}
    g_t = torch.Generator(device=process.device)
    g_t.manual_seed(torch.randint(0, 2**63-1, (1,)).item())
    t0 = torch.randint(0, num_timesteps // B, (B,), generator=g_t,
                       device=process.device)
    t  = t0 + torch.arange(B, device=process.device) * (num_timesteps // B)

    # (ii) Default “all points valid” mask
    mask_target = (torch.zeros_like(batch.x_target[..., 0])
                   if batch.mask_target is None else batch.mask_target)

    # (iii) vmap via list comprehension (Python loop fine for small B)
    losses = []
    for bi in range(B):
        g = torch.Generator()
        g.manual_seed(torch.randint(0, 2**63-1, (1,)).item())
        losses.append(single_point_loss(
            g, int(t[bi].item()),
            batch.y_target[bi],
            batch.x_target[bi],
            mask_target[bi])
        )

    return torch.tensor(losses, device=process.device).mean()

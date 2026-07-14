"""Gaussian diffusion utilities for TabICL-style regression NDPs."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def _expand_to(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape ``a`` so it broadcasts against ``ref``.

    Diffusion schedules are stored as 1-D tensors indexed by timestep, while
    targets usually have shape ``[B, N, C]``. This helper appends singleton
    dimensions to the schedule value so element-wise arithmetic works without
    manually spelling out the target rank each time.
    """
    return a.reshape(a.shape + (1,) * (ref.ndim - a.ndim))


def _as_int(value: int | torch.Tensor) -> int:
    """Convert a scalar int or scalar tensor timestep to a Python ``int``."""
    if isinstance(value, torch.Tensor):
        return int(value.detach().reshape(-1)[0].item())
    return int(value)


def _randn_like(
    ref: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return Gaussian noise matching ``ref`` with generator compatibility.

    Some PyTorch builds do not support ``generator`` in ``torch.randn_like``.
    The fallback uses ``torch.randn`` with explicit shape, dtype, and device so
    callers can still request deterministic sampling where supported.
    """
    if generator is not None and hasattr(generator, "randn_like"):
        return generator.randn_like(ref)
    try:
        return torch.randn_like(ref, generator=generator)
    except TypeError:
        return torch.randn(
            ref.shape,
            dtype=ref.dtype,
            device=ref.device,
            generator=generator,
        )


def _build_reverse_schedule(
    num_timesteps: int,
    num_sample_steps: int | None,
) -> list[int]:
    """Build descending reverse-process timesteps.

    ``None``, non-positive, or overlarge ``num_sample_steps`` means use the
    full DDPM schedule ``T-1, ..., 0``. Smaller values create a coarse schedule
    for accelerated DDIM-style sampling while always including both endpoints.
    """
    if (
        num_sample_steps is None
        or int(num_sample_steps) <= 0
        or int(num_sample_steps) >= num_timesteps
    ):
        return list(range(num_timesteps - 1, -1, -1))

    steps = int(num_sample_steps)
    schedule = torch.linspace(num_timesteps - 1, 0, steps, dtype=torch.long)
    values = sorted({int(v) for v in schedule.tolist()}, reverse=True)
    if values[0] != num_timesteps - 1:
        values.insert(0, num_timesteps - 1)
    if values[-1] != 0:
        values.append(0)
    return values


def cosine_schedule(
    beta_start: float,
    beta_end: float,
    timesteps: int,
    s: float = 0.008,
) -> torch.Tensor:
    """Construct a normalized cosine beta schedule.

    The raw cosine schedule defines cumulative signal retention values
    ``alpha_bar``. Consecutive ratios are converted into betas, clamped for
    numerical stability, then linearly rescaled into
    ``[beta_start, beta_end]``.
    """
    x = torch.linspace(0, timesteps, timesteps + 1)
    f = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = f / f[0]
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = betas.clamp(1e-4, 0.9999)
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    return betas * (beta_end - beta_start) + beta_start


class GaussianDiffusion:
    """Fixed-schedule Gaussian forward and reverse diffusion.

    The class owns the beta schedule and precomputed products needed for both
    training-time noising, ``q(y_t | y_0)``, and inference-time denoising,
    ``p(y_{t-1} | y_t)``.
    """

    def __init__(self, betas: torch.Tensor) -> None:
        """Precompute schedule tensors used throughout diffusion.

        Args:
            betas: 1-D tensor of per-timestep noise variances. The tensor's
                device and dtype become the default sampling device and dtype.
        """
        self.device = betas.device
        self.dtype = betas.dtype
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _normalize_sampling_method(method: str) -> str:
        """Validate and normalize a reverse sampler name."""
        method = str(method).lower()
        if method not in {"ddpm", "ddim"}:
            raise ValueError(f"Unknown sampling method: {method}")
        return method

    def pt0(
        self,
        y0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and variance of ``q(y_t | y_0)``.

        For the forward process, ``y_t`` is Gaussian with mean
        ``sqrt(alpha_bar_t) * y0`` and variance ``1 - alpha_bar_t``. This
        method returns those closed-form moments without sampling.
        """
        alpha_bar_t = self.alpha_bars[t].to(y0.device)
        mean = torch.sqrt(alpha_bar_t) * y0
        var = 1.0 - alpha_bar_t
        return mean, var

    def forward(
        self,
        key: torch.Generator | None,
        y0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample noised targets ``y_t`` for training.

        Returns both the noised target and the exact epsilon noise used to
        produce it. The model is trained to predict that epsilon from
        ``x``, ``y_t``, and ``t``.
        """
        mean, var = self.pt0(y0, t)
        noise = _randn_like(y0, generator=key)
        yt = mean + torch.sqrt(var.clamp(min=1e-8)) * noise
        return yt, noise

    def predict_x0(
        self,
        noise: torch.Tensor,
        yt: torch.Tensor,
        t: int | torch.Tensor,
    ) -> torch.Tensor:
        """Recover the implied clean target from ``y_t`` and predicted noise.

        This is the standard epsilon-parameterized inverse:
        ``x0_hat = (y_t - sqrt(1 - alpha_bar_t) * eps_hat)
        / sqrt(alpha_bar_t)``. DDIM uses this estimate directly.
        """
        t_int = _as_int(t)
        alpha_bar_t = _expand_to(self.alpha_bars[t_int].to(yt.device), yt)
        return (
            yt - torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-8)) * noise
        ) / torch.sqrt(alpha_bar_t.clamp(min=1e-8))

    def ddpm_backward_step(
        self,
        key: torch.Generator | None,
        noise: torch.Tensor,
        yt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Run one stochastic DDPM reverse step.

        Computes the usual epsilon-parameterized reverse mean, then adds
        posterior Gaussian noise except at ``t == 0``. This is the high-quality
        but slower ancestral sampler used when every reverse step is visited.
        """
        beta_t = _expand_to(self.betas[t].to(yt.device), yt)
        alpha_t = _expand_to(self.alphas[t].to(yt.device), yt)
        alpha_bar_t = _expand_to(self.alpha_bars[t].to(yt.device), yt)

        if int(t.item()) > 0:
            alpha_bar_prev = _expand_to(self.alpha_bars[t - 1].to(yt.device), yt)
            z = _randn_like(yt, generator=key)
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)
            z = torch.zeros_like(yt)

        posterior_var = (
            (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_t).clamp(min=1e-8)
            * beta_t
        )
        mean = (yt - beta_t / torch.sqrt((1.0 - alpha_bar_t).clamp(min=1e-8)) * noise)
        mean = mean / torch.sqrt(alpha_t.clamp(min=1e-8))
        return mean + torch.sqrt(posterior_var.clamp(min=0.0)) * z

    def ddim_backward_step(
        self,
        key: torch.Generator | None,
        noise: torch.Tensor,
        yt: torch.Tensor,
        t: int | torch.Tensor,
        *,
        next_t: int | torch.Tensor | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Run one DDIM reverse step.

        DDIM first estimates ``x0`` from the current noised value and predicted
        epsilon, then jumps to ``next_t``. With ``eta == 0`` the step is
        deterministic; larger eta injects controlled stochasticity.
        """
        if float(eta) < 0.0:
            raise ValueError(f"DDIM eta must be non-negative, got {eta}")

        t_int = _as_int(t)
        next_t_int = t_int - 1 if next_t is None else _as_int(next_t)
        x0_hat = self.predict_x0(noise, yt, t_int)
        if next_t_int < 0:
            return x0_hat

        alpha_bar_t = _expand_to(self.alpha_bars[t_int].to(yt.device), yt)
        alpha_bar_next = _expand_to(self.alpha_bars[next_t_int].to(yt.device), yt)

        eta_value = float(eta)
        if eta_value == 0.0:
            sigma = torch.zeros_like(alpha_bar_t)
        else:
            sigma = eta_value * torch.sqrt(
                (
                    (1.0 - alpha_bar_next)
                    / (1.0 - alpha_bar_t).clamp(min=1e-8)
                ).clamp(min=0.0)
                * (
                    1.0
                    - alpha_bar_t / alpha_bar_next.clamp(min=1e-8)
                ).clamp(min=0.0)
            )

        direction = torch.sqrt(
            (1.0 - alpha_bar_next - sigma.pow(2)).clamp(min=0.0)
        ) * noise
        y_next = torch.sqrt(alpha_bar_next.clamp(min=1e-8)) * x0_hat + direction
        if eta_value > 0.0:
            y_next = y_next + sigma * _randn_like(yt, generator=key)
        return y_next

    def reverse_step(
        self,
        key: torch.Generator | None,
        noise: torch.Tensor,
        yt: torch.Tensor,
        t: int | torch.Tensor,
        *,
        method: str = "ddpm",
        next_t: int | torch.Tensor | None = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Dispatch to the configured reverse sampler.

        ``ddpm`` ignores ``next_t`` and steps to ``t-1``. ``ddim`` uses
        ``next_t`` to support sparse accelerated schedules.
        """
        method = self._normalize_sampling_method(method)
        if method == "ddpm":
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(int(t), device=yt.device, dtype=torch.long)
            return self.ddpm_backward_step(key, noise, yt, t.to(device=yt.device))
        return self.ddim_backward_step(
            key,
            noise,
            yt,
            t,
            next_t=next_t,
            eta=eta,
        )

    def ddpm_backward_mean_var(
        self,
        noise: torch.Tensor,
        yt: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return diagnostic DDPM reverse mean and variance.

        This mirrors the DDPM mean calculation but does not sample. It is useful
        for inspecting reverse-process moments or adding tests around the math.
        """
        beta_t = _expand_to(self.betas[t].to(yt.device), yt)
        alpha_t = _expand_to(self.alphas[t].to(yt.device), yt)
        alpha_bar_t = _expand_to(self.alpha_bars[t].to(yt.device), yt)
        mean = (yt - beta_t / torch.sqrt((1 - alpha_bar_t).clamp(min=1e-8)) * noise)
        mean = mean / torch.sqrt(alpha_t.clamp(min=1e-8))
        var = torch.clamp(beta_t * (t > 0), min=1e-3)
        return mean, var

    @torch.inference_mode()
    def sample(
        self,
        key: torch.Generator | None,
        x: torch.Tensor,
        *,
        x_context: torch.Tensor | None,
        y_context: torch.Tensor | None,
        model,
        output_dim: int = 1,
        num_sample_steps: int | None = None,
        method: str = "ddpm",
        eta: float = 0.0,
        amp: bool = False,
        amp_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Draw target values from the learned reverse process.

        Starts from Gaussian noise shaped like the requested targets, evaluates
        the regression model at each reverse timestep, and repeatedly applies
        ``reverse_step`` until a clean prediction is produced.
        """
        method = self._normalize_sampling_method(method)
        batch_size, num_points, _ = x.shape
        y = _randn_like(
            torch.empty(
                batch_size,
                num_points,
                output_dim,
                device=x.device,
                dtype=x.dtype,
            ),
            generator=key,
        )
        schedule = _build_reverse_schedule(len(self.betas), num_sample_steps)
        target_mask = torch.zeros(
            (batch_size, num_points), dtype=torch.bool, device=x.device
        )
        context_mask = (
            None
            if x_context is None
            else torch.zeros(
                x_context.shape[:2], dtype=torch.bool, device=x_context.device
            )
        )
        amp_enabled = (
            bool(amp)
            and x.device.type == "cuda"
            and amp_dtype is not None
            and amp_dtype != torch.float32
        )
        for step_idx, t_int in enumerate(schedule):
            t_batch = torch.full(
                (batch_size,),
                t_int,
                device=x.device,
                dtype=torch.long,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype or torch.float16,
                enabled=amp_enabled,
            ):
                eps_hat = model(
                    x_target=x,
                    y_target=y,
                    t=t_batch,
                    mask_target=target_mask,
                    x_context=x_context,
                    y_context=y_context,
                    mask_context=context_mask,
                )
            eps_hat = eps_hat.float()
            next_t = schedule[step_idx + 1] if step_idx + 1 < len(schedule) else -1
            y = self.reverse_step(
                key,
                eps_hat,
                y,
                t_int,
                method=method,
                next_t=next_t,
                eta=eta,
            )
        return y


def stratified_timesteps(
    batch_size: int,
    num_timesteps: int,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample low-discrepancy training timesteps.

    Instead of drawing every timestep independently, the batch is divided into
    evenly spaced bins over ``[0, num_timesteps)`` and one random offset is
    sampled per bin. This gives each batch broader timestep coverage.
    """
    if device is None:
        device = "cpu"
    step = float(num_timesteps) / float(batch_size)
    offset = torch.rand(batch_size, device=device, generator=generator) * step
    bins = step * torch.arange(batch_size, device=device)
    return (offset + bins).clamp(0, num_timesteps - 1 - 1e-6).long()

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import DataConfig
from .runtime import derived_seed, make_generator
from .types import ConditioningSet, RegressionBatch


def matern52_kernel(
    x1: Tensor,
    x2: Tensor,
    *,
    lengthscale: float = 0.25,
    variance: float = 1.0,
) -> Tensor:
    sqrt5 = math.sqrt(5.0)
    distance = (x1[:, None, :] - x2[None, :, :]).square().sum(-1).sqrt()
    r = distance / lengthscale
    return variance * (1.0 + sqrt5 * r + 5.0 * r.square() / 3.0) * torch.exp(-sqrt5 * r)


def tensor_fingerprint(*tensors: Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().contiguous().cpu()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def historical_tensor_fingerprint(*tensors: Tensor) -> str:
    """Fingerprint tensors with the field order used by the old evaluator."""
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().contiguous().cpu()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def noisy_gp_sample(
    x: Tensor,
    generator: torch.Generator,
    *,
    lengthscale: float,
    variance: float,
    observation_noise_std: float,
    jitter: float,
) -> Tensor:
    count = x.shape[0]
    covariance = matern52_kernel(
        x, x, lengthscale=lengthscale, variance=variance
    ) + jitter * torch.eye(count, device=x.device, dtype=x.dtype)
    chol = torch.linalg.cholesky(covariance)
    latent = chol @ torch.randn(
        (count,), generator=generator, device=x.device, dtype=x.dtype
    )
    noise = observation_noise_std * torch.randn(
        (count,), generator=generator, device=x.device, dtype=x.dtype
    )
    return (latent + noise).unsqueeze(-1)


def _sample_count(low: int, high: int, generator: torch.Generator) -> int:
    if low == high:
        return low
    return int(torch.randint(low, high + 1, (), generator=generator).item())


def sample_training_batch(generator: torch.Generator, config: DataConfig) -> RegressionBatch:
    batch_size = config.batch_size
    num_context = _sample_count(
        config.train_num_context_min, config.train_num_context_max, generator
    )
    num_target = _sample_count(
        config.train_num_target_min, config.train_num_target_max, generator
    )
    x_context = config.x_min + (config.x_max - config.x_min) * torch.rand(
        (batch_size, num_context, 1), generator=generator
    )
    x_target = config.x_min + (config.x_max - config.x_min) * torch.rand(
        (batch_size, num_target, 1), generator=generator
    )
    x_joint = torch.cat((x_context, x_target), dim=1)
    y_joint = torch.stack(
        [
            noisy_gp_sample(
                x_joint[index],
                generator,
                lengthscale=config.lengthscale,
                variance=config.variance,
                observation_noise_std=config.observation_noise_std,
                jitter=config.jitter,
            )
            for index in range(batch_size)
        ],
        dim=0,
    )
    batch = RegressionBatch(
        x_target=x_target,
        y_target=y_joint[:, num_context:],
        x_context=x_context,
        y_context=y_joint[:, :num_context],
        mask_target=torch.zeros((batch_size, num_target), dtype=torch.bool),
        mask_context=torch.zeros((batch_size, num_context), dtype=torch.bool),
    )
    batch.validate()
    return batch


@dataclass(frozen=True)
class MasterTask:
    x_context: Tensor
    y_context: Tensor
    context_ids: Tensor
    core_membership: Tensor
    extra_membership: Tensor
    x_target: Tensor
    task_seed: int
    base_separation: float
    prefix_count: int
    prefix_separation: float
    fingerprint: str


@dataclass(frozen=True)
class PrefixTask:
    context_size: int
    selected_ids: Tensor
    conditioning: ConditioningSet
    x_target: Tensor
    posterior_mean: Tensor
    posterior_covariance: Tensor
    fingerprint: str


def historical_stable_cholesky(
    matrix: Tensor,
    *,
    jitter: float,
    max_tries: int = 6,
) -> Tensor:
    """Match the adaptive Cholesky stabilization used by historical metrics."""
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    base = jitter * matrix.diagonal().median().clamp_min(1.0).item()
    error: RuntimeError | None = None
    for attempt in range(max_tries):
        try:
            return torch.linalg.cholesky(matrix + (10.0**attempt) * base * eye)
        except RuntimeError as exc:
            error = exc
    assert error is not None
    raise error


def _accept_points(
    *,
    accepted: list[Tensor],
    target_count: int,
    separation: float,
    generator: torch.Generator,
    device: torch.device,
    x_min: float,
    x_max: float,
    proposal_budget: list[int],
) -> None:
    while len(accepted) < target_count:
        if proposal_budget[0] <= 0:
            raise RuntimeError("Context rejection sampler exhausted 100,000 proposals")
        proposal_budget[0] -= 1
        candidate = torch.empty((), device=device).uniform_(
            x_min, x_max, generator=generator
        )
        if not accepted or all(
            abs(float((candidate - existing).item())) >= separation
            for existing in accepted
        ):
            accepted.append(candidate)


def build_master_task(
    *,
    max_context: int,
    num_targets: int,
    task_seed: int,
    base_separation: float,
    prefix_count: int,
    prefix_separation: float,
    data: DataConfig,
    device: torch.device,
) -> MasterTask:
    if not 0 < prefix_count <= max_context:
        raise ValueError("prefix_count must lie within the master context size")
    if prefix_separation < base_separation:
        raise ValueError("prefix separation must be at least the full-set separation")
    accepted: list[Tensor] = []
    proposals = [100_000]
    location_generator = make_generator(device, derived_seed(task_seed, "context_locations"))
    _accept_points(
        accepted=accepted,
        target_count=prefix_count,
        separation=prefix_separation,
        generator=location_generator,
        device=device,
        x_min=data.x_min,
        x_max=data.x_max,
        proposal_budget=proposals,
    )


    _accept_points(
        accepted=accepted,
        target_count=max_context,
        separation=base_separation,
        generator=location_generator,
        device=device,
        x_min=data.x_min,
        x_max=data.x_max,
        proposal_budget=proposals,
    )
    x_context = torch.stack(accepted).unsqueeze(-1)
    y_context = noisy_gp_sample(
        x_context,
        make_generator(device, derived_seed(task_seed, "context_values")),
        lengthscale=data.lengthscale,
        variance=data.variance,
        observation_noise_std=data.observation_noise_std,
        jitter=data.jitter,
    )
    context_ids = torch.arange(max_context, device=device, dtype=torch.long)
    core_membership = torch.randperm(
        prefix_count,
        generator=make_generator(device, derived_seed(task_seed, "core_membership")),
        device=device,
    )
    extra_membership = prefix_count + torch.randperm(
        max_context - prefix_count,
        generator=make_generator(device, derived_seed(task_seed, "extra_membership")),
        device=device,
    )
    x_target = torch.linspace(
        data.x_min, data.x_max, num_targets, device=device
    ).unsqueeze(-1)
    return MasterTask(
        x_context=x_context,
        y_context=y_context,
        context_ids=context_ids,
        core_membership=core_membership,
        extra_membership=extra_membership,
        x_target=x_target,
        task_seed=task_seed,
        base_separation=base_separation,
        prefix_count=prefix_count,
        prefix_separation=prefix_separation,
        fingerprint=tensor_fingerprint(x_context, y_context, x_target),
    )


def historical_analytic_posterior(
    x_context: Tensor,
    y_context: Tensor,
    x_target: Tensor,
    *,
    data: DataConfig,
) -> tuple[Tensor, Tensor]:
    """Reproduce the float64 noisy-GP posterior from the historical evaluator."""
    x_context = x_context.double()
    y_context = y_context.double()
    x_target = x_target.double()
    context_covariance = matern52_kernel(
        x_context,
        x_context,
        lengthscale=data.lengthscale,
        variance=data.variance,
    ) + data.observation_noise_std**2 * torch.eye(
        x_context.shape[0], device=x_context.device, dtype=x_context.dtype
    )
    context_target = matern52_kernel(
        x_context,
        x_target,
        lengthscale=data.lengthscale,
        variance=data.variance,
    )
    target_covariance = matern52_kernel(
        x_target,
        x_target,
        lengthscale=data.lengthscale,
        variance=data.variance,
    )
    context_cholesky = historical_stable_cholesky(
        context_covariance, jitter=data.jitter
    )
    alpha = torch.cholesky_solve(y_context.reshape(-1, 1), context_cholesky)
    mean = (context_target.T @ alpha).reshape(-1, 1)
    solved = torch.cholesky_solve(context_target, context_cholesky)
    covariance = target_covariance - context_target.T @ solved
    covariance = covariance + data.observation_noise_std**2 * torch.eye(
        x_target.shape[0], device=x_target.device, dtype=x_target.dtype
    )
    return mean, covariance


def _historical_noisy_gp_context(
    x_context: Tensor,
    generator: torch.Generator,
    *,
    data: DataConfig,
) -> Tensor:
    x = x_context.double()
    covariance = matern52_kernel(
        x,
        x,
        lengthscale=data.lengthscale,
        variance=data.variance,
    ) + data.observation_noise_std**2 * torch.eye(
        x.shape[0], device=x.device, dtype=x.dtype
    )
    cholesky = historical_stable_cholesky(covariance, jitter=data.jitter)
    noise = torch.randn(
        (1, x.shape[0]),
        generator=generator,
        device=x.device,
        dtype=x.dtype,
    )
    return (noise @ cholesky.T)[0].unsqueeze(-1).to(x_context.dtype)


def build_independent_task(
    *,
    context_size: int,
    num_targets: int,
    task_seed: int,
    context_separation: float,
    data: DataConfig,
    device: torch.device,
) -> PrefixTask:
    """Build one independent task using the old shared task RNG stream."""
    if context_size <= 0 or num_targets <= 0:
        raise ValueError("Context and target counts must be positive")
    if context_separation < 0.0:
        raise ValueError("Context separation cannot be negative")
    width = data.x_max - data.x_min
    maximum = (
        math.floor(width / context_separation + 1e-12) + 1
        if context_separation
        else None
    )
    if maximum is not None and context_size > maximum:
        raise ValueError(
            f"Cannot place {context_size} points at least {context_separation} apart"
        )

    generator = make_generator(device, task_seed)
    accepted: list[Tensor] = []
    for _ in range(100_000):
        candidate = torch.empty((1,), device=device, dtype=torch.float32).uniform_(
            data.x_min, data.x_max, generator=generator
        )
        if not accepted or all(
            abs(float((candidate[0] - value[0]).item())) >= context_separation
            for value in accepted
        ):
            accepted.append(candidate)
            if len(accepted) == context_size:
                break
    if len(accepted) != context_size:
        raise RuntimeError(
            "Historical context rejection sampler exhausted 100,000 proposals"
        )

    x_context = torch.stack(accepted, dim=0)
    y_context = _historical_noisy_gp_context(x_context, generator, data=data)
    x_target = torch.linspace(
        data.x_min,
        data.x_max,
        num_targets,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(-1)
    mean, covariance = historical_analytic_posterior(
        x_context, y_context, x_target, data=data
    )
    selected_ids = torch.arange(context_size, device=device, dtype=torch.long)
    return PrefixTask(
        context_size=context_size,
        selected_ids=selected_ids,
        conditioning=ConditioningSet(x_context, y_context, None),
        x_target=x_target,
        posterior_mean=mean,
        posterior_covariance=covariance,
        fingerprint=historical_tensor_fingerprint(
            x_target, x_context, y_context, mean[:, 0], covariance
        ),
    )


def analytic_posterior(
    x_context: Tensor,
    y_context: Tensor,
    x_target: Tensor,
    *,
    data: DataConfig,
) -> tuple[Tensor, Tensor]:
    count = x_context.shape[0]
    identity_context = torch.eye(count, device=x_context.device, dtype=x_context.dtype)
    k_cc = matern52_kernel(
        x_context, x_context, lengthscale=data.lengthscale, variance=data.variance
    ) + (data.observation_noise_std**2 + data.jitter) * identity_context
    k_ct = matern52_kernel(
        x_context, x_target, lengthscale=data.lengthscale, variance=data.variance
    )
    k_tt = matern52_kernel(
        x_target, x_target, lengthscale=data.lengthscale, variance=data.variance
    ) + data.observation_noise_std**2 * torch.eye(
        x_target.shape[0], device=x_target.device, dtype=x_target.dtype
    )
    chol = torch.linalg.cholesky(k_cc)
    alpha = torch.cholesky_solve(y_context, chol)
    mean = k_ct.transpose(0, 1) @ alpha
    solved = torch.cholesky_solve(k_ct, chol)
    covariance = k_tt - k_ct.transpose(0, 1) @ solved
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    return mean, covariance


def build_prefix_task(master: MasterTask, context_size: int, data: DataConfig) -> PrefixTask:
    if not 0 < context_size <= master.x_context.shape[0]:
        raise ValueError("Requested context size is outside the master task")
    if context_size <= master.prefix_count:
        membership = master.core_membership[:context_size]
    else:
        membership = torch.cat(
            (
                master.core_membership,
                master.extra_membership[: context_size - master.prefix_count],
            )
        )
    order = torch.randperm(
        membership.numel(),
        generator=make_generator(
            membership.device, derived_seed(master.task_seed, f"input_order:{context_size}")
        ),
        device=membership.device,
    )
    selected_ids = membership[order]
    if selected_ids.numel() >= 3:
        ordered_x = master.x_context[selected_ids, 0]
        spatially_sorted = bool(
            torch.all(ordered_x[:-1] <= ordered_x[1:])
            or torch.all(ordered_x[:-1] >= ordered_x[1:])
        )
        if spatially_sorted:
            # Preserve membership while making the model-facing rows explicitly
            # unordered, even for the rare seed whose random permutation is monotone.
            selected_ids = torch.roll(selected_ids, shifts=1)
    x_context = master.x_context[selected_ids]
    y_context = master.y_context[selected_ids]
    mean, covariance = analytic_posterior(
        x_context, y_context, master.x_target, data=data
    )
    return PrefixTask(
        context_size=context_size,
        selected_ids=selected_ids,
        conditioning=ConditioningSet(x_context, y_context, None),
        x_target=master.x_target,
        posterior_mean=mean,
        posterior_covariance=covariance,
        fingerprint=tensor_fingerprint(selected_ids, x_context, y_context, master.x_target),
    )


def draw_posterior_samples(
    task: PrefixTask,
    *,
    count: int,
    seed: int,
    jitter: float,
    rng_semantics: str = "batched",
) -> Tensor:
    if rng_semantics in {"sample-major", "historical-batched"}:
        cholesky = historical_stable_cholesky(
            task.posterior_covariance.double(), jitter=jitter
        )
        generator = make_generator(task.x_target.device, seed)
        if rng_semantics == "sample-major":
            noise = torch.cat(
                [
                    torch.randn(
                        (1, task.x_target.shape[0]),
                        generator=generator,
                        device=task.x_target.device,
                        dtype=torch.float64,
                    )
                    for _ in range(count)
                ],
                dim=0,
            )
        else:
            noise = torch.randn(
                (count, task.x_target.shape[0]),
                generator=generator,
                device=task.x_target.device,
                dtype=torch.float64,
            )
        samples = task.posterior_mean.double().reshape(1, -1) + noise @ cholesky.T
        result = samples.unsqueeze(-1)
        if rng_semantics == "historical-batched":
            return result
        return result.to(task.x_target.dtype)
    if rng_semantics != "batched":
        raise ValueError(f"Unknown GP sample RNG semantics: {rng_semantics}")
    covariance = task.posterior_covariance + jitter * torch.eye(
        task.x_target.shape[0],
        device=task.x_target.device,
        dtype=task.x_target.dtype,
    )
    chol = torch.linalg.cholesky(covariance)
    noise = torch.randn(
        (count, task.x_target.shape[0]),
        generator=make_generator(task.x_target.device, seed),
        device=task.x_target.device,
        dtype=task.x_target.dtype,
    )
    return (task.posterior_mean[:, 0].unsqueeze(0) + noise @ chol.transpose(0, 1)).unsqueeze(-1)

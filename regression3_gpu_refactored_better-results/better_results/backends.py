from __future__ import annotations

from math import ceil
from typing import Any, Callable

import torch
from torch import Tensor, nn

from .config import Config
from .integrators import integrate_fixed
from .models import ConditionalNDPModel, FlowNPModel, JointNDPModel
from .processes.conditional import (
    GaussianDiffusion as ConditionalGaussianDiffusion,
    cosine_schedule as conditional_cosine_schedule,
    stratified_timesteps as conditional_stratified_timesteps,
)
from .processes.unconditional import (
    GaussianDiffusion as UnconditionalGaussianDiffusion,
    cosine_schedule as unconditional_cosine_schedule,
    stratified_timesteps as unconditional_stratified_timesteps,
)
from .runtime import SampleMajorGenerator, make_generator
from .types import ConditioningSet, RegressionBatch, SamplingRequest


def _masked_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    loss_type: str,
) -> Tensor:
    if loss_type == "l1":
        point_loss = (prediction - target).abs().sum(dim=-1)
    elif loss_type in {"l2", "mse"}:
        point_loss = (prediction - target).square().sum(dim=-1)
    else:
        raise ValueError(f"Unknown loss: {loss_type}")
    valid = ~mask.to(device=point_loss.device, dtype=torch.bool)
    counts = valid.sum(dim=1).clamp_min(1)
    return ((point_loss * valid).sum(dim=1) / counts).mean()


def _valid_conditioning(
    conditioning: ConditioningSet,
    reference: Tensor,
) -> ConditioningSet:
    x_context = conditioning.x_context.to(reference)
    y_context = conditioning.y_context.to(reference)
    if conditioning.mask_context is None:
        return ConditioningSet(x_context, y_context, None)
    valid = ~conditioning.mask_context.to(device=reference.device, dtype=torch.bool)
    return ConditioningSet(x_context[valid], y_context[valid], None)


class _AdaptiveBackend:
    def __init__(self) -> None:
        self._metadata: dict[str, dict[str, Any]] = {}

    def execution_metadata(self, operation: str = "sampling") -> dict[str, Any]:
        return dict(self._metadata.get(operation, {}))

    def _adaptive_batches(
        self,
        *,
        total: int,
        device: torch.device,
        run_chunk: Callable[[int, int], Tensor],
        requested_batch_size: int | None,
        generator: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Tensor:
        requested = total if requested_batch_size is None else min(total, requested_batch_size)
        if total <= 0 or requested <= 0:
            raise ValueError("Sampling counts and batch size must be positive")
        batch_size = requested
        retries = 0
        initial_state = generator.get_state() if generator is not None else None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        while True:
            if generator is not None and initial_state is not None:
                generator.set_state(initial_state)
            chunks: list[Tensor] = []
            try:
                for start in range(0, total, batch_size):
                    chunks.append(run_chunk(start, min(total, start + batch_size)))
                result = torch.cat(chunks, dim=0)
                details: dict[str, Any] = {
                    "vectorized": batch_size > 1,
                    "requested_batch_size": requested,
                    "effective_batch_size": batch_size,
                    "batch_count": ceil(total / batch_size),
                    "oom_retries": retries,
                }
                if metadata:
                    details.update(metadata)
                if device.type == "cuda":
                    details["peak_cuda_memory_bytes"] = int(
                        torch.cuda.max_memory_allocated(device)
                    )
                self._metadata["sampling"] = details
                return result
            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda" or batch_size == 1:
                    raise
                chunks.clear()
                retries += 1
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()


class _NDPBackend(_AdaptiveBackend):
    name: str
    conditioning_route: str
    model_type: type[nn.Module]

    def build_model(self, config: Config, device: torch.device) -> nn.Module:
        return self.model_type(
            n_layers=config.network.n_layers,
            hidden_dim=config.network.hidden_dim,
            num_heads=config.network.num_heads,
            num_timesteps=config.diffusion.timesteps,
        ).to(device)

    @staticmethod
    def _times(
        batch_size: int,
        config: Config,
        generator: torch.Generator,
        device: torch.device,
        stratified,
    ) -> Tensor:
        if config.ndp.timestep_sampling == "stratified":
            return stratified(
                batch_size,
                config.diffusion.timesteps,
                device=device,
                generator=generator,
            )
        return torch.randint(
            config.diffusion.timesteps,
            (batch_size,),
            generator=generator,
            device=device,
        )

    def sample_conditional(
        self,
        model: nn.Module,
        process,
        x_target: Tensor,
        conditioning: ConditioningSet,
        request: SamplingRequest,
    ) -> Tensor:
        if request.sampler not in {"ddpm", "ddim"}:
            raise ValueError("NDP sampler must be ddpm or ddim")
        if request.sampler == "ddpm" and request.num_steps != process.betas.numel():
            raise ValueError("DDPM requires the complete diffusion schedule")
        if request.rng_semantics not in {"stable", "historical"}:
            raise ValueError(f"Unknown model RNG semantics: {request.rng_semantics}")
        model.eval()
        conditioning = _valid_conditioning(conditioning, x_target)
        historical = request.rng_semantics == "historical"
        generators = (
            make_generator(x_target.device, request.seed)
            if historical
            else SampleMajorGenerator.from_base_seed(
                x_target.device, request.seed, request.num_samples
            )
        )

        def run_chunk(start: int, end: int) -> Tensor:
            batch = end - start
            x_batch = x_target.unsqueeze(0).expand(batch, -1, -1).contiguous()
            x_context = conditioning.x_context.unsqueeze(0).expand(batch, -1, -1).contiguous()
            y_context = conditioning.y_context.unsqueeze(0).expand(batch, -1, -1).contiguous()
            return self._sample_chunk(
                model,
                process,
                x_batch,
                x_context,
                y_context,
                request,
                generators if historical else generators.subset(start, end),
            )

        metadata = {
            "sampler": request.sampler,
            "conditioning_route": self.conditioning_route,
            "sample_seed_semantics": (
                "historical_shared_batch"
                if historical
                else "counter_based_sample_major"
            ),
        }
        if self.name == "ndp_uncond":
            metadata.update(
                conditioning_algorithm="repaint",
                repaint_inner_steps=int(process.repaint_inner_steps),
                repaint_context_noise="fresh",
            )
        return self._adaptive_batches(
            total=request.num_samples,
            device=x_target.device,
            run_chunk=run_chunk,
            requested_batch_size=request.batch_size,
            generator=generators,
            metadata=metadata,
        )

    def _sample_chunk(self, model, process, x_target, x_context, y_context, request, generator):
        raise NotImplementedError


class ConditionalNDPBackend(_NDPBackend):
    name = "ndp_cond"
    conditioning_route = "direct_clean_context"
    model_type = ConditionalNDPModel

    def build_process(self, config: Config, device: torch.device) -> ConditionalGaussianDiffusion:
        betas = conditional_cosine_schedule(
            config.diffusion.beta_start,
            config.diffusion.beta_end,
            config.diffusion.timesteps,
        ).to(device)
        process = ConditionalGaussianDiffusion(betas)
        process.ddim_eta = config.ndp.ddim_eta
        return process

    def training_loss(self, model, process, batch, generator, config) -> Tensor:
        times = self._times(
            batch.batch_size,
            config,
            generator,
            batch.y_target.device,
            conditional_stratified_timesteps,
        )
        noisy, noise = process.forward(
            generator, batch.y_target.float(), times.view(-1, 1, 1)
        )
        prediction = model(
            batch.x_target,
            noisy,
            times,
            batch.mask_target,
            x_context=batch.x_context,
            y_context=batch.y_context,
            mask_context=batch.mask_context,
        )
        return _masked_loss(
            prediction, noise, batch.mask_target, loss_type=config.training.loss_type
        )

    def _sample_chunk(self, model, process, x_target, x_context, y_context, request, generator):
        return process.sample(
            generator,
            x_target,
            x_context=x_context,
            y_context=y_context,
            model=model,
            output_dim=request.y_dim,
            num_sample_steps=request.num_steps,
            method=request.sampler,
            eta=float(process.ddim_eta),
        )


class UnconditionalNDPBackend(_NDPBackend):
    name = "ndp_uncond"
    conditioning_route = "repaint"
    model_type = JointNDPModel

    def build_process(self, config: Config, device: torch.device) -> UnconditionalGaussianDiffusion:
        betas = unconditional_cosine_schedule(
            config.diffusion.beta_start,
            config.diffusion.beta_end,
            config.diffusion.timesteps,
        ).to(device)
        process = UnconditionalGaussianDiffusion(betas)
        process.repaint_inner_steps = config.ndp.repaint_inner_steps
        process.repaint_context_noise = config.ndp.repaint_context_noise
        process.ddim_eta = config.ndp.ddim_eta
        return process

    def training_loss(self, model, process, batch, generator, config) -> Tensor:
        x_joint = torch.cat((batch.x_context, batch.x_target), dim=1)
        y_joint = torch.cat((batch.y_context, batch.y_target), dim=1)
        mask_joint = torch.cat((batch.mask_context, batch.mask_target), dim=1)
        times = self._times(
            batch.batch_size,
            config,
            generator,
            y_joint.device,
            unconditional_stratified_timesteps,
        )
        noisy, noise = process.forward(
            generator, y_joint.float(), times.view(-1, 1, 1)
        )
        prediction = model(x_joint, noisy, times, mask_joint)
        return _masked_loss(
            prediction, noise, mask_joint, loss_type=config.training.loss_type
        )

    def _sample_chunk(self, model, process, x_target, x_context, y_context, request, generator):
        return process.conditional_sample(
            generator,
            x_target,
            x_context=x_context,
            y_context=y_context,
            model=model,
            num_inner_steps=int(process.repaint_inner_steps),
            method=request.sampler,
            num_sample_steps=request.num_steps,
            eta=float(process.ddim_eta),
        )


class FlowNPBackend(_AdaptiveBackend):
    name = "flownp"

    def build_model(self, config: Config, device: torch.device) -> nn.Module:
        model = FlowNPModel(config.data.input_dim, config.flownp).to(device)
        model.backend_name = self.name
        return model

    def build_process(self, config: Config, device: torch.device) -> None:
        del config, device
        return None

    def training_loss(self, model, process, batch, generator, config) -> Tensor:
        del process
        noise = torch.randn(
            batch.y_target.shape,
            generator=generator,
            device=batch.y_target.device,
            dtype=batch.y_target.dtype,
        )
        if config.flownp.time_distribution == "stratified":
            offsets = torch.rand(
                (batch.batch_size,),
                generator=generator,
                device=batch.y_target.device,
                dtype=batch.y_target.dtype,
            )
            bins = torch.arange(
                batch.batch_size,
                device=batch.y_target.device,
                dtype=batch.y_target.dtype,
            )
            time = ((bins + offsets) / float(batch.batch_size)).view(-1, 1, 1)
        else:
            shape = (
                batch.y_target.shape
                if config.flownp.time_sampling == "per_target"
                else (batch.batch_size, 1, 1)
            )
            time = torch.rand(
                shape,
                generator=generator,
                device=batch.y_target.device,
                dtype=batch.y_target.dtype,
            )
        intermediate = (1.0 - time) * noise + time * batch.y_target
        target = batch.y_target - noise
        prediction = model(
            batch.x_target,
            intermediate,
            time,
            batch.mask_target,
            x_context=batch.x_context,
            y_context=batch.y_context,
            mask_context=batch.mask_context,
        )
        return _masked_loss(
            prediction,
            target,
            batch.mask_target,
            loss_type=config.training.loss_type,
        )

    @staticmethod
    def _velocity(model, x_target, conditioning, batch_size):
        x_batch = x_target.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        target_mask = torch.zeros(
            (batch_size, x_target.shape[0]), device=x_target.device, dtype=torch.bool
        )
        x_context = conditioning.x_context.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        y_context = conditioning.y_context.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        context_mask = torch.zeros(
            (batch_size, x_context.shape[1]), device=x_target.device, dtype=torch.bool
        )

        def velocity(value: Tensor, time: Tensor) -> Tensor:
            return model(
                x_batch,
                value,
                time,
                target_mask,
                x_context=x_context,
                y_context=y_context,
                mask_context=context_mask,
            )

        return velocity

    def sample_conditional(self, model, process, x_target, conditioning, request) -> Tensor:
        del process
        if request.sampler not in {"euler", "midpoint", "rk4"}:
            raise ValueError("FlowNP sampler must be euler, midpoint, or rk4")
        if request.rng_semantics not in {"stable", "historical"}:
            raise ValueError(f"Unknown model RNG semantics: {request.rng_semantics}")
        model.eval()
        conditioning = _valid_conditioning(conditioning, x_target)
        initial = torch.stack(
            [
                torch.randn(
                    (x_target.shape[0], request.y_dim),
                    generator=make_generator(x_target.device, request.seed + index),
                    device=x_target.device,
                    dtype=x_target.dtype,
                )
                for index in range(request.num_samples)
            ]
        )

        def run_chunk(start: int, end: int) -> Tensor:
            velocity = self._velocity(model, x_target, conditioning, end - start)
            return integrate_fixed(
                velocity,
                initial[start:end],
                steps=request.num_steps,
                method=request.sampler,
            )

        return self._adaptive_batches(
            total=request.num_samples,
            device=x_target.device,
            run_chunk=run_chunk,
            requested_batch_size=request.batch_size,
            metadata={
                "sampler": request.sampler,
                "conditioning_route": "direct_clean_context",
                "sample_seed_semantics": "base_plus_sample_index",
            },
        )


BACKENDS = {
    "ndp_cond": ConditionalNDPBackend,
    "ndp_uncond": UnconditionalNDPBackend,
    "flownp": FlowNPBackend,
}


def get_backend(name: str):
    try:
        return BACKENDS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown model: {name}") from exc

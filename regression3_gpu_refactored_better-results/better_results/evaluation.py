from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .backends import get_backend
from .checkpoints import (
    config_from_checkpoint,
    file_sha256,
    load_model_state,
    torch_load,
)
from .config import Config, MODELS
from .gp import (
    MasterTask,
    PrefixTask,
    build_independent_task,
    build_master_task,
    build_prefix_task,
    draw_posterior_samples,
    historical_tensor_fingerprint,
    tensor_fingerprint,
)
from .plotting import PANEL_ORDER, plot_comparison
from .runtime import resolve_device
from .training import EXPECTED_PARAMETERS
from .types import SamplingRequest


SAMPLING_SEMANTICS_VERSION = 1
TASK_CONSTRUCTION_VERSION = "nested_unordered_v1"
INDEPENDENT_TASK_CONSTRUCTION_VERSION = "independent_historical_v1"


def _gp_settings() -> dict[str, Any]:
    data = Config.for_model("ndp_cond").data
    return {
        "kernel": "matern52",
        "lengthscale": data.lengthscale,
        "variance": data.variance,
        "observation_noise_std": data.observation_noise_std,
        "target_noise": "observed",
        "cholesky_jitter": data.jitter,
        "x_bounds": [data.x_min, data.x_max],
        "query_layout": "linspace",
    }


@dataclass(frozen=True)
class EvaluationSpec:
    runs_root: Path
    contexts: tuple[int, ...]
    num_targets: int
    samples: int
    task_seed: int
    gp_sample_seed: int
    model_sample_seed: int
    context_separation: float
    prefix_count: int
    prefix_separation: float
    sampling_steps: int
    batch_size: int
    output_dir: Path
    task_construction: str = "nested"
    model_rng_semantics: str = "stable"
    gp_sample_rng_semantics: str = "batched"
    context_separation_overrides: tuple[tuple[int, float], ...] = ()

    def separation_for(self, context: int) -> float:
        return float(dict(self.context_separation_overrides).get(context, self.context_separation))

    @property
    def task_construction_version(self) -> str:
        if self.task_construction == "independent":
            return INDEPENDENT_TASK_CONSTRUCTION_VERSION
        return TASK_CONSTRUCTION_VERSION

    def validate(self) -> None:
        if not self.contexts or tuple(sorted(set(self.contexts))) != self.contexts:
            raise ValueError("Contexts must be unique and sorted")
        if self.contexts[0] <= 0:
            raise ValueError("Contexts must be positive")
        if self.task_construction not in {"nested", "independent"}:
            raise ValueError("Task construction must be nested or independent")
        if self.model_rng_semantics not in {"stable", "historical"}:
            raise ValueError("Model RNG semantics must be stable or historical")
        if self.gp_sample_rng_semantics not in {
            "batched",
            "sample-major",
            "historical-batched",
        }:
            raise ValueError(
                "GP sample RNG semantics must be batched, sample-major, or "
                "historical-batched"
            )
        if self.task_construction == "nested" and self.contexts[-1] < self.prefix_count:
            raise ValueError("Maximum context must include the separated prefix")
        if min(self.num_targets, self.samples, self.sampling_steps, self.batch_size) <= 0:
            raise ValueError("Evaluation counts must be positive")
        if min(self.context_separation, self.prefix_separation) < 0.0:
            raise ValueError("Separations cannot be negative")
        if self.task_construction == "nested" and self.prefix_separation < self.context_separation:
            raise ValueError("Prefix separation cannot be smaller than full-set separation")
        overrides = dict(self.context_separation_overrides)
        if len(overrides) != len(self.context_separation_overrides):
            raise ValueError("Context separation overrides must be unique")
        if any(context <= 0 or separation < 0.0 for context, separation in overrides.items()):
            raise ValueError("Context separation overrides are invalid")
        if any(context not in self.contexts for context in overrides):
            raise ValueError("Context separation overrides must name requested contexts")

    def serializable(self) -> dict[str, Any]:
        result = asdict(self)
        result["implementation_version"] = 4
        result["sampling_semantics_version"] = SAMPLING_SEMANTICS_VERSION
        result["task_construction_version"] = self.task_construction_version
        result["gp"] = _gp_settings()
        result["runs_root"] = str(self.runs_root.resolve())
        result["output_dir"] = str(self.output_dir.resolve())
        result["contexts"] = list(self.contexts)
        result["context_separation_overrides"] = [
            [context, separation]
            for context, separation in self.context_separation_overrides
        ]
        return result


@dataclass
class Runtime:
    name: str
    backend: Any
    model: nn.Module
    process: Any
    config: Config
    checkpoint: Path
    checkpoint_sha256: str
    step: int


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return __import__("hashlib").sha256(encoded).hexdigest()


def _evaluation_semantics(payload: dict[str, Any]) -> dict[str, Any]:
    models = payload.get("models", {})
    return {
        "sampling_semantics_version": int(
            payload.get("sampling_semantics_version", SAMPLING_SEMANTICS_VERSION)
        ),
        "task_construction": payload.get("task_construction", "nested"),
        "task_construction_version": payload.get(
            "task_construction_version", TASK_CONSTRUCTION_VERSION
        ),
        "model_rng_semantics": payload.get("model_rng_semantics", "stable"),
        "gp_sample_rng_semantics": payload.get(
            "gp_sample_rng_semantics", "batched"
        ),
        "contexts": list(payload["contexts"]),
        "num_targets": int(payload["num_targets"]),
        "samples": int(payload["samples"]),
        "task_seed": int(payload["task_seed"]),
        "gp_sample_seed": int(payload["gp_sample_seed"]),
        "model_sample_seed": int(payload["model_sample_seed"]),
        "context_separation": float(payload["context_separation"]),
        "context_separation_overrides": [
            [int(context), float(separation)]
            for context, separation in payload.get("context_separation_overrides", [])
        ],
        "prefix_count": int(payload["prefix_count"]),
        "prefix_separation": float(payload["prefix_separation"]),
        "sampling_steps": int(payload["sampling_steps"]),
        "gp": payload.get("gp", _gp_settings()),
        "samplers": {
            "gp": "analytic_gp_posterior",
            "ndp_cond": "ddpm",
            "ndp_uncond": "repaint_ddpm",
            "flownp": "euler",
        },
        "models": {
            name: {
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "step": int(metadata["step"]),
                "ema_weights": bool(metadata["ema_weights"]),
                "parameters": int(metadata["parameters"]),
            }
            for name, metadata in sorted(models.items())
        },
    }


def _sample_semantics(
    *,
    name: str,
    task: PrefixTask,
    spec: EvaluationSpec,
    runtime: Runtime | None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "sampling_semantics_version": SAMPLING_SEMANTICS_VERSION,
        "task_construction": spec.task_construction_version,
        "source": name,
        "task_fingerprint_sha256": task.fingerprint,
        "shape": [spec.samples, spec.num_targets, 1],
    }
    if name == "gp":
        identity.update(
            sampler="analytic_gp_posterior",
            seed=spec.gp_sample_seed,
            rng=spec.gp_sample_rng_semantics,
            gp=_gp_settings(),
        )
        if spec.gp_sample_rng_semantics == "historical-batched":
            identity["historical_gp_storage_version"] = 2
        return identity
    if runtime is None:
        raise ValueError(f"Runtime is required for model source {name}")
    sampler = "euler" if name == "flownp" else "ddpm"
    identity.update(
        sampler=sampler,
        steps=spec.sampling_steps,
        seed=spec.model_sample_seed,
        checkpoint_sha256=runtime.checkpoint_sha256,
        checkpoint_step=runtime.step,
        ema_weights=True,
        rng=(
            "base_plus_sample_index"
            if name == "flownp"
            else (
                "historical_shared_batch"
                if spec.model_rng_semantics == "historical"
                else "counter_based_sample_major"
            )
        ),
    )
    if name == "ndp_uncond":
        identity["repaint"] = {
            "inner_steps": runtime.config.ndp.repaint_inner_steps,
            "context_noise": runtime.config.ndp.repaint_context_noise,
        }
    return identity


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _validate_or_write_json(path: Path, payload: dict[str, Any]) -> Path:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Existing metadata is incompatible: {path}")
        return path
    return _write_json(path, payload)


def _validate_or_update_evaluation_spec(
    path: Path, payload: dict[str, Any]
) -> Path:
    if not path.exists():
        return _write_json(path, payload)
    existing = json.loads(path.read_text(encoding="utf-8"))
    if _evaluation_semantics(existing) != _evaluation_semantics(payload):
        raise ValueError(f"Existing evaluation semantics are incompatible: {path}")
    if existing != payload:
        return _write_json(path, payload)
    return path


def _load_runtime(name: str, runs_root: Path, device: torch.device) -> Runtime:
    checkpoint = runs_root / name / "checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    payload = torch_load(checkpoint, map_location=device)
    config = config_from_checkpoint(payload, expected_backend=name)
    backend = get_backend(name)
    model = backend.build_model(config, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise ValueError(f"{name} has {parameter_count} parameters, expected {EXPECTED_PARAMETERS}")
    load_model_state(
        model,
        payload,
        expected_backend=name,
        prefer_ema=True,
    )
    model.eval()
    return Runtime(
        name=name,
        backend=backend,
        model=model,
        process=backend.build_process(config, device),
        config=config,
        checkpoint=checkpoint,
        checkpoint_sha256=file_sha256(checkpoint),
        step=int(payload.get("step", -1)) if isinstance(payload, dict) else -1,
    )


def _minimum_separation(x: Tensor) -> float | None:
    if x.shape[0] < 2:
        return None
    ordered = torch.sort(x[:, 0]).values
    return float((ordered[1:] - ordered[:-1]).min())


def _master_payload(master: MasterTask) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_seed": master.task_seed,
        "context_count": int(master.x_context.shape[0]),
        "target_count": int(master.x_target.shape[0]),
        "base_separation": master.base_separation,
        "prefix_count": master.prefix_count,
        "prefix_separation": master.prefix_separation,
        "observed_prefix_separation": _minimum_separation(master.x_context[: master.prefix_count]),
        "observed_full_separation": _minimum_separation(master.x_context),
        "context_ids": master.context_ids.cpu().tolist(),
        "core_membership": master.core_membership.cpu().tolist(),
        "extra_membership": master.extra_membership.cpu().tolist(),
        "x_context": master.x_context.cpu().tolist(),
        "y_context": master.y_context.cpu().tolist(),
        "x_target": master.x_target.cpu().tolist(),
        "fingerprint_sha256": master.fingerprint,
    }


def _prefix_payload(task: PrefixTask, master: MasterTask) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "master_fingerprint_sha256": master.fingerprint,
        "context_size": task.context_size,
        "selected_context_ids_in_model_order": task.selected_ids.cpu().tolist(),
        "x_context": task.conditioning.x_context.cpu().tolist(),
        "y_context": task.conditioning.y_context.cpu().tolist(),
        "x_target": task.x_target.cpu().tolist(),
        "observed_minimum_separation": _minimum_separation(task.conditioning.x_context),
        "context_rows_spatially_sorted": bool(
            task.context_size > 1
            and (
                torch.all(
                    task.conditioning.x_context[1:, 0]
                    >= task.conditioning.x_context[:-1, 0]
                )
                or torch.all(
                    task.conditioning.x_context[1:, 0]
                    <= task.conditioning.x_context[:-1, 0]
                )
            )
        ),
        "fingerprint_sha256": task.fingerprint,
    }


def _independent_payload(
    task: PrefixTask,
    *,
    task_seed: int,
    configured_separation: float,
) -> dict[str, Any]:
    x_context = task.conditioning.x_context
    y_context = task.conditioning.y_context
    return {
        "schema_version": 2,
        "task_construction": INDEPENDENT_TASK_CONSTRUCTION_VERSION,
        "task_seed": task_seed,
        "context_size": task.context_size,
        "selected_context_ids_in_model_order": task.selected_ids.cpu().tolist(),
        "x_context": x_context.cpu().tolist(),
        "y_context": y_context.cpu().tolist(),
        "x_target": task.x_target.cpu().tolist(),
        "configured_minimum_separation": configured_separation,
        "observed_minimum_separation": _minimum_separation(x_context),
        "context_rows_spatially_sorted": bool(
            task.context_size > 1
            and (
                torch.all(x_context[1:, 0] >= x_context[:-1, 0])
                or torch.all(x_context[1:, 0] <= x_context[:-1, 0])
            )
        ),
        "conditioning_fingerprint_sha256": historical_tensor_fingerprint(
            x_context, y_context
        ),
        "fingerprint_sha256": task.fingerprint,
    }


def _sample_payload(
    *,
    name: str,
    samples: Tensor,
    task: PrefixTask,
    spec_hash: str,
    sampling_semantics_hash: str,
    metadata: dict[str, Any],
    samples_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "source": name,
        "spec_sha256": spec_hash,
        "sampling_semantics_sha256": sampling_semantics_hash,
        "task_fingerprint_sha256": task.fingerprint,
        "shape": list(samples.shape),
        "samples_fingerprint_sha256": samples_fingerprint,
        "metadata": metadata,
        "samples": samples.detach().cpu(),
    }


def _sample_fingerprint(name: str, samples: Tensor, spec: EvaluationSpec) -> str:
    historical = (
        spec.gp_sample_rng_semantics in {"sample-major", "historical-batched"}
        if name == "gp"
        else spec.model_rng_semantics == "historical"
    )
    function = historical_tensor_fingerprint if historical else tensor_fingerprint
    return function(samples)


def _legacy_sample_metadata_matches(
    *,
    name: str,
    payload: dict[str, Any],
    spec: EvaluationSpec,
    runtime: Runtime | None,
) -> bool:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return False
    if name == "gp":
        return (
            metadata.get("sampler") == "analytic_gp_posterior"
            and int(metadata.get("seed", -1)) == spec.gp_sample_seed
            and metadata.get("sample_seed_semantics", "batched")
            == spec.gp_sample_rng_semantics
        )
    if runtime is None:
        return False
    sampler = "euler" if name == "flownp" else "ddpm"
    execution = metadata.get("execution", {})
    expected_rng = "base_plus_sample_index" if name == "flownp" else (
        "historical_shared_batch"
        if spec.model_rng_semantics == "historical"
        else "counter_based_sample_major"
    )
    matches = (
        metadata.get("sampler") == sampler
        and int(metadata.get("steps", -1)) == spec.sampling_steps
        and int(metadata.get("seed", -1)) == spec.model_sample_seed
        and metadata.get("checkpoint_sha256") == runtime.checkpoint_sha256
        and int(metadata.get("checkpoint_step", -1)) == runtime.step
        and bool(metadata.get("ema_weights"))
        and isinstance(execution, dict)
        and execution.get("sample_seed_semantics") == expected_rng
    )
    if name == "ndp_uncond":
        matches = matches and (
            execution.get("conditioning_algorithm") == "repaint"
            and int(execution.get("repaint_inner_steps", -1))
            == runtime.config.ndp.repaint_inner_steps
            and execution.get("repaint_context_noise")
            == runtime.config.ndp.repaint_context_noise
        )
    return bool(matches)


def _load_or_generate_samples(
    path: Path,
    *,
    name: str,
    task: PrefixTask,
    spec: EvaluationSpec,
    spec_hash: str,
    runtime: Runtime | None,
) -> tuple[Tensor, dict[str, Any]]:
    sampling_semantics_hash = _canonical_hash(
        _sample_semantics(name=name, task=task, spec=spec, runtime=runtime)
    )
    if path.exists():
        payload = torch_load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid sample artifact: {path}")
        samples = payload.get("samples")
        common_valid = (
            payload.get("source") == name
            and payload.get("task_fingerprint_sha256") == task.fingerprint
            and torch.is_tensor(samples)
            and list(samples.shape) == [spec.samples, spec.num_targets, 1]
            and bool(torch.isfinite(samples).all())
            and payload.get("samples_fingerprint_sha256")
            == _sample_fingerprint(name, samples, spec)
        )
        semantic_valid = (
            int(payload.get("schema_version", 1)) >= 2
            and payload.get("sampling_semantics_sha256")
            == sampling_semantics_hash
        )
        legacy_valid = (
            int(payload.get("schema_version", 1)) == 1
            and _legacy_sample_metadata_matches(
                name=name,
                payload=payload,
                spec=spec,
                runtime=runtime,
            )
        )
        if not common_valid or not (semantic_valid or legacy_valid):
            raise ValueError(f"Existing sample artifact is incompatible: {path}")
        return samples, dict(payload.get("metadata", {}))

    if name == "gp":
        samples = draw_posterior_samples(
            task,
            count=spec.samples,
            seed=spec.gp_sample_seed,
            jitter=1e-6,
            rng_semantics=spec.gp_sample_rng_semantics,
        )
        metadata = {
            "sampler": "analytic_gp_posterior",
            "seed": spec.gp_sample_seed,
            "sample_seed_semantics": spec.gp_sample_rng_semantics,
            "ema_weights": False,
        }
        if spec.gp_sample_rng_semantics == "historical-batched":
            metadata.update(
                {
                    "stored_dtype": str(samples.dtype).removeprefix("torch."),
                    "historical_scored_shape": list(samples[..., 0].shape),
                    "historical_scored_samples_fingerprint_sha256": (
                        historical_tensor_fingerprint(samples[..., 0])
                    ),
                }
            )
    else:
        assert runtime is not None
        sampler = "euler" if name == "flownp" else "ddpm"
        request = SamplingRequest(
            sampler=sampler,
            num_steps=spec.sampling_steps,
            num_samples=spec.samples,
            seed=spec.model_sample_seed,
            batch_size=spec.batch_size,
            rng_semantics=spec.model_rng_semantics,
        )
        with torch.inference_mode():
            samples = runtime.backend.sample_conditional(
                runtime.model,
                runtime.process,
                task.x_target,
                task.conditioning,
                request,
            )
        metadata = {
            "sampler": sampler,
            "steps": spec.sampling_steps,
            "seed": spec.model_sample_seed,
            "ema_weights": True,
            "checkpoint": str(runtime.checkpoint.resolve()),
            "checkpoint_sha256": runtime.checkpoint_sha256,
            "checkpoint_step": runtime.step,
            "execution": runtime.backend.execution_metadata("sampling"),
        }
    if list(samples.shape) != [spec.samples, spec.num_targets, 1]:
        raise ValueError(f"Wrong sample shape for {name}: {tuple(samples.shape)}")
    if not bool(torch.isfinite(samples).all()):
        raise FloatingPointError(f"Non-finite samples from {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _sample_payload(
            name=name,
            samples=samples,
            task=task,
            spec_hash=spec_hash,
            sampling_semantics_hash=sampling_semantics_hash,
            metadata=metadata,
            samples_fingerprint=_sample_fingerprint(name, samples, spec),
        ),
        path,
    )
    return samples.detach().cpu(), metadata


def evaluate(spec: EvaluationSpec, *, device: str = "auto") -> list[Path]:
    spec.validate()
    target_device = resolve_device(device)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    runtimes = {
        name: _load_runtime(name, spec.runs_root, target_device) for name in MODELS
    }
    spec_payload = spec.serializable()
    spec_payload["models"] = {
        name: {
            "checkpoint": str(runtime.checkpoint.resolve()),
            "checkpoint_sha256": runtime.checkpoint_sha256,
            "step": runtime.step,
            "ema_weights": True,
            "parameters": EXPECTED_PARAMETERS,
        }
        for name, runtime in runtimes.items()
    }
    spec_hash = _canonical_hash(_evaluation_semantics(spec_payload))
    spec_payload["sampling_semantics_sha256"] = spec_hash
    spec_payload["spec_sha256"] = spec_hash
    files = [
        _validate_or_update_evaluation_spec(
            spec.output_dir / "evaluation_spec.json", spec_payload
        )
    ]

    data = Config.for_model("ndp_cond").data
    master: MasterTask | None = None
    if spec.task_construction == "nested":
        master = build_master_task(
            max_context=spec.contexts[-1],
            num_targets=spec.num_targets,
            task_seed=spec.task_seed,
            base_separation=spec.context_separation,
            prefix_count=spec.prefix_count,
            prefix_separation=spec.prefix_separation,
            data=data,
            device=target_device,
        )
        files.append(
            _validate_or_write_json(
                spec.output_dir / "master_task.json", _master_payload(master)
            )
        )

    previous_ids: set[int] = set()
    for context_size in spec.contexts:
        if master is None:
            configured_separation = spec.separation_for(context_size)
            task = build_independent_task(
                context_size=context_size,
                num_targets=spec.num_targets,
                task_seed=spec.task_seed,
                context_separation=configured_separation,
                data=data,
                device=target_device,
            )
            task_payload = _independent_payload(
                task,
                task_seed=spec.task_seed,
                configured_separation=configured_separation,
            )
        else:
            task = build_prefix_task(master, context_size, data)
            current_ids = set(int(value) for value in task.selected_ids.cpu().tolist())
            if not previous_ids.issubset(current_ids):
                raise AssertionError("Nested context membership was not preserved")
            previous_ids = current_ids
            task_payload = _prefix_payload(task, master)
        context_dir = spec.output_dir / f"ctx_{context_size}"
        files.append(
            _validate_or_write_json(
                context_dir / "task.json", task_payload
            )
        )
        panel_samples: dict[str, Tensor] = {}
        for name in PANEL_ORDER:
            path = context_dir / f"{name}_samples.pt"
            values, _ = _load_or_generate_samples(
                path,
                name=name,
                task=task,
                spec=spec,
                spec_hash=spec_hash,
                runtime=runtimes.get(name),
            )
            panel_samples[name] = values
            files.append(path)
        plot_path = context_dir / "comparison.png"
        task_label = (
            "Historical independent task"
            if spec.task_construction == "independent"
            else "Nested unordered context"
        )
        plot_comparison(task, panel_samples, plot_path, task_label=task_label)
        files.append(plot_path)
    return files

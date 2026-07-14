from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .config import Config, DataConfig, MODELS
from .evaluation import _evaluation_semantics
from .gp import (
    analytic_posterior,
    historical_analytic_posterior,
    historical_tensor_fingerprint,
    tensor_fingerprint,
)


MODEL_LABELS = {
    "ndp_cond": "Conditional NDP",
    "ndp_uncond": "Unconditional NDP",
    "flownp": "FlowNP",
}


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_or_validate_json(path: Path, payload: dict[str, Any]) -> Path:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Existing MMG report is incompatible: {path}")
        return path
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def moment_matched_gaussian(
    samples: Tensor,
    *,
    correction: int = 1,
) -> tuple[Tensor, Tensor]:
    values = samples.detach().cpu().double()
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("MMG fitting requires samples shaped [K,N,1] with K >= 2")
    if correction not in {0, 1}:
        raise ValueError("MMG covariance correction must be 0 or 1")
    denominator = values.shape[0] - correction
    if denominator <= 0:
        raise ValueError("MMG sample count must exceed covariance correction")
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / denominator
    return mean, covariance


def mvn_log_likelihood_batch(
    values: Tensor,
    mean: Tensor,
    covariance: Tensor,
    *,
    jitter: float,
) -> Tensor:
    if jitter < 0.0:
        raise ValueError("Likelihood jitter cannot be negative")
    observations = values.detach().cpu().double()
    if observations.ndim == 3 and observations.shape[-1] == 1:
        observations = observations[..., 0]
    location = mean.detach().cpu().double().reshape(-1)
    covariance = covariance.detach().cpu().double()
    if observations.ndim != 2 or observations.shape[1] != location.numel():
        raise ValueError("Likelihood values and mean have incompatible shapes")
    if covariance.shape != (location.numel(), location.numel()):
        raise ValueError("Likelihood covariance has an incompatible shape")
    stabilized = covariance + jitter * torch.eye(
        location.numel(), dtype=torch.float64
    )
    cholesky = torch.linalg.cholesky(stabilized)
    centered = observations - location.unsqueeze(0)
    whitened = torch.linalg.solve_triangular(
        cholesky, centered.T, upper=False
    )
    mahalanobis = whitened.square().sum(dim=0)
    log_determinant = 2.0 * torch.log(cholesky.diagonal()).sum()
    normalization = location.numel() * math.log(2.0 * math.pi)
    return -0.5 * (normalization + log_determinant + mahalanobis)


def _series(values: Tensor) -> dict[str, Any]:
    values = values.detach().cpu().double().reshape(-1)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise FloatingPointError("MMG likelihood series is empty or non-finite")
    population_std = values.std(unbiased=False) if values.numel() > 1 else values.new_zeros(())
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "population_std": float(population_std),
        "standard_error": float(population_std / math.sqrt(values.numel())),
        "per_sample": values.tolist(),
    }


def _gp_config(evaluation_spec: dict[str, Any]) -> DataConfig:
    defaults = Config.for_model("ndp_cond").data
    gp = evaluation_spec.get("gp")
    if gp is None:
        if int(evaluation_spec.get("implementation_version", -1)) != 2:
            raise ValueError("Evaluation spec does not record GP settings")
        return defaults
    if gp.get("kernel") != "matern52" or gp.get("target_noise") != "observed":
        raise ValueError("MMG reporting supports the saved noisy Matérn-5/2 task")
    defaults.lengthscale = float(gp["lengthscale"])
    defaults.variance = float(gp["variance"])
    defaults.observation_noise_std = float(gp["observation_noise_std"])
    defaults.jitter = float(gp["cholesky_jitter"])
    defaults.x_min, defaults.x_max = map(float, gp["x_bounds"])
    return defaults


def _validate_saved_samples(
    path: Path,
    *,
    source: str,
    task_fingerprint: str,
    evaluation_spec: dict[str, Any],
) -> tuple[Tensor, dict[str, Any]]:
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid saved sample payload: {path}")
    samples = payload.get("samples")
    expected_shape = [
        int(evaluation_spec["samples"]),
        int(evaluation_spec["num_targets"]),
        1,
    ]
    historical_fingerprint = (
        evaluation_spec.get("gp_sample_rng_semantics")
        in {"sample-major", "historical-batched"}
        if source == "gp"
        else evaluation_spec.get("model_rng_semantics") == "historical"
    )
    fingerprint_function = (
        historical_tensor_fingerprint if historical_fingerprint else tensor_fingerprint
    )
    common_valid = (
        payload.get("source") == source
        and payload.get("task_fingerprint_sha256") == task_fingerprint
        and torch.is_tensor(samples)
        and list(samples.shape) == expected_shape
        and bool(torch.isfinite(samples).all())
        and payload.get("samples_fingerprint_sha256")
        == fingerprint_function(samples)
    )
    if not common_valid:
        raise ValueError(f"Saved samples failed fingerprint validation: {path}")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Saved samples have invalid metadata: {path}")
    if source == "gp":
        metadata_valid = (
            metadata.get("sampler") == "analytic_gp_posterior"
            and int(metadata.get("seed", -1))
            == int(evaluation_spec["gp_sample_seed"])
            and metadata.get("sample_seed_semantics", "batched")
            == evaluation_spec.get("gp_sample_rng_semantics", "batched")
        )
    else:
        model_spec = evaluation_spec["models"][source]
        sampler = "euler" if source == "flownp" else "ddpm"
        execution = metadata.get("execution", {})
        expected_rng = "base_plus_sample_index" if source == "flownp" else (
            "historical_shared_batch"
            if evaluation_spec.get("model_rng_semantics") == "historical"
            else "counter_based_sample_major"
        )
        metadata_valid = (
            metadata.get("sampler") == sampler
            and int(metadata.get("steps", -1))
            == int(evaluation_spec["sampling_steps"])
            and int(metadata.get("seed", -1))
            == int(evaluation_spec["model_sample_seed"])
            and metadata.get("checkpoint_sha256")
            == model_spec["checkpoint_sha256"]
            and int(metadata.get("checkpoint_step", -1))
            == int(model_spec["step"])
            and bool(metadata.get("ema_weights"))
            and isinstance(execution, dict)
            and execution.get("sample_seed_semantics") == expected_rng
        )
        if source == "ndp_uncond":
            metadata_valid = metadata_valid and (
                execution.get("conditioning_algorithm") == "repaint"
                and int(execution.get("repaint_inner_steps", -1)) == 5
                and execution.get("repaint_context_noise") == "fresh"
            )
    if not metadata_valid:
        raise ValueError(f"Saved samples have incompatible semantics: {path}")
    return samples, payload


@dataclass(frozen=True)
class MMGReportSpec:
    input_dir: Path
    covariance: str = "unbiased"
    score_jitter: float = 1e-6

    def validate(self) -> None:
        if self.covariance not in {"unbiased", "mle"}:
            raise ValueError("MMG covariance must be unbiased or mle")
        if self.score_jitter < 0.0:
            raise ValueError("MMG score jitter cannot be negative")


def _format_cell(summary: dict[str, Any]) -> str:
    return f"{summary['mean']:.3f} ± {summary['standard_error']:.3f}"


def _markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Moment-matched Gaussian comparison",
        "",
        "Values are per-target log likelihood in nats, reported as mean ± standard error.",
        "Model samples fit each Gaussian; saved ground-truth GP samples are scored.",
        (
            "The standard error is across GP reference functions within one fixed "
            f"{comparison['task_construction']} task."
        ),
        "",
        "| Context | Ground Truth GP | Conditional NDP | Unconditional NDP | FlowNP |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        model_means = {
            name: row["models"][name]["mean"] for name in MODELS
        }
        best = max(model_means.values())
        cells = []
        for name in MODELS:
            cell = _format_cell(row["models"][name])
            if model_means[name] == best:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(
            f"| {row['context']} | {_format_cell(row['ground_truth_gp'])} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            f"Model samples per fit: {comparison['model_sample_count']}.",
            f"Ground-truth GP references per score: {comparison['reference_count']}.",
            f"Covariance estimator: {comparison['covariance_estimator']} "
            f"(denominator {comparison['covariance_denominator']}).",
            f"Likelihood jitter: {comparison['score_jitter']:.1e}.",
        ]
    )
    return "\n".join(lines) + "\n"


def report_mmg(spec: MMGReportSpec) -> list[Path]:
    spec.validate()
    root = spec.input_dir.resolve()
    evaluation_path = root / "evaluation_spec.json"
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Missing evaluation specification: {evaluation_path}")
    evaluation_spec = json.loads(evaluation_path.read_text(encoding="utf-8"))
    contexts = tuple(int(value) for value in evaluation_spec["contexts"])
    data = _gp_config(evaluation_spec)
    correction = 1 if spec.covariance == "unbiased" else 0
    report_spec = {
        "schema_version": 1,
        "metric": "moment_matched_gaussian_log_likelihood",
        "input_sampling_semantics_sha256": evaluation_spec.get(
            "sampling_semantics_sha256",
            _canonical_hash(_evaluation_semantics(evaluation_spec)),
        ),
        "contexts": list(contexts),
        "model_sample_count": int(evaluation_spec["samples"]),
        "reference_count": int(evaluation_spec["samples"]),
        "num_targets": int(evaluation_spec["num_targets"]),
        "model_sample_role": "fit_moment_matched_mean_and_covariance_only",
        "gp_sample_role": "ground_truth_references_scored_under_each_distribution",
        "covariance_estimator": spec.covariance,
        "covariance_correction": correction,
        "covariance_denominator": (
            "K-1" if correction == 1 else "K"
        ),
        "fit_jitter": 0.0,
        "score_jitter": spec.score_jitter,
        "standard_error_scope": (
            "gp_references_within_one_fixed_independent_task"
            if evaluation_spec.get("task_construction") == "independent"
            else "gp_references_within_one_fixed_nested_task"
        ),
        "task_construction": evaluation_spec.get("task_construction", "nested"),
        "gp": {
            "kernel": "matern52",
            "lengthscale": data.lengthscale,
            "variance": data.variance,
            "observation_noise_std": data.observation_noise_std,
            "target_noise": "observed",
            "posterior_jitter": data.jitter,
        },
    }
    report_spec["report_spec_sha256"] = _canonical_hash(report_spec)
    rows: list[dict[str, Any]] = []
    files: list[Path] = []
    for context in contexts:
        context_dir = root / f"ctx_{context}"
        task_path = context_dir / "task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task_fingerprint = task["fingerprint_sha256"]
        x_context = torch.tensor(task["x_context"], dtype=torch.float32)
        y_context = torch.tensor(task["y_context"], dtype=torch.float32)
        x_target = torch.tensor(task["x_target"], dtype=torch.float32)
        posterior_function = (
            historical_analytic_posterior
            if evaluation_spec.get("task_construction") == "independent"
            else analytic_posterior
        )
        posterior_mean, posterior_covariance = posterior_function(
            x_context, y_context, x_target, data=data
        )
        references, reference_payload = _validate_saved_samples(
            context_dir / "gp_samples.pt",
            source="gp",
            task_fingerprint=task_fingerprint,
            evaluation_spec=evaluation_spec,
        )
        ground_truth_joint = mvn_log_likelihood_batch(
            references,
            posterior_mean,
            posterior_covariance,
            jitter=spec.score_jitter,
        )
        ground_truth = _series(ground_truth_joint / x_target.shape[0])
        models: dict[str, dict[str, Any]] = {}
        for name in MODELS:
            model_samples, model_payload = _validate_saved_samples(
                context_dir / f"{name}_samples.pt",
                source=name,
                task_fingerprint=task_fingerprint,
                evaluation_spec=evaluation_spec,
            )
            fitted_mean, fitted_covariance = moment_matched_gaussian(
                model_samples, correction=correction
            )
            likelihood = mvn_log_likelihood_batch(
                references,
                fitted_mean,
                fitted_covariance,
                jitter=spec.score_jitter,
            )
            models[name] = {
                **_series(likelihood / x_target.shape[0]),
                "model_samples_fingerprint_sha256": model_payload[
                    "samples_fingerprint_sha256"
                ],
                "fitted_mean_shape": list(fitted_mean.shape),
                "fitted_covariance_shape": list(fitted_covariance.shape),
            }
        context_payload = {
            **report_spec,
            "context": context,
            "task_fingerprint_sha256": task_fingerprint,
            "reference_samples_fingerprint_sha256": reference_payload[
                "metadata"
            ].get(
                "historical_scored_samples_fingerprint_sha256",
                reference_payload["samples_fingerprint_sha256"],
            ),
            "stored_reference_tensor_fingerprint_sha256": reference_payload[
                "samples_fingerprint_sha256"
            ],
            "ground_truth_gp": ground_truth,
            "models": models,
        }
        context_path = context_dir / "mmg.json"
        files.append(_write_or_validate_json(context_path, context_payload))
        rows.append(
            {
                "context": context,
                "ground_truth_gp": ground_truth,
                "models": models,
            }
        )
    comparison = {**report_spec, "rows": rows}
    comparison_json = root / "comparison_mmg.json"
    files.append(_write_or_validate_json(comparison_json, comparison))
    comparison_markdown = root / "comparison_mmg.md"
    markdown = _markdown(comparison)
    if comparison_markdown.exists():
        if comparison_markdown.read_text(encoding="utf-8") != markdown:
            raise ValueError(f"Existing MMG report is incompatible: {comparison_markdown}")
    else:
        comparison_markdown.write_text(markdown, encoding="utf-8")
    files.append(comparison_markdown)
    return files

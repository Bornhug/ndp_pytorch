from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from better_results.cli import main
from better_results.config import Config, MODELS
from better_results.evaluation import (
    EvaluationSpec,
    _canonical_hash,
    _evaluation_semantics,
    _load_or_generate_samples,
    _sample_semantics,
    _validate_or_update_evaluation_spec,
)
from better_results.gp import tensor_fingerprint
from better_results.reporting import (
    MMGReportSpec,
    moment_matched_gaussian,
    mvn_log_likelihood_batch,
    report_mmg,
)


def _sample_payload(
    source: str,
    samples: torch.Tensor,
    *,
    task_fingerprint: str,
    metadata: dict,
) -> dict:
    return {
        "schema_version": 1,
        "source": source,
        "spec_sha256": "legacy-path-sensitive-hash",
        "task_fingerprint_sha256": task_fingerprint,
        "shape": list(samples.shape),
        "samples_fingerprint_sha256": tensor_fingerprint(samples),
        "metadata": metadata,
        "samples": samples,
    }


def _write_synthetic_evaluation(root: Path) -> None:
    task_fingerprint = "synthetic-task"
    evaluation_spec = {
        "implementation_version": 2,
        "contexts": [1],
        "num_targets": 2,
        "samples": 3,
        "task_seed": 5,
        "gp_sample_seed": 11,
        "model_sample_seed": 7,
        "context_separation": 0.0,
        "prefix_count": 1,
        "prefix_separation": 0.0,
        "sampling_steps": 2,
        "batch_size": 3,
        "spec_sha256": "synthetic-spec",
        "models": {
            name: {
                "checkpoint": f"unused/{name}/checkpoint.pt",
                "checkpoint_sha256": f"sha-{name}",
                "step": 128000,
                "ema_weights": True,
                "parameters": 549441,
            }
            for name in MODELS
        },
    }
    root.mkdir(parents=True)
    (root / "evaluation_spec.json").write_text(json.dumps(evaluation_spec))
    context_dir = root / "ctx_1"
    context_dir.mkdir()
    task = {
        "context_size": 1,
        "x_context": [[0.0]],
        "y_context": [[0.25]],
        "x_target": [[-0.5], [0.5]],
        "fingerprint_sha256": task_fingerprint,
    }
    (context_dir / "task.json").write_text(json.dumps(task))
    references = torch.tensor(
        [[[-0.2], [0.4]], [[0.1], [0.7]], [[-0.5], [0.2]]],
        dtype=torch.float32,
    )
    torch.save(
        _sample_payload(
            "gp",
            references,
            task_fingerprint=task_fingerprint,
            metadata={
                "sampler": "analytic_gp_posterior",
                "seed": 11,
                "ema_weights": False,
            },
        ),
        context_dir / "gp_samples.pt",
    )
    base_samples = torch.tensor(
        [[[-0.4], [0.2]], [[0.0], [0.9]], [[0.5], [0.4]]],
        dtype=torch.float32,
    )
    for index, name in enumerate(MODELS):
        samples = base_samples + 0.1 * index
        execution = {
            "sample_seed_semantics": (
                "base_plus_sample_index"
                if name == "flownp"
                else "counter_based_sample_major"
            )
        }
        if name == "ndp_uncond":
            execution.update(
                conditioning_algorithm="repaint",
                repaint_inner_steps=5,
                repaint_context_noise="fresh",
            )
        torch.save(
            _sample_payload(
                name,
                samples,
                task_fingerprint=task_fingerprint,
                metadata={
                    "sampler": "euler" if name == "flownp" else "ddpm",
                    "steps": 2,
                    "seed": 7,
                    "ema_weights": True,
                    "checkpoint_sha256": f"sha-{name}",
                    "checkpoint_step": 128000,
                    "execution": execution,
                },
            ),
            context_dir / f"{name}_samples.pt",
        )


class MMGMathTests(unittest.TestCase):
    def test_unbiased_moments_and_likelihood_match_torch(self) -> None:
        samples = torch.tensor(
            [[[0.0], [0.0]], [[2.0], [1.0]], [[4.0], [4.0]]]
        )
        mean, covariance = moment_matched_gaussian(samples, correction=1)
        squeezed = samples[..., 0].double()
        self.assertTrue(torch.allclose(mean, squeezed.mean(dim=0)))
        self.assertTrue(torch.allclose(covariance, torch.cov(squeezed.T, correction=1)))
        references = torch.tensor([[[1.0], [0.5]], [[3.0], [2.0]]])
        actual = mvn_log_likelihood_batch(
            references, mean, covariance, jitter=1e-3
        )
        expected = torch.distributions.MultivariateNormal(
            mean,
            covariance_matrix=covariance + 1e-3 * torch.eye(2, dtype=torch.float64),
        ).log_prob(references[..., 0].double())
        self.assertTrue(torch.allclose(actual, expected))

    def test_model_samples_fit_moments_and_gp_samples_only_change_scores(self) -> None:
        model_samples = torch.tensor(
            [[[0.0], [0.0]], [[1.0], [1.0]], [[2.0], [0.5]]]
        )
        mean, covariance = moment_matched_gaussian(model_samples)
        original_mean = mean.clone()
        references_a = torch.tensor([[[0.0], [0.0]], [[0.5], [0.5]]])
        references_b = references_a + 5.0
        scores_a = mvn_log_likelihood_batch(
            references_a, mean, covariance, jitter=1e-3
        )
        scores_b = mvn_log_likelihood_batch(
            references_b, mean, covariance, jitter=1e-3
        )
        self.assertTrue(torch.equal(mean, original_mean))
        self.assertFalse(torch.allclose(scores_a, scores_b))
        shifted_mean, _ = moment_matched_gaussian(model_samples + 2.0)
        self.assertFalse(torch.allclose(mean, shifted_mean))


class SavedArtifactTests(unittest.TestCase):
    def test_semantic_identity_excludes_paths_output_and_batch_size(self) -> None:
        base = EvaluationSpec(
            runs_root=Path("first/runs"),
            contexts=(1,),
            num_targets=3,
            samples=2,
            task_seed=5,
            gp_sample_seed=11,
            model_sample_seed=7,
            context_separation=0.0,
            prefix_count=1,
            prefix_separation=0.0,
            sampling_steps=2,
            batch_size=2,
            output_dir=Path("first/output"),
        )
        moved = replace(
            base,
            runs_root=Path("second/runs"),
            output_dir=Path("second/output"),
            batch_size=1,
        )
        task = SimpleNamespace(fingerprint="task")
        config = Config.for_model("ndp_cond")
        runtime = SimpleNamespace(
            checkpoint_sha256="checkpoint-sha",
            step=128000,
            config=config,
        )
        first = _canonical_hash(
            _sample_semantics(
                name="ndp_cond", task=task, spec=base, runtime=runtime
            )
        )
        second = _canonical_hash(
            _sample_semantics(
                name="ndp_cond", task=task, spec=moved, runtime=runtime
            )
        )
        changed_seed = _canonical_hash(
            _sample_semantics(
                name="ndp_cond",
                task=task,
                spec=replace(base, model_sample_seed=8),
                runtime=runtime,
            )
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed_seed)

    def test_evaluation_metadata_relocation_updates_paths_without_incompatibility(self) -> None:
        base = {
            "contexts": [1],
            "num_targets": 3,
            "samples": 2,
            "task_seed": 5,
            "gp_sample_seed": 11,
            "model_sample_seed": 7,
            "context_separation": 0.0,
            "prefix_count": 1,
            "prefix_separation": 0.0,
            "sampling_steps": 2,
            "batch_size": 2,
            "runs_root": "first/runs",
            "output_dir": "first/output",
            "models": {
                name: {
                    "checkpoint": f"first/{name}/checkpoint.pt",
                    "checkpoint_sha256": f"sha-{name}",
                    "step": 128000,
                    "ema_weights": True,
                    "parameters": 549441,
                }
                for name in MODELS
            },
        }
        moved = json.loads(json.dumps(base))
        moved["runs_root"] = "second/runs"
        moved["output_dir"] = "second/output"
        moved["batch_size"] = 1
        for name in MODELS:
            moved["models"][name]["checkpoint"] = f"second/{name}/checkpoint.pt"
        self.assertEqual(_evaluation_semantics(base), _evaluation_semantics(moved))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation_spec.json"
            path.write_text(json.dumps(base))
            _validate_or_update_evaluation_spec(path, moved)
            self.assertEqual(json.loads(path.read_text()), moved)

    def test_legacy_artifact_reuses_samples_despite_spec_hash_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ndp_cond_samples.pt"
            samples = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1)
            torch.save(
                _sample_payload(
                    "ndp_cond",
                    samples,
                    task_fingerprint="task",
                    metadata={
                        "sampler": "ddpm",
                        "steps": 2,
                        "seed": 7,
                        "ema_weights": True,
                        "checkpoint_sha256": "checkpoint-sha",
                        "checkpoint_step": 128000,
                        "execution": {
                            "sample_seed_semantics": "counter_based_sample_major"
                        },
                    },
                ),
                path,
            )
            spec = EvaluationSpec(
                runs_root=Path("moved/runs"),
                contexts=(1,),
                num_targets=3,
                samples=2,
                task_seed=5,
                gp_sample_seed=11,
                model_sample_seed=7,
                context_separation=0.0,
                prefix_count=1,
                prefix_separation=0.0,
                sampling_steps=2,
                batch_size=1,
                output_dir=Path("moved/output"),
            )
            runtime = SimpleNamespace(
                checkpoint_sha256="checkpoint-sha",
                step=128000,
                config=Config.for_model("ndp_cond"),
            )
            loaded, _ = _load_or_generate_samples(
                path,
                name="ndp_cond",
                task=SimpleNamespace(fingerprint="task"),
                spec=spec,
                spec_hash="different-new-hash",
                runtime=runtime,
            )
            self.assertTrue(torch.equal(loaded, samples))

    def test_report_cli_never_calls_model_training_sampling_or_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation"
            _write_synthetic_evaluation(root)
            with (
                mock.patch("better_results.cli.evaluate", side_effect=AssertionError),
                mock.patch("better_results.cli.train_model", side_effect=AssertionError),
                mock.patch("better_results.cli.train_all", side_effect=AssertionError),
                mock.patch(
                    "torch.cuda.is_available",
                    side_effect=AssertionError("CUDA must not be queried"),
                ),
            ):
                exit_code = main(
                    [
                        "report",
                        "--input-dir",
                        str(root),
                        "--metrics",
                        "mmg",
                        "--mmg-covariance",
                        "unbiased",
                        "--mmg-score-jitter",
                        "1e-3",
                    ]
                )
            self.assertEqual(exit_code, 0)
            comparison = json.loads((root / "comparison_mmg.json").read_text())
            self.assertEqual(comparison["model_sample_count"], 3)
            self.assertEqual(comparison["reference_count"], 3)
            self.assertEqual(comparison["rows"][0]["context"], 1)
            self.assertTrue((root / "comparison_mmg.md").exists())

    def test_direct_report_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation"
            _write_synthetic_evaluation(root)
            spec = MMGReportSpec(root, score_jitter=1e-3)
            first = report_mmg(spec)
            second = report_mmg(spec)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

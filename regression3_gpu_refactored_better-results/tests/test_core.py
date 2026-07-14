from __future__ import annotations

import math
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from better_results.backends import _AdaptiveBackend, get_backend
from better_results.cli import DEFAULT_PRETRAINED_ROOT, build_parser
from better_results.checkpoints import (
    config_from_checkpoint,
    load_model_state,
    torch_load,
)
from better_results.config import Config, MODELS
from better_results.gp import (
    analytic_posterior,
    build_independent_task,
    build_master_task,
    build_prefix_task,
    draw_posterior_samples,
    historical_stable_cholesky,
    matern52_kernel,
)
from better_results.plotting import PANEL_ORDER, context_marker_area, plot_comparison
from better_results.training import EXPECTED_PARAMETERS, _finite
from better_results.types import ConditioningSet, SamplingRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_RUNS = PROJECT_ROOT / "pretrained" / "matched_549441_seed42"


class ModelTests(unittest.TestCase):
    def test_parameter_counts(self) -> None:
        for name in MODELS:
            config = Config.for_model(name)
            model = get_backend(name).build_model(config, torch.device("cpu"))
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters()),
                EXPECTED_PARAMETERS,
            )

    def test_bundled_ema_checkpoints_load_strictly(self) -> None:
        self.assertTrue(PRETRAINED_RUNS.exists())
        for name in MODELS:
            payload = torch_load(PRETRAINED_RUNS / name / "checkpoint.pt", map_location="cpu")
            self.assertEqual(payload["step"], 128_000)
            config = config_from_checkpoint(payload, expected_backend=name)
            model = get_backend(name).build_model(config, torch.device("cpu"))
            load_model_state(model, payload, expected_backend=name, prefer_ema=True)

    def test_bundle_manifest_hashes_and_default_cli_path(self) -> None:
        manifest = json.loads((PRETRAINED_RUNS / "manifest.json").read_text())
        self.assertEqual(set(manifest["models"]), set(MODELS))
        for name, entry in manifest["models"].items():
            self.assertEqual(entry["trainable_parameters"], EXPECTED_PARAMETERS)
            self.assertEqual(entry["checkpoint_step"], 128_000)
            for kind in ("checkpoint", "config"):
                path = PRETRAINED_RUNS / entry[kind]["path"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(path.stat().st_size, entry[kind]["bytes"])
                self.assertEqual(digest, entry[kind]["sha256"])
        arguments = build_parser().parse_args(
            ["evaluate", "--contexts", "1", "--output-dir", "unused"]
        )
        self.assertEqual(arguments.runs_root, DEFAULT_PRETRAINED_ROOT)

    def test_historical_evaluation_cli_options(self) -> None:
        arguments = build_parser().parse_args(
            [
                "evaluate",
                "--contexts",
                "50",
                "75",
                "--task-construction",
                "independent",
                "--model-rng-semantics",
                "historical",
                "--gp-sample-rng-semantics",
                "historical-batched",
                "--context-separation-override",
                "75=0.01",
                "--output-dir",
                "unused",
            ]
        )
        self.assertEqual(arguments.task_construction, "independent")
        self.assertEqual(arguments.model_rng_semantics, "historical")
        self.assertEqual(arguments.gp_sample_rng_semantics, "historical-batched")
        self.assertEqual(arguments.context_separation_override, [(75, 0.01)])

    def test_source_has_no_original_package_import(self) -> None:
        forbidden = "regression3_gpu_" + "refactored"
        for path in (PROJECT_ROOT / "better_results").rglob("*.py"):
            self.assertNotIn(forbidden, path.read_text(encoding="utf-8"), path)

    def test_nonfinite_guard(self) -> None:
        with self.assertRaises(FloatingPointError):
            _finite("loss", torch.tensor(float("nan")), 3)
        with self.assertRaises(FloatingPointError):
            _finite("gradient", torch.tensor(float("inf")), 4)


class GPTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = Config.for_model("ndp_cond").data
        cls.master = build_master_task(
            max_context=100,
            num_targets=50,
            task_seed=10,
            base_separation=0.01,
            prefix_count=50,
            prefix_separation=0.05,
            data=cls.data,
            device=torch.device("cpu"),
        )

    def test_matern_covariance_and_observation_shapes(self) -> None:
        x = torch.tensor([[-1.0], [0.0], [1.0]])
        covariance = matern52_kernel(x, x, lengthscale=0.25, variance=1.0)
        self.assertTrue(torch.allclose(covariance, covariance.T))
        self.assertTrue(torch.allclose(covariance.diag(), torch.ones(3)))
        self.assertEqual(self.master.x_context.shape, (100, 1))
        self.assertEqual(self.master.y_context.shape, (100, 1))

    def test_staged_spacing_nested_ids_and_unordered_rows(self) -> None:
        core_x = torch.sort(self.master.x_context[:50, 0]).values
        full_x = torch.sort(self.master.x_context[:, 0]).values
        self.assertGreaterEqual(float(torch.diff(core_x).min()), 0.05 - 1e-7)
        self.assertGreaterEqual(float(torch.diff(full_x).min()), 0.01 - 1e-7)
        previous: set[int] = set()
        for count in (1, 5, 25, 50, 75, 100):
            task = build_prefix_task(self.master, count, self.data)
            ids = set(task.selected_ids.tolist())
            self.assertEqual(len(ids), count)
            self.assertTrue(previous.issubset(ids))
            previous = ids
            if count >= 3:
                x = task.conditioning.x_context[:, 0]
                monotone = torch.all(x[:-1] <= x[1:]) or torch.all(x[:-1] >= x[1:])
                self.assertFalse(bool(monotone))

    def test_deterministic_master_and_shared_posterior(self) -> None:
        repeat = build_master_task(
            max_context=100,
            num_targets=50,
            task_seed=10,
            base_separation=0.01,
            prefix_count=50,
            prefix_separation=0.05,
            data=self.data,
            device=torch.device("cpu"),
        )
        self.assertEqual(self.master.fingerprint, repeat.fingerprint)
        task = build_prefix_task(self.master, 25, self.data)
        mean, covariance = analytic_posterior(
            task.conditioning.x_context,
            task.conditioning.y_context,
            task.x_target,
            data=self.data,
        )
        self.assertTrue(torch.equal(task.posterior_mean, mean))
        self.assertTrue(torch.equal(task.posterior_covariance, covariance))
        samples = draw_posterior_samples(task, count=7, seed=98775, jitter=1e-6)
        self.assertEqual(samples.shape, (7, 50, 1))
        self.assertTrue(bool(torch.isfinite(samples).all()))

    def test_independent_tasks_and_historical_gp_sample_major_stream(self) -> None:
        first = build_independent_task(
            context_size=25,
            num_targets=50,
            task_seed=10,
            context_separation=0.05,
            data=self.data,
            device=torch.device("cpu"),
        )
        repeat = build_independent_task(
            context_size=25,
            num_targets=50,
            task_seed=10,
            context_separation=0.05,
            data=self.data,
            device=torch.device("cpu"),
        )
        self.assertEqual(first.fingerprint, repeat.fingerprint)
        self.assertTrue(torch.equal(first.conditioning.x_context, repeat.conditioning.x_context))
        self.assertTrue(torch.equal(first.conditioning.y_context, repeat.conditioning.y_context))
        ordered = torch.sort(first.conditioning.x_context[:, 0]).values
        self.assertGreaterEqual(float(torch.diff(ordered).min()), 0.05 - 1e-7)

        actual = draw_posterior_samples(
            first,
            count=4,
            seed=98775,
            jitter=1e-6,
            rng_semantics="sample-major",
        )
        generator = torch.Generator().manual_seed(98775)
        noise = torch.cat(
            [torch.randn((1, 50), generator=generator, dtype=torch.float64) for _ in range(4)]
        )
        cholesky = historical_stable_cholesky(
            first.posterior_covariance, jitter=1e-6
        )
        expected = (
            first.posterior_mean.reshape(1, -1) + noise @ cholesky.T
        ).unsqueeze(-1).float()
        self.assertTrue(torch.equal(actual, expected))

        historical_batched = draw_posterior_samples(
            first,
            count=4,
            seed=98775,
            jitter=1e-6,
            rng_semantics="historical-batched",
        )
        batched_noise = torch.randn(
            (4, 50),
            generator=torch.Generator().manual_seed(98775),
            dtype=torch.float64,
        )
        batched_expected = (
            first.posterior_mean.reshape(1, -1) + batched_noise @ cholesky.T
        ).unsqueeze(-1)
        self.assertTrue(torch.equal(historical_batched, batched_expected))
        self.assertEqual(historical_batched.dtype, torch.float64)


class SamplingAndPlotTests(unittest.TestCase):
    def test_native_sampler_shapes_and_context_permutation_invariance(self) -> None:
        torch.manual_seed(7)
        x_target = torch.linspace(-1.0, 1.0, 4).unsqueeze(-1)
        x_context = torch.tensor([[-0.8], [0.1], [0.7]])
        y_context = torch.tensor([[0.2], [-0.4], [0.8]])
        order = torch.tensor([2, 0, 1])
        base_conditioning = ConditioningSet(x_context, y_context)
        shuffled_conditioning = ConditioningSet(x_context[order], y_context[order])
        for name in MODELS:
            config = Config.for_model(name)
            config.diffusion.timesteps = 2
            config.ndp.repaint_inner_steps = 1
            backend = get_backend(name)
            model = backend.build_model(config, torch.device("cpu")).eval()
            process = backend.build_process(config, torch.device("cpu"))
            request = SamplingRequest(
                sampler="euler" if name == "flownp" else "ddpm",
                num_steps=2,
                num_samples=2,
                seed=19,
                batch_size=2,
            )
            with torch.inference_mode():
                first = backend.sample_conditional(
                    model, process, x_target, base_conditioning, request
                )
                second = backend.sample_conditional(
                    model, process, x_target, shuffled_conditioning, request
                )
                sequential = backend.sample_conditional(
                    model,
                    process,
                    x_target,
                    base_conditioning,
                    replace(request, batch_size=1),
                )
            self.assertEqual(first.shape, (2, 4, 1), name)
            self.assertTrue(bool(torch.isfinite(first).all()), name)
            self.assertTrue(torch.allclose(first, second, atol=2e-5, rtol=2e-5), name)
            self.assertTrue(
                torch.allclose(first, sequential, atol=2e-5, rtol=2e-5), name
            )

    def test_oom_halving_preserves_order(self) -> None:
        backend = _AdaptiveBackend()

        def run_chunk(start: int, end: int) -> torch.Tensor:
            if end - start > 2:
                raise torch.cuda.OutOfMemoryError("simulated")
            return torch.arange(start, end).unsqueeze(-1)

        with (
            mock.patch("torch.cuda.reset_peak_memory_stats"),
            mock.patch("torch.cuda.empty_cache"),
            mock.patch("torch.cuda.max_memory_allocated", return_value=1234),
        ):
            result = backend._adaptive_batches(
                total=5,
                device=torch.device("cuda"),
                run_chunk=run_chunk,
                requested_batch_size=5,
            )
        self.assertEqual(result[:, 0].tolist(), [0, 1, 2, 3, 4])
        metadata = backend.execution_metadata()
        self.assertEqual(metadata["effective_batch_size"], 2)
        self.assertEqual(metadata["oom_retries"], 1)

    def test_historical_ndp_rng_is_deterministic_and_recorded(self) -> None:
        config = Config.for_model("ndp_cond")
        config.diffusion.timesteps = 2
        backend = get_backend("ndp_cond")
        model = backend.build_model(config, torch.device("cpu")).eval()
        process = backend.build_process(config, torch.device("cpu"))
        x_target = torch.linspace(-1.0, 1.0, 3).unsqueeze(-1)
        conditioning = ConditioningSet(torch.tensor([[0.0]]), torch.tensor([[0.25]]))
        request = SamplingRequest(
            sampler="ddpm",
            num_steps=2,
            num_samples=3,
            seed=10,
            batch_size=3,
            rng_semantics="historical",
        )
        with torch.inference_mode():
            first = backend.sample_conditional(
                model, process, x_target, conditioning, request
            )
            second = backend.sample_conditional(
                model, process, x_target, conditioning, request
            )
        self.assertTrue(torch.equal(first, second))
        metadata = backend.execution_metadata()
        self.assertEqual(metadata["sample_seed_semantics"], "historical_shared_batch")
        self.assertEqual(metadata["effective_batch_size"], 3)

    def test_plot_has_exact_context_offsets_in_every_panel(self) -> None:
        data = Config.for_model("ndp_cond").data
        master = build_master_task(
            max_context=5,
            num_targets=8,
            task_seed=10,
            base_separation=0.01,
            prefix_count=5,
            prefix_separation=0.05,
            data=data,
            device=torch.device("cpu"),
        )
        task = build_prefix_task(master, 5, data)
        samples = {
            name: torch.zeros((3, 8, 1), dtype=task.x_target.dtype)
            for name in PANEL_ORDER
        }
        import matplotlib.axes

        original = matplotlib.axes.Axes.scatter
        counts: list[int] = []

        def recording_scatter(axis, x, y, *args, **kwargs):
            counts.append(len(x))
            return original(axis, x, y, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plot.png"
            with mock.patch.object(matplotlib.axes.Axes, "scatter", recording_scatter):
                plot_comparison(task, samples, output)
            self.assertGreater(output.stat().st_size, 0)
        self.assertEqual(counts, [5, 5, 5, 5])
        self.assertEqual(context_marker_area(1), 8.0)
        self.assertAlmostEqual(context_marker_area(100), 3.0)


if __name__ == "__main__":
    unittest.main()

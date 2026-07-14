from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import MODELS
from .evaluation import EvaluationSpec, evaluate
from .reporting import MMGReportSpec, report_mmg
from .training import train_all, train_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAINED_ROOT = PROJECT_ROOT / "pretrained" / "matched_549441_seed42"


def _prefix_separation(value: str) -> tuple[int, float]:
    try:
        count, separation = value.split("=", 1)
        result = int(count), float(separation)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("prefix separation must use COUNT=VALUE") from exc
    if min(result) <= 0:
        raise argparse.ArgumentTypeError("prefix separation values must be positive")
    return result


def _context_separation_override(value: str) -> tuple[int, float]:
    try:
        context, separation = value.split("=", 1)
        result = int(context), float(separation)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "context separation override must use CONTEXT=VALUE"
        ) from exc
    if result[0] <= 0 or result[1] < 0.0:
        raise argparse.ArgumentTypeError(
            "context must be positive and separation cannot be negative"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone matched GP regression plots")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train or resume one model")
    train.add_argument("--model", choices=MODELS, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--max-steps", type=int)

    train_queue = commands.add_parser("train-all", help="Train all three models sequentially")
    train_queue.add_argument("--runs-root", type=Path, required=True)
    train_queue.add_argument("--device", default="auto")
    train_queue.add_argument("--max-steps", type=int)

    evaluate_parser = commands.add_parser("evaluate", help="Create GP comparison plots")
    evaluate_parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_PRETRAINED_ROOT,
        help=f"Model-run root (default: {DEFAULT_PRETRAINED_ROOT})",
    )
    evaluate_parser.add_argument("--contexts", nargs="+", type=int, required=True)
    evaluate_parser.add_argument("--num-targets", type=int, default=50)
    evaluate_parser.add_argument("--samples", type=int, default=100)
    evaluate_parser.add_argument("--task-seed", type=int, default=10)
    evaluate_parser.add_argument("--gp-sample-seed", type=int, default=98775)
    evaluate_parser.add_argument("--model-sample-seed", type=int, default=10)
    evaluate_parser.add_argument(
        "--task-construction", choices=("nested", "independent"), default="nested"
    )
    evaluate_parser.add_argument(
        "--model-rng-semantics", choices=("stable", "historical"), default="stable"
    )
    evaluate_parser.add_argument(
        "--gp-sample-rng-semantics",
        choices=("batched", "sample-major", "historical-batched"),
        default="batched",
    )
    evaluate_parser.add_argument("--context-separation", type=float, default=0.01)
    evaluate_parser.add_argument(
        "--context-separation-override",
        type=_context_separation_override,
        action="append",
        default=[],
    )
    evaluate_parser.add_argument(
        "--prefix-separation", type=_prefix_separation, default=(50, 0.05)
    )
    evaluate_parser.add_argument("--sampling-steps", type=int, default=500)
    evaluate_parser.add_argument("--batch-size", type=int, default=100)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--device", default="auto")

    report_parser = commands.add_parser(
        "report", help="Compute metrics from existing saved sample tensors"
    )
    report_parser.add_argument("--input-dir", type=Path, required=True)
    report_parser.add_argument(
        "--metrics", nargs="+", choices=("mmg",), required=True
    )
    report_parser.add_argument(
        "--mmg-covariance", choices=("unbiased", "mle"), default="unbiased"
    )
    report_parser.add_argument("--mmg-score-jitter", type=float, default=1e-6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        result = train_model(
            args.model,
            run_dir=args.run_dir,
            device=args.device,
            max_steps=args.max_steps,
        )
        print(f"{args.model}: step={result.step} checkpoint={result.checkpoint.resolve()}")
        return 0
    if args.command == "train-all":
        for result in train_all(
            runs_root=args.runs_root,
            device=args.device,
            max_steps=args.max_steps,
        ):
            print(f"{result.run_dir.name}: step={result.step} checkpoint={result.checkpoint.resolve()}")
        return 0
    if args.command == "report":
        if tuple(args.metrics) != ("mmg",):
            raise ValueError("Only one MMG metric pass is supported")
        files = report_mmg(
            MMGReportSpec(
                input_dir=args.input_dir,
                covariance=args.mmg_covariance,
                score_jitter=args.mmg_score_jitter,
            )
        )
        print("Created or validated without model sampling:")
        for path in files:
            print(f"  {path.resolve()}")
        return 0
    prefix_count, prefix_separation = args.prefix_separation
    spec = EvaluationSpec(
        runs_root=args.runs_root,
        contexts=tuple(args.contexts),
        num_targets=args.num_targets,
        samples=args.samples,
        task_seed=args.task_seed,
        gp_sample_seed=args.gp_sample_seed,
        model_sample_seed=args.model_sample_seed,
        context_separation=args.context_separation,
        prefix_count=prefix_count,
        prefix_separation=prefix_separation,
        sampling_steps=args.sampling_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        task_construction=args.task_construction,
        model_rng_semantics=args.model_rng_semantics,
        gp_sample_rng_semantics=args.gp_sample_rng_semantics,
        context_separation_overrides=tuple(args.context_separation_override),
    )
    files = evaluate(spec, device=args.device)
    print("Created or validated:")
    for path in files:
        print(f"  {path.resolve()}")
    return 0

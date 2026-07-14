# Better-results GP regression plots

This standalone package trains the three parameter-matched GP regression models
and renders GP context comparisons. It supports both nested unordered tasks and
the historical independent-task recipe. PF likelihood remains intentionally excluded.

## Evaluate the existing final checkpoints

```powershell
conda activate ndp_pytorch
cd C:\Users\apex\Code\Python\ndp_pytorch_ctx\regression3_gpu_refactored_better-results
python -m better_results evaluate `
  --contexts 1 5 25 50 75 100 `
  --num-targets 50 --samples 100 `
  --task-seed 10 --gp-sample-seed 98775 --model-sample-seed 10 `
  --context-separation 0.01 --prefix-separation 50=0.05 `
  --sampling-steps 500 --batch-size 100 `
  --output-dir logs\eval_unordered_nested_ctx1_5_25_50_75_100 `
  --device cuda
```

Evaluation defaults to the bundled checkpoints in
`pretrained\matched_549441_seed42`. Pass `--runs-root` only to evaluate a
different set of locally trained runs.

## Reproduce the historical independent-task evaluation

```powershell
python -m better_results evaluate `
  --task-construction independent `
  --model-rng-semantics historical `
  --gp-sample-rng-semantics historical-batched `
  --contexts 1 5 25 50 75 100 `
  --num-targets 50 --samples 100 `
  --task-seed 10 --gp-sample-seed 98775 --model-sample-seed 10 `
  --context-separation 0.05 `
  --context-separation-override 75=0.01 `
  --context-separation-override 100=0.01 `
  --sampling-steps 500 --batch-size 100 `
  --output-dir logs\eval_old_independent_ctx1_5_25_50_75_100 `
  --device cuda
```

Each context size restarts the historical task RNG and is therefore an
independent fixed task rather than a subset of a shared master task.

## Report MMG from saved samples

The reporting command reuses the saved model and analytic-GP tensors. It does
not load a checkpoint, run a sampler, or require CUDA.

```powershell
python -m better_results report `
  --input-dir logs\eval_old_independent_ctx1_5_25_50_75_100 `
  --metrics mmg `
  --mmg-covariance unbiased `
  --mmg-score-jitter 1e-6
```

Model tensors fit the moment-matched Gaussians. The separate `gp_samples.pt`
tensors are the ground-truth references scored under those Gaussians.
`historical-batched` reproduces the archived evaluator's single float64
`[K,N]` CUDA random draw while storing samples in the common `[K,N,1]` format.
See [MMG_METRIC_METHODOLOGY.md](MMG_METRIC_METHODOLOGY.md) for the equations,
negative-cross-entropy interpretation, and complete computation procedure.

## Training

Train one model with `python -m better_results train --model MODEL --run-dir DIR`.
Train or resume the full sequential queue with
`python -m better_results train-all --runs-root DIR`. The default recipe is
128,000 steps; use `--max-steps 2` for a smoke run.

# Moment-matched Gaussian comparison

Values are per-target log likelihood in nats, reported as mean ± standard error.
Model samples fit each Gaussian; saved ground-truth GP samples are scored.
The standard error is across GP reference functions within one fixed independent task.

| Context | Ground Truth GP | Conditional NDP | Unconditional NDP | FlowNP |
|---:|---:|---:|---:|---:|
| 1 | -0.157 ± 0.010 | -0.515 ± 0.024 | **-0.461 ± 0.022** | -0.625 ± 0.034 |
| 5 | 0.038 ± 0.010 | -0.379 ± 0.026 | **-0.298 ± 0.027** | -0.467 ± 0.036 |
| 25 | 0.753 ± 0.010 | 0.235 ± 0.023 | **0.317 ± 0.029** | 0.214 ± 0.039 |
| 50 | 1.179 ± 0.010 | **0.764 ± 0.023** | 0.718 ± 0.033 | 0.635 ± 0.038 |
| 75 | 1.213 ± 0.010 | **0.700 ± 0.029** | 0.504 ± 0.040 | 0.588 ± 0.040 |
| 100 | 1.293 ± 0.010 | **0.743 ± 0.030** | 0.722 ± 0.034 | 0.742 ± 0.038 |

Model samples per fit: 100.
Ground-truth GP references per score: 100.
Covariance estimator: unbiased (denominator K-1).
Likelihood jitter: 1.0e-06.

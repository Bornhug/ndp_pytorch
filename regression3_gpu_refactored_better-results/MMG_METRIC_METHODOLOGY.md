# Moment-Matched Gaussian Metric: Mathematics and Computation

This document specifies the moment-matched Gaussian (MMG) metric used by the
standalone `better_results` project. It describes the historical independent-task
evaluation in:

```text
logs/eval_old_independent_ctx1_5_25_50_75_100
```

The metric answers this question:

> If a model's predictive distribution is approximated by a full multivariate
> Gaussian fitted from its generated functions, how much probability does that
> Gaussian assign to functions drawn from the analytic GP posterior?

Higher log likelihood is better.

## 1. Notation and evaluation setting

For each context size \(M\in\{1,5,25,50,75,100\}\), the evaluator constructs one
fixed GP regression task:

- Context locations and values:
  \(X_C\in\mathbb R^{M\times1}\) and \(Y_C\in\mathbb R^{M\times1}\).
- Target locations:
  \(X_T\in\mathbb R^{N\times1}\), with \(N=50\).
- Model samples:
  \(Y_m^{(1)},\ldots,Y_m^{(K)}\), with \(K=100\).
- Analytic-GP reference samples:
  \(Y_*^{(1)},\ldots,Y_*^{(R)}\), with \(R=100\).

Although \(K=R=100\) in this run, they have different roles:

- The \(K\) model samples are used only to estimate a model mean and covariance.
- The separate \(R\) GP samples are the held-out functions whose likelihood is
  evaluated.

The saved tensors all have shape `[100,50,1]`. For the multivariate likelihood,
the final singleton dimension is removed, so each function is one vector in
\(\mathbb R^{50}\). This is one joint 50-dimensional distribution, not 50
independently fitted univariate distributions.

## 2. Analytic noisy-GP posterior

The prior uses a Matérn-\(5/2\) kernel:

\[
k(x,x')
=
\sigma_f^2
\left(
1+\frac{\sqrt5r}{\ell}+\frac{5r^2}{3\ell^2}
\right)
\exp\left(-\frac{\sqrt5r}{\ell}\right),
\qquad r=|x-x'|,
\]

with length scale \(\ell=0.25\), variance \(\sigma_f^2=1\), and observation-noise
standard deviation \(\sigma_{\mathrm{obs}}=0.05\).

Let

\[
K_{CC}=k(X_C,X_C),\qquad
K_{CT}=k(X_C,X_T),\qquad
K_{TT}=k(X_T,X_T).
\]

The posterior over noisy target observations is

\[
\mu_*
=
K_{CT}^{\mathsf T}
\left(K_{CC}+\sigma_{\mathrm{obs}}^2I\right)^{-1}Y_C,
\]

\[
\Sigma_*
=
K_{TT}
-K_{CT}^{\mathsf T}
\left(K_{CC}+\sigma_{\mathrm{obs}}^2I\right)^{-1}K_{CT}
+\sigma_{\mathrm{obs}}^2I.
\]

Thus the analytic posterior is

\[
p_*(Y_T\mid X_T,X_C,Y_C)
=
\mathcal N(\mu_*,\Sigma_*).
\]

The displayed posterior equations omit numerical stabilization for readability.
The implementation factorizes the context covariance with an adaptive initial
jitter

\[
\delta(A)=10^{-6}\max\left(\operatorname{median}(\operatorname{diag}A),1\right),
\]

and multiplies it by 10 only if a Cholesky attempt fails. Observation noise and
numerical jitter are different quantities: \(0.05^2I\) is part of the statistical
model, whereas \(\delta(A)I\) exists only for numerical stability.

The historical run draws all \(R=100\) references using one batched float64 CUDA
normal draw with seed 98775. It factorizes the target posterior covariance using
the same adaptive rule, so the sampled reference distribution is

\[
p_{\mathrm{ref}}
=
\mathcal N(\mu_*,\Sigma_*+\delta_{\mathrm{ref}}I).
\]

All current tasks succeed on the first Cholesky attempt. Here
\(\delta_{\mathrm{ref}}=10^{-6}\) for contexts 5–100 and
\(1.0025\times10^{-6}\) for context 1.

## 3. Fitting each model's moment-matched Gaussian

For model \(m\), flatten each generated target function to
\(Y_m^{(k)}\in\mathbb R^{50}\). Estimate the predictive mean by

\[
\widehat\mu_m
=
\frac1K\sum_{k=1}^{K}Y_m^{(k)}.
\]

Estimate the full unbiased covariance by

\[
\widehat\Sigma_m
=
\frac1{K-1}
\sum_{k=1}^{K}
\left(Y_m^{(k)}-\widehat\mu_m\right)
\left(Y_m^{(k)}-\widehat\mu_m\right)^{\mathsf T}.
\]

For this run, the denominator is \(K-1=99\), and
\(\widehat\Sigma_m\in\mathbb R^{50\times50}\). Off-diagonal entries retain the
model's estimated correlation between different target locations.

No jitter is added while fitting these moments. During likelihood scoring, the
evaluator uses

\[
q_{m,\lambda}(Y_T)
=
\mathcal N\left(
Y_T;\widehat\mu_m,\widehat\Sigma_m+\lambda I
\right),
\qquad \lambda=10^{-6}.
\]

This MMG is an evaluation approximation to the model's sampled distribution. It
is not the model's probability-flow likelihood and is not used during training.

## 4. Per-reference MMG log likelihood

Every model is evaluated on the same \(R=100\) analytic-GP reference functions.
For reference \(r\), the joint log likelihood is

\[
L_{m,r}
=
\log\mathcal N\left(
Y_*^{(r)};\widehat\mu_m,\widehat\Sigma_m+\lambda I
\right).
\]

For a general Gaussian with covariance \(A\), this is computed as

\[
\log\mathcal N(y;\mu,A)
=
-\frac12\left[
N\log(2\pi)
+\log\det A
+(y-\mu)^{\mathsf T}A^{-1}(y-\mu)
\right].
\]

The report divides the joint value by \(N=50\):

\[
\ell_{m,r}=\frac{L_{m,r}}{50}.
\]

Therefore the reported unit is nats per target value. This normalization makes
scores easier to compare when the number of target locations changes, although
all tasks in this run use \(N=50\).

## 5. Ground-truth GP score and negative cross-entropy

The ground-truth column scores each reference under the stabilized analytic GP
density

\[
p_{\mathrm{score}}
=
\mathcal N(\mu_*,\Sigma_*+\lambda I),
\qquad \lambda=10^{-6},
\]

using

\[
\ell_{*,r}
=
\frac1N
\log p_{\mathrm{score}}\left(Y_*^{(r)}\right),
\qquad Y_*^{(r)}\sim p_{\mathrm{ref}}.
\]

For distributions \(p\) and \(q\), define cross-entropy as

\[
H(p,q)=-\mathbb E_{Y\sim p}[\log q(Y)].
\]

Therefore the expected per-target ground-truth score is the **negative
cross-entropy**

\[
-\frac1N
H\!\left(p_{\mathrm{ref}},p_{\mathrm{score}}\right)
=
\frac1N
\mathbb E_{p_{\mathrm{ref}}}
\left[\log p_{\mathrm{score}}(Y)\right].
\]

The finite report mean approximates this expectation using the 100 saved GP
references. It should be described as negative cross-entropy even when the two
distributions happen to be equal.

For a fitted model Gaussian \(q_{m,\lambda}\), the corresponding expected score is

\[
-\frac1N
H\!\left(p_{\mathrm{ref}},q_{m,\lambda}\right)
=
\frac1N
\mathbb E_{p_{\mathrm{ref}}}
\left[\log q_{m,\lambda}(Y)\right].
\]

Higher likelihood is better because a higher negative cross-entropy means a
lower cross-entropy between the GP reference distribution and the evaluated
density.

The expected ground-truth-minus-model gap is

\[
\frac1N
\left[
H\!\left(p_{\mathrm{ref}},q_{m,\lambda}\right)
-H\!\left(p_{\mathrm{ref}},p_{\mathrm{score}}\right)
\right].
\]

For contexts 5–100, \(p_{\mathrm{ref}}=p_{\mathrm{score}}\) because their
reference-sampling and scoring jitters are both \(10^{-6}\). At context 1, the
jitters differ by only \(2.5\times10^{-9}\), which is numerically negligible.
On a finite set of references, a model can occasionally have a higher realized
sample mean due to Monte Carlo variation. The fitted MMG also varies with its
finite \(K=100\) model samples.

## 6. Mean, standard deviation, and standard error

For one model and one fixed context task, the report has \(R=100\) per-reference
scores \(\ell_{m,1},\ldots,\ell_{m,R}\).

The reported mean is

\[
\overline\ell_m
=
\frac1R\sum_{r=1}^{R}\ell_{m,r}.
\]

The stored `population_std` is

\[
s_{\mathrm{pop},m}
=
\sqrt{
\frac1R
\sum_{r=1}^{R}
\left(\ell_{m,r}-\overline\ell_m\right)^2
}.
\]

It describes how much log likelihood varies across GP posterior reference
functions within this one fixed task.

The reported standard error is

\[
\mathrm{SE}_m
=
\frac{s_{\mathrm{pop},m}}{\sqrt R}.
\]

With \(R=100\), this is \(s_{\mathrm{pop},m}/10\). It estimates the Monte Carlo
uncertainty of the mean score for this fixed fitted Gaussian and fixed task.

It is **not**:

- the standard deviation of the \(K=100\) model samples;
- uncertainty in the fitted model mean or covariance;
- a bootstrap confidence interval;
- variability across different context tasks;
- variability across training seeds.

The report writes `mean ± standard_error`. It does not multiply the standard
error by 1.96.

## 7. Step-by-step computation in this run

For each requested context size \(M\):

1. Reset the task generator to seed 10.
2. Propose \(x_C\sim\operatorname{Uniform}[-2,2]\) and reject a proposal if it is
   too close to an accepted context location.
3. Use minimum separation 0.05 for \(M=1,5,25,50\), and 0.01 for
   \(M=75,100\).
4. Continue the same task generator and jointly draw the \(M\) noisy context
   values from the Matérn-\(5/2\) GP prior.
5. Construct \(N=50\) evenly spaced target locations over \([-2,2]\).
6. Compute the analytic noisy-GP posterior mean and full covariance at those
   target locations.
7. Load or generate \(K=100\) samples from each step-128,000 EMA model:
   - Conditional NDP: direct DDPM with 500 steps.
   - Unconditional NDP: RePaint DDPM with 500 steps, five inner cycles, and
     fresh context noise.
   - FlowNP: Euler integration with 500 steps.
8. Fit one full \(50\times50\) Gaussian to each model's 100 saved samples using
   the unbiased denominator 99.
9. Load or draw \(R=100\) common analytic-GP posterior references using seed
   98775 and the historical batched float64 RNG stream.
10. Score every GP reference under the analytic posterior and under each of the
    three fitted model Gaussians, using \(10^{-6}I\) scoring jitter.
11. Divide every joint log likelihood by 50.
12. Compute the mean, population standard deviation, and standard error across
    the 100 reference scores.
13. Save all per-reference values and summaries to `ctx_M/mmg.json`, then build
    `comparison_mmg.json` and `comparison_mmg.md`.

The same context, target locations, and GP references are used for all three
models at a given \(M\), making their comparison paired at the reference-function
level.

Restarting seed 10 at every context size can make location sequences share
deterministic prefixes when they use the same spacing rule. Nevertheless, there
is no shared master GP realization: each context size performs its own
size-dependent proposal stream and GP value draw, and is treated as a separate
fixed task.

## 8. Scope and interpretation

This report is a **single-fixed-task MMG evaluation at each context size**. Each
context size constructs a separate historical-style task, but the standard
error is still calculated across 100 GP references inside that task. It is not
the official multi-task protocol in which scores are aggregated across many
independently constructed tasks.

The MMG score evaluates both predictive mean and joint covariance. A model can
have a good mean prediction but receive a poor MMG score if its uncertainty or
cross-target correlations are wrong. Conversely, this metric does not prove
that the model distribution is Gaussian: it deliberately discards all moments
beyond the fitted mean and covariance.

This metric is also distinct from:

- **Analytic GP likelihood of model samples:** model-generated functions are
  scored under the GP posterior, reversing the direction used here.
- **Probability-flow likelihood:** evaluates density using the model's learned
  continuous-time vector field rather than a sample-fitted Gaussian.
- **RMSE:** measures pointwise mean or standard-deviation error rather than a
  joint distributional log likelihood.

## 9. Reproducing the report from saved samples

No model loading, CUDA computation, or resampling is needed once the sample
tensors exist:

```powershell
python -m better_results report `
  --input-dir logs\eval_old_independent_ctx1_5_25_50_75_100 `
  --metrics mmg `
  --mmg-covariance unbiased `
  --mmg-score-jitter 1e-6
```

The implementation is in
[`better_results/reporting.py`](better_results/reporting.py); historical GP
posterior construction and reference sampling are in
[`better_results/gp.py`](better_results/gp.py).

perfect — here’s a side-by-side, “code ↔ math” walkthrough of what the **fixed-hyperparameter GP evaluator** is doing. I’ll use the function names from the files we added (`gp_likelihood_core.py`) and show exactly which math each line implements.

---

# 1) Using the same GP as your data generator

### Code (imports/constants)

* `from data import _LENGTHSCALE, _KERNEL_VAR, _NOISE_VAR, _rbf_kernel, _matern52_kernel`

### Math

You fix the GP hyperparameters to match `data.py`:

* base lengthscale: $\ell_0$ (e.g. 0.25),
* signal variance: $\sigma_f^2$,
* observation noise variance: $\sigma_n^2$,
* kernel: SE (RBF) or Matérn-5/2.

If only $D_\text{active}$ input dims are used, `data.py` scales the effective lengthscale:

$$
\ell \;=\; \ell_{\text{eff}} \;=\; \ell_0 \sqrt{D_\text{active}}.
$$

Kernel formulas:

* **SE/RBF** (“se”):

$$
k(x,x') \;=\; \sigma_f^2 \exp\!\bigg(-\frac{\|x-x'\|^2}{2\ell^2}\bigg).
$$

* **Matérn-5/2** (“matern”):

$$
r=\|x-x'\|,\qquad
k(x,x') \;=\; \sigma_f^2 \Big( 1+\tfrac{\sqrt5 r}{\ell} + \tfrac{5r^2}{3\ell^2} \Big) e^{-\sqrt5 r/\ell}.
$$

---

# 2) Posterior given the context

### Code (`gp_posterior_fixed`)

```python
Kdd = k(xC, xC) + θ.noise * I
KdT = k(xC, xT);  KTd = KdT.T;  KTT = k(xT, xT)
L = cholesky(Kdd)
alpha = cholesky_solve(yC, L)         # Kdd^{-1} y_c
m = (KTd @ alpha).view(-1)            # K_*c K_dd^{-1} y_c
v = cholesky_solve(KdT, L)            # Kdd^{-1} K_c*
S = KTT - (KTd @ v)                   # K** - K_*c Kdd^{-1} K_c*
if include_obs_noise: S += σ_n^2 I
```

### Math

With context $(X_c, y_c)$ and query inputs $X_*$:

$$
\begin{aligned}
K_{cc} &= k(X_c,X_c) + \sigma_n^2 I_M,\\
K_{c*} &= k(X_c,X_*), \quad K_{*c}=K_{c*}^\top,\quad
K_{**}=k(X_*,X_*).
\end{aligned}
$$

Zero-mean GP posterior:

$$
\boxed{\,m = K_{*c} K_{cc}^{-1} y_c\,}, \qquad
\boxed{\,S_f = K_{**} - K_{*c} K_{cc}^{-1} K_{c*}\,}.
$$

If you want the distribution of noisy observations $y_* = f_* + \eta$ with $\eta\!\sim\!\mathcal N(0,\sigma_n^2 I)$:

$$
\boxed{\,S = S_f + \sigma_n^2 I\,}.
$$

**Numerics:** Instead of forming $K_{cc}^{-1}$, the code uses

* Cholesky $K_{cc}=LL^\top$,
* triangular solves for $K_{cc}^{-1}y_c$ and $K_{cc}^{-1}K_{c*}$.
  An adaptive jitter guarantees $K_{cc}$ is SPD in finite precision.

---

# 3) Likelihood of a generated continuation

### Code (`mvn_loglik`)

```python
L = cholesky(S)
diff = (y_hat - m)
alpha = cholesky_solve(diff, L)        # S^{-1} diff
quad = diff^T alpha                    # (y_hat - m)^T S^{-1} (y_hat - m)
logdet = 2 * sum(log(diag(L)))         # log det S
loglik = -0.5 * (quad + logdet + N*log(2π))
```

### Math

For one generated continuation $\hat y_* \in \mathbb{R}^N$ at $X_*$, the evaluator computes:

$$
\boxed{
\log \mathcal N(\hat y_*; m, S)
= -\tfrac12 \big[
(\hat y_* - m)^\top S^{-1} (\hat y_* - m)
+ \log\det(2\pi S)
\big].
}
$$

* Quadratic term via solves with $L$: if $S = LL^\top$, then
  $\alpha = S^{-1}(\hat y_* - m)$ is obtained by two triangular solves.
* $\log\det S = 2\sum_i \log L_{ii}$.

Across $K$ samples:

$$
\overline{\text{LL}} \;=\; \frac1K \sum_{k=1}^K \log \mathcal N(\hat y_*^{(k)}; m, S).
$$

Larger $\overline{\text{LL}}$ ⇒ more GP-like outputs.

---

# 4) Calibration check (Mahalanobis)

### Code (`mahalanobis2`)

```python
L = cholesky(S)
v = solve(L, y_hat - m)         # v = L^{-1}(y_hat - m)
Q = (v**2).sum()                # ||S^{-1/2}(y_hat - m)||^2
```

### Math

$$
\boxed{
Q \;=\; \| S^{-1/2}(\hat y_* - m)\|_2^2
}
$$

Under a perfect GP match, $Q \sim \chi^2_N$ (mean $=N$). It reveals under/over-dispersion separately from likelihood.

---

# 5) Sorting/masking (to line up with your sampler output)

Your sampler returns $K$ curves over the **sorted** concatenation $\tilde X=\mathrm{sort}([X_c;X_*])$.
The helper builds a boolean mask $M\in\{0,1\}^{M+N}$ that is `True` exactly at the indices corresponding to $X_*$ within $\tilde X$. Then:

$$
\hat y_*^{(k)} \;=\; \tilde Y^{(k)}[M] \in \mathbb{R}^N,
$$

so the likelihood uses exactly the query segment that matches $m,S$ computed at $X_*$.

---

# 6) Why “fixed hypers” here is critical

Because your data generator in `data.py` uses **known** $(\ell_0,\sigma_f^2,\sigma_n^2)$ and a **specific kernel**, the evaluator **does not fit** anything: it **reuses the same kernel and hyperparameters** (including $\ell_{\text{eff}}=\ell_0\sqrt{D_\text{active}}$). So you are truly answering:

> *How probable are my generated continuations under the same GP that produced the dataset?*

---

# 7) Costs & stability (what the code’s numerics buy you)

* One Cholesky on $M\times M$ for the context ($O(M^3)$), reused for all $K$ samples.
* One Cholesky on $N\times N$ for the predictive covariance $S$ ($O(N^3)$), reused for all $K$.
* Per sample: $O(N^2)$ (two triangular solves + a few dot products).
* Everything in **float64** with **adaptive jittered Cholesky**.

---

## Tiny “cheat sheet”: code → formula

* `alpha = cholesky_solve(yC, L)`
  ↔ $K_{cc}^{-1} y_c$

* `v = cholesky_solve(KdT, L)`
  ↔ $K_{cc}^{-1} K_{c*}$

* `m = K_*c @ (K_{cc}^{-1} y_c)`
  ↔ $m$

* `S = K_{**} - K_*c @ (K_{cc}^{-1} K_{c*}) [+ \sigma_n^2 I]\)`
  ↔ $S_f$ (and $S$ if including noise)

* `loglik(y_hat; m, S)`
  ↔ $-\tfrac12[(y-m)^\top S^{-1}(y-m) + \log\det(2\pi S)]$

* `mahalanobis2(y_hat; m, S)`
  ↔ $\|S^{-1/2}(y-m)\|_2^2 \sim \chi^2_N$

That’s the whole evaluator, line-for-line with the math.

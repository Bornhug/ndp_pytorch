from __future__ import annotations

import time
import sys
import math
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# add project root so imports from 'regression' and 'neural_diffusion_processes' work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regression.config import Config
from neural_diffusion_processes.model import BiDimensionalAttentionModel
from neural_diffusion_processes.process_pd import GaussianDiffusionPD, cosine_schedule

from regression.likelihood import (
    estimate_kl_from_pf_lls,
    gaussian_entropy_from_cov,
    gp_posterior_fixed,
    gp_prior_fixed,
    mvn_loglik,
    mahalanobis2,
    GPHypersFixed,
    sample_gp_conditional_fixed,
    vp_probflow_loglik_conditional,
)

from regression.data import _DATASET_FACTORIES, _JITTER, _NOISE_VAR

# ======================= Defaults (PD) ========================
DEFAULT_MODE = "uncond"         # "uncond" or "cond"
DEFAULT_N_POINTS = 50
DEFAULT_N_FUNCS = 8
DEFAULT_SEED = 0
DEFAULT_OUT = Path("progressive_distillation/logs") / "out_ddim.png"
DEFAULT_LOG_ROOT = Path("progressive_distillation") / "logs"

# DDIM-only (generalized DDPM)
DEFAULT_NUM_STEPS = 50
DEFAULT_K = 14
# =============================================================


def build_process(cfg: Config, device: torch.device) -> GaussianDiffusionPD:
    betas = cosine_schedule(cfg.diffusion.beta_start, cfg.diffusion.beta_end, cfg.diffusion.timesteps).to(device)
    return GaussianDiffusionPD(betas)


def build_network(cfg: Config, device: torch.device) -> BiDimensionalAttentionModel:
    return BiDimensionalAttentionModel(
        n_layers=cfg.network.n_layers,
        hidden_dim=cfg.network.hidden_dim,
        num_heads=cfg.network.num_heads,
    ).to(device)


def load_ema_model(cfg: Config, device: torch.device, ckpt_path: Path):
    model = build_network(cfg, device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def make_x0_model_adapter(model: torch.nn.Module, T: int):
    """
    Adapter so unified sampler can call: fn(t, yt, x_aug, mask, *, key).
    Our PD model predicts x0; we ignore masks (contexts handled explicitly).
    """
    def x0_model(t, yt, xx, mask, *, key):
        t_tensor = torch.as_tensor(t, device=xx.device).long().clamp_(0, T - 1).view(1)
        # Split xx/yt by mask if desired; here we pass them as combined (handled upstream).
        # Keep API compatible with training forward: (x_tgt, y_tgt, t, mask_tgt, x_context, y_context, mask_context)
        # For unconditional path, pass None for contexts.
        if yt.ndim == 2:
            yt_b = yt.unsqueeze(0)
            xx_b = xx.unsqueeze(0)
        else:
            yt_b, xx_b = yt, xx
        return model(xx_b, yt_b, t_tensor, None).squeeze(0)
    return x0_model


def _save_plot_conditional(xs, ys, x_ctx, y_ctx, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = xs.detach().cpu().view(-1)
    ys = ys.detach().cpu()
    plt.figure(figsize=(5.2, 3.5))
    if ys.ndim == 1:
        plt.plot(xs, ys, alpha=0.9, label="sample")
    elif ys.ndim == 2:
        N = xs.numel()
        if ys.shape[0] == N:
            plt.plot(xs, ys, alpha=0.7)
        elif ys.shape[1] == N:
            plt.plot(xs, ys.T, alpha=0.7)
        else:
            raise ValueError(f"ys has incompatible shape {tuple(ys.shape)} for xs length {N}")
    else:
        raise ValueError(f"ys must be 1D or 2D, got {ys.ndim}D")
    plt.scatter(x_ctx.detach().cpu(), y_ctx.detach().cpu(), s=35, c="k", zorder=5, label="context")
    plt.title(title); plt.legend(frameon=False, fontsize=8)
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


@torch.no_grad()
def evaluate_gp_scores(
    dataset_name: str,
    input_dim: int,
    *,
    xs_sorted: torch.Tensor,          # [L]
    ys_sorted: torch.Tensor,          # [S,L]
    include_obs_noise: bool = True,
    x_ctx: torch.Tensor | None = None,
    y_ctx: torch.Tensor | None = None,
    x_query: torch.Tensor | None = None,
):
    """Unified prior/posterior GP scoring based on whether context is provided (M>0)."""
    active_dims = list(range(input_dim))
    if x_ctx is not None and y_ctx is not None and x_query is not None and x_ctx.numel() > 0:
        m, S, θ = gp_posterior_fixed(dataset_name, x_ctx, y_ctx, x_query, active_dims, include_obs_noise)
        x_concat = torch.cat([x_ctx, x_query], dim=0).squeeze(-1)
        M, N = x_ctx.size(0), x_query.size(0)
        order = torch.argsort(x_concat)
        is_query = torch.zeros(M + N, dtype=torch.bool, device=x_ctx.device); is_query[M:M+N] = True
        mask_query = is_query[order]
        y_eval = ys_sorted[:, mask_query]
    else:
        x = xs_sorted.view(-1, 1)
        m, S, θ = gp_prior_fixed(dataset_name, x, active_dims, include_obs_noise)
        y_eval = ys_sorted

    ll_list, q_list = [], []
    y_eval = y_eval.double()
    for k in range(y_eval.size(0)):
        yk = y_eval[k]
        ll_list.append(mvn_loglik(yk, m, S).detach().cpu().item())
        q_list.append(mahalanobis2(yk, m, S).detach().cpu().item())

    ll = torch.tensor(ll_list); q = torch.tensor(q_list)
    stats = {
        "ell_eff": float(θ.ell_eff), "var": float(θ.var), "noise": float(θ.noise),
        "mean_ll": float(ll.mean().item()), "std_ll": float(ll.std(unbiased=False).item()),
        "mean_mahal": float(q.mean().item()), "std_mahal": float(q.std(unbiased=False).item()),
        "count": int(y_eval.size(0)), "N": int(y_eval.size(1)),
        "include_obs_noise": bool(include_obs_noise),
    }
    return stats, ll_list, q_list


def find_latest_ckpt(root: Path = DEFAULT_LOG_ROOT) -> Path | None:
    if not root.exists():
        return None
    cks = list(root.rglob("model_ema.pt"))
    if not cks:
        return None
    return max(cks, key=lambda p: p.stat().st_mtime)


@torch.no_grad()
def sample_cond_ddim(cfg, model, process, device, x_context, y_context, x_query,
                     K: int = 14, seed: int = 0, num_steps: int = 50, eta: float = 0.0):
    T = int(process.betas.numel())
    net_fn = make_x0_model_adapter(model, T)

    x_plot = torch.cat([x_context, x_query], dim=0).squeeze(-1)
    order = torch.argsort(x_plot)
    xs_sorted = x_plot[order]

    ys_rows = []
    for s_id in range(K):
        g = torch.Generator(device=device).manual_seed(seed + s_id)
        y_q = process.sample(
            g, x_query,
            model_fn=net_fn,
            x_context=x_context, y_context=y_context,
            num_sample_steps=num_steps, eta=eta,
        )
        y_plot = torch.cat([y_context.squeeze(-1), y_q.squeeze(-1)], dim=0)
        ys_rows.append(y_plot[order])
    ys_sorted = torch.stack(ys_rows, dim=0)
    return xs_sorted, ys_sorted


def _dump_json(path: Path, payload: dict, msg: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    if msg:
        print(f"✓ saved {msg}: {path}")


def _iter_with_progress(K: int, desc: str) -> Iterable[int]:
    if K > 5:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=K, desc=desc, leave=False)
            def wrapper():
                for i in range(K):
                    yield i
                    pbar.update(1)
            return wrapper()
        except Exception:
            return range(K)
    return range(K)


def run_evaluations(*, mode, cfg, model, process, device, n_points, n_funcs, seed,
                    num_steps, K, include_obs_noise, out_path, x_ctx, y_ctx, x_query,
                    sampler,
                    compute_fixed, compute_real, compute_ceiling, compute_gp_sample_loglik, compute_noise):
    posterior = None

    def _ensure_posterior():
        nonlocal posterior
        if posterior is None:
            posterior = gp_posterior_fixed(
                cfg.dataset, x_ctx, y_ctx, x_query, list(range(cfg.input_dim)), include_obs_noise=include_obs_noise
            )
        return posterior

    if compute_fixed:
        # Derive effective DDIM parameters from sampler choice
        # DDPM ≈ DDIM with full schedule and eta=1.0
        steps_eff = int(process.betas.numel()) if sampler == "ddpm" else num_steps
        eta_eff = 1.0 if sampler == "ddpm" else 0.0
        # Use conditional path always; when x_ctx has 0 rows this reduces to unconditional
        xs_sorted, ys_sorted = sample_cond_ddim(
            cfg, model, process, device,
            x_ctx, y_ctx, x_query,
            K=K, seed=seed,
            num_steps=steps_eff, eta=eta_eff,
        )
        out_png = out_path.with_suffix(".cond.png")
        _save_plot_conditional(
            xs_sorted, ys_sorted, x_ctx.squeeze(-1), y_ctx.squeeze(-1), out_png,
            title=f"Conditional {sampler.upper()} samples (K={K})"
        )
        print(f"✓ saved: {out_png}")

        stats, per_ll, per_mahal = evaluate_gp_scores(
            dataset_name=cfg.dataset, input_dim=cfg.input_dim,
            xs_sorted=xs_sorted, ys_sorted=ys_sorted, include_obs_noise=include_obs_noise,
            x_ctx=x_ctx, y_ctx=y_ctx, x_query=x_query,
        )
        payload_cond = {
            "sampler": "ddim",
            "num_steps": num_steps,
            "seed": seed,
            "K": K,
            "M": x_ctx.size(0),
            "per_ll": per_ll,
            "per_mahalanobis2": per_mahal,
            "stats": stats,
        }
        _dump_json(out_path.with_suffix(".gp_eval.fixed.json"), payload_cond, "GP posterior eval (cond)")

        # Optional diffusion PF-ODE scoring on GP posterior samples
        if compute_gp_sample_loglik:
            if cfg.dataset not in {"se", "matern"}:
                print("[gp-sample/loglik] Skipping: dataset not GP (requires 'se' or 'matern').")
            elif x_ctx.size(0) == 0:
                print("[gp-sample/loglik] Skipping: no context points available.")
            else:
                pf_gp_lls: list[float] = []
                gp_rng = torch.Generator(device=x_query.device).manual_seed(seed + 12345)
                for _ in _iter_with_progress(K, desc="PF-ODE (GP samples)"):
                    y_gp_sample, _ = sample_gp_conditional_fixed(
                        dataset=cfg.dataset,
                        x_target=x_query,
                        x_context=x_ctx,
                        y_context=y_ctx,
                        include_obs_noise=include_obs_noise,
                        generator=gp_rng,
                    )
                    logp_gp = vp_probflow_loglik_conditional(
                        model=model, process=process, x_target=x_query, y_target0=y_gp_sample,
                        x_context=x_ctx, y_context=y_ctx, mask_context=None, steps=500, hutch="rademacher",
                    )
                    pf_gp_lls.append(logp_gp.item())
                pf_gp_lls_t = torch.tensor(pf_gp_lls) if pf_gp_lls else torch.tensor([0.0])
                payload_gp_pf = {
                    "sampler": "ddim",
                    "num_steps": num_steps,
                    "seed": seed,
                    "K": len(pf_gp_lls),
                    "M": x_ctx.size(0),
                    "pf_loglik": {
                        "mean": float(pf_gp_lls_t.mean().item()),
                        "std": float(pf_gp_lls_t.std(unbiased=False).item()) if pf_gp_lls_t.numel() > 1 else 0.0,
                        "per_sample": pf_gp_lls,
                    },
                }
                _dump_json(out_path.with_suffix(f".gp_eval.fixed_pf_ctx{int(x_ctx.size(0))}.json"), payload_gp_pf,
                            "GP sample diffusion eval")

                # KL estimate via entropy + E_p[log q]
                m_post, S_post, _ = gp_posterior_fixed(
                    cfg.dataset, x_ctx=x_ctx, y_ctx=y_ctx, x_tgt=x_query,
                    active_dims=list(range(cfg.input_dim)), include_obs_noise=include_obs_noise,
                )
                kl_stats = estimate_kl_from_pf_lls(torch.tensor(pf_gp_lls, dtype=torch.float64), S_post)
                _dump_json(out_path.with_suffix(f".gp_eval.fixed_pf_ctx{int(x_ctx.size(0))}.kl.json"), kl_stats,
                            "KL estimate (fixed PF vs model)")

    if compute_noise:
        pf_noise_lls: list[float] = []
        noise_rng = torch.Generator(device=device).manual_seed(seed + 54321)
        x_range = (-2.0, 2.0)
        y_range = (-1.0, 1.0)
        for _ in _iter_with_progress(K, desc="PF-ODE (uniform noise)"):
            x_noise = torch.rand((n_points, cfg.input_dim), dtype=torch.float32, device=device, generator=noise_rng)
            x_noise = x_noise.mul_(x_range[1] - x_range[0]).add_(x_range[0])
            y_noise = torch.rand((n_points, 1), dtype=torch.float32, device=device, generator=noise_rng)
            y_noise = y_noise.mul_(y_range[1] - y_range[0]).add_(y_range[0])
            logp_noise = vp_probflow_loglik_conditional(
                model=model, process=process, x_target=x_noise, y_target0=y_noise,
                x_context=None, y_context=None, mask_context=None, steps=500, hutch="rademacher",
            )
            pf_noise_lls.append(logp_noise.item())
        pf_noise_t = torch.tensor(pf_noise_lls) if pf_noise_lls else torch.tensor([0.0])
        payload_noise = {
            "sampler": "ddim",
            "num_steps": num_steps,
            "seed": seed,
            "K": len(pf_noise_lls),
            "x_range": x_range,
            "y_range": y_range,
            "pf_loglik": {
                "mean": float(pf_noise_t.mean().item()),
                "std": float(pf_noise_t.std(unbiased=False).item()) if pf_noise_t.numel() > 1 else 0.0,
                "per_sample": pf_noise_lls,
            },
        }
        _dump_json(out_path.with_suffix(".noise_pf.json"), payload_noise, "uniform noise diffusion eval")

    if compute_real:
        m, S, θ = _ensure_posterior()
        Nt = m.numel()
        jitter = 1e-9
        try:
            L = torch.linalg.cholesky(S)
        except RuntimeError:
            L = torch.linalg.cholesky(S + jitter * torch.eye(Nt, dtype=S.dtype, device=S.device))
        z = torch.randn((K, Nt), dtype=m.dtype, device=m.device)
        y_samps = m.unsqueeze(0) + z @ L.T
        x_plot = torch.cat([x_ctx, x_query], dim=0).squeeze(-1)
        order = torch.argsort(x_plot)
        xs_sorted_real = x_plot[order]
        ys_rows_real = []
        for k in range(K):
            y_plot = torch.cat([y_ctx.squeeze(-1), y_samps[k].squeeze(-1)], dim=0)
            ys_rows_real.append(y_plot[order])
        ys_sorted_real = torch.stack(ys_rows_real, dim=0)
        out_path_real_png = out_path.with_suffix(".real.png")
        _save_plot_conditional(xs_sorted_real, ys_sorted_real, x_ctx.squeeze(-1), y_ctx.squeeze(-1), out_path_real_png,
                               title="Real GP baseline samples")
        per_ll2, per_mahal2 = [], []
        for k in _iter_with_progress(K, desc="PF-ODE (real GP)"):
            yk = y_samps[k]
            per_ll2.append(mvn_loglik(yk, m, S).detach().cpu().item())
            per_mahal2.append(mahalanobis2(yk, m, S).detach().cpu().item())
        payload_real = {
            "ell_eff": float(θ.ell_eff), "var": float(θ.var), "noise": float(θ.noise),
            "mean_ll": float(torch.tensor(per_ll2).mean().item()),
            "std_ll": float(torch.tensor(per_ll2).std(unbiased=False).item()) if K > 1 else 0.0,
            "mean_mahal": float(torch.tensor(per_mahal2).mean().item()),
            "std_mahal": float(torch.tensor(per_mahal2).std(unbiased=False).item()) if K > 1 else 0.0,
            "K": int(K), "N": int(Nt), "include_obs_noise": bool(include_obs_noise),
        }
        _dump_json(out_path.with_suffix(".gp_eval.real.json"), payload_real, "GP eval (real)")

    if compute_ceiling:
        m, S, θ = _ensure_posterior()
        x_plot = torch.cat([x_ctx, x_query], dim=0).squeeze(-1)
        order = torch.argsort(x_plot)
        xs_sorted = x_plot[order]
        y_plot = torch.cat([y_ctx.squeeze(-1), m.squeeze(-1)], dim=0)
        y_sorted = y_plot[order]
        out_path_ceiling_png = out_path.with_suffix(".ceiling.png")
        _save_plot_conditional(xs_sorted, y_sorted, x_ctx.squeeze(-1), y_ctx.squeeze(-1), out_path_ceiling_png,
                               title="GP posterior mean (ceiling)")
        payload3 = {
            "sampler": "none", "num_steps": 0, "seed": seed, "K": K, "M": x_ctx.size(0),
            "model_score": {
                "mean": 0.0, "std": 0.0, "per_sample": [0.0 for _ in range(K)],
            },
        }
        _dump_json(out_path.with_suffix(".gp_eval.ceiling.json"), payload3, "GP eval (ceiling)")

    return False


def main(
    ckpt: Path | None = None,
    mode: str = DEFAULT_MODE,
    n_points: int = DEFAULT_N_POINTS,
    n_funcs: int = DEFAULT_N_FUNCS,
    seed: int = DEFAULT_SEED,
    out_path: Path = DEFAULT_OUT,
    num_steps: int = DEFAULT_NUM_STEPS,
    K: int = DEFAULT_K,
    compute_fixed: bool = True,
    compute_real: bool = False,
    compute_ceiling: bool = False,
    include_obs_noise: bool = True,
    compute_gp_sample_loglik: bool = False,
    compute_noise: bool = False,
    num_context: int = 10,
):
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(ckpt) if ckpt is not None else find_latest_ckpt()
    if ckpt_path is None or not ckpt_path.exists():
        raise SystemExit(
            f"[!] No checkpoint found.\n"
            f"    Looked for latest under: {DEFAULT_LOG_ROOT}\n"
            f"    Or pass an explicit path to main(ckpt=...)."
        )
    model = load_ema_model(cfg, device, ckpt_path)
    process = build_process(cfg, device)

    if cfg.dataset not in {"se", "matern"}:
        raise ValueError("This evaluation path supports only GP datasets ('se' or 'matern').")

    rng_device = "cpu" if device.type == "cpu" else f"{device.type}:{device.index or 0}"
    prior_rng = torch.Generator(device=rng_device)
    device = torch.device(rng_device)
    prior_rng.manual_seed(seed)

    num_context = max(0, int(num_context))
    if num_context > 0:
        x_ctx = torch.rand((num_context, cfg.input_dim), device=device, dtype=torch.float32, generator=prior_rng)
        x_ctx = x_ctx.mul_(4.0).sub_(2.0)
        y_ctx, _ = sample_gp_conditional_fixed(
            dataset=cfg.dataset, x_target=x_ctx, include_obs_noise=include_obs_noise, generator=prior_rng
        )
    else:
        x_ctx = torch.empty((0, cfg.input_dim), device=device, dtype=torch.float32)
        y_ctx = torch.empty((0, 1), device=device, dtype=torch.float32)

    x_query = torch.linspace(-2, 2, n_points, device=device, dtype=torch.float32).unsqueeze(-1)
    out_path = out_path.with_name(f"{out_path.stem}_ctx{num_context}{out_path.suffix}")

    _ = run_evaluations(
        mode=mode, cfg=cfg, model=model, process=process, device=device,
        n_points=n_points, n_funcs=n_funcs, seed=seed,
        num_steps=num_steps, K=K, include_obs_noise=include_obs_noise, out_path=out_path,
        x_ctx=x_ctx, y_ctx=y_ctx, x_query=x_query,
        sampler=sampler,
        compute_fixed=compute_fixed, compute_real=compute_real, compute_ceiling=compute_ceiling,
        compute_gp_sample_loglik=compute_gp_sample_loglik, compute_noise=compute_noise,
    )


if __name__ == "__main__":
    # Specify the checkpoint to evaluate (example below). If None, falls back to latest under DEFAULT_LOG_ROOT.
    # Example (change to your run directory):
    ckpt_file = Path(f"progressive_distillation/logs/progressive_distill/pd_Oct21_020506_from512_to1/00_512to256/model_ema.pt")

    sampler = "ddim"  # or "ddpm"
    num_steps = 128
    K = 30

    # Build an output path matching your run folder naming
    out_path = Path(f"progressive_distillation/logs/progressive_distill/pd_Oct21_020506_from512_to1/00_512to256/out_{sampler}_steps{num_steps}_K{K}.png")

    main(
        ckpt=ckpt_file,
        mode="cond",
        n_points=50,
        n_funcs=30,
        seed=10,
        num_steps=num_steps,
        K=K,
        num_context=5,  # Try different numbers of context points
        out_path=out_path,
        compute_fixed=True, compute_real=False, compute_ceiling=False,
        compute_gp_sample_loglik=False,
    )

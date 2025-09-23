# ─────────────────── interpolation.py (no-CLI needed) ───────────────────
from __future__ import annotations
import time

# --- make imports work when run directly in PyCharm ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # project root …/ndp_pytorch
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- plotting backend (non-interactive) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

# project imports – same as training
from config import Config
from neural_diffusion_processes.model   import BiDimensionalAttentionModel
from neural_diffusion_processes.process import GaussianDiffusion, cosine_schedule
from data import get_batch, Batch

from neural_diffusion_processes.samplers_ddim import DDIMSampler
from neural_diffusion_processes.samplers_ode  import EulerHeunSampler

import json  # NEW

# NEW: fixed-hyper GP core
from likelihood import gp_posterior_fixed, mvn_loglik, mahalanobis2, GPHypersFixed

# NEW: toggle & options
DEFAULT_EVAL_GP = True                 # compute GP-likeness when in "cond" mode
DEFAULT_INCLUDE_OBS_NOISE = True       # True: score y (with noise), False: latent f

# ========================== USER DEFAULTS ==========================
# You can change these defaults and just click "Run" in PyCharm.
DEFAULT_MODE      = "uncond"   # "uncond" or "cond"
DEFAULT_N_POINTS  = 50        # number of x points to sample on
DEFAULT_N_FUNCS   = 8          # curves for unconditional mode
DEFAULT_SEED      = 0
DEFAULT_OUT       = Path("samples") / "out_ddpm_steps500.png"
DEFAULT_LOG_ROOT  = Path("logs") / "regression"   # where runs live


DEFAULT_SAMPLER   = "ddpm"   # "ddpm" | "ddim" | "euler" | "heun"
DEFAULT_NUM_STEPS = 50       # used by ddim/euler/heun
DEFAULT_K         = 14       # number of conditional samples
# ==================================================================


# --------------------------- Builders ---------------------------

def build_process(cfg: Config, device: torch.device) -> GaussianDiffusion:
    betas = cosine_schedule(cfg.diffusion.beta_start,
                            cfg.diffusion.beta_end,
                            cfg.diffusion.timesteps).to(device)
    return GaussianDiffusion(betas)

def build_network(cfg: Config, device: torch.device) -> BiDimensionalAttentionModel:
    net = BiDimensionalAttentionModel(
        n_layers   = cfg.network.n_layers,
        hidden_dim = cfg.network.hidden_dim,
        num_heads  = cfg.network.num_heads,
    ).to(device)
    return net

def load_ema_model(cfg: Config, device: torch.device, ckpt_path: Path):
    model = build_network(cfg, device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model



# in interpolation_speed.py (and use the same idea for the training-time _plot_samples adapter if needed)
def make_eps_model(model: torch.nn.Module, T: int):
    """
    Adapter so the samplers can call: fn(t, yt, x_aug, mask_aug, *, key)
    We split x_aug, yt by mask into (context, target) and pass them to the model.
    """
    def eps_model(t, yt, xx, mask, *, key):
        # t → [1] Long within [0, T-1]
        t_tensor = torch.as_tensor(t, device=xx.device).long().clamp_(0, T - 1).view(1)

        # mask semantics in our code: 1 == "context row" (kept fixed); 0 == "target row"
        is_ctx = (mask > 0.5)
        x_ctx,  y_ctx  = xx[is_ctx],  yt[is_ctx]          # [M,D], [M,1]
        x_tgt,  y_tgt  = xx[~is_ctx], yt[~is_ctx]         # [N,D], [N,1]

        # add batch dim; allow "no context" case gracefully
        x_tgt = x_tgt.unsqueeze(0); y_tgt = y_tgt.unsqueeze(0)
        m_tgt = mask[~is_ctx].unsqueeze(0) if mask is not None else None

        if x_ctx.numel() == 0:
            x_ctx_b = y_ctx_b = m_ctx_b = None
        else:
            x_ctx_b = x_ctx.unsqueeze(0); y_ctx_b = y_ctx.unsqueeze(0)
            m_ctx_b = torch.zeros(x_ctx_b.size(1), device=xx.device).unsqueeze(0)  # all valid

        out = model(x_tgt, y_tgt, t_tensor, m_tgt,
                    x_context=x_ctx_b, y_context=y_ctx_b, mask_context=m_ctx_b)
        return out.squeeze(0)  # [N,1]
    return eps_model



# ----------------------- Evaluating GP(Computing Log likelihood) ------------------------

def _mask_query_columns(x_ctx: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
    x_concat = torch.cat([x_ctx, x_query], dim=0).squeeze(-1)  # [M+N]
    M, N = x_ctx.size(0), x_query.size(0)
    order = torch.argsort(x_concat)                            # [M+N]
    is_query = torch.zeros(M + N, dtype=torch.bool, device=x_ctx.device)
    is_query[M:M+N] = True
    return is_query[order]


def evaluate_gp_likeness_from_sorted_fixed(
    dataset_name: str,
    input_dim: int,
    x_ctx: torch.Tensor, y_ctx: torch.Tensor,
    x_query: torch.Tensor,
    xs_sorted: torch.Tensor, ys_sorted: torch.Tensor,
    include_obs_noise: bool = True,
):
    """
    Scores each generated continuation (rows of ys_sorted restricted to query columns)
    under the *fixed-parameter* GP posterior used by data.py.
    """
    active_dims = list(range(input_dim))

    # GP posterior with fixed hypers & matching kernel
    m, S, θ = gp_posterior_fixed(dataset_name, x_ctx, y_ctx, x_query, active_dims,
                                 include_obs_noise=include_obs_noise)

    # Extract only the query part from xs/ys, aligned with xs_sorted
    mask_query = _mask_query_columns(x_ctx, x_query)   # [M+N]
    yq_all = ys_sorted[:, mask_query]                  # [K, N]

    # Score each sample
    ll_list, q_list = [], []
    for k in range(yq_all.size(0)):
        yk = yq_all[k].double()
        ll_list.append(mvn_loglik(yk, m, S).detach().cpu().item())
        q_list.append(mahalanobis2(yk, m, S).detach().cpu().item())

    ll = torch.tensor(ll_list)
    q  = torch.tensor(q_list)
    stats = {
        "ell_eff": float(θ.ell_eff), "var": float(θ.var), "noise": float(θ.noise),
        "mean_ll": float(ll.mean().item()), "std_ll": float(ll.std(unbiased=False).item()),
        "mean_mahal": float(q.mean().item()), "std_mahal": float(q.std(unbiased=False).item()),
        "K": int(yq_all.size(0)), "N": int(yq_all.size(1)),
        "include_obs_noise": bool(include_obs_noise),
    }
    return stats, ll.tolist(), q.tolist()



# ----------------------- Auto checkpoint ------------------------

def find_latest_ckpt(root: Path = DEFAULT_LOG_ROOT) -> Path | None:
    """Find the most recently modified model_ema.pt under logs/regression/**/."""
    if not root.exists():
        return None
    cks = list(root.rglob("model_ema.pt"))
    if not cks:
        return None
    return max(cks, key=lambda p: p.stat().st_mtime)


# ----------------------- Unconditional -------------------------

@torch.no_grad()
def sample_uncond(cfg, model, process, device, x_min=-2, x_max=2, n_points=60, n_funcs=8,
                  seed=42, sampler: str = "ddpm", num_steps: int = 50):
    x = torch.linspace(x_min, x_max, n_points, device=device).unsqueeze(-1)  # [N,1]
    m = torch.zeros(n_points, device=device)  # explicit mask
    gen = torch.Generator(device=device).manual_seed(seed)

    T = int(process.betas.numel())
    net_fn = make_eps_model(model, T)

    ys = []
    if sampler == "ddpm":
        # original full-step sampler
        for _ in range(n_funcs):
            ys.append(process.sample(gen, x, None, model_fn=net_fn))
    elif sampler == "ddim":
        s = DDIMSampler(process, num_sample_steps=num_steps)
        for _ in range(n_funcs):
            ys.append(s.sample_uncond(gen, x, net_fn, m, y_dim=1))
    elif sampler in ("euler", "heun"):
        s = EulerHeunSampler(process, num_sample_steps=num_steps)
        for _ in range(n_funcs):
            ys.append(s.sample_uncond(gen, x, net_fn, m, y_dim=1, method=sampler))
    else:
        raise ValueError(f"Unknown sampler: {sampler}")
    return x, torch.stack(ys).squeeze(-1)  # [S,N]




def plot_uncond(x, ys, out_path: Path, title="Unconditional samples"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3.5))
    plt.plot(x.detach().cpu(), ys.detach().cpu().T, alpha=0.6)
    plt.title(title); plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def assert_same_schedule(cfg, process):
    T_cfg = cfg.diffusion.timesteps
    T_proc = int(process.betas.numel())
    assert T_cfg == T_proc, f"Timesteps mismatch: cfg={T_cfg} vs process={T_proc}"
    print(f"[schedule] T={T_proc}, beta0={float(process.betas[0]):.3e}, betaT={float(process.betas[-1]):.3e}")

@torch.no_grad()
def plot_exact_like_training(model, process, cfg, device, out_path, title="step_like_training"):
    from pathlib import Path
    out_path = Path(out_path)  # ✅ make sure it's a Path

    assert not model.training, "Call model.eval() before sampling."
    x = torch.linspace(-2, 2, 60, device=device).unsqueeze(-1)
    net_fn = lambda t, yt, xx, m, *, key: (
        model(xx.unsqueeze(0), yt.unsqueeze(0), t.view(1),
              m.unsqueeze(0) if m is not None else m).squeeze(0)
    )
    gen = torch.Generator(device=device).manual_seed(0)
    ys = torch.stack([process.sample(gen, x, None, model_fn=net_fn) for _ in range(8)]
                     ).squeeze(-1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.plot(x.detach().cpu(), ys.detach().cpu().T, color="C0", alpha=0.5)
    plt.title(title); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()
# ------------------------ Conditional --------------------------

@torch.no_grad()
def sample_cond(cfg, model, process, device, x_context, y_context, x_query,
                K: int = 14, seed: int = 0, sampler: str = "ddpm", num_steps: int = 50):
    """
    Returns:
      xs_sorted: [M+N]
      ys_sorted: [K, M+N]  (each row is one conditional sample aligned with xs_sorted)
    """
    T = int(process.betas.numel())
    net_fn = make_eps_model(model, T)

    m_ctx = torch.zeros(x_context.size(0), device=device)  # [M]
    m_tgt = torch.zeros(x_query.size(0),   device=device)  # [N]

    # Build plotting x once and its permutation
    x_plot = torch.cat([x_context, x_query], dim=0).squeeze(-1)  # [M+N]
    order  = torch.argsort(x_plot)                               # [M+N]
    xs_sorted = x_plot[order]                                    # [M+N]

    # Prepare sampler object once
    ddim_sampler = None
    ode_sampler  = None
    if sampler == "ddim":
        ddim_sampler = DDIMSampler(process, num_sample_steps=num_steps)
    elif sampler in ("euler", "heun"):
        ode_sampler  = EulerHeunSampler(process, num_sample_steps=num_steps)

    ys_rows = []
    for s_id in range(K):
        g = torch.Generator(device=device).manual_seed(seed + s_id)

        if sampler == "ddpm":
            y_q = process.conditional_sample(
                g, x_query, m_tgt,
                x_context=x_context, y_context=y_context,
                mask_context=m_ctx, model_fn=net_fn
            )  # [N,1]
        elif sampler == "ddim":
            y_q = ddim_sampler.sample_cond(
                g, x_query, m_tgt,
                x_context=x_context, y_context=y_context, mask_context=m_ctx,
                model_fn=net_fn, y_dim=1
            )  # [N,1]
        elif sampler in ("euler", "heun"):
            y_q = ode_sampler.sample_cond(
                g, x_query, m_tgt,
                x_context=x_context, y_context=y_context, mask_context=m_ctx,
                model_fn=net_fn, y_dim=1, method=sampler
            )  # [N,1]
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        # Combine context + query predictions → length M+N, then reorder
        y_plot = torch.cat([y_context.squeeze(-1), y_q.squeeze(-1)], dim=0)  # [M+N]
        ys_rows.append(y_plot[order])                                        # [M+N]

    ys_sorted = torch.stack(ys_rows, dim=0)  # [K, M+N]
    return xs_sorted, ys_sorted





def plot_conditional(xs, ys, x_ctx, y_ctx, out_path: Path, title="Conditional sample"):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xs = xs.detach().cpu().view(-1)     # [N]
    ys = ys.detach().cpu()              # [N] or [K,N] or [N,K]

    plt.figure(figsize=(5.2, 3.5))

    if ys.ndim == 1:
        # single curve: [N]
        plt.plot(xs, ys, alpha=0.9, label="sample")
    elif ys.ndim == 2:
        N = xs.numel()
        if ys.shape[0] == N:
            # shape [N, K] → columns are K curves
            plt.plot(xs, ys, alpha=0.7)
        elif ys.shape[1] == N:
            # shape [K, N] → rows are K curves, transpose for plotting
            plt.plot(xs, ys.T, alpha=0.7)
        else:
            raise ValueError(f"ys has incompatible shape {tuple(ys.shape)} for xs length {N}")
    else:
        raise ValueError(f"ys must be 1D or 2D, got {ys.ndim}D")

    plt.scatter(x_ctx.detach().cpu(), y_ctx.detach().cpu(), s=35, c="k", zorder=5, label="context")
    plt.title(title)
    plt.legend(frameon=False, fontsize=8)
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()





# ----------------------------- Main -----------------------------

def main(
    ckpt: Path | None = None,
    mode: str = DEFAULT_MODE,
    n_points: int = DEFAULT_N_POINTS,
    n_funcs: int = DEFAULT_N_FUNCS,
    seed: int = DEFAULT_SEED,
    out_path: Path = DEFAULT_OUT,
    sampler: str = DEFAULT_SAMPLER,
    num_steps: int = DEFAULT_NUM_STEPS,
    K: int = DEFAULT_K,
):

    cfg = Config()  # or Config.from_file(...)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-find checkpoint if not provided
    ckpt_path = Path(ckpt) if ckpt is not None else find_latest_ckpt()
    if ckpt_path is None or not ckpt_path.exists():
        raise SystemExit(
            f"[!] No checkpoint found.\n"
            f"    Looked for latest under: {DEFAULT_LOG_ROOT}\n"
            f"    Or pass an explicit path to main(ckpt=...)."
        )
    print(f"Using checkpoint: {ckpt_path}")

    model = load_ema_model(cfg, device, ckpt_path) # Class: BiDimentionalAttentionModel
    process = build_process(cfg, device) # class GaussianDiffusion

    t0 = time.time()
    #plot_exact_like_training(model, process, cfg, device, "samples/like_training.png")
    print(f"[timing] plot_exact_like_training took {time.time() - t0:.2f} sec")

    # Unconditional
    if mode == "uncond":
        if cfg.input_dim != 1:
            raise ValueError("Unconditional plotting assumes input_dim == 1.")
        print(f"[timing] → start sample_uncond/{sampler} "
              f"({('T=' + str(process.betas.numel())) if sampler=='ddpm' else ('S=' + str(num_steps))}, "
              f"n_funcs={n_funcs})")
        t0 = time.perf_counter()
        x, ys = sample_uncond(cfg, model, process, device,
                              n_points=n_points, n_funcs=n_funcs, seed=seed,
                              sampler=sampler, num_steps=num_steps)
        print(f"[timing] ← end   sample_uncond/{sampler}: {time.perf_counter()-t0:.2f}s")
        plot_uncond(x, ys, out_path, title="Unconditional samples")
        print(f"✓ saved: {out_path}")
        return


    # Conditional
    if cfg.input_dim != 1:
        raise ValueError("Conditional plotting assumes input_dim == 1.")

    # Build a random context from the generator (batch_size=1)
    batch: Batch = get_batch(
        torch.Generator().manual_seed(cfg.seed+1),
        batch_size=1,
        name=cfg.dataset,
        task="interpolation",
        input_dim=cfg.input_dim,
        device=device,
    )
    x_ctx, y_ctx = batch.x_context[0], batch.y_context[0]  # [M,1], [M,1]
    x_query = torch.linspace(-2, 2, n_points, device=device).unsqueeze(-1)  # [N,1]

    print(f"[timing] → start sample_cond/{sampler} "
          f"({('T=' + str(process.betas.numel())) if sampler=='ddpm' else ('S=' + str(num_steps))}, K={K})")
    t0 = time.perf_counter()
    xs, ys = sample_cond(cfg, model, process, device,
                         x_context=x_ctx, y_context=y_ctx, x_query=x_query, seed=seed,
                         sampler=sampler, num_steps=num_steps, K=K)
    print(f"[timing] ← end   sample_cond/{sampler}: {time.perf_counter()-t0:.2f}s")
    plot_conditional(xs, ys, x_ctx.squeeze(-1), y_ctx.squeeze(-1), out_path, title="Conditional sample")
    print(f"✓ saved: {out_path}")


    # ==== GP-likeness under the *fixed* GP (matches data.py) ====
    t1 = time.perf_counter()
    stats, per_ll, per_mahal = evaluate_gp_likeness_from_sorted_fixed(
        dataset_name=cfg.dataset,
        input_dim=cfg.input_dim,
        x_ctx=x_ctx, y_ctx=y_ctx,
        x_query=x_query,
        xs_sorted=xs, ys_sorted=ys,
        include_obs_noise=True,     # score y (noisy observations); set False to score latent f
    )
    print("[gp-like/fixed] GP hypers:",
          f"ell_eff={stats['ell_eff']:.4g}, var={stats['var']:.4g}, noise={stats['noise']:.4g}")
    print("[gp-like/fixed] LL (mean±std over K): "
          f"{stats['mean_ll']:.3f} ± {stats['std_ll']:.3f}")
    print("[gp-like/fixed] Mahalanobis^2 (mean±std): "
          f"{stats['mean_mahal']:.3f} ± {stats['std_mahal']:.3f}")
    print(f"[gp-like/fixed] computed in {time.perf_counter() - t1:.2f}s")

    # Save numbers next to the plot
    out_json = out_path.with_suffix(".gp_eval.fixed.json")
    payload = {
        "sampler": sampler, "num_steps": num_steps, "seed": seed,
        "K": stats["K"], "N": stats["N"], "include_obs_noise": stats["include_obs_noise"],
        "gp_hypers": {"ell_eff": stats["ell_eff"], "var": stats["var"], "noise": stats["noise"]},
        "likelihood": {"mean": stats["mean_ll"], "std": stats["std_ll"], "per_sample": per_ll},
        "mahalanobis2": {"mean": stats["mean_mahal"], "std": stats["std_mahal"], "per_sample": per_mahal},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✓ saved GP eval (fixed): {out_json}")


if __name__ == "__main__":
    ckpt_file = Path("logs/regression/Sep13_142401_zoqr_t_outblocks/model_ema.pt")
    sampler = "ddpm"
    num_steps = 25
    main(
        ckpt=ckpt_file,
        mode="cond",          # or "uncond"
        n_points=50,
        n_funcs=8,
        seed=10,
        sampler=sampler,       # "ddpm" | "ddim" | "euler" | "heun"
        num_steps=num_steps,         # (used by ddim/euler/heun)
        K=14,
        out_path=Path(f"logs/regression/Sep13_142401_zoqr_t_outblocks/out_{sampler}_steps{num_steps}.png"),
    )


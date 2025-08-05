# generate_data_torch.py
import os, sys
sys.path.append('.')                       # so "import data" resolves

import torch
import numpy as np
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.nn.functional as F

from data import get_batch, DATASETS, TASKS, _DATASET_CONFIGS

# ------------------------ user-tunable flags ------------------------ #
DRYRUN       = False
PLOT         = False
BATCH_SIZE   = 4
DATASET_SIZE = {"training": 2**14, "interpolation": 128}
SEED         = 0
# -------------------------------------------------------------------- #

g = torch.Generator().manual_seed(SEED)     # reproducible PRNG

# ------------ helper ------------------------------------------------- #
# Pads a (batch, N, …) or (batch, N) tensor on dim=1 up to max_len ---- #
# Pads along dim=1 up to max_len (appends zeros) ---------------------- #
def _pad_ctx(t: torch.Tensor, max_len: int) -> torch.Tensor:
    cur = t.size(1)
    if cur == max_len:
        return t
    pad_shape = list(t.shape)
    pad_shape[1] = max_len - cur           # rows to add
    pad = torch.zeros(*pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=1)


# --- plotting setup (optional) -------------------------------------- #
if PLOT:
    fig, axes = plt.subplots(8, 3, figsize=(8, 20))
    axes      = axes.flatten()
    ax_idx    = 0

# -------------------------------------------------------------------- #
for dataset, task in product(DATASETS, TASKS):
    for input_dim in range(1, _DATASET_CONFIGS[dataset].max_input_dim + 1):
        print(dataset, task, input_dim)

        batches = []
        n_iter  = DATASET_SIZE[task] // BATCH_SIZE
        for _ in tqdm(range(n_iter)):
            batch = get_batch(
                g,
                batch_size=BATCH_SIZE,
                name=dataset,
                task=task,
                input_dim=input_dim,
            )
            batches.append(batch)

        # ---------------- cat along batch dimension ------------------ #
        # 1️⃣ work out the longest context length in this macro-batch
        max_ctx = max(b.x_context.size(1) for b in batches)
        x_context = torch.cat([_pad_ctx(b.x_context, max_ctx) for b in batches], dim=0)
        y_context = torch.cat([_pad_ctx(b.y_context, max_ctx) for b in batches], dim=0)
        x_target    = torch.cat([b.x_target    for b in batches], dim=0)
        y_target    = torch.cat([b.y_target    for b in batches], dim=0)
        mask_target = torch.cat([b.mask_target for b in batches], dim=0)
        mask_context = torch.cat([_pad_ctx(b.mask_context, max_ctx) for b in batches], dim=0)

        print(f"{dataset} {input_dim} {task}")
        print(x_context.shape, y_context.shape,
              x_target.shape,  y_target.shape,
              mask_target.shape, mask_context.shape)

        # ---------------- save to .npz (unless DRYRUN) --------------- #
        if not DRYRUN:
            os.makedirs("data", exist_ok=True)
            np.savez(
                f"data/{dataset}_{input_dim}_{task}.npz",
                x_context   = x_context.cpu().numpy(),
                y_context   = y_context.cpu().numpy(),
                x_target    = x_target.cpu().numpy(),
                y_target    = y_target.cpu().numpy(),
                mask_target = mask_target.cpu().numpy(),
                mask_context= mask_context.cpu().numpy(),
            )

        # ---------------- optional histogram plot -------------------- #
        if PLOT:
            num_context = mask_context.shape[1] - mask_context.bool().sum(dim=1)
            num_target  = mask_target.shape[1] - mask_target.bool().sum(dim=1)
            axes[ax_idx].hist(num_context.cpu(), bins=20, label="context")
            axes[ax_idx].hist(num_target.cpu(),  bins=20, label="target")
            axes[ax_idx].set_title(f"{dataset} {input_dim} {task}", fontsize=8)
            if ax_idx == 0:
                axes[ax_idx].legend()
            ax_idx += 1

# ---------------- save figure --------------------------------------- #
if PLOT:
    plt.tight_layout()
    plt.savefig("num_data.png")
    plt.close()

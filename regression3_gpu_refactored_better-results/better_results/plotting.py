from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from torch import Tensor

from .gp import PrefixTask


PANEL_ORDER = ("gp", "ndp_cond", "ndp_uncond", "flownp")
PANEL_TITLES = {
    "gp": "Analytic GP posterior",
    "ndp_cond": "Conditional NDP (DDPM)",
    "ndp_uncond": "Unconditional NDP (RePaint)",
    "flownp": "FlowNP (Euler)",
}


def context_marker_area(context_size: int) -> float:
    return max(3.0, min(8.0, 12.0 / math.sqrt(float(context_size))))


def plot_comparison(
    task: PrefixTask,
    samples: dict[str, Tensor],
    path: str | Path,
    *,
    task_label: str = "Nested unordered context",
) -> Path:
    if tuple(samples) != PANEL_ORDER:
        raise ValueError(f"Samples must follow panel order {PANEL_ORDER}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_target = task.x_target[:, 0].detach().cpu().numpy()
    x_context = task.conditioning.x_context[:, 0].detach().cpu().numpy()
    y_context = task.conditioning.y_context[:, 0].detach().cpu().numpy()
    all_values = [task.conditioning.y_context.detach().cpu().flatten()]
    all_values.extend(value.detach().cpu().flatten() for value in samples.values())
    combined = torch.cat(all_values)
    y_min = float(combined.min())
    y_max = float(combined.max())
    padding = max(0.15, 0.05 * max(y_max - y_min, 1e-6))

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 7.0),
        dpi=160,
        sharex=True,
        sharey=True,
    )
    marker_area = context_marker_area(task.context_size)
    for axis, name in zip(axes.flat, PANEL_ORDER):
        values = samples[name].detach().cpu().squeeze(-1).numpy()
        axis.plot(x_target, values.T, linewidth=0.6, alpha=0.13, color="C0")
        axis.scatter(
            x_context,
            y_context,
            s=marker_area,
            color="black",
            alpha=0.9,
            linewidths=0,
            zorder=5,
        )
        axis.set_title(PANEL_TITLES[name])
        axis.set_xlim(float(task.x_target.min()), float(task.x_target.max()))
        axis.set_ylim(y_min - padding, y_max + padding)
        axis.grid(alpha=0.15, linewidth=0.5)
    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[1, 0].set_ylabel("y")
    legend_marker = Line2D(
        [], [], marker="o", linestyle="None", color="black", markersize=4.5, label="context"
    )
    axes[0, 0].legend(handles=[legend_marker], loc="best", frameon=True)
    figure.suptitle(
        f"{task_label}: M={task.context_size}, N={task.x_target.shape[0]}, "
        f"K={next(iter(samples.values())).shape[0]}"
    )
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path

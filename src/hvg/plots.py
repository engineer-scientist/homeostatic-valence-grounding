"""Publication-style plots of official (or smoke) metrics — never invented data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK = "#1a1f24"
MUTED = "#5a6572"
ACCENT = "#b3541e"
BLUE = "#2c5f8a"
LINE = "#d9dee3"


def _style():
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.edgecolor": LINE,
            "axes.labelcolor": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_auc_curves(checkpoints: list, values: dict[str, list], path: str | Path) -> Path:
    """values maps a series name to AUC values aligned with checkpoints."""
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = [ACCENT, BLUE, MUTED, "#2d6a4f"]
    for (name, ys), c in zip(values.items(), colors):
        ax.plot(checkpoints, ys, "-o", label=name, color=c, lw=2)
    ax.axhline(0.5, color=LINE, lw=1)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Held-out AUC")
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_delta_p_bars(series: dict[str, tuple[float, float]], path: str | Path, threshold: float = 0.15) -> Path:
    """series: name -> (mean ΔP, error)."""
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    names = list(series)
    means = [series[n][0] for n in names]
    errs = [series[n][1] for n in names]
    xs = np.arange(len(names))
    ax.bar(xs, means, yerr=errs, color=[ACCENT, "#d9b5a4", BLUE, "#a9c0d1"][: len(names)], capsize=4)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(threshold, color=ACCENT, ls=":", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("ΔP")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path

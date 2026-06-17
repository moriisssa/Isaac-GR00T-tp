# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


REFERENCE_FIGSIZE = (9.5, 5.2)
REFERENCE_DPI = 160
REFERENCE_GRID_COLOR = "#b0b0b0"
REFERENCE_GRID_ALPHA = 0.25


def apply_reference_plot_style() -> None:
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": REFERENCE_GRID_COLOR,
            "grid.alpha": REFERENCE_GRID_ALPHA,
            "grid.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def style_reference_axes(ax) -> None:
    ax.grid(True, color=REFERENCE_GRID_COLOR, alpha=REFERENCE_GRID_ALPHA, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_reference_figure(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=REFERENCE_DPI)

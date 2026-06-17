# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from gr00t.eval.plot_style import (
    REFERENCE_FIGSIZE,
    apply_reference_plot_style,
    save_reference_figure,
    style_reference_axes,
)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_progress_curve_rows(
    csv_path: str | Path, *, success_only: bool = False
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid = _parse_bool(row["valid"])
            success = _parse_bool(row["success"])
            if not valid or (success_only and not success):
                continue
            rows.append(
                {
                    "episode": int(row["episode"]),
                    "step": int(row["step"]),
                    "target_progress": float(row["target_progress"]),
                    "progress_pred": float(row["progress_pred"]),
                }
            )
    return rows


def write_progress_curve_plot(
    rows: list[dict[str, Any]],
    *,
    png_path: str | Path,
    target: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    apply_reference_plot_style()
    fig, ax = plt.subplots(figsize=REFERENCE_FIGSIZE)
    for episode in sorted({row["episode"] for row in rows}):
        episode_rows = sorted(
            [row for row in rows if row["episode"] == episode],
            key=lambda row: row["step"],
        )
        if not episode_rows:
            continue
        xs = [row["target_progress"] for row in episode_rows]
        ys = [row["progress_pred"] for row in episode_rows]
        ax.plot(xs, ys, alpha=0.18, linewidth=1.0)

    if rows:
        bins = np.linspace(0.0, 1.0, 21)
        target_values = np.asarray([row["target_progress"] for row in rows], dtype=np.float32)
        pred = np.asarray([row["progress_pred"] for row in rows], dtype=np.float32)
        bin_indices = np.clip(np.digitize(target_values, bins) - 1, 0, len(bins) - 2)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        mean_pred = [
            float(np.mean(pred[bin_indices == idx])) if np.any(bin_indices == idx) else np.nan
            for idx in range(len(bin_centers))
        ]
        ax.plot(
            bin_centers,
            mean_pred,
            marker="o",
            linewidth=2.0,
            label="binned prediction mean",
        )

    ax.plot([0.0, 1.0], [0.0, 1.0], "--", linewidth=1.5, label="ideal")
    ax.set_xlim(0.0, 1.0)
    if rows:
        pred_values = np.asarray([row["progress_pred"] for row in rows], dtype=np.float32)
        y_min = float(min(0.0, np.nanmin(pred_values)))
        y_max = float(max(1.0, np.nanmax(pred_values)))
        margin = max(0.05, 0.05 * (y_max - y_min))
        ax.set_ylim(y_min - margin, y_max + margin)
    else:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("normalized rollout progress")
    ax.set_ylabel("predicted progress")
    ax.set_title(f"Progress prediction curve ({target})")
    ax.legend(loc="best", frameon=True)
    style_reference_axes(ax)
    save_reference_figure(fig, png_path)
    plt.close(fig)

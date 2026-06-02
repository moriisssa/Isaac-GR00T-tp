#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import csv
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Literal

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import compute_progress_label, extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.open_loop_eval import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch
import tyro


ProgressTarget = Literal["current", "chunk_end"]


@dataclass
class OfflineProgressEvalConfig:
    model_path: str
    """Path to the trained checkpoint directory."""

    dataset_path: str = "examples/SimplerEnv/fractal20220817_data_lerobot/"
    """Path to the LeRobot-format dataset."""

    embodiment_tag: str = "SIMPLER_ENV_GOOGLE"
    """Embodiment tag used by the checkpoint."""

    output_dir: str | None = None
    """Directory for prediction CSVs, metrics JSON, and progress curve plots."""

    traj_ids: list[int] = field(default_factory=lambda: [0, 1, 2])
    """Episode indices to evaluate."""

    samples_per_traj: int = 8
    """Number of uniformly spaced timesteps sampled from each episode."""

    progress_target: ProgressTarget = "chunk_end"
    """Target definition used during progress-head training."""

    tail_shrink_action_chunk: bool = False
    """If True, evaluate tail observations with shortened action chunks."""

    progress_num_bins: int = 10
    """Number of bins used for progress classification metrics."""

    device: str | None = None
    """Torch device. Defaults to cuda if available, else cpu."""

    video_backend: str = "torchcodec"
    """Video backend for LeRobotEpisodeLoader."""


def _build_policy_observation(
    data_point: Any,
    modality_configs: dict[str, Any],
) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    for key, value in data_point.states.items():
        obs[f"state.{key}"] = value
    for key, value in data_point.images.items():
        obs[f"video.{key}"] = np.asarray(value)
    for language_key in modality_configs["language"].modality_keys:
        obs[language_key] = data_point.text
    return parse_observation_gr00t(obs, modality_configs)


def _sample_step_indices(
    episode_length: int,
    action_horizon: int,
    samples_per_traj: int,
    tail_shrink_action_chunk: bool = False,
) -> list[int]:
    effective_length = (
        max(1, episode_length)
        if tail_shrink_action_chunk
        else max(1, episode_length - action_horizon + 1)
    )
    if samples_per_traj >= effective_length:
        return list(range(effective_length))
    return np.linspace(0, effective_length - 1, samples_per_traj, dtype=int).tolist()


def _summarize(
    rows: list[dict[str, Any]], *, target_key: str = "target_progress"
) -> dict[str, Any]:
    pred = np.asarray([row["progress_pred"] for row in rows], dtype=np.float32)
    target = np.asarray([row[target_key] for row in rows], dtype=np.float32)
    error = pred - target
    finite = np.isfinite(pred) & np.isfinite(target)
    if not finite.any():
        return {"count": len(rows), "valid_count": 0}

    pred = pred[finite]
    target = target[finite]
    error = error[finite]
    summary = {
        "count": len(rows),
        "valid_count": int(finite.sum()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "pred_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(target)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "corr": float(np.corrcoef(target, pred)[0, 1]) if len(pred) > 1 else float("nan"),
    }
    target_class_key = {
        "target_progress": "target_class",
        "current_progress": "current_class",
        "chunk_end_progress": "chunk_end_class",
    }.get(target_key)
    if target_class_key and rows and "progress_class_pred" in rows[0]:
        pred_class = np.asarray([row["progress_class_pred"] for row in rows], dtype=np.int64)
        target_class = np.asarray([row[target_class_key] for row in rows], dtype=np.int64)
        finite_class = np.isfinite(pred_class) & np.isfinite(target_class)
        if finite_class.any():
            summary.update(
                {
                    "class_count": int(finite_class.sum()),
                    "accuracy": float(
                        np.mean(pred_class[finite_class] == target_class[finite_class])
                    ),
                    "pred_class_mean": float(np.mean(pred_class[finite_class])),
                    "target_class_mean": float(np.mean(target_class[finite_class])),
                }
            )
    return summary


def _summarize_pairwise(
    rows: list[dict[str, Any]],
    *,
    target_key: str = "target_progress",
    score_key: str = "progress_score",
    gap_min: float = 0.05,
) -> dict[str, Any]:
    total = 0
    correct = 0
    gaps = []
    for traj_id in sorted({row["traj_id"] for row in rows}):
        traj_rows = sorted(
            [row for row in rows if row["traj_id"] == traj_id],
            key=lambda row: row["step_index"],
        )
        for i, row_a in enumerate(traj_rows):
            for row_b in traj_rows[i + 1 :]:
                target_diff = float(row_b[target_key]) - float(row_a[target_key])
                if abs(target_diff) < gap_min:
                    continue
                score_diff = float(row_b.get(score_key, row_b["progress_pred"])) - float(
                    row_a.get(score_key, row_a["progress_pred"])
                )
                if not np.isfinite(score_diff):
                    continue
                total += 1
                gaps.append(abs(target_diff))
                correct += int((score_diff > 0.0) == (target_diff > 0.0))
    return {
        "count": total,
        "accuracy": float(correct / total) if total else float("nan"),
        "gap_min": gap_min,
        "gap_mean": float(np.mean(gaps)) if gaps else float("nan"),
    }


def _progress_to_class(progress: float, num_bins: int) -> int:
    return int(np.clip(np.floor(progress * num_bins), 0, num_bins - 1))


def _compute_binned_curve(
    rows: list[dict[str, Any]],
    *,
    target_key: str = "target_progress",
    num_bins: int = 20,
) -> list[dict[str, Any]]:
    target = np.asarray([row[target_key] for row in rows], dtype=np.float32)
    pred = np.asarray([row["progress_pred"] for row in rows], dtype=np.float32)
    finite = np.isfinite(target) & np.isfinite(pred)
    target = target[finite]
    pred = pred[finite]

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.clip(np.digitize(target, bins) - 1, 0, num_bins - 1)
    binned_rows = []
    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        center = float((bins[bin_idx] + bins[bin_idx + 1]) / 2.0)
        if not np.any(mask):
            binned_rows.append(
                {
                    "bin_index": bin_idx,
                    "target_start": float(bins[bin_idx]),
                    "target_end": float(bins[bin_idx + 1]),
                    "target_center": center,
                    "count": 0,
                    "pred_mean": None,
                    "pred_std": None,
                    "target_mean": None,
                    "mae": None,
                    "bias": None,
                }
            )
            continue
        pred_bin = pred[mask]
        target_bin = target[mask]
        error = pred_bin - target_bin
        binned_rows.append(
            {
                "bin_index": bin_idx,
                "target_start": float(bins[bin_idx]),
                "target_end": float(bins[bin_idx + 1]),
                "target_center": center,
                "count": int(mask.sum()),
                "pred_mean": float(np.mean(pred_bin)),
                "pred_std": float(np.std(pred_bin)),
                "target_mean": float(np.mean(target_bin)),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
            }
        )
    return binned_rows


def _write_binned_curve_csv(binned_curve: list[dict[str, Any]], path: Path) -> None:
    binned_fieldnames = [
        "bin_index",
        "target_start",
        "target_end",
        "target_center",
        "count",
        "target_mean",
        "pred_mean",
        "pred_std",
        "mae",
        "bias",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=binned_fieldnames)
        writer.writeheader()
        writer.writerows(binned_curve)


def _plot_progress_curve(
    rows: list[dict[str, Any]],
    binned_curve: list[dict[str, Any]],
    *,
    target_key: str,
    target_label: str,
    png_path: Path,
    svg_path: Path,
) -> str | None:
    try:
        from matplotlib import pyplot as plt

        traj_ids = sorted({row["traj_id"] for row in rows})
        fig, ax = plt.subplots(figsize=(7, 5))
        for traj_id in traj_ids:
            traj_rows = sorted(
                [row for row in rows if row["traj_id"] == traj_id],
                key=lambda row: row[target_key],
            )
            ax.plot(
                [row[target_key] for row in traj_rows],
                [row["progress_pred"] for row in traj_rows],
                marker="o" if len(traj_ids) <= 10 else None,
                linewidth=1.0,
                alpha=0.28,
                label=f"traj {traj_id}" if len(traj_ids) <= 10 else None,
            )
        curve_rows = [row for row in binned_curve if row["count"] > 0]
        curve_x = [row["target_center"] for row in curve_rows]
        curve_y = [row["pred_mean"] for row in curve_rows]
        curve_std = [row["pred_std"] for row in curve_rows]
        if curve_rows:
            ax.plot(
                curve_x,
                curve_y,
                marker="o",
                linewidth=2.5,
                color="tab:red",
                label="binned prediction mean",
            )
            ax.fill_between(
                curve_x,
                np.asarray(curve_y) - np.asarray(curve_std),
                np.asarray(curve_y) + np.asarray(curve_std),
                color="tab:red",
                alpha=0.14,
                linewidth=0,
                label="prediction std",
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="ideal")
        ax.set_xlabel(target_label)
        ax.set_ylabel("predicted progress")
        ax.set_title(f"Offline progress prediction ({target_label})")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(png_path)
        fig.savefig(svg_path)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is best effort.
        return repr(exc)
    return None


def _write_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    progress_target: ProgressTarget,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "progress_predictions.csv"
    binned_csv_path = output_dir / "progress_binned_curve.csv"
    binned_current_csv_path = output_dir / "progress_binned_curve_current.csv"
    binned_chunk_end_csv_path = output_dir / "progress_binned_curve_chunk_end.csv"
    metrics_path = output_dir / "progress_metrics.json"
    png_path = output_dir / "progress_curve.png"
    svg_path = output_dir / "progress_curve.svg"
    current_png_path = output_dir / "progress_curve_current.png"
    current_svg_path = output_dir / "progress_curve_current.svg"
    chunk_end_png_path = output_dir / "progress_curve_chunk_end.png"
    chunk_end_svg_path = output_dir / "progress_curve_chunk_end.svg"

    fieldnames = [
        "traj_id",
        "step_index",
        "episode_length",
        "target_progress",
        "current_progress",
        "chunk_end_progress",
        "progress_score",
        "progress_pred",
        "progress_class_pred",
        "target_class",
        "current_class",
        "chunk_end_class",
        "class_correct",
        "abs_error",
        "abs_error_current",
        "abs_error_chunk_end",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    binned_curve = _compute_binned_curve(rows)
    binned_current_curve = _compute_binned_curve(rows, target_key="current_progress")
    binned_chunk_end_curve = _compute_binned_curve(rows, target_key="chunk_end_progress")
    _write_binned_curve_csv(binned_curve, binned_csv_path)
    _write_binned_curve_csv(binned_current_curve, binned_current_csv_path)
    _write_binned_curve_csv(binned_chunk_end_curve, binned_chunk_end_csv_path)

    metrics = {
        "progress_target": progress_target,
        "overall": _summarize(rows),
        "overall_current_axis": _summarize(rows, target_key="current_progress"),
        "overall_chunk_end_axis": _summarize(rows, target_key="chunk_end_progress"),
        "pairwise": _summarize_pairwise(rows),
        "pairwise_current_axis": _summarize_pairwise(rows, target_key="current_progress"),
        "pairwise_chunk_end_axis": _summarize_pairwise(rows, target_key="chunk_end_progress"),
        "binned_curve": binned_curve,
        "binned_curve_current": binned_current_curve,
        "binned_curve_chunk_end": binned_chunk_end_curve,
        "by_traj": {
            str(traj_id): _summarize([row for row in rows if row["traj_id"] == traj_id])
            for traj_id in sorted({row["traj_id"] for row in rows})
        },
        "by_traj_current_axis": {
            str(traj_id): _summarize(
                [row for row in rows if row["traj_id"] == traj_id],
                target_key="current_progress",
            )
            for traj_id in sorted({row["traj_id"] for row in rows})
        },
        "by_traj_chunk_end_axis": {
            str(traj_id): _summarize(
                [row for row in rows if row["traj_id"] == traj_id],
                target_key="chunk_end_progress",
            )
            for traj_id in sorted({row["traj_id"] for row in rows})
        },
        "files": {
            "csv": str(csv_path),
            "binned_csv": str(binned_csv_path),
            "binned_current_csv": str(binned_current_csv_path),
            "binned_chunk_end_csv": str(binned_chunk_end_csv_path),
            "png": str(png_path),
            "svg": str(svg_path),
            "current_png": str(current_png_path),
            "current_svg": str(current_svg_path),
            "chunk_end_png": str(chunk_end_png_path),
            "chunk_end_svg": str(chunk_end_svg_path),
        },
    }

    plot_errors = {}
    selected_error = _plot_progress_curve(
        rows,
        binned_curve,
        target_key="target_progress",
        target_label=f"{progress_target} progress",
        png_path=png_path,
        svg_path=svg_path,
    )
    if selected_error is not None:
        plot_errors["selected"] = selected_error
    current_error = _plot_progress_curve(
        rows,
        binned_current_curve,
        target_key="current_progress",
        target_label="current progress",
        png_path=current_png_path,
        svg_path=current_svg_path,
    )
    if current_error is not None:
        plot_errors["current"] = current_error
    chunk_end_error = _plot_progress_curve(
        rows,
        binned_chunk_end_curve,
        target_key="chunk_end_progress",
        target_label="chunk_end progress",
        png_path=chunk_end_png_path,
        svg_path=chunk_end_svg_path,
    )
    if chunk_end_error is not None:
        plot_errors["chunk_end"] = chunk_end_error
    if plot_errors:
        metrics["plot_errors"] = plot_errors

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main(config: OfflineProgressEvalConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    model_path = Path(config.model_path)
    output_dir = (
        Path(config.output_dir) if config.output_dir else model_path / "offline_progress_eval"
    )
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)

    logging.info("Loading policy from %s on %s", model_path, device)
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=str(model_path),
        device=device,
    )
    modality_configs = policy.get_modality_config()
    action_horizon = len(modality_configs["action"].delta_indices)

    logging.info("Loading dataset from %s", config.dataset_path)
    loader = LeRobotEpisodeLoader(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        video_backend=config.video_backend,
        video_backend_kwargs=None,
    )

    rows: list[dict[str, Any]] = []
    for traj_id in config.traj_ids:
        if traj_id >= len(loader):
            logging.warning(
                "Skipping out-of-range traj_id=%s; dataset length=%s", traj_id, len(loader)
            )
            continue

        episode = loader[traj_id]
        step_indices = _sample_step_indices(
            len(episode),
            action_horizon,
            config.samples_per_traj,
            tail_shrink_action_chunk=config.tail_shrink_action_chunk,
        )
        logging.info("Evaluating traj_id=%s with steps=%s", traj_id, step_indices)

        for step_index in step_indices:
            data_point = extract_step_data(
                episode,
                step_index,
                deepcopy(modality_configs),
                embodiment_tag,
                allow_padding=False,
                progress_target=config.progress_target,
                tail_shrink_action_chunk=config.tail_shrink_action_chunk,
            )
            observation = _build_policy_observation(data_point, modality_configs)
            _action, info = policy.get_action(observation)
            if "progress" not in info:
                raise RuntimeError("Checkpoint did not return progress predictions")

            pred = float(np.asarray(info["progress"], dtype=np.float32).reshape(-1)[0])
            if "progress_score" in info:
                score = float(np.asarray(info["progress_score"], dtype=np.float32).reshape(-1)[0])
            else:
                clipped_pred = np.clip(pred, 1e-6, 1.0 - 1e-6)
                score = float(np.log(clipped_pred / (1.0 - clipped_pred)))
            if "progress_class" in info:
                pred_class = int(np.asarray(info["progress_class"], dtype=np.int64).reshape(-1)[0])
            else:
                pred_class = _progress_to_class(pred, config.progress_num_bins)
            current_progress = float(
                compute_progress_label(
                    episode,
                    step_index,
                    action_horizon=action_horizon,
                    target="current",
                )[0]
            )
            chunk_end_progress = float(
                compute_progress_label(
                    episode,
                    step_index,
                    action_horizon=action_horizon,
                    target="chunk_end",
                )[0]
            )
            target = (
                chunk_end_progress if config.progress_target == "chunk_end" else current_progress
            )
            target_class = _progress_to_class(target, config.progress_num_bins)
            current_class = _progress_to_class(current_progress, config.progress_num_bins)
            chunk_end_class = _progress_to_class(chunk_end_progress, config.progress_num_bins)
            rows.append(
                {
                    "traj_id": traj_id,
                    "step_index": step_index,
                    "episode_length": len(episode),
                    "target_progress": target,
                    "current_progress": current_progress,
                    "chunk_end_progress": chunk_end_progress,
                    "progress_score": score,
                    "progress_pred": pred,
                    "progress_class_pred": pred_class,
                    "target_class": target_class,
                    "current_class": current_class,
                    "chunk_end_class": chunk_end_class,
                    "class_correct": pred_class == target_class,
                    "abs_error": abs(pred - target),
                    "abs_error_current": abs(pred - current_progress),
                    "abs_error_chunk_end": abs(pred - chunk_end_progress),
                }
            )

    if not rows:
        raise RuntimeError("No rows were evaluated")

    metrics = _write_outputs(rows, output_dir, progress_target=config.progress_target)
    print(json.dumps(metrics["overall"], indent=2))
    print(f"Saved progress eval outputs to: {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(OfflineProgressEvalConfig))

#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.plot_style import (
    REFERENCE_FIGSIZE,
    apply_reference_plot_style,
    save_reference_figure,
    style_reference_axes,
)
from gr00t.eval.rollout_policy import create_gr00t_sim_policy


STATE_KEYS = ("x", "y", "z", "rx", "ry", "rz", "rw", "gripper")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a GR00T progress head on same-scene rollout videos."
    )
    parser.add_argument("--episodes-csv", required=True)
    parser.add_argument(
        "--model-path",
        default=(
            "output/progress_vlm_layerwise_pairwise/"
            "layerwise_1k_current_pairwise_bt_20260602_200859/"
            "layer_00/fractal_progress_vlm_layer_00_pairwise_bt_1k_current_20260602_200859"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--language", default="place the object in the closed drawer")
    parser.add_argument("--samples-per-video", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pair-gap-min", type=float, default=0.05)
    parser.add_argument(
        "--plot-title",
        default=None,
        help="Title for the score curve plot. Defaults to a title inferred from the model/output path.",
    )
    return parser.parse_args()


def _read_episodes(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row["success"] = str(row["success"]).lower() in {"true", "1", "yes"}
            row["scene_id"] = int(row["scene_id"])
            row["repeat_id"] = int(row["repeat_id"])
            row["policy_seed"] = int(row["policy_seed"])
            row["policy_steps"] = int(row["policy_steps"])
            row["primitive_steps"] = int(row["primitive_steps"])
            rows.append(row)
    return rows


def _sample_video_frames(video_path: Path, samples_per_video: int) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    indices = np.linspace(0, frame_count - 1, min(samples_per_video, frame_count), dtype=np.int64)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"Failed to sample frames from {video_path}")
    progress = indices[: len(frames)].astype(np.float32) / max(frame_count - 1, 1)
    return np.stack(frames, axis=0).astype(np.uint8), progress


def _make_obs(frames: np.ndarray, language: str) -> dict[str, Any]:
    batch = frames.shape[0]
    obs: dict[str, Any] = {
        "video.image": frames[:, None, ...],
        "annotation.human.action.task_description": [language] * batch,
    }
    for key in STATE_KEYS:
        obs[f"state.{key}"] = np.zeros((batch, 1, 1), dtype=np.float32)
    return obs


def _predict_scores(policy, frames: np.ndarray, language: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    preds = []
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        _, info = policy.get_action(_make_obs(batch_frames, language))
        if "progress_score" in info:
            batch_scores = np.asarray(info["progress_score"], dtype=np.float32).reshape(-1)
        elif "progress" in info:
            batch_scores = np.asarray(info["progress"], dtype=np.float32).reshape(-1)
        else:
            raise RuntimeError("Model did not return progress_score/progress")
        batch_preds = np.asarray(info.get("progress", batch_scores), dtype=np.float32).reshape(-1)
        scores.extend(batch_scores.tolist())
        preds.extend(batch_preds.tolist())
    return np.asarray(scores, dtype=np.float32), np.asarray(preds, dtype=np.float32)


def _pair_accuracy(progress: np.ndarray, scores: np.ndarray, pair_gap_min: float) -> float | None:
    correct = 0.0
    total = 0
    for i in range(len(progress)):
        for j in range(i + 1, len(progress)):
            if progress[j] - progress[i] < pair_gap_min:
                continue
            diff = float(scores[j] - scores[i])
            if diff > 0:
                correct += 1.0
            elif diff == 0:
                correct += 0.5
            total += 1
    return float(correct / total) if total else None


def _corr(progress: np.ndarray, scores: np.ndarray) -> float | None:
    if len(progress) < 2 or np.std(scores) == 0 or np.std(progress) == 0:
        return None
    return float(np.corrcoef(progress, scores)[0, 1])


def _write_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "success",
        "frame_progress",
        "score",
        "pred",
        "video_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _infer_plot_label(model_path: str, output_dir: Path) -> str:
    for text in (output_dir.name, model_path):
        lower = text.lower()
        if "cosmos" in lower and "layer0" in lower:
            return "Cosmos-pretrained-L00"
        if "cosmos_pretrained_layer_00" in lower:
            return "Cosmos-pretrained-L00"

        import re

        match = re.search(r"(?:gr00t[_-]fractal[_-])?layer[_-](\d{1,2})", lower)
        if match:
            return f"GR00T-Fractal-L{int(match.group(1)):02d}"
    return "GR00T progress head"


def _write_svg(
    path: Path,
    video_metrics: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    apply_reference_plot_style()
    fig, ax = plt.subplots(figsize=REFERENCE_FIGSIZE)
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in score_rows:
        by_video.setdefault(str(row["video_id"]), []).append(row)

    used_labels: set[str] = set()
    for metric in video_metrics:
        rows = sorted(
            by_video.get(str(metric["video_id"]), []),
            key=lambda row: float(row["frame_progress"]),
        )
        if len(rows) < 2:
            continue
        label = "success" if metric["success"] else "failure"
        color = "tab:blue" if metric["success"] else "tab:red"
        linestyle = "-" if metric["success"] else "--"
        ax.plot(
            [float(row["frame_progress"]) for row in rows],
            [float(row["score"]) for row in rows],
            color=color,
            linestyle=linestyle,
            alpha=0.70 if metric["success"] else 0.85,
            linewidth=1.8,
            label=label if label not in used_labels else None,
        )
        used_labels.add(label)

    ax.set_xlim(0.0, 1.0)
    if score_rows:
        scores = np.asarray([row["score"] for row in score_rows], dtype=np.float32)
        y_min = float(np.nanmin(scores))
        y_max = float(np.nanmax(scores))
        if y_max <= y_min:
            y_max = y_min + 1.0
        margin = max(0.1, 0.05 * (y_max - y_min))
        ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel("normalized video time")
    ax.set_ylabel("pairwise score/logit")
    ax.set_title(title)
    if used_labels:
        ax.legend(loc="best", frameon=True)
    style_reference_axes(ax)
    save_reference_figure(fig, path)
    plt.close(fig)


def _mean(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def main() -> None:
    args = _parse_args()
    episodes_csv = Path(args.episodes_csv)
    output_dir = Path(args.output_dir) if args.output_dir else episodes_csv.parent / "l00_video_pair_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_label = _infer_plot_label(args.model_path, output_dir)
    plot_title = args.plot_title or f"Same-scene {plot_label} progress score"

    policy = create_gr00t_sim_policy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag.SIMPLER_ENV_GOOGLE,
    )

    video_metrics: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for episode in _read_episodes(episodes_csv):
        video_path = Path(episode["video_path"])
        frames, progress = _sample_video_frames(video_path, args.samples_per_video)
        scores, preds = _predict_scores(policy, frames, args.language, args.batch_size)
        progress = progress[: len(scores)]
        video_id = video_path.stem
        pair_acc = _pair_accuracy(progress, scores, args.pair_gap_min)
        metric = {
            "video_id": video_id,
            "success": episode["success"],
            "scene_id": episode["scene_id"],
            "repeat_id": episode["repeat_id"],
            "policy_seed": episode["policy_seed"],
            "num_samples": int(len(scores)),
            "pair_accuracy": pair_acc,
            "score_corr": _corr(progress, scores),
            "score_start": float(scores[0]),
            "score_end": float(scores[-1]),
            "score_delta": float(scores[-1] - scores[0]),
            "pred_start": float(preds[0]),
            "pred_end": float(preds[-1]),
            "video_path": str(video_path),
        }
        video_metrics.append(metric)
        for x, score, pred in zip(progress, scores, preds):
            score_rows.append(
                {
                    "video_id": video_id,
                    "success": episode["success"],
                    "frame_progress": float(x),
                    "score": float(score),
                    "pred": float(pred),
                    "video_path": str(video_path),
                }
            )
        print(
            f"{video_id}: success={int(episode['success'])} "
            f"pair_acc={pair_acc} corr={metric['score_corr']}"
        )

    success_metrics = [m for m in video_metrics if m["success"]]
    failure_metrics = [m for m in video_metrics if not m["success"]]
    summary = {
        "episodes_csv": str(episodes_csv),
        "model_path": args.model_path,
        "plot_title": plot_title,
        "language": args.language,
        "samples_per_video": args.samples_per_video,
        "pair_gap_min": args.pair_gap_min,
        "num_videos": len(video_metrics),
        "num_success_videos": len(success_metrics),
        "num_failure_videos": len(failure_metrics),
        "success_mean_pair_accuracy": _mean([m["pair_accuracy"] for m in success_metrics]),
        "failure_mean_pair_accuracy": _mean([m["pair_accuracy"] for m in failure_metrics]),
        "success_mean_corr": _mean([m["score_corr"] for m in success_metrics]),
        "failure_mean_corr": _mean([m["score_corr"] for m in failure_metrics]),
        "success_mean_score_delta": _mean([m["score_delta"] for m in success_metrics]),
        "failure_mean_score_delta": _mean([m["score_delta"] for m in failure_metrics]),
        "videos": video_metrics,
    }

    metrics_path = output_dir / "l00_video_pair_metrics.json"
    scores_csv = output_dir / "l00_video_scores.csv"
    curve_svg = output_dir / "l00_same_scene_score_curves.svg"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _write_scores_csv(scores_csv, score_rows)
    _write_svg(curve_svg, video_metrics, score_rows, plot_title)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved scores: {scores_csv}")
    print(f"Saved plot: {curve_svg}")


if __name__ == "__main__":
    main()

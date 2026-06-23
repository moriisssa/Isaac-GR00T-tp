#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.rollout_policy import create_gr00t_sim_policy


STATE_KEYS = ("x", "y", "z", "rx", "ry", "rz", "rw", "gripper")
TASK_LANGUAGE = {
    "google_robot_pick_object": "pick the object",
    "google_robot_open_drawer": "open the drawer",
    "google_robot_place_in_closed_drawer": "place the object in the closed drawer",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize learned attention-pooling token weights for same-scene "
            "success/failure rollout videos."
        )
    )
    parser.add_argument("--model-path", action="append", required=True)
    parser.add_argument("--same-scene-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-video", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--embodiment-tag", default="SIMPLER_ENV_GOOGLE")
    return parser.parse_args()


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _resolve_model_path(path: str) -> Path:
    model_path = Path(path)
    if (model_path / "config.json").exists():
        return model_path
    candidates = sorted(p for p in model_path.iterdir() if (p / "config.json").exists())
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"Could not find config.json under model path: {model_path}")


def _infer_task_name(path: Path) -> str:
    text = str(path)
    for task in TASK_LANGUAGE:
        if task in text:
            return task
    if "pick_object" in text:
        return "google_robot_pick_object"
    if "open_drawer" in text:
        return "google_robot_open_drawer"
    if "place_in_closed_drawer" in text:
        return "google_robot_place_in_closed_drawer"
    return path.name


def _find_same_scene_pair(path: Path) -> tuple[Path, Path, int | None]:
    videos = sorted(path.rglob("*.mp4"))
    by_scene: dict[int | None, dict[str, list[Path]]] = {}
    for video in videos:
        lower = video.name.lower()
        if "success" in lower:
            role = "success"
        elif "failure" in lower:
            role = "failure"
        else:
            continue
        match = re.search(r"scene(\d+)", lower)
        scene_id = int(match.group(1)) if match else None
        by_scene.setdefault(scene_id, {"success": [], "failure": []})[role].append(video)

    for scene_id in sorted(by_scene, key=lambda value: (-1 if value is None else value)):
        group = by_scene[scene_id]
        if group["success"] and group["failure"]:
            return group["success"][0], group["failure"][0], scene_id
    raise FileNotFoundError(f"No same-scene success/failure mp4 pair found under {path}")


def _sample_video(video_path: Path, samples_per_video: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    indices = np.linspace(0, frame_count - 1, min(samples_per_video, frame_count), dtype=np.int64)
    frames: list[np.ndarray] = []
    used_indices: list[int] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        used_indices.append(int(idx))
    cap.release()
    if not frames:
        raise ValueError(f"Failed to sample frames from {video_path}")
    progress = np.asarray(used_indices, dtype=np.float32) / max(frame_count - 1, 1)
    return np.stack(frames, axis=0).astype(np.uint8), np.asarray(used_indices), progress


def _make_obs(frames: np.ndarray, language: str) -> dict[str, Any]:
    batch = frames.shape[0]
    obs: dict[str, Any] = {
        "video.image": frames[:, None, ...],
        "annotation.human.action.task_description": [language] * batch,
    }
    for key in STATE_KEYS:
        obs[f"state.{key}"] = np.zeros((batch, 1, 1), dtype=np.float32)
    return obs


def _get_tokenizer(policy: Any) -> Any | None:
    base_policy = getattr(policy, "policy", policy)
    processor = getattr(base_policy, "processor", None)
    nested_processor = getattr(processor, "processor", None)
    for candidate in (nested_processor, processor):
        tokenizer = getattr(candidate, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
    return None


def _decode_token(tokenizer: Any | None, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return str(token_id)


def _classify_token(
    token_id: int,
    *,
    valid: bool,
    image: bool,
    tokenizer: Any | None,
) -> tuple[str, str]:
    if not valid:
        return "pad", "<pad>"
    if image:
        return "image", "<image>"
    text = _decode_token(tokenizer, token_id)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if int(token_id) in special_ids or text.startswith("<|"):
        return "special", text
    return "text", text


def _entropy(weights: np.ndarray) -> float:
    positive = weights[weights > 0]
    if positive.size == 0:
        return 0.0
    return float(-(positive * np.log(positive + 1e-12)).sum())


def _grid_shape(num_tokens: int, image_grid_thw: np.ndarray | None) -> tuple[int, int]:
    if image_grid_thw is not None and image_grid_thw.size >= 3:
        grid = np.asarray(image_grid_thw).reshape(-1, 3)[0]
        h, w = int(grid[1]), int(grid[2])
        if h > 0 and w > 0:
            if h * w == num_tokens:
                return h, w
            if (h // 2) * (w // 2) == num_tokens:
                return max(h // 2, 1), max(w // 2, 1)
            if (h // 4) * (w // 4) == num_tokens:
                return max(h // 4, 1), max(w // 4, 1)
    h = int(math.sqrt(num_tokens))
    while h > 1 and num_tokens % h != 0:
        h -= 1
    return max(h, 1), max(num_tokens // max(h, 1), 1)


def _write_overlay(
    frame: np.ndarray,
    image_weights: np.ndarray,
    image_grid_thw: np.ndarray | None,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_weights.size == 0:
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return
    grid_h, grid_w = _grid_shape(int(image_weights.size), image_grid_thw)
    usable = grid_h * grid_w
    values = image_weights[:usable]
    if values.size < usable:
        values = np.pad(values, (0, usable - values.size))
    heat = values.reshape(grid_h, grid_w)
    if float(heat.max()) > float(heat.min()):
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    else:
        heat = np.zeros_like(heat)
    heat = cv2.resize(heat.astype(np.float32), (frame.shape[1], frame.shape[0]))
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.58 * frame.astype(np.float32) + 0.42 * colored.astype(np.float32), 0, 255)
    cv2.imwrite(str(path), cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))


def _write_token_bar(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    title: str,
    top_k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    selected = sorted(
        [row for row in rows if row["token_type"] != "pad"],
        key=lambda row: float(row["weight"]),
        reverse=True,
    )[:top_k]
    labels = [
        f'{row["token_index"]}:{row["token_type"]}:{str(row["token_text"]).strip()[:18]}'
        for row in selected
    ]
    values = [float(row["weight"]) for row in selected]
    fig_h = max(3.5, 0.28 * max(len(selected), 1))
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    y = np.arange(len(selected))
    ax.barh(y, values, color="#4c78a8")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("attention-pool token weight")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_curves(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.0), sharex=True)
    style = {
        "success": {"color": "#2f6db3", "linestyle": "-", "label": "success"},
        "failure": {"color": "#c43b3b", "linestyle": "--", "label": "failure"},
    }
    for role in ("success", "failure"):
        role_rows = sorted(
            [row for row in rows if row["role"] == role],
            key=lambda row: float(row["normalized_time"]),
        )
        if not role_rows:
            continue
        x = [float(row["normalized_time"]) for row in role_rows]
        axes[0].plot(
            x,
            [float(row["progress_pred"]) for row in role_rows],
            **style[role],
        )
        axes[1].plot(
            x,
            [float(row["image_weight_sum"]) for row in role_rows],
            **style[role],
        )
        axes[1].plot(
            x,
            [float(row["text_weight_sum"]) for row in role_rows],
            color=style[role]["color"],
            linestyle=":" if role == "success" else "-.",
            label=f"{role} text",
        )
        axes[2].plot(
            x,
            [float(row["attention_entropy"]) for row in role_rows],
            **style[role],
        )
    axes[0].set_ylabel("progress pred")
    axes[1].set_ylabel("weight mass")
    axes[2].set_ylabel("entropy")
    axes[2].set_xlabel("normalized video time")
    axes[0].set_ylim(-0.03, 1.03)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _predict_video(
    policy: Any,
    tokenizer: Any | None,
    video_path: Path,
    *,
    role: str,
    language: str,
    output_dir: Path,
    samples_per_video: int,
    batch_size: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames, frame_indices, progress = _sample_video(video_path, samples_per_video)
    frame_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        _, info = policy.get_action(
            _make_obs(batch_frames, language),
            options={"return_progress_token_weights": True},
        )
        weights_batch = np.asarray(info["progress_token_weights"], dtype=np.float32)
        scores_batch = np.asarray(info["progress_token_scores"], dtype=np.float32)
        valid_batch = np.asarray(info["progress_token_mask"]).astype(bool)
        image_batch = np.asarray(info["progress_token_image_mask"]).astype(bool)
        input_ids_batch = np.asarray(info["progress_token_input_ids"])
        image_grid_batch = info.get("progress_token_image_grid_thw")
        pred_batch = np.asarray(info.get("progress", np.zeros((len(batch_frames), 1))), dtype=np.float32)
        score_batch = np.asarray(info.get("progress_score", pred_batch), dtype=np.float32)

        for local_idx in range(len(batch_frames)):
            global_idx = start + local_idx
            weights = weights_batch[local_idx].reshape(-1)
            scores = scores_batch[local_idx].reshape(-1)
            valid = valid_batch[local_idx].reshape(-1)
            image_mask = image_batch[local_idx].reshape(-1)
            input_ids = input_ids_batch[local_idx].reshape(-1)
            token_types: list[str] = []
            per_frame_token_rows: list[dict[str, Any]] = []
            for token_index, (token_id, weight, score, is_valid, is_image) in enumerate(
                zip(input_ids, weights, scores, valid, image_mask, strict=False)
            ):
                token_type, token_text = _classify_token(
                    int(token_id),
                    valid=bool(is_valid),
                    image=bool(is_image),
                    tokenizer=tokenizer,
                )
                token_types.append(token_type)
                row = {
                    "role": role,
                    "video_path": str(video_path),
                    "frame_sample_index": global_idx,
                    "frame_index": int(frame_indices[global_idx]),
                    "normalized_time": float(progress[global_idx]),
                    "token_index": int(token_index),
                    "token_id": int(token_id),
                    "token_type": token_type,
                    "token_text": token_text,
                    "weight": float(weight),
                    "score": float(score),
                    "valid": bool(is_valid),
                }
                per_frame_token_rows.append(row)
                token_rows.append(row)

            type_arr = np.asarray(token_types)
            image_weights = weights[(type_arr == "image") & valid]
            text_weights = weights[(type_arr == "text") & valid]
            special_weights = weights[(type_arr == "special") & valid]
            top_weights = np.sort(weights[valid])[::-1] if np.any(valid) else np.asarray([])
            grid = None
            if image_grid_batch is not None:
                grid = np.asarray(image_grid_batch)
                if grid.ndim >= 2 and grid.shape[0] == len(batch_frames):
                    grid = grid[local_idx]
            frame_row = {
                "role": role,
                "video_path": str(video_path),
                "frame_sample_index": global_idx,
                "frame_index": int(frame_indices[global_idx]),
                "normalized_time": float(progress[global_idx]),
                "progress_pred": float(pred_batch.reshape(-1)[local_idx]),
                "progress_score": float(score_batch.reshape(-1)[local_idx]),
                "image_weight_sum": float(image_weights.sum()),
                "text_weight_sum": float(text_weights.sum()),
                "special_weight_sum": float(special_weights.sum()),
                "top1_weight": float(top_weights[0]) if top_weights.size else 0.0,
                "top5_weight_sum": float(top_weights[:5].sum()) if top_weights.size else 0.0,
                "attention_entropy": _entropy(weights[valid]),
                "num_valid_tokens": int(valid.sum()),
                "num_image_tokens": int(((type_arr == "image") & valid).sum()),
                "num_text_tokens": int(((type_arr == "text") & valid).sum()),
            }
            frame_rows.append(frame_row)

            stem = f"{role}_sample{global_idx:03d}_frame{int(frame_indices[global_idx]):05d}"
            _write_overlay(
                batch_frames[local_idx],
                image_weights,
                grid,
                output_dir / role / "overlays" / f"{stem}.png",
            )
            _write_token_bar(
                per_frame_token_rows,
                output_dir / role / "token_bars" / f"{stem}.svg",
                title=f"{role} t={progress[global_idx]:.2f} pred={frame_row['progress_pred']:.3f}",
                top_k=top_k,
            )
    return frame_rows, token_rows


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_dir)
    same_scene_dirs = [Path(path) for path in args.same_scene_dir]
    tasks = []
    for same_scene_dir in same_scene_dirs:
        success_video, failure_video, scene_id = _find_same_scene_pair(same_scene_dir)
        task_name = _infer_task_name(same_scene_dir)
        tasks.append(
            {
                "task_name": task_name,
                "language": TASK_LANGUAGE.get(task_name, task_name.replace("_", " ")),
                "same_scene_dir": same_scene_dir,
                "scene_id": scene_id,
                "success_video": success_video,
                "failure_video": failure_video,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "selected_videos.json").write_text(json.dumps(tasks, indent=2, default=str))

    all_summary: list[dict[str, Any]] = []
    for raw_model_path in args.model_path:
        model_path = _resolve_model_path(raw_model_path)
        model_slug = _safe_name(model_path.name)
        print(f"Loading model: {model_path}")
        policy = create_gr00t_sim_policy(
            model_path=str(model_path),
            embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
        )
        tokenizer = _get_tokenizer(policy)
        for task in tasks:
            task_slug = _safe_name(task["task_name"])
            task_output = output_root / model_slug / task_slug
            print(f"Visualizing {model_slug} / {task_slug}")
            frame_rows: list[dict[str, Any]] = []
            token_rows: list[dict[str, Any]] = []
            for role, video_path in (
                ("success", task["success_video"]),
                ("failure", task["failure_video"]),
            ):
                rows, tokens = _predict_video(
                    policy,
                    tokenizer,
                    Path(video_path),
                    role=role,
                    language=task["language"],
                    output_dir=task_output,
                    samples_per_video=args.samples_per_video,
                    batch_size=args.batch_size,
                    top_k=args.top_k,
                )
                frame_rows.extend(rows)
                token_rows.extend(tokens)
            _write_csv(task_output / "token_weight_timeseries.csv", frame_rows)
            _write_csv(task_output / "token_weights_long.csv", token_rows)
            _write_curves(
                frame_rows,
                task_output / "compare" / "success_failure_token_weight_curves.svg",
                title=f"{model_slug} / {task_slug}",
            )
            summary = {
                "model_path": str(model_path),
                "model_slug": model_slug,
                "task_name": task["task_name"],
                "scene_id": task["scene_id"],
                "language": task["language"],
                "success_video": str(task["success_video"]),
                "failure_video": str(task["failure_video"]),
                "output_dir": str(task_output),
            }
            for role in ("success", "failure"):
                role_rows = [row for row in frame_rows if row["role"] == role]
                if role_rows:
                    summary[f"{role}_pred_mean"] = float(
                        np.mean([row["progress_pred"] for row in role_rows])
                    )
                    summary[f"{role}_pred_min"] = float(
                        np.min([row["progress_pred"] for row in role_rows])
                    )
                    summary[f"{role}_pred_max"] = float(
                        np.max([row["progress_pred"] for row in role_rows])
                    )
                    summary[f"{role}_image_mass_mean"] = float(
                        np.mean([row["image_weight_sum"] for row in role_rows])
                    )
                    summary[f"{role}_text_mass_mean"] = float(
                        np.mean([row["text_weight_sum"] for row in role_rows])
                    )
                    summary[f"{role}_entropy_mean"] = float(
                        np.mean([row["attention_entropy"] for row in role_rows])
                    )
            (task_output / "summary.json").write_text(json.dumps(summary, indent=2))
            all_summary.append(summary)
        del policy
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    (output_root / "summary.json").write_text(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()

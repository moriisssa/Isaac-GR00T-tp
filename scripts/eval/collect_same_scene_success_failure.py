#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np


@dataclass
class EpisodeRecord:
    env_name: str
    task_name: str
    phase: str
    scene_id: int
    repeat_id: int
    env_seed: int
    policy_seed: int
    model_id: str | None
    model_scale: float | None
    drawer_id: str | None
    success: bool
    policy_steps: int
    primitive_steps: int
    video_path: str
    reset_options: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect same-scene success/failure SimplerEnv rollouts by first finding "
            "a successful scene, then repeating that scene with different policy seeds."
        )
    )
    parser.add_argument(
        "--env-name",
        default="simpler_env_google/google_robot_place_in_closed_drawer",
        help="Gym-style env name, e.g. simpler_env_google/google_robot_close_drawer.",
    )
    parser.add_argument(
        "--model-path",
        default="checkpoints/GR00T-N1.7-SimplerEnv-Fractal",
        help="Local GR00T model path. Ignored if --policy-client-host/port is set.",
    )
    parser.add_argument("--policy-client-host", default="")
    parser.add_argument("--policy-client-port", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to output/same_scene_success_failure_<task>_<timestamp>.",
    )
    parser.add_argument("--scene-ids", type=int, nargs="*", default=None)
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-count", type=int, default=60)
    parser.add_argument("--search-attempts-per-scene", type=int, default=2)
    parser.add_argument("--max-fixed-repeats", type=int, default=30)
    parser.add_argument("--target-successes", type=int, default=1)
    parser.add_argument("--target-failures", type=int, default=3)
    parser.add_argument(
        "--env-seed-base",
        type=int,
        default=None,
        help=(
            "Base seed for deterministic environment initialization. The per-scene "
            "env seed is env_seed_base + scene_id. Defaults to --policy-seed-base."
        ),
    )
    parser.add_argument(
        "--randomize-search-env-seed",
        action="store_true",
        help=(
            "Use a fresh deterministic-random env seed for each search rollout. "
            "After the first success, fixed rollouts reuse that successful env seed."
        ),
    )
    parser.add_argument("--policy-seed-base", type=int, default=20260615)
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional top-level SimplerEnv model_id to fix the object identity.",
    )
    parser.add_argument(
        "--model-scale",
        type=float,
        default=None,
        help="Optional top-level SimplerEnv model_scale to fix object scale.",
    )
    parser.add_argument("--shuffle-scenes", action="store_true", default=True)
    parser.add_argument("--no-shuffle-scenes", action="store_false", dest="shuffle_scenes")
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--robot-init-x", type=float, default=0.75)
    parser.add_argument("--robot-init-y", type=float, default=0.0)
    parser.add_argument("--robot-init-roll", type=float, default=0.0)
    parser.add_argument("--robot-init-pitch", type=float, default=0.0)
    parser.add_argument("--robot-init-yaw", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _task_name(env_name: str) -> str:
    if "/" in env_name:
        prefix, task = env_name.split("/", 1)
        if prefix != "simpler_env_google":
            raise ValueError(f"Only simpler_env_google is supported by this script, got {env_name}")
        return task
    return env_name


def _make_scene_ids(args: argparse.Namespace) -> list[int]:
    if args.scene_ids:
        scene_ids = list(args.scene_ids)
    else:
        scene_ids = list(range(args.scene_start, args.scene_start + args.scene_count))
    if args.shuffle_scenes:
        rng = random.Random(args.policy_seed_base)
        rng.shuffle(scene_ids)
    return scene_ids


def _build_reset_options(args: argparse.Namespace, scene_id: int) -> dict[str, Any]:
    quat = _euler_to_quat_wxyz(
        args.robot_init_roll,
        args.robot_init_pitch,
        args.robot_init_yaw,
    )
    reset_options = {
        "robot_init_options": {
            "init_xy": np.asarray([args.robot_init_x, args.robot_init_y], dtype=np.float64),
            "init_rot_quat": quat,
        },
        "obj_init_options": {
            "episode_id": int(scene_id),
        },
    }
    if args.model_id is not None:
        reset_options["model_id"] = args.model_id
    if args.model_scale is not None:
        reset_options["model_scale"] = args.model_scale
    return reset_options


def _env_seed(args: argparse.Namespace, scene_id: int) -> int:
    base = args.env_seed_base if args.env_seed_base is not None else args.policy_seed_base
    return (int(base) + int(scene_id)) % (2**32 - 1)


def _search_env_seed_rng(args: argparse.Namespace) -> random.Random:
    base = args.env_seed_base if args.env_seed_base is not None else args.policy_seed_base
    return random.Random(int(base) % (2**32 - 1))


def _nested_attr(obj: Any, name: str) -> Any:
    seen: set[int] = set()
    stack = [obj]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if hasattr(current, name):
            return getattr(current, name)
        for child_name in ("unwrapped", "env"):
            child = getattr(current, child_name, None)
            if child is not None and id(child) not in seen:
                stack.append(child)
    return None


def _euler_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _set_policy_seed(seed: int, *, local_policy: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if local_policy:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _batch_observation(obs: dict[str, Any]) -> dict[str, Any]:
    batched: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            arr = value
            if key.startswith("state"):
                arr = arr.astype(np.float32, copy=False)
            batched[key] = arr[None, ...]
        elif isinstance(value, str):
            batched[key] = [value]
        else:
            batched[key] = value
    return batched


def _env_action(actions: dict[str, np.ndarray], n_action_steps: int) -> dict[str, np.ndarray]:
    env_action: dict[str, np.ndarray] = {}
    for key, value in actions.items():
        arr = np.asarray(value)
        if arr.ndim != 3:
            raise ValueError(f"Expected action {key} to have shape [B,T,D], got {arr.shape}")
        env_action[key] = arr[0, :n_action_steps].astype(np.float32, copy=False)
    return env_action


def _latest_frame(obs: dict[str, Any]) -> np.ndarray:
    for key in ("video.image", "video.image_0"):
        if key not in obs:
            continue
        frame = np.asarray(obs[key])
        if frame.ndim == 4:
            frame = frame[-1]
        if frame.ndim != 3:
            raise ValueError(f"Expected video frame rank 3 or 4 for {key}, got {frame.shape}")
        return frame.astype(np.uint8, copy=False)
    raise KeyError("No video key found in observation")


def _info_success(info: dict[str, Any]) -> bool:
    success = info.get("success", False)
    if isinstance(success, np.ndarray):
        return bool(np.any(success))
    if isinstance(success, list):
        return bool(any(success))
    return bool(success)


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import cv2
    import shutil
    import subprocess

    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("Cannot write video without frames")
    h, w = frames[0].shape[:2]
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    tmp_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {tmp_path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        tmp_path.replace(path)
        return

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(tmp_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        str(path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (OSError, subprocess.CalledProcessError):
        tmp_path.replace(path)
    else:
        tmp_path.unlink(missing_ok=True)


def _run_episode(
    *,
    env: Any,
    policy,
    args: argparse.Namespace,
    task_name: str,
    reset_options: dict[str, Any],
    scene_id: int,
    repeat_id: int,
    env_seed: int,
    policy_seed: int,
    phase: str,
    videos_dir: Path,
    local_policy: bool,
) -> EpisodeRecord:
    _set_policy_seed(policy_seed, local_policy=local_policy)
    policy.reset()
    obs, _ = env.reset(seed=env_seed, options=reset_options)
    model_id = _nested_attr(env, "model_id")
    model_scale = _nested_attr(env, "model_scale")
    drawer_id = _nested_attr(env, "drawer_id")

    frames = [_latest_frame(obs)]
    success = False
    done = False
    truncated = False
    policy_steps = 0

    while not (done or truncated):
        actions, _ = policy.get_action(_batch_observation(obs))
        obs, _, done, truncated, info = env.step(_env_action(actions, args.n_action_steps))
        success = success or _info_success(info)
        frames.append(_latest_frame(obs))
        policy_steps += 1
        if policy_steps >= args.max_episode_steps:
            truncated = True

    status = "success" if success else "failure"
    stem = f"{phase}_scene{scene_id:04d}_repeat{repeat_id:03d}_policy{policy_seed}_{status}"
    video_path = videos_dir / f"{stem}.mp4"
    _write_video(video_path, frames, args.fps)

    return EpisodeRecord(
        env_name=args.env_name,
        task_name=task_name,
        phase=phase,
        scene_id=scene_id,
        repeat_id=repeat_id,
        env_seed=env_seed,
        policy_seed=policy_seed,
        model_id=model_id,
        model_scale=model_scale,
        drawer_id=drawer_id,
        success=success,
        policy_steps=policy_steps,
        primitive_steps=policy_steps * args.n_action_steps,
        video_path=str(video_path),
        reset_options=_jsonable(reset_options),
    )


def _append_record(csv_path: Path, jsonl_path: Path, record: EpisodeRecord) -> None:
    row = asdict(record)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    with jsonl_path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.rollout_policy import create_gr00t_sim_policy
    from gr00t.eval.sim.SimplerEnv.simpler_env import GoogleFractalEnv
    from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper

    task_name = _task_name(args.env_name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("output") / f"same_scene_success_failure_{task_name}_{timestamp}"
    )
    videos_dir = output_dir / "videos"
    csv_path = output_dir / "episodes.csv"
    jsonl_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    env = MultiStepWrapper(
        GoogleFractalEnv(task_name, image_size=(256, 320)),
        video_delta_indices=np.asarray([0]),
        state_delta_indices=np.asarray([0]),
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        terminate_on_success=True,
    )

    local_policy = not (args.policy_client_host and args.policy_client_port is not None)
    policy = create_gr00t_sim_policy(
        model_path=args.model_path if local_policy else "",
        embodiment_tag=EmbodimentTag.SIMPLER_ENV_GOOGLE,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
    )

    records: list[EpisodeRecord] = []
    selected_scene_id: int | None = None
    selected_reset_options: dict[str, Any] | None = None
    selected_env_seed: int | None = None
    search_env_seed_rng = _search_env_seed_rng(args)
    repeat_id = 0

    try:
        for scene_id in _make_scene_ids(args):
            reset_options = _build_reset_options(args, scene_id)
            for attempt in range(args.search_attempts_per_scene):
                scene_env_seed = (
                    search_env_seed_rng.randrange(0, 2**32 - 1)
                    if args.randomize_search_env_seed
                    else _env_seed(args, scene_id)
                )
                policy_seed = args.policy_seed_base + repeat_id
                record = _run_episode(
                    env=env,
                    policy=policy,
                    args=args,
                    task_name=task_name,
                    reset_options=reset_options,
                    scene_id=scene_id,
                    repeat_id=repeat_id,
                    env_seed=scene_env_seed,
                    policy_seed=policy_seed,
                    phase="search",
                    videos_dir=videos_dir,
                    local_policy=local_policy,
                )
                repeat_id += 1
                records.append(record)
                _append_record(csv_path, jsonl_path, record)
                print(
                    f"[search] scene={scene_id} env_seed={scene_env_seed} "
                    f"policy_seed={policy_seed} model={record.model_id} drawer={record.drawer_id} "
                    f"success={int(record.success)} video={record.video_path}",
                    flush=True,
                )
                if record.success:
                    selected_scene_id = scene_id
                    selected_reset_options = reset_options
                    selected_env_seed = scene_env_seed
                    break
            if selected_scene_id is not None:
                break

        if selected_scene_id is None or selected_reset_options is None or selected_env_seed is None:
            raise RuntimeError("No successful scene found in the search budget.")

        fixed_successes = sum(
            1
            for record in records
            if (
                record.scene_id == selected_scene_id
                and record.env_seed == selected_env_seed
                and record.success
            )
        )
        fixed_failures = sum(
            1
            for record in records
            if (
                record.scene_id == selected_scene_id
                and record.env_seed == selected_env_seed
                and not record.success
            )
        )

        fixed_repeat = 0
        while (
            fixed_repeat < args.max_fixed_repeats
            and (
                fixed_successes < args.target_successes
                or fixed_failures < args.target_failures
            )
        ):
            policy_seed = args.policy_seed_base + repeat_id
            record = _run_episode(
                env=env,
                policy=policy,
                args=args,
                task_name=task_name,
                reset_options=selected_reset_options,
                scene_id=selected_scene_id,
                repeat_id=repeat_id,
                env_seed=selected_env_seed,
                policy_seed=policy_seed,
                phase="fixed",
                videos_dir=videos_dir,
                local_policy=local_policy,
            )
            repeat_id += 1
            fixed_repeat += 1
            records.append(record)
            _append_record(csv_path, jsonl_path, record)
            fixed_successes += int(record.success)
            fixed_failures += int(not record.success)
            print(
                f"[fixed] scene={selected_scene_id} env_seed={selected_env_seed} "
                f"policy_seed={policy_seed} model={record.model_id} drawer={record.drawer_id} "
                f"success={int(record.success)} "
                f"counts=s{fixed_successes}/f{fixed_failures} video={record.video_path}",
                flush=True,
            )

    finally:
        env.close()

    selected_records = [
        record
        for record in records
        if record.scene_id == selected_scene_id and record.env_seed == selected_env_seed
    ]
    summary = {
        "output_dir": str(output_dir),
        "episodes_csv": str(csv_path),
        "episodes_jsonl": str(jsonl_path),
        "selected_scene_id": selected_scene_id,
        "selected_env_seed": selected_env_seed,
        "num_records": len(records),
        "selected_scene_successes": sum(int(record.success) for record in selected_records),
        "selected_scene_failures": sum(int(not record.success) for record in selected_records),
        "target_successes": args.target_successes,
        "target_failures": args.target_failures,
        "complete": (
            sum(int(record.success) for record in selected_records) >= args.target_successes
            and sum(int(not record.success) for record in selected_records) >= args.target_failures
        ),
        "records": [asdict(record) for record in selected_records],
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

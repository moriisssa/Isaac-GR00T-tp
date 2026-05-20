# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import csv
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.progress_curve_plot import write_progress_curve_plot
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy import BasePolicy
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import tyro


class TrtMode(str, Enum):
    """TensorRT inference modes."""

    N17_FULL_PIPELINE = "n17_full_pipeline"
    VIT_LLM_ONLY = "vit_llm_only"
    ACTION_HEAD = "action_head"


@dataclass
class VideoConfig:
    """Configuration for video recording settings.

    Attributes:
        video_dir: Directory to save videos (if None, no videos are saved)
        steps_per_render: Number of steps between each call to env.render() while recording
            during rollout
        fps: Frames per second for the output video
        codec: Video codec to use for compression
        input_pix_fmt: Input pixel format
        crf: Constant Rate Factor for video compression (lower = better quality)
        thread_type: Threading strategy for video encoding
        thread_count: Number of threads to use for encoding
    """

    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = 720
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    overlay_text: bool = True
    n_action_steps: int = 8


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        video_delta_indices: Indices of video observations to stack
        state_delta_indices: Indices of state observations to stack
        n_action_steps: Number of action steps to execute
        max_episode_steps: Maximum number of steps per episode
    """

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 720
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    """Container for various environment wrapper configurations.

    Attributes:
        video: Configuration for video recording
        multistep: Configuration for multi-step processing
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


@dataclass
class ProgressCurveConfig:
    """Configuration for optional progress prediction curve logging."""

    output_dir: str | None = None
    success_only: bool = False
    target: str = "current"
    target_offset_steps: int = 0


def get_simpler_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

        register_simpler_envs()
        return gym.make(env_name)

    return env_fn


def get_libero_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

        register_libero_envs()
        return gym.make(env_name)

    return env_fn


def get_gym_env(env_name: str, env_idx: int, total_n_envs: int):
    """Create Ray environment factory function without wrappers."""

    env_embodiment = get_embodiment_tag_from_env_name(env_name)

    if env_embodiment in (EmbodimentTag.SIMPLER_ENV_GOOGLE, EmbodimentTag.SIMPLER_ENV_WIDOWX):
        env_fn = get_simpler_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.LIBERO_PANDA,):
        env_fn = get_libero_env_fn(env_name)

    else:
        raise ValueError(f"Invalid environment name: {env_name}")

    return env_fn()


def create_eval_env(
    env_name: str, env_idx: int, total_n_envs: int, wrapper_configs: WrapperConfigs
) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        idx: Environment index (used to determine video recording)
        wrapper_configs: Configuration for environment wrappers
    Returns:
        Wrapped gymnasium environment
    """

    env = get_gym_env(env_name, env_idx, total_n_envs)
    if wrapper_configs.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            overlay_text=wrapper_configs.video.overlay_text,
        )

    env = MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


class _RobustAsyncVectorEnv(gym.vector.AsyncVectorEnv):
    """AsyncVectorEnv that tolerates variable-shaped info arrays across envs.

    Gymnasium's default _add_info pre-allocates a numpy array based on the
    first env's value shape and then assigns subsequent envs into it.  When
    envs return differently-shaped values (e.g. variable-length contact arrays)
    the assignment raises ValueError.  We catch that and fall back to a plain
    Python list for that key so the rest of the step can proceed normally.
    """

    def _add_info(self, infos, info, env_num):
        for k, v in info.items():
            if k not in infos:
                infos[k] = [None] * self.num_envs
                infos[f"_{k}"] = np.zeros(self.num_envs, dtype=bool)
            if isinstance(infos[k], np.ndarray):
                try:
                    infos[k][env_num] = v
                except (ValueError, TypeError):
                    lst = list(infos[k])
                    lst[env_num] = v
                    infos[k] = lst
            else:
                infos[k][env_num] = v
            infos[f"_{k}"][env_num] = True
        return infos


def _extract_progress_predictions(policy_info: dict[str, Any], n_envs: int) -> list[float | None]:
    if "progress" not in policy_info:
        return [None] * n_envs

    progress = np.asarray(policy_info["progress"], dtype=np.float32).reshape(-1)
    if progress.size == 1 and n_envs > 1:
        progress = np.repeat(progress, n_envs)

    predictions: list[float | None] = []
    for env_idx in range(n_envs):
        if env_idx >= progress.size or not np.isfinite(progress[env_idx]):
            predictions.append(None)
        else:
            predictions.append(float(progress[env_idx]))
    return predictions


def _finalize_progress_episode(
    episode_rows: list[dict[str, Any]],
    *,
    episode_index: int,
    success: bool,
    valid: bool,
    target: str,
    target_offset_steps: int,
    final_primitive_step: int,
) -> list[dict[str, Any]]:
    episode_length = len(episode_rows)
    denominator = max(final_primitive_step, 1)
    finalized_rows = []

    for row in episode_rows:
        if target == "current":
            target_step = row["primitive_step"]
        elif target == "chunk_end":
            target_step = min(row["primitive_step"] + target_offset_steps, denominator)
        else:
            raise ValueError(f"Unsupported progress curve target: {target}")
        target_progress = float(target_step / denominator)
        progress_pred = row["progress_pred"]
        finalized_rows.append(
            {
                "episode": episode_index,
                "env_idx": row["env_idx"],
                "step": row["step"],
                "primitive_step": row["primitive_step"],
                "episode_length": episode_length,
                "final_primitive_step": final_primitive_step,
                "target_progress": target_progress,
                "progress_pred": progress_pred,
                "abs_error": abs(progress_pred - target_progress),
                "success": success,
                "valid": valid,
            }
        )
    return finalized_rows


def _summarize_progress_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_samples": 0,
            "num_episodes": 0,
            "mae": None,
            "rmse": None,
            "pred_mean": None,
            "target_mean": None,
            "negative_delta_fraction": None,
        }

    pred = np.asarray([row["progress_pred"] for row in rows], dtype=np.float32)
    target = np.asarray([row["target_progress"] for row in rows], dtype=np.float32)
    errors = pred - target

    negative_deltas = 0
    total_deltas = 0
    for episode in sorted({row["episode"] for row in rows}):
        episode_pred = [
            row["progress_pred"] for row in rows if row["episode"] == episode and row["valid"]
        ]
        if len(episode_pred) <= 1:
            continue
        deltas = np.diff(np.asarray(episode_pred, dtype=np.float32))
        negative_deltas += int(np.sum(deltas < -1e-3))
        total_deltas += int(deltas.size)

    return {
        "num_samples": int(len(rows)),
        "num_episodes": int(len({row["episode"] for row in rows})),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
        "pred_mean": float(np.mean(pred)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
        "target_mean": float(np.mean(target)),
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "corr": float(np.corrcoef(target, pred)[0, 1]) if len(pred) > 1 else None,
        "negative_delta_fraction": (
            float(negative_deltas / total_deltas) if total_deltas > 0 else None
        ),
    }


def _write_progress_curve_outputs(
    rows: list[dict[str, Any]],
    *,
    output_dir: str,
    success_only: bool,
    target: str,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "progress_predictions.csv"
    metrics_path = output_path / "progress_metrics.json"
    png_path = output_path / "progress_curve.png"
    svg_path = output_path / "progress_curve.svg"

    fieldnames = [
        "episode",
        "env_idx",
        "step",
        "primitive_step",
        "episode_length",
        "final_primitive_step",
        "target_progress",
        "progress_pred",
        "abs_error",
        "success",
        "valid",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["valid"]]
    metric_rows = [row for row in valid_rows if row["success"]] if success_only else valid_rows
    plot_path: Path | None = None
    plot_backend: str | None = None

    if rows:
        try:
            write_progress_curve_plot(metric_rows, png_path=png_path, target=target)
            plot_path = png_path
            plot_backend = "matplotlib"
        except ImportError:
            if _write_progress_curve_plot_subprocess(
                csv_path=csv_path,
                png_path=png_path,
                success_only=success_only,
                target=target,
            ):
                plot_path = png_path
                plot_backend = "matplotlib"
            else:
                _write_progress_curve_svg(metric_rows, svg_path)
                plot_path = svg_path
                plot_backend = "svg"

    metrics = {
        "csv_path": str(csv_path),
        "metrics_path": str(metrics_path),
        "plot_path": str(plot_path) if plot_path is not None else None,
        "plot_backend": plot_backend,
        "success_only": success_only,
        "target": target,
        "all": _summarize_progress_rows(rows),
        "valid": _summarize_progress_rows(valid_rows),
        "selected": _summarize_progress_rows(metric_rows),
    }

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Progress curve CSV saved to: {csv_path}")
    print(f"Progress curve metrics saved to: {metrics_path}")
    if plot_path is not None:
        print(f"Progress curve plot saved to: {plot_path}")
    return metrics


def _write_progress_curve_plot_subprocess(
    *,
    csv_path: Path,
    png_path: Path,
    success_only: bool,
    target: str,
) -> bool:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "eval" / "plot_progress_curve.py"
    if not script_path.exists():
        return False

    python_path = repo_root / ".venv" / "bin" / "python"
    if python_path.exists():
        cmd = [str(python_path), str(script_path)]
    else:
        uv_path = shutil.which("uv")
        if uv_path is None:
            return False
        cmd = [uv_path, "run", "python", str(script_path)]

    cmd.extend(
        [
            "--csv-path",
            str(csv_path),
            "--png-path",
            str(png_path),
            "--target",
            target,
        ]
    )
    if success_only:
        cmd.append("--success-only")

    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return False
    return png_path.exists()


def _write_progress_curve_svg(rows: list[dict[str, Any]], svg_path: Path) -> None:
    width = 840
    height = 560
    margin = 64
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    def point(x: float, y: float) -> tuple[float, float]:
        return margin + x * plot_width, height - margin - y * plot_height

    def polyline(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    episode_lines = []
    for episode in sorted({row["episode"] for row in rows}):
        episode_rows = sorted(
            [row for row in rows if row["episode"] == episode],
            key=lambda row: row["step"],
        )
        if len(episode_rows) < 2:
            continue
        points = [point(row["target_progress"], row["progress_pred"]) for row in episode_rows]
        episode_lines.append(
            f'<polyline points="{polyline(points)}" '
            'fill="none" stroke="#7a9cc6" stroke-width="1" opacity="0.22" />'
        )

    mean_line = ""
    if rows:
        bins = np.linspace(0.0, 1.0, 21)
        target_values = np.asarray([row["target_progress"] for row in rows], dtype=np.float32)
        pred = np.asarray([row["progress_pred"] for row in rows], dtype=np.float32)
        bin_indices = np.clip(np.digitize(target_values, bins) - 1, 0, len(bins) - 2)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        mean_points = []
        for idx, bin_center in enumerate(bin_centers):
            if np.any(bin_indices == idx):
                mean_points.append(
                    point(float(bin_center), float(np.mean(pred[bin_indices == idx])))
                )
        if len(mean_points) >= 2:
            mean_line = (
                f'<polyline points="{polyline(mean_points)}" '
                'fill="none" stroke="#d55e00" stroke-width="3" opacity="0.95" />'
            )

    ideal_points = polyline([point(0.0, 0.0), point(1.0, 1.0)])
    x0, y0 = point(0.0, 0.0)
    x1, y1 = point(1.0, 1.0)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y0:.2f}" stroke="#222" stroke-width="1"/>
  <line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y1:.2f}" stroke="#222" stroke-width="1"/>
  <polyline points="{ideal_points}" fill="none" stroke="#444" stroke-width="2" stroke-dasharray="6 6"/>
  {"".join(episode_lines)}
  {mean_line}
  <text x="{width / 2:.2f}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">Progress prediction curve</text>
  <text x="{width / 2:.2f}" y="{height - 16}" text-anchor="middle" font-family="sans-serif" font-size="14">normalized rollout progress</text>
  <text x="18" y="{height / 2:.2f}" transform="rotate(-90 18,{height / 2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="14">predicted progress</text>
  <text x="{x1:.2f}" y="{y0 + 24:.2f}" text-anchor="middle" font-family="sans-serif" font-size="12">1.0</text>
  <text x="{x0:.2f}" y="{y0 + 24:.2f}" text-anchor="middle" font-family="sans-serif" font-size="12">0.0</text>
  <text x="{x0 - 10:.2f}" y="{y1 + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">1.0</text>
  <text x="{x0 - 10:.2f}" y="{y0 + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">0.0</text>
  <text x="{x1 - 12:.2f}" y="{y1 + 24:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#444">ideal</text>
  <text x="{x1 - 12:.2f}" y="{y1 + 44:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#d55e00">binned prediction mean</text>
</svg>
"""
    svg_path.write_text(svg)


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    n_episodes: int = 10,
    n_envs: int = 1,
    progress_curve_config: ProgressCurveConfig | None = None,
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy: Policy instance
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        wrapper_configs: Configuration for environment wrappers
        ray_env: Whether to use ray gym env to create each env.
    Returns:
        Collection results from running the episodes
    """
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_name,
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
        )
        for idx in range(n_envs)
    ]

    if n_envs == 1:
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = _RobustAsyncVectorEnv(
            env_fns,
            shared_memory=False,
            context="spawn",
        )

    # Storage for results
    episode_lengths = []
    current_rewards = [0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)
    progress_rows: list[dict[str, Any]] = []
    current_progress_rows: list[list[dict[str, Any]]] = [[] for _ in range(n_envs)]

    # Initial reset
    observations, _ = env.reset()
    policy.reset()
    i = 0

    pbar = tqdm(total=n_episodes, desc="Episodes")
    while completed_episodes < n_episodes:
        actions, policy_info = policy.get_action(observations)
        progress_preds = _extract_progress_predictions(policy_info, n_envs)
        if progress_curve_config and progress_curve_config.output_dir:
            for env_idx, progress_pred in enumerate(progress_preds):
                if progress_pred is None:
                    continue
                current_progress_rows[env_idx].append(
                    {
                        "env_idx": env_idx,
                        "step": current_lengths[env_idx],
                        "primitive_step": current_lengths[env_idx]
                        * wrapper_configs.multistep.n_action_steps,
                        "progress_pred": progress_pred,
                    }
                )
        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        # NOTE (FY): Currently we don't properly handle policy reset. For now, our policy are stateless,
        # but in the future if we need policy to be stateful, we need to detect env reset and call policy.reset()
        i += 1
        # Update episode tracking
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, list):
                    env_success = np.any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            else:
                current_successes[env_idx] = False

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                env_success = env_infos["final_info"][env_idx]["success"]
                if isinstance(env_success, list):
                    env_success = any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1

            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                if "final_info" in env_infos:
                    current_successes[env_idx] |= any(env_infos["final_info"][env_idx]["success"])
                if "task_progress" in env_infos:
                    episode_infos["task_progress"].append(env_infos["task_progress"][env_idx][-1])
                if "q_score" in env_infos:
                    episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                episode_valid = True
                if "valid" in env_infos:
                    episode_valid = all(env_infos["valid"][env_idx])
                    episode_infos["valid"].append(episode_valid)
                # Accumulate results
                episode_lengths.append(current_lengths[env_idx])
                episode_index = len(episode_successes)
                episode_successes.append(current_successes[env_idx])
                if progress_curve_config and progress_curve_config.output_dir:
                    progress_rows.extend(
                        _finalize_progress_episode(
                            current_progress_rows[env_idx],
                            episode_index=episode_index,
                            success=current_successes[env_idx],
                            valid=episode_valid,
                            target=progress_curve_config.target,
                            target_offset_steps=progress_curve_config.target_offset_steps,
                            final_primitive_step=max(
                                current_lengths[env_idx] * wrapper_configs.multistep.n_action_steps
                                - 1,
                                1,
                            ),
                        )
                    )
                    current_progress_rows[env_idx] = []
                # Reset trackers for this environment.
                current_successes[env_idx] = False
                # only update completed_episodes if valid
                if "valid" in episode_infos:
                    if episode_infos["valid"][-1]:
                        completed_episodes += 1
                        pbar.update(1)
                else:
                    # envs don't return valid
                    completed_episodes += 1
                    pbar.update(1)
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0
        observations = next_obs
    pbar.close()

    env.reset()
    env.close()
    print(f"Collecting {n_episodes} episodes took {time.time() - start_time} seconds")

    assert len(episode_successes) >= n_episodes, (
        f"Expected at least {n_episodes} episodes, got {len(episode_successes)}"
    )

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        assert len(value) == len(episode_successes), (
            f"Length of {key} is not equal to the number of episodes"
        )

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    if progress_curve_config and progress_curve_config.output_dir:
        episode_infos["progress_curve"] = _write_progress_curve_outputs(
            progress_rows,
            output_dir=progress_curve_config.output_dir,
            success_only=progress_curve_config.success_only,
            target=progress_curve_config.target,
        )

    return env_name, episode_successes, episode_infos


def create_gr00t_sim_policy(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    trt_engine_path: str = "",
    trt_mode: TrtMode = TrtMode.N17_FULL_PIPELINE,
) -> BasePolicy:
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    if policy_client_host and policy_client_port:
        from gr00t.policy.server_client import PolicyClient

        policy = PolicyClient(host=policy_client_host, port=policy_client_port)
    else:
        gr00t_policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=0,
        )
        if trt_engine_path:
            deploy_dir = str(Path(__file__).resolve().parents[2] / "scripts" / "deployment")
            if deploy_dir not in sys.path:
                sys.path.insert(0, deploy_dir)
            from trt_model_forward import setup_tensorrt_engines

            setup_tensorrt_engines(gr00t_policy, trt_engine_path, mode=trt_mode)
        policy = Gr00tSimPolicyWrapper(gr00t_policy)
    return policy


def run_gr00t_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    video_dir: str | None = None,
    trt_engine_path: str = "",
    trt_mode: TrtMode = TrtMode.N17_FULL_PIPELINE,
    progress_curve_dir: str | None = None,
    progress_curve_success_only: bool = False,
    progress_curve_target: str = "current",
    record_video: bool = True,
):
    if progress_curve_target not in {"current", "chunk_end"}:
        raise ValueError(
            f"Unsupported progress_curve_target={progress_curve_target!r}; "
            "expected 'current' or 'chunk_end'"
        )
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    if not record_video:
        video_dir = None
    elif video_dir is None:
        if model_path:
            parts = model_path.split("/")
            model_slug = parts[-3] if len(parts) >= 3 else parts[-1]
            video_dir = f"/tmp/sim_eval_videos_{model_slug}_ac{n_action_steps}_{uuid.uuid4()}"
        else:
            video_dir = f"/tmp/sim_eval_videos_{env_name}_ac{n_action_steps}_{uuid.uuid4()}"
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )

    policy = create_gr00t_sim_policy(
        model_path,
        embodiment_tag,
        policy_client_host,
        policy_client_port,
        trt_engine_path=trt_engine_path,
        trt_mode=trt_mode,
    )
    progress_target_offset_steps = 0
    if progress_curve_target == "chunk_end":
        modality_config = policy.get_modality_config()
        model_action_horizon = len(modality_config["action"].delta_indices)
        progress_target_offset_steps = max(model_action_horizon - 1, 0)
        print(
            "Progress curve chunk_end target uses "
            f"model action horizon={model_action_horizon} "
            f"(offset={progress_target_offset_steps} primitive steps)"
        )

    results = run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=n_episodes,
        n_envs=n_envs,
        progress_curve_config=ProgressCurveConfig(
            output_dir=progress_curve_dir,
            success_only=progress_curve_success_only,
            target=progress_curve_target,
            target_offset_steps=progress_target_offset_steps,
        ),
    )
    print("Video saved to: ", wrapper_configs.video.video_dir)
    return results


@dataclass
class RolloutConfig:
    """Configuration for rollout policy evaluation."""

    max_episode_steps: int = 504
    """Maximum number of steps per episode."""

    n_episodes: int = 50
    """Number of episodes to run."""

    model_path: str = ""
    """Path to model checkpoint."""

    policy_client_host: str = ""
    """Host for policy client."""

    policy_client_port: int | None = None
    """Port for policy client."""

    env_name: str = "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    """Environment name."""

    n_envs: int = 8
    """Number of parallel environments."""

    n_action_steps: int = 8
    """Number of action steps."""

    video_dir: str | None = None
    """Directory to save videos. If None, uses /tmp/sim_eval_videos_<env>_<uuid>."""

    trt_engine_path: str = ""
    """Path to TRT engine directory. If set, uses TRT inference instead of PyTorch."""

    trt_mode: TrtMode = TrtMode.N17_FULL_PIPELINE
    """TRT mode: 'n17_full_pipeline' (all engines), 'vit_llm_only', or 'action_head'."""

    progress_curve_dir: str | None = None
    """Directory to save progress prediction CSV, metrics JSON, and curve PNG."""

    progress_curve_success_only: bool = False
    """If true, compute the plotted/selected progress metrics only on successful valid episodes."""

    progress_curve_target: str = "current"
    """Progress target used for the curve: 'current' or 'chunk_end'."""

    record_video: bool = True
    """Whether to record rollout videos."""


if __name__ == "__main__":
    args = tyro.cli(RolloutConfig)

    # validate policy configuration
    assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
        not args.model_path and args.policy_client_host and args.policy_client_port is not None
    ), (
        "Invalid policy configuration: You must provide EITHER model_path OR (policy_client_host & policy_client_port), not both.\n"
        "If all 3 arguments are provided, explicitly choose one:\n"
        '  - To use policy client: set --policy-client-host and --policy-client-port, and set --model-path ""\n'
        '  - To use model path: set --model-path, and set --policy-client-host "" (and leave --policy-client-port unset)'
    )

    results = run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        video_dir=args.video_dir,
        trt_engine_path=args.trt_engine_path,
        trt_mode=args.trt_mode,
        progress_curve_dir=args.progress_curve_dir,
        progress_curve_success_only=args.progress_curve_success_only,
        progress_curve_target=args.progress_curve_target,
        record_video=args.record_video,
    )
    print("results: ", results)
    print("success rate: ", np.mean(results[1]))

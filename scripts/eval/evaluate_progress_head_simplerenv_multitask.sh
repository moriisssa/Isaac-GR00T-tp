#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-output/fractal_progress_head_full_5k_chunk_end_suffix_token_masked_mlp_sigmoid/fractal_progress_head_chunk_end_suffix_token_masked_mlp_sigmoid}
POLICY_CLIENT_HOST=${POLICY_CLIENT_HOST:-127.0.0.1}
POLICY_CLIENT_PORT=${POLICY_CLIENT_PORT:-5564}
OUTPUT_ROOT=${OUTPUT_ROOT:-${MODEL_PATH}/simplerenv_multitask_eval}
N_EPISODES=${N_EPISODES:-5}
N_ENVS=${N_ENVS:-1}
N_ACTION_STEPS=${N_ACTION_STEPS:-4}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-300}
PROGRESS_CURVE_TARGET=${PROGRESS_CURVE_TARGET:-chunk_end}
PYTHON_BIN=${PYTHON_BIN:-gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python}

TASKS=${TASKS:-"simpler_env_google/google_robot_pick_coke_can simpler_env_google/google_robot_pick_object simpler_env_google/google_robot_move_near"}

mkdir -p "${OUTPUT_ROOT}"

for env_name in ${TASKS}; do
  task_name=${env_name##*/}
  task_output_dir="${OUTPUT_ROOT}/${task_name}"
  mkdir -p "${task_output_dir}"

  echo "=== Evaluating ${env_name} ==="
  "${PYTHON_BIN}" gr00t/eval/rollout_policy.py \
    --n-episodes "${N_EPISODES}" \
    --policy-client-host "${POLICY_CLIENT_HOST}" \
    --policy-client-port "${POLICY_CLIENT_PORT}" \
    --max-episode-steps "${MAX_EPISODE_STEPS}" \
    --env-name "${env_name}" \
    --n-action-steps "${N_ACTION_STEPS}" \
    --n-envs "${N_ENVS}" \
    --video-dir "${task_output_dir}/videos" \
    --progress-curve-dir "${task_output_dir}/progress_curve" \
    --progress-curve-target "${PROGRESS_CURVE_TARGET}" \
    2>&1 | tee "${task_output_dir}/rollout.log"
done

echo "Saved SimplerEnv multitask evaluation outputs to: ${OUTPUT_ROOT}"

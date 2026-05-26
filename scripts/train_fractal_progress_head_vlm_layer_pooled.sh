#!/usr/bin/env bash

set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

PROGRESS_VLM_LAYER="${PROGRESS_VLM_LAYER:-}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --progress-vlm-layer)
      PROGRESS_VLM_LAYER="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage:
  PROGRESS_VLM_LAYER=8 bash scripts/train_fractal_progress_head_vlm_layer_pooled.sh
  bash scripts/train_fractal_progress_head_vlm_layer_pooled.sh --progress-vlm-layer 8
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PROGRESS_VLM_LAYER" ]]; then
  echo "Missing required PROGRESS_VLM_LAYER or --progress-vlm-layer." >&2
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29580}"
MAX_STEPS="${MAX_STEPS:-1000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-1}"
USE_WANDB="${USE_WANDB:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-512}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
PROGRESS_TARGET="${PROGRESS_TARGET:-current}"
LOAD_BF16="${LOAD_BF16:-1}"
RESUME="${RESUME:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-progress}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-checkpoints/GR00T-N1.7-SimplerEnv-Fractal}"
DATASET_PATH="${DATASET_PATH:-examples/SimplerEnv/fractal20220817_data_lerobot/}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-SIMPLER_ENV_GOOGLE}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fractal_progress_head_vlm_layer_${PROGRESS_VLM_LAYER}_1k_current_regression}"
OUTPUT_DIR="${OUTPUT_DIR:-output/progress_vlm_layerwise/${EXPERIMENT_NAME}}"

if [[ -d "$OUTPUT_DIR" && "$RESUME" != "1" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  echo "Use a new OUTPUT_DIR, or set RESUME=1 to continue from it." >&2
  exit 1
fi

EXTRA_PROGRESS_ARGS=()
if [[ "$LOAD_BF16" == "1" ]]; then
  EXTRA_PROGRESS_ARGS+=(--load-bf16)
fi

WANDB_FLAG=()
if [[ "$USE_WANDB" == "1" ]]; then
  WANDB_FLAG+=(--use_wandb)
fi

SAVE_ONLY_MODEL_FLAG=()
if [[ "$SAVE_ONLY_MODEL" == "1" ]]; then
  SAVE_ONLY_MODEL_FLAG+=(--save_only_model)
fi

uv run torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  gr00t/experiment/launch_finetune.py \
  --base_model_path "$BASE_MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --embodiment_tag "$EMBODIMENT_TAG" \
  --num_gpus "$NUM_GPUS" \
  --output_dir "$OUTPUT_DIR" \
  --experiment_name "$EXPERIMENT_NAME" \
  --wandb_project "$WANDB_PROJECT" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --max_steps "$MAX_STEPS" \
  --warmup_ratio 0.1 \
  --weight_decay 1e-5 \
  --learning_rate "$LEARNING_RATE" \
  "${WANDB_FLAG[@]}" \
  --global_batch_size "$GLOBAL_BATCH_SIZE" \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --shard_size "$SHARD_SIZE" \
  --num_shards_per_epoch "$NUM_SHARDS_PER_EPOCH" \
  --episode_sampling_rate "$EPISODE_SAMPLING_RATE" \
  --state_dropout_prob 0.0 \
  "${SAVE_ONLY_MODEL_FLAG[@]}" \
  --enable-progress-head \
  --tune-progress-head \
  --progress-head-source vlm_layer_pooled \
  --progress-vlm-layer "$PROGRESS_VLM_LAYER" \
  --no-tune-projector \
  --no-tune-diffusion-model \
  --no-tune-vlln \
  --use-ddp \
  --progress-loss-weight 1.0 \
  --progress-output-type scalar \
  --progress-target "$PROGRESS_TARGET" \
  --tail-shrink-action-chunk \
  "${EXTRA_PROGRESS_ARGS[@]}"

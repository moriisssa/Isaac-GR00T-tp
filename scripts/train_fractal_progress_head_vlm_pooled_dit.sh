#!/usr/bin/env bash

set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29506}"
MAX_STEPS="${MAX_STEPS:-1000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
USE_WANDB="${USE_WANDB:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-output/fractal_progress_head_vlm_pooled_dit_final_token_linear_1k_current}"
PROGRESS_TARGET="${PROGRESS_TARGET:-current}"
RESUME="${RESUME:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-gr00t-progress}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fractal_progress_head_vlm_pooled_dit_final_token_linear_1k_current}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-checkpoints/GR00T-N1.7-SimplerEnv-Fractal}"
DATASET_PATH="${DATASET_PATH:-examples/SimplerEnv/fractal20220817_data_lerobot/}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-SIMPLER_ENV_GOOGLE}"

if [[ -d "$OUTPUT_DIR" && "$RESUME" != "1" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  echo "Use a new OUTPUT_DIR, or set RESUME=1 to continue from it." >&2
  exit 1
fi

export NUM_GPUS
export MASTER_PORT
export MAX_STEPS
export GLOBAL_BATCH_SIZE
export SAVE_STEPS
export SAVE_TOTAL_LIMIT
export USE_WANDB
export DATALOADER_NUM_WORKERS
export NUM_SHARDS_PER_EPOCH
export LEARNING_RATE

uv run bash examples/finetune.sh \
  --base-model-path "$BASE_MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name "$EXPERIMENT_NAME" \
  --wandb-project "$WANDB_PROJECT" \
  --state-dropout-prob 0.0 \
  -- --enable-progress-head \
     --tune-progress-head \
     --progress-head-source vlm_pooled_dit \
     --no-tune-projector \
     --no-tune-diffusion-model \
     --no-tune-vlln \
     --use-ddp \
     --progress-loss-weight 1.0 \
     --progress-target "$PROGRESS_TARGET"

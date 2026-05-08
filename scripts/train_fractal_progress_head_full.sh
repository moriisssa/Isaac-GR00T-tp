#!/usr/bin/env bash

set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"
MAX_STEPS="${MAX_STEPS:-5000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
SAVE_STEPS="${SAVE_STEPS:-500}"
USE_WANDB="${USE_WANDB:-0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-output/fractal_progress_head_full}"
PROGRESS_TARGET="${PROGRESS_TARGET:-current}"

export NUM_GPUS
export MASTER_PORT
export MAX_STEPS
export GLOBAL_BATCH_SIZE
export SAVE_STEPS
export USE_WANDB
export DATALOADER_NUM_WORKERS
export NUM_SHARDS_PER_EPOCH
export OUTPUT_DIR
export PROGRESS_TARGET

uv run bash examples/finetune.sh \
  --base-model-path /data-ssd/xucx/Isaac-GR00T/checkpoints/GR00T-N1.7-SimplerEnv-Fractal \
  --dataset-path examples/SimplerEnv/fractal20220817_data_lerobot/ \
  --embodiment-tag SIMPLER_ENV_GOOGLE \
  --output-dir "$OUTPUT_DIR" \
  --state-dropout-prob 0.0 \
  -- --enable-progress-head \
     --tune-progress-head \
     --no-tune-projector \
     --no-tune-diffusion-model \
     --no-tune-vlln \
     --use-ddp \
     --progress-loss-weight 1.0 \
     --progress-target "$PROGRESS_TARGET"

#!/usr/bin/env bash

set -euo pipefail

VARIANT="${1:-stronger_bs64_5k}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

case "$VARIANT" in
  stronger_bs64_5k)
    EXPERIMENT_SUFFIX="scalar05_bs64_5k_current_lr5e6_wd002_reg_${STAMP}"
    export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
    export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
    ;;
  stronger_bs256_accum_5k)
    EXPERIMENT_SUFFIX="scalar05_bs256_accum4_5k_current_lr5e6_wd002_reg_${STAMP}"
    # This codebase treats GLOBAL_BATCH_SIZE as the per-update micro batch
    # across GPUs. Accumulating 4 steps makes the effective batch 64 x 4 = 256.
    export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
    export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    echo "Expected: stronger_bs64_5k or stronger_bs256_accum_5k" >&2
    exit 1
    ;;
esac

export NUM_GPUS="${NUM_GPUS:-4}"
export MASTER_PORT="${MASTER_PORT:-29585}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-1}"
export EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
export EVAL_STEPS="${EVAL_STEPS:-500}"
export EVAL_SET_SPLIT_RATIO="${EVAL_SET_SPLIT_RATIO:-0.1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export NUM_EVAL_SHARDS_PER_EPOCH="${NUM_EVAL_SHARDS_PER_EPOCH:-32}"
export SAVE_BEST_EVAL_METRIC_NAME="${SAVE_BEST_EVAL_METRIC_NAME:-eval_loss}"
export SAVE_BEST_EVAL_METRIC_GREATER_IS_BETTER="${SAVE_BEST_EVAL_METRIC_GREATER_IS_BETTER:-false}"
export USE_WANDB="${USE_WANDB:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export SHARD_SIZE="${SHARD_SIZE:-1024}"
export NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-512}"
export EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
export PROGRESS_TARGET="${PROGRESS_TARGET:-current}"
export LOAD_BF16="${LOAD_BF16:-1}"
export RESUME="${RESUME:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-gr00t-progress}"

export BASE_MODEL_PATH="${BASE_MODEL_PATH:-checkpoints/GR00T-N1.7-SimplerEnv-Fractal}"
export DATASET_PATH="${DATASET_PATH:-examples/SimplerEnv/fractal20220817_data_lerobot/}"
export EMBODIMENT_TAG="${EMBODIMENT_TAG:-SIMPLER_ENV_GOOGLE}"

export PROGRESS_VLM_LAYER="${PROGRESS_VLM_LAYER:-16}"
export PROGRESS_HEAD_SOURCE="${PROGRESS_HEAD_SOURCE:-vlm_concat_attention_pool}"
export PROGRESS_LOSS_TYPE="${PROGRESS_LOSS_TYPE:-pairwise_bt}"
export PROGRESS_PAIR_GAP_MIN="${PROGRESS_PAIR_GAP_MIN:-0.08}"
export PROGRESS_PAIR_MARGIN_ALPHA="${PROGRESS_PAIR_MARGIN_ALPHA:-0.2}"
export PROGRESS_PAIR_SCALAR_LOSS_WEIGHT="${PROGRESS_PAIR_SCALAR_LOSS_WEIGHT:-0.5}"
export PROGRESS_LOGIT_L2_WEIGHT="${PROGRESS_LOGIT_L2_WEIGHT:-1e-4}"
export PROGRESS_LOGIT_VARIANCE_WEIGHT="${PROGRESS_LOGIT_VARIANCE_WEIGHT:-1e-3}"
export PROGRESS_PAIR_SMOOTHNESS_WEIGHT="${PROGRESS_PAIR_SMOOTHNESS_WEIGHT:-0.05}"
export PROGRESS_PAIR_SMOOTHNESS_MARGIN="${PROGRESS_PAIR_SMOOTHNESS_MARGIN:-0.05}"
export PROGRESS_PAIR_MONOTONIC_WEIGHT="${PROGRESS_PAIR_MONOTONIC_WEIGHT:-0.05}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-fractal_progress_vlm_concat_attention_pool_pairwise_bt_${EXPERIMENT_SUFFIX}}"
export OUTPUT_DIR="${OUTPUT_DIR:-output/progress_vlm_layerwise_pairwise/overfit_control_20260629/${VARIANT}_${EXPERIMENT_SUFFIX}}"

mkdir -p output/logs
LOG_PATH="${LOG_PATH:-output/logs/${EXPERIMENT_NAME}.train.log}"

echo "Running $VARIANT"
echo "Experiment: $EXPERIMENT_NAME"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_PATH"
echo "Effective batch: $((GLOBAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) = ${GLOBAL_BATCH_SIZE} x ${GRADIENT_ACCUMULATION_STEPS}"

bash scripts/train_fractal_progress_head_vlm_layer_pooled.sh 2>&1 | tee "$LOG_PATH"

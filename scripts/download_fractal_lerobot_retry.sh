#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ID="${REPO_ID:-IPEC-COMMUNITY/fractal20220817_data_lerobot}"
LOCAL_DIR="${LOCAL_DIR:-examples/SimplerEnv/fractal20220817_data_lerobot/}"
MAX_WORKERS="${MAX_WORKERS:-4}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-30}"
MAX_RETRIES="${MAX_RETRIES:-0}"
HF_DOWNLOAD_EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts/download_fractal_lerobot_retry.sh [-- <extra hf download args>...]

Environment variables:
  REPO_ID                HuggingFace dataset repo id.
                         Default: IPEC-COMMUNITY/fractal20220817_data_lerobot
  LOCAL_DIR              Local output directory.
                         Default: examples/SimplerEnv/fractal20220817_data_lerobot/
  MAX_WORKERS            Number of parallel download workers.
                         Default: 4
  RETRY_SLEEP_SECONDS    Seconds to wait before retrying after a failed download.
                         Default: 30
  MAX_RETRIES            Maximum number of retries. 0 means retry forever.
                         Default: 0

Examples:
  bash scripts/download_fractal_lerobot_retry.sh
  RETRY_SLEEP_SECONDS=10 MAX_RETRIES=20 bash scripts/download_fractal_lerobot_retry.sh
  bash scripts/download_fractal_lerobot_retry.sh -- --include "meta/*" "data/chunk-000/*"
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            HF_DOWNLOAD_EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$MAX_RETRIES" in
    ''|*[!0-9]*)
        echo "MAX_RETRIES must be a non-negative integer: $MAX_RETRIES" >&2
        exit 1
        ;;
esac

case "$RETRY_SLEEP_SECONDS" in
    ''|*[!0-9]*)
        echo "RETRY_SLEEP_SECONDS must be a non-negative integer: $RETRY_SLEEP_SECONDS" >&2
        exit 1
        ;;
esac

trap 'echo "Received interrupt signal; stopping retry loop."; exit 130' INT TERM

attempt=1
mkdir -p "$LOCAL_DIR"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download attempt ${attempt}"
    echo "Repo: ${REPO_ID}"
    echo "Local dir: ${LOCAL_DIR}"

    if uv run hf download \
        --repo-type dataset "$REPO_ID" \
        --local-dir "$LOCAL_DIR" \
        --max-workers "$MAX_WORKERS" \
        "${HF_DOWNLOAD_EXTRA_ARGS[@]}"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download finished successfully."
        exit 0
    else
        status=$?
    fi

    if [ "$MAX_RETRIES" -gt 0 ] && [ "$attempt" -ge "$MAX_RETRIES" ]; then
        echo "Download failed with exit code ${status}; reached MAX_RETRIES=${MAX_RETRIES}." >&2
        exit "$status"
    fi

    echo "Download failed with exit code ${status}; retrying in ${RETRY_SLEEP_SECONDS}s..." >&2
    sleep "$RETRY_SLEEP_SECONDS"
    attempt=$((attempt + 1))
done

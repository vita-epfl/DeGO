#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/dego_30e_vggt_after20.py}"
CACHE_DIR="${CACHE_DIR:-data/vggt_cache_spatial_temporal_block22}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:--1}"
DEVICE="${DEVICE:-cuda:0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "$CACHE_DIR"

args=(
  generate_vggt_cache.py
  --config "$CONFIG"
  --cache-dir "$CACHE_DIR"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --start-idx "$START_IDX"
  --end-idx "$END_IDX"
  --device "$DEVICE"
  --cache-selected-only
)

if [[ "$SKIP_EXISTING" == "1" ]]; then
  args+=(--skip-existing)
fi

python "${args[@]}"

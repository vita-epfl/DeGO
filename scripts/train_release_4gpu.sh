#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/dego_30e_vggt_after20.py}"
GPUS="${GPUS:-4}"
WORK_DIR="${WORK_DIR:-work_dirs/dego_30e_vggt_after20_release}"
PORT="${PORT:-29500}"
AUTO_RESUME="${AUTO_RESUME:-1}"

args=(
  "$CONFIG"
  "$GPUS"
  --work-dir "$WORK_DIR"
)

if [[ "$AUTO_RESUME" == "1" ]]; then
  args+=(--auto-resume)
fi

PORT="$PORT" bash tools/dist_train.sh "${args[@]}"

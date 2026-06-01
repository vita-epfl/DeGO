#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

VERSION="${VERSION:-v1.0-trainval}"
SPLIT="${SPLIT:-train}"
METHOD="${METHOD:-separate}"
INFO_ROOT="${INFO_ROOT:-data}"
SAVE_ROOT="${SAVE_ROOT:-data/grounded_sam_nusc}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-ckpts/sam_vit_h_4b8939.pth}"
MAX_SIZE="${MAX_SIZE:-800}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OVERWRITE="${OVERWRITE:-0}"
SCENE_PREFIXES="${SCENE_PREFIXES:-}"

args=(
  groundedsam/generate_grounded_sam.py
  --single-gpu
  --version "$VERSION"
  --split "$SPLIT"
  --method "$METHOD"
  --info-root "$INFO_ROOT"
  --save-root "$SAVE_ROOT"
  --sam-checkpoint "$SAM_CHECKPOINT"
  --max-size "$MAX_SIZE"
  --num-workers "$NUM_WORKERS"
)

if [[ "$OVERWRITE" == "1" ]]; then
  args+=(--overwrite)
fi

if [[ -n "$SCENE_PREFIXES" ]]; then
  args+=(--scene-prefixes)
  for prefix in $SCENE_PREFIXES; do
    args+=("$prefix")
  done
fi

python "${args[@]}"

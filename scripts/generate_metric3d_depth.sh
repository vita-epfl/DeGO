#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

ROOT="${ROOT:-data}"
VERSION="${VERSION:-v1.0-trainval}"
SCENE_PREFIX="${SCENE_PREFIX:-scene}"
MODEL="${MODEL:-metric3d_vit_large}"
TARGET_DIR="${TARGET_DIR:-metric_3d_nusc}"
OVERWRITE="${OVERWRITE:-0}"

args=(
  tools/generate_m3d_nusc.py
  --root "$ROOT"
  --version "$VERSION"
  --model "$MODEL"
  --target-dir "$TARGET_DIR"
)

args+=(--scene-prefix)
for prefix in $SCENE_PREFIX; do
  args+=("$prefix")
done

if [[ "$OVERWRITE" == "1" ]]; then
  args+=(--overwrite)
fi

python "${args[@]}"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/dego_30e_vggt_after20.py}"
CHECKPOINT="${CHECKPOINT:-}"
NBH="${NBH:-5}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "CHECKPOINT must point to a .pth file" >&2
  exit 2
fi

python tools/test.py "$CONFIG" "$CHECKPOINT" --eval mIoU --nbh "$NBH"

#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${DEGO_ENV_NAME:-dego_cu113}"
CONDA_SH="${CONDA_SH:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECREATE=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--recreate]

Create the DeGO CUDA 11.3 conda environment and install editable repo code.

Environment variables:
  DEGO_ENV_NAME  Conda environment name. Default: dego_cu113
  CONDA_SH       Optional path to conda.sh if conda is not already on PATH.
  MAX_JOBS       Build parallelism for CUDA extensions. Default: 8
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate)
      RECREATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$CONDA_SH" ]]; then
  if [[ ! -f "$CONDA_SH" ]]; then
    echo "Conda init script not found: $CONDA_SH" >&2
    exit 1
  fi
  source "$CONDA_SH"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Could not find conda. Set CONDA_SH=/path/to/etc/profile.d/conda.sh and rerun." >&2
  exit 1
fi
CONDA_BASE="$(conda info --base)"
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [[ "$RECREATE" -eq 1 ]]; then
    echo "Removing existing conda env: $ENV_NAME"
    conda env remove -n "$ENV_NAME" -y
    if [[ -d "$ENV_PREFIX" ]]; then
      STALE_PREFIX="${ENV_PREFIX}.stale.$(date +%Y%m%d%H%M%S)"
      echo "Existing env prefix still exists after removal: $ENV_PREFIX"
      echo "Moving stale prefix to: $STALE_PREFIX"
      mv "$ENV_PREFIX" "$STALE_PREFIX"
    fi
  else
    echo "Conda env already exists: $ENV_NAME"
    echo "Use --recreate to rebuild it from scratch."
    exit 0
  fi
elif [[ "$RECREATE" -eq 1 && -d "$ENV_PREFIX" ]]; then
  STALE_PREFIX="${ENV_PREFIX}.stale.$(date +%Y%m%d%H%M%S)"
  echo "Found unregistered env prefix: $ENV_PREFIX"
  echo "Moving stale prefix to: $STALE_PREFIX"
  mv "$ENV_PREFIX" "$STALE_PREFIX"
fi

echo "Creating conda env: $ENV_NAME"
conda env create -f "$REPO_ROOT/environment_dego_cu113.yml" -n "$ENV_NAME"

conda activate "$ENV_NAME"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export MAX_JOBS="${MAX_JOBS:-8}"

python -m pip install --upgrade pip==24.2 setuptools==60.2.0 wheel==0.44.0
python -m pip install -r "$REPO_ROOT/requirements/dego_cu113.txt"
# Some transitive packages pull the GUI OpenCV wheel even though DeGO only
# needs headless OpenCV. The CUDA devel image does not ship libGL/libxcb.
python -m pip uninstall -y opencv-python || true
python -m pip install --force-reinstall --no-deps opencv-python-headless==4.8.1.78
python -m pip install -e "$REPO_ROOT" --no-deps

python - <<'PY'
import torch
import torchvision
import mmcv
import mmdet
import mmseg
import numpy as np
import gsplat

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("mmcv", mmcv.__version__)
print("mmdet", mmdet.__version__)
print("mmseg", mmseg.__version__)
print("numpy", np.__version__)
print("gsplat", getattr(gsplat, "__version__", "unknown"))
PY

echo "Environment is ready: $ENV_NAME"

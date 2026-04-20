#!/usr/bin/env bash
set -euo pipefail

# Fast Phase 4 environment install:
# 1. create a minimal conda env with Python
# 2. install PyTorch/cu121 and the remaining dependencies via pip
#
# Usage:
#   bash scripts/install_phase4_env.sh
#   bash scripts/install_phase4_env.sh pathology-phase4
#
# Optional overrides:
#   PYTHON_VERSION=3.11
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
#   TORCH_SPEC="torch torchvision"

ENV_NAME="${1:-pathology-phase4}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_SPEC="${TORCH_SPEC:-torch torchvision}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists."
else
  echo "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"

echo "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing PyTorch and torchvision from ${TORCH_INDEX_URL}..."
python -m pip install --upgrade --index-url "${TORCH_INDEX_URL}" ${TORCH_SPEC}

echo "Installing remaining Phase 4 dependencies..."
python -m pip install --upgrade \
  "numpy>=1.24" \
  "scipy>=1.10" \
  "opencv-python>=4.8" \
  "pillow>=10" \
  "tqdm>=4.66" \
  "tensorboard>=2.15" \
  "pyyaml>=6.0" \
  "typing_extensions>=4.8"

echo
echo "Environment '${ENV_NAME}' is ready."
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Quick check:"
echo "  python - <<'PY'"
echo "  import torch"
echo "  print(torch.__version__)"
echo "  print(torch.cuda.is_available())"
echo "  if torch.cuda.is_available():"
echo "      print(torch.cuda.get_device_name(0))"
echo "  PY"

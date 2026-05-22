#!/usr/bin/env bash
set -euo pipefail

# Stage 4 baseline environment install.
# Creates a conda env with PyTorch/cu121 and the minimal deps for segmentator.
#
# Usage:
#   bash scripts/install_segmentator_stage4_env.sh
#   bash scripts/install_segmentator_stage4_env.sh pathology-segmentator-stage4
#
# Optional overrides:
#   PYTHON_VERSION=3.11
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
#   TORCH_SPEC="torch torchvision"

ENV_NAME="${1:-pathology-segmentator-stage4}"
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

echo "Installing Stage 4 dependencies..."
python -m pip install --upgrade \
  -r segmentator/requirements-stage4.txt \
  "scipy>=1.10" \
  "tensorboard>=2.15" \
  "typing_extensions>=4.8"

echo
echo "Environment '${ENV_NAME}' is ready."
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"

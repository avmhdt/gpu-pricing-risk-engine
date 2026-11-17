#!/usr/bin/env bash
# Rebuild the gpre environment on a fresh cloud GPU instance (NFR-12: sessions stay resumable).
#
# Usage on the instance:
#   curl -LsSf https://raw.githubusercontent.com/avmhdt/gpu-pricing-risk-engine/main/scripts/cloud_setup.sh -o cloud_setup.sh
#   bash cloud_setup.sh [branch]
#
# It installs nothing system-wide except uv (in ~/.local/bin), clones the public repo, installs
# the dev and gpu dependency groups, writes a .env with non-secret values, and runs the checks
# that must pass before any timed run. No keys are needed or written.
set -euo pipefail

BRANCH="${1:-main}"
REPO_URL="https://github.com/avmhdt/gpu-pricing-risk-engine.git"
DIR="$HOME/gpre"
FLAVOR="${GPRE_FLAVOR:-n3-RTX-A4000x1}"

echo "== GPU and driver"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
CUDA_DIR="${CUDA_HOME:-/usr/local/cuda}"
if [ -x "$CUDA_DIR/bin/nvcc" ]; then "$CUDA_DIR/bin/nvcc" --version | tail -2; fi

echo "== uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "== repository ($BRANCH)"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch --quiet origin
  git -C "$DIR" checkout --quiet "$BRANCH"
  git -C "$DIR" pull --quiet --ff-only origin "$BRANCH"
else
  git clone --quiet --branch "$BRANCH" "$REPO_URL" "$DIR"
fi
cd "$DIR"
git log --oneline -1

echo "== python environment (dev + gpu groups)"
uv sync --frozen --group dev --group gpu

echo "== .env (non-secret values only)"
if [ ! -f .env ]; then
  cp .env.example .env
  sed -i "s|^GPRE_CLOUD_PROVIDER=.*|GPRE_CLOUD_PROVIDER=hyperstack/${FLAVOR}|" .env
  sed -i "s|^DATA_ROOT=.*|DATA_ROOT=~/data/gpu-pricing-risk-engine|" .env
fi

echo "== backend checks (must pass before any timed run)"
uv run gpre env --require cuda
uv run gpre env --require gpu
echo "setup complete in $DIR"

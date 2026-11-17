#!/usr/bin/env bash
# P-04: can Nsight Compute read GPU performance counters on this instance?
# Runs a short kernel under ncu, first as the user, then with sudo, and prints a verdict.
# Writes results/profiles/ncu_check_<host>.txt (small text, committed with the session's results).
set -uo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
CUDA_DIR="${CUDA_HOME:-/usr/local/cuda}"
NCU="$(command -v ncu || true)"
[ -z "$NCU" ] && [ -x "$CUDA_DIR/bin/ncu" ] && NCU="$CUDA_DIR/bin/ncu"
OUT="results/profiles/ncu_check_$(hostname).txt"
mkdir -p results/profiles

if [ -z "$NCU" ]; then
  echo "ncu not found (no CUDA toolkit with Nsight Compute on this image)" | tee "$OUT"
  exit 2
fi
PY="$(uv run python -c 'import sys; print(sys.executable)')"
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"

{
  echo "host: $(hostname)  date: $(date -u +%FT%TZ)"
  "$NCU" --version | tail -1
  echo "--- as user"
  "$NCU" --metrics "$METRICS" --kernel-name gbm_european_fused --launch-count 1 \
    "$PY" scripts/ncu_probe.py 2>&1 | tail -15
  echo "--- with sudo"
  sudo -E "$NCU" --metrics "$METRICS" --kernel-name gbm_european_fused --launch-count 1 \
    "$PY" scripts/ncu_probe.py 2>&1 | tail -15
} | tee "$OUT"

if grep -q "ERR_NVGPUCTRPERM" "$OUT" && ! grep -q "dram__throughput" "$OUT"; then
  echo "VERDICT: counters blocked even with sudo; the analytic roofline fallback applies (FR-28)" | tee -a "$OUT"
elif grep -q "dram__throughput" "$OUT"; then
  echo "VERDICT: counters available (see which run above succeeded)" | tee -a "$OUT"
else
  echo "VERDICT: unclear; read the output above" | tee -a "$OUT"
fi

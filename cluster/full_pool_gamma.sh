#!/usr/bin/env bash
set -euo pipefail
# WORKDIR defaults to the repository root (parent of this script's directory).
# Override with `WORKDIR=/your/path cluster/full_pool_gamma.sh` if your
# staged data and FITS images live elsewhere.
WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$WORKDIR"
echo "[start] host=$(hostname) date=$(date)"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tolman}"
mkdir -p "$MPLCONFIGDIR" results/li_ddr
echo "[env] python=$(command -v python3)"

python3 cluster/full_pool_gamma.py
echo "[done] $(date)"

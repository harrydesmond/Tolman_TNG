#!/usr/bin/env bash
set -euo pipefail
WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$WORKDIR"
echo "[start] host=$(hostname) date=$(date)"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tolman-tng}"
mkdir -p "$MPLCONFIGDIR" results/li_ddr/cosmic_variance
echo "[env] python=$(command -v python3)"

python3 cluster/cosmic_variance_subblocks.py
echo "[done] $(date)"

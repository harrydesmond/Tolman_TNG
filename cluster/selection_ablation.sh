#!/usr/bin/env bash
set -euo pipefail
WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$WORKDIR"
echo "[start] host=$(hostname) date=$(date)"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tolman-tng}"
mkdir -p "$MPLCONFIGDIR" results/systematics
echo "[env] python=$(command -v python3)"

python3 cluster/selection_ablation.py
echo "[done] $(date)"

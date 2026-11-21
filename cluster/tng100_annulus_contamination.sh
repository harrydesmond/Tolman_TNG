#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
FIELD=tng100-7-6-xyz_137sqarcmin
SAMPLE=tng100_7_6_xyz_hst_jwst_imageap_annulus
N_WORKERS="${N_WORKERS:-4}"

cd "$WORKDIR"
echo "[start] host=$(hostname) date=$(date)"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tolman-tng}"
mkdir -p "$MPLCONFIGDIR" "results/$SAMPLE" Plots
echo "[env] python=$(command -v python3)"

python3 -m tolman.cli annulus-contamination \
  --catalog data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv \
  --fits-glob "data/raw/illustris/$FIELD/*.fits" \
  --out-dir "results/$SAMPLE" \
  --plot-dir Plots \
  --sample-name "$SAMPLE" \
  --n-workers "$N_WORKERS"

cat "results/$SAMPLE/summary.md"
echo "[done] $(date)"

#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
FIELD=tng100-7-6-xyz_137sqarcmin
SAMPLE=tng100_7_6_xyz_hst_jwst_imageap_sweep

cd "$WORKDIR"
echo "[start] host=$(hostname) date=$(date)"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tolman-tng}"
mkdir -p "$MPLCONFIGDIR" "results/$SAMPLE" Plots
echo "[env] python=$(command -v python3)"

echo "[analysis] aperture radius sweep"
python3 -m tolman.cli aperture-sweep \
  --catalog data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv \
  --fits-glob "data/raw/illustris/$FIELD/*.fits" \
  --out-dir "results/$SAMPLE" \
  --plot-dir Plots \
  --sample-name "$SAMPLE" \
  --scale-factors 0.5 1.0 1.5 2.0 3.0 4.0 6.0

cat "results/$SAMPLE/summary.md"
echo "[done] $(date)"

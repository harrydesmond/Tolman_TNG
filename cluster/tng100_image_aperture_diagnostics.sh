#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SAMPLE=tng100_7_6_xyz_hst_jwst_imageap_shift

cd "$WORKDIR"
echo "[start] host=$(hostname) date=$(date)"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tolman-tng}"
mkdir -p "$MPLCONFIGDIR" "results/$SAMPLE" Plots
echo "[env] python=$(command -v python3)"

echo "[analysis] decompose catalogue-vs-image-aperture slope shift"
python3 -m tolman.cli image-aperture-diagnostics \
  --catalogue-measured "results/tng100_7_6_xyz_hst_jwst_specmodel/measured_catalog.csv" \
  --image-measured "results/tng100_7_6_xyz_hst_jwst_imageap_specmodel/measured_catalog.csv" \
  --out-dir "results/$SAMPLE" \
  --plot-dir Plots \
  --sample-name "$SAMPLE"

echo "[analysis] summary"
cat "results/$SAMPLE/summary.md"

echo "[done] $(date)"

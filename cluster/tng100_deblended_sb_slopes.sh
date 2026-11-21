#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SAMPLE=tng100_7_6_xyz_hst_jwst_imageap_deblended_sb

cd "$WORKDIR"
echo "[start] host=$(hostname) date=$(date)"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tolman-tng}"
mkdir -p "$MPLCONFIGDIR" "results/$SAMPLE" Plots
echo "[env] python=$(command -v python3)"

echo "[analysis] Voronoi deblended SB slopes"
python3 -m tolman.cli deblended-tng-slopes \
  --catalog data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv \
  --per-filter-csv results/tng100_7_6_xyz_hst_jwst_imageap_deblended/deblending_per_filter.csv \
  --config configs/tng100_7_6_xyz_hst_jwst_imageap.toml \
  --out-dir "results/$SAMPLE" \
  --plot-dir Plots \
  --sample-name "$SAMPLE"

cat "results/$SAMPLE/slope_table.csv"
echo "[done] $(date)"

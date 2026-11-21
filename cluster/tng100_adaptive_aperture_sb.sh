#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
FIELD=tng100-7-6-xyz_137sqarcmin
SAMPLE=tng100_7_6_xyz_hst_jwst_imageap_astrodeep_aperture

cd "$WORKDIR"
echo "[start] host=$(hostname) date=$(date)"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tolman-tng}"
mkdir -p "$MPLCONFIGDIR" "results/$SAMPLE" Plots
echo "[env] python=$(command -v python3)"

echo "[analysis] TNG adaptive aperture SB (ASTRODEEP-equivalent)"
python3 -m tolman.cli adaptive-aperture-sb \
  --catalog data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv \
  --fits-glob "data/raw/illustris/$FIELD/*.fits" \
  --config configs/tng100_7_6_xyz_hst_jwst_imageap.toml \
  --out-dir "results/$SAMPLE" \
  --plot-dir Plots \
  --sample-name "$SAMPLE"

cat "results/$SAMPLE/slope_table.csv"
echo "[done] $(date)"

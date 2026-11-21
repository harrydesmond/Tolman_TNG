#!/usr/bin/env bash
set -euo pipefail

# Reproduce the main Tolman/TNG paper products from staged data.
#
# This script documents the intended command sequence. Some steps require the
# full TNG100 FITS image set, which is too large for the GitHub repository and
# was run on a compute cluster (see README.md
# "Cluster Resource Requirements" for indicative cores/memory/time).

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tolman-tng}"
mkdir -p "$MPLCONFIGDIR" data results Plots

ASTRO_CDS_DIR="${ASTRO_CDS_DIR:-data/raw/astrodeep/cds}"
TNG100_FITS_GLOB="${TNG100_FITS_GLOB:-data/raw/illustris/tng100-7-6-xyz_137sqarcmin/*.fits}"

echo "1. ASTRODEEP/CDS normalization and catalogue reproduction"
echo "   Expected CDS tables: ${ASTRO_CDS_DIR}/*.tsv"
echo "   The field-level normalization loop is intentionally explicit in README.md."

python3 -m tolman.cli measure \
  --catalog data/astrodeep_cds_all.csv \
  --config configs/astrodeep_cds_tng_common.toml \
  --result-dir results/astrodeep_cds_tng_common \
  --plot-dir Plots \
  --sample-name astrodeep_cds_tng_common

echo "2. Fit empirical ASTRODEEP spectroscopic-selection model"
python3 -m tolman.cli fit-spec-selection \
  --catalog results/astrodeep_cds/measured_catalog.csv \
  --out results/selection/astrodeep_spec_selection.joblib

echo "3. TNG100 catalogue-flux and image-aperture products"
echo "   Requires staged TNG100 FITS images matching: ${TNG100_FITS_GLOB}"

python3 -m tolman.cli mast-normalize \
  --fits-glob "${TNG100_FITS_GLOB}" \
  --out data/processed_tng100_7_6_xyz_hst_jwst.csv \
  --config-out configs/tng100_7_6_xyz_hst_jwst.toml \
  --sample-name tng100_7_6_xyz

python3 -m tolman.cli apply-spec-selection \
  --catalog data/processed_tng100_7_6_xyz_hst_jwst.csv \
  --model results/selection/astrodeep_spec_selection.joblib \
  --mode deterministic \
  --out data/processed_tng100_7_6_xyz_hst_jwst_specmodel.csv

python3 -m tolman.cli measure \
  --catalog data/processed_tng100_7_6_xyz_hst_jwst_specmodel.csv \
  --config configs/tng100_7_6_xyz_hst_jwst.toml \
  --result-dir results/tng100_7_6_xyz_hst_jwst_specmodel \
  --plot-dir Plots \
  --sample-name tng100_7_6_xyz_hst_jwst_specmodel

python3 -m tolman.cli tng-image-aperture \
  --fits-glob "${TNG100_FITS_GLOB}" \
  --out data/processed_tng100_7_6_xyz_hst_jwst_imageap.csv \
  --config-out configs/tng100_7_6_xyz_hst_jwst_imageap.toml \
  --diagnostics-out results/tng100_7_6_xyz_hst_jwst_imageap/image_diagnostics.csv \
  --sample-name tng100_7_6_xyz \
  --n-workers 4

python3 -m tolman.cli apply-spec-selection \
  --catalog data/processed_tng100_7_6_xyz_hst_jwst_imageap.csv \
  --model results/selection/astrodeep_spec_selection.joblib \
  --mode deterministic \
  --out data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv

python3 -m tolman.cli measure \
  --catalog data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv \
  --config configs/tng100_7_6_xyz_hst_jwst_imageap.toml \
  --result-dir results/tng100_7_6_xyz_hst_jwst_imageap_specmodel \
  --plot-dir Plots \
  --sample-name tng100_7_6_xyz_hst_jwst_imageap_specmodel

echo "4. Main systematics diagnostics"
python3 -m tolman.cli image-aperture-diagnostics \
  --catalogue-measured results/tng100_7_6_xyz_hst_jwst_specmodel/measured_catalog.csv \
  --image-measured results/tng100_7_6_xyz_hst_jwst_imageap_specmodel/measured_catalog.csv \
  --out-dir results/tng100_7_6_xyz_hst_jwst_imageap_shift \
  --plot-dir Plots \
  --sample-name tng100_7_6_xyz_hst_jwst_imageap_shift

python3 -m tolman.cli deblending-diagnostics \
  --catalog data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv \
  --fits-glob "${TNG100_FITS_GLOB}" \
  --out-dir results/tng100_7_6_xyz_hst_jwst_imageap_deblended \
  --plot-dir Plots \
  --sample-name tng100_7_6_xyz_hst_jwst_imageap_deblended

python3 -m tolman.cli deblended-tng-slopes \
  --catalog data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv \
  --per-filter-csv results/tng100_7_6_xyz_hst_jwst_imageap_deblended/deblending_per_filter.csv \
  --config configs/tng100_7_6_xyz_hst_jwst_imageap.toml \
  --out-dir results/tng100_7_6_xyz_hst_jwst_imageap_deblended_sb \
  --plot-dir Plots \
  --sample-name tng100_7_6_xyz_hst_jwst_imageap_deblended_sb

python3 -m tolman.cli adaptive-aperture-sb \
  --catalog data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv \
  --fits-glob "${TNG100_FITS_GLOB}" \
  --config configs/tng100_7_6_xyz_hst_jwst_imageap.toml \
  --out-dir results/tng100_7_6_xyz_hst_jwst_imageap_astrodeep_aperture \
  --plot-dir Plots \
  --sample-name tng100_7_6_xyz_hst_jwst_imageap_astrodeep_aperture

echo "Done. Compare slope tables under results/ and plots under Plots/."

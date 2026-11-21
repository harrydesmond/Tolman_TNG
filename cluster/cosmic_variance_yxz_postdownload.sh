#!/usr/bin/env bash
# Post-download stage of the yxz cosmic-variance run.
#
# Runs the analysis stages of cosmic_variance_yxz.sh on FITS images that are
# already staged under data/raw/illustris/, without downloading them again.

set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
FIELD=tng100-7-6-yxz_137sqarcmin
RAW_DIR="$WORKDIR/data/raw/illustris/$FIELD"
SAMPLE_CAT=tng100_7_6_yxz_hst_jwst                 # catalogue-flux pipeline (variant i)
SAMPLE_IMG=tng100_7_6_yxz_hst_jwst_imageap         # image-aperture pipeline (variant ii)
N_WORKERS="${N_WORKERS:-4}"
MODEL_PATH="$WORKDIR/results/selection/astrodeep_spec_selection.joblib"

cd "$WORKDIR"
echo "[start] host=$(hostname) date=$(date)"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tolman-tng}"
mkdir -p "$MPLCONFIGDIR" "results/$SAMPLE_CAT" "results/${SAMPLE_CAT}_specmodel" \
         "results/$SAMPLE_IMG" "results/${SAMPLE_IMG}_specmodel" \
         "results/li_ddr/cosmic_variance" Plots
echo "[env] python=$(command -v python3)"

echo "[check] FITS files present in $RAW_DIR"
test "$(find "$RAW_DIR" -maxdepth 1 -name '*.fits' | wc -l)" -eq 13

# ---------- Catalogue-flux pipeline (variant i) ----------
echo "[analysis] mast-normalize (catalogue flux)"
python3 -m tolman.cli mast-normalize \
  --fits-glob "data/raw/illustris/$FIELD/*.fits" \
  --sample-name "$SAMPLE_CAT" \
  --out "data/processed_${SAMPLE_CAT}.csv" \
  --config-out "configs/${SAMPLE_CAT}.toml"

echo "[analysis] apply spec-selection (catalogue flux)"
python3 -m tolman.cli apply-spec-selection \
  --catalog "data/processed_${SAMPLE_CAT}.csv" \
  --model "$MODEL_PATH" \
  --mode deterministic \
  --out "data/processed_${SAMPLE_CAT}_specmodel.csv"

echo "[analysis] measure (catalogue flux)"
python3 -m tolman.cli measure \
  --catalog "data/processed_${SAMPLE_CAT}_specmodel.csv" \
  --config "configs/${SAMPLE_CAT}.toml" \
  --result-dir "results/${SAMPLE_CAT}_specmodel" \
  --plot-dir Plots \
  --sample-name "${SAMPLE_CAT}_specmodel" \
  --save-measured-catalog

# ---------- Image-aperture pipeline (variant ii) ----------
echo "[analysis] tng-image-aperture (image-aperture flux)"
python3 -m tolman.cli tng-image-aperture \
  --fits-glob "data/raw/illustris/$FIELD/*.fits" \
  --sample-name "$SAMPLE_IMG" \
  --n-workers "$N_WORKERS" \
  --out "data/processed_${SAMPLE_IMG}.csv" \
  --config-out "configs/${SAMPLE_IMG}.toml" \
  --diagnostics-out "results/${SAMPLE_IMG}/image_diagnostics.csv"

echo "[analysis] apply spec-selection (image aperture)"
python3 -m tolman.cli apply-spec-selection \
  --catalog "data/processed_${SAMPLE_IMG}.csv" \
  --model "$MODEL_PATH" \
  --mode deterministic \
  --out "data/processed_${SAMPLE_IMG}_specmodel.csv"

echo "[analysis] measure (image aperture)"
python3 -m tolman.cli measure \
  --catalog "data/processed_${SAMPLE_IMG}_specmodel.csv" \
  --config "configs/${SAMPLE_IMG}.toml" \
  --result-dir "results/${SAMPLE_IMG}_specmodel" \
  --plot-dir Plots \
  --sample-name "${SAMPLE_IMG}_specmodel" \
  --save-measured-catalog

# ---------- gamma_rho on volume-controlled (intrinsic photrad_kpc) ----------
echo "[gamma] volume-controlled gamma_rho on yxz lightcone"
python3 - <<PY
from pathlib import Path
import numpy as np
import pandas as pd

from tolman.li_ddr.li_reproduction import COSMO_LI, luminosity_at_band

ROOT = Path("${WORKDIR}")
FILTER_NAME = "F200W"
ALPHA_SPEC = -0.3  # Li convention; equivalent to paper alpha_SED = 0.3.
N_BOOT = 200
RNG = np.random.default_rng(20260509)

VARIANTS = [
    ("specmodel (catalogue flux)",
     ROOT / "results/${SAMPLE_CAT}_specmodel/measured_catalog.csv"),
    ("imageap_specmodel (canonical)",
     ROOT / "results/${SAMPLE_IMG}_specmodel/measured_catalog.csv"),
]

def _spec_select(df):
    keep = df["mock_spec_selected"].astype(bool)
    if "image_success_F200W" in df.columns:
        keep &= df["image_success_F200W"].astype(bool)
    keep &= df[f"flux_{FILTER_NAME}_uJy"] > 0
    keep &= df[f"aperture_radius_arcsec_{FILTER_NAME}"] > 0
    return df.loc[keep].copy()

def _gamma(z, log_rho):
    finite = np.isfinite(z) & np.isfinite(log_rho) & (z > 0)
    if finite.sum() < 10:
        return np.nan
    g, _ = np.polyfit(np.log10(1.0 + z[finite]), log_rho[finite], 1)
    return float(g)

def _bootstrap(z, log_rho):
    n = len(z)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        boots[i] = _gamma(z[idx], log_rho[idx])
    return float(np.mean(boots)), float(np.std(boots, ddof=1))

rows = []
for label, path in VARIANTS:
    if not path.exists():
        print(f"!! {label} skipped: {path} missing")
        continue
    df = pd.read_csv(path, low_memory=False)
    df = _spec_select(df)
    z = df["z"].to_numpy(dtype=float)
    flux_jy = df[f"flux_{FILTER_NAME}_uJy"].to_numpy(dtype=float) * 1e-6
    photrad_kpc = (df["aperture_radius_arcsec_F200W"].to_numpy(dtype=float)
                   * df["kpc_per_arcsec"].to_numpy(dtype=float))
    L = luminosity_at_band(flux_jy, z, np.full_like(z, ALPHA_SPEC), cosmo=COSMO_LI)
    r_cm = photrad_kpc * 3.0857e21
    rho_L = L / ((4.0/3.0) * np.pi * r_cm**3)
    finite = np.isfinite(rho_L) & (rho_L > 0) & np.isfinite(z) & (z > 0)
    z_f = z[finite]; lr = np.log10(rho_L[finite])
    g0 = _gamma(z_f, lr)
    gb, gbs = _bootstrap(z_f, lr)
    rows.append(dict(lightcone="tng100-7-6-yxz", variant=label, n=int(finite.sum()),
                     gamma=g0, gamma_boot_mean=gb, gamma_boot_std=gbs,
                     alpha_spec_li=ALPHA_SPEC, alpha_sed_paper=-ALPHA_SPEC))
    print(f"{label:35s} n={finite.sum():5d} gamma={g0:+.3f} (boot={gb:+.3f}+/-{gbs:.3f})")

table = pd.DataFrame(rows)
out = ROOT / "results/li_ddr/cosmic_variance/yxz_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(out, index=False)
print(f"\nWrote -> {out}")
PY
echo "[done] $(date)"

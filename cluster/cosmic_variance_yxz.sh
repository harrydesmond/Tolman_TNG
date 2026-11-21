#!/usr/bin/env bash
# Cosmic-variance probe via the TNG100 yxz orientation lightcone.
#
# MAST hosts only two TNG100 137 arcmin^2 lightcones:
#   tng100-7-6-xyz (canonical, used in the paper)
#   tng100-7-6-yxz (same matter cube, different LOS axis ordering)
# The two share an underlying simulation volume but the LOS structure
# integrated along the LOS differs (different filaments at different z).
# Delta gamma_yxz - gamma_xyz is therefore a partial cosmic-variance
# proxy that bounds the LOS-projection contribution.
#
# Resources: ~4 cores, ~32 GB memory total (8 GB/core), full end-to-end
# wall time ~12-24 h including the ~24 GiB FITS download.
# IMPORTANT: submit as a single Python process, not one copy per core --
# the tng-image-aperture step uses a ProcessPoolExecutor internally, so
# per-core script replication spawns nested worker pools and OOM-kills
# the job at the FITS-pixel stage.

set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
FIELD=tng100-7-6-yxz_137sqarcmin
TAG=tng100-7-6-yxz
SAMPLE_CAT=tng100_7_6_yxz_hst_jwst                 # catalogue-flux pipeline (variant i)
SAMPLE_IMG=tng100_7_6_yxz_hst_jwst_imageap         # image-aperture pipeline (variant ii)
RAW_DIR="$WORKDIR/data/raw/illustris/$FIELD"
LOG_DIR="$WORKDIR/logs"
MAX_DOWNLOADS="${MAX_DOWNLOADS:-4}"
N_WORKERS="${N_WORKERS:-4}"
MODEL_PATH="$WORKDIR/results/selection/astrodeep_spec_selection.joblib"

mkdir -p "$RAW_DIR" "$WORKDIR/data" "$WORKDIR/configs" \
         "$WORKDIR/results/$SAMPLE_CAT" "$WORKDIR/results/${SAMPLE_CAT}_specmodel" \
         "$WORKDIR/results/$SAMPLE_IMG" "$WORKDIR/results/${SAMPLE_IMG}_specmodel" \
         "$WORKDIR/results/li_ddr/cosmic_variance" "$WORKDIR/Plots" "$LOG_DIR"
cd "$WORKDIR"

echo "[start] host=$(hostname) date=$(date)"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tolman-tng}"
mkdir -p "$MPLCONFIGDIR"
echo "[env] python=$(command -v python3)"

# ---------- Phase 1: download all 13 FITS (24 GiB total) ----------
download_one() {
  local url="$1"
  local out="$2"
  local tmp="${out}.part"
  if [[ -s "$out" ]]; then
    echo "[download] exists $out"
    return 0
  fi
  echo "[download] start $out"
  python3 - "$url" "$tmp" <<'PY'
from pathlib import Path
import sys, time, urllib.error, urllib.request

url = sys.argv[1]
path = Path(sys.argv[2])
existing = path.stat().st_size if path.exists() else 0

for attempt in range(1, 6):
    headers = {}
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status = getattr(response, "status", response.getcode())
            if existing > 0 and status != 206:
                mode = "wb"; existing = 0
            downloaded = existing
            next_report = downloaded + 256 * 1024 * 1024
            with path.open(mode) as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        print(f"[download-python] {path.name}: {downloaded/1024**3:.2f} GiB", flush=True)
                        next_report = downloaded + 256 * 1024 * 1024
        break
    except Exception as exc:
        if attempt == 5:
            raise
        print(f"[download-python] retry {attempt} after {exc!r}", flush=True)
        time.sleep(30 * attempt)
        existing = path.stat().st_size if path.exists() else 0
PY
  mv "$tmp" "$out"
  echo "[download] done $out"
}

base="https://archive.stsci.edu/hlsps/illustris/${FIELD}"
files=(
  "hlsp_illustris_hst_acs_${TAG}_f435w_v1_sim.fits"
  "hlsp_illustris_hst_acs_${TAG}_f606w_v1_sim.fits"
  "hlsp_illustris_hst_acs_${TAG}_f814w_v1_sim.fits"
  "hlsp_illustris_hst_wfc3_${TAG}_f125w_v1_sim.fits"
  "hlsp_illustris_hst_wfc3_${TAG}_f140w_v1_sim.fits"
  "hlsp_illustris_hst_wfc3_${TAG}_f160w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_${TAG}_f090w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_${TAG}_f115w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_${TAG}_f150w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_${TAG}_f200w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_${TAG}_f277w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_${TAG}_f356w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_${TAG}_f444w_v1_sim.fits"
)

for fname in "${files[@]}"; do
  download_one "${base}/${fname}" "$RAW_DIR/$fname" &
  while (( "$(jobs -rp | wc -l)" >= MAX_DOWNLOADS )); do
    wait -n
  done
done
wait

echo "[check] downloaded FITS files"
find "$RAW_DIR" -maxdepth 1 -name '*.fits' -printf '%f %s\n' | sort
test "$(find "$RAW_DIR" -maxdepth 1 -name '*.fits' | wc -l)" -eq 13

# ---------- Phase 2: catalogue-flux pipeline (variant i) ----------
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

# ---------- Phase 3: image-aperture pipeline (variant ii) ----------
echo "[analysis] tng-image-aperture (image-aperture flux), N_WORKERS=$N_WORKERS"
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

# ---------- Phase 4: volume-controlled gamma_rho ----------
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

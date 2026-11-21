#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
FIELD=tng100-7-6-xyz_137sqarcmin
SAMPLE=tng100_7_6_xyz_hst_jwst
RAW_DIR="$WORKDIR/data/raw/illustris/$FIELD"
LOG_DIR="$WORKDIR/logs"
MAX_DOWNLOADS="${MAX_DOWNLOADS:-4}"

mkdir -p "$RAW_DIR" "$WORKDIR/data" "$WORKDIR/configs" "$WORKDIR/results/$SAMPLE" "$WORKDIR/Plots" "$LOG_DIR"
cd "$WORKDIR"

echo "[start] host=$(hostname) date=$(date)"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tolman-tng}"
mkdir -p "$MPLCONFIGDIR"
echo "[env] python=$(command -v python3)"

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
import shutil
import sys
import time
import urllib.error
import urllib.request

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
                mode = "wb"
                existing = 0
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
                        print(f"[download-python] {path.name}: {downloaded / 1024**3:.2f} GiB", flush=True)
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

files=(
  "hlsp_illustris_hst_acs_tng100-7-6-xyz_f435w_v1_sim.fits https://archive.stsci.edu/hlsps/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_hst_acs_tng100-7-6-xyz_f435w_v1_sim.fits"
  "hlsp_illustris_hst_acs_tng100-7-6-xyz_f606w_v1_sim.fits https://archive.stsci.edu/hlsps/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_hst_acs_tng100-7-6-xyz_f606w_v1_sim.fits"
  "hlsp_illustris_hst_acs_tng100-7-6-xyz_f814w_v1_sim.fits https://archive.stsci.edu/hlsps/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_hst_acs_tng100-7-6-xyz_f814w_v1_sim.fits"
  "hlsp_illustris_hst_wfc3_tng100-7-6-xyz_f125w_v1_sim.fits https://archive.stsci.edu/hlsps/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_hst_wfc3_tng100-7-6-xyz_f125w_v1_sim.fits"
  "hlsp_illustris_hst_wfc3_tng100-7-6-xyz_f140w_v1_sim.fits https://archive.stsci.edu/hlsps/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_hst_wfc3_tng100-7-6-xyz_f140w_v1_sim.fits"
  "hlsp_illustris_hst_wfc3_tng100-7-6-xyz_f160w_v1_sim.fits https://archive.stsci.edu/hlsps/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_hst_wfc3_tng100-7-6-xyz_f160w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f090w_v1_sim.fits https://archive.stsci.edu/hlsps/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f090w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f115w_v1_sim.fits https://archive.stsci.edu/hlsps/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f115w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f150w_v1_sim.fits https://archive.stsci.edu/hlsps/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f150w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f200w_v1_sim.fits https://archive.stsci.edu/hlsps/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f200w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f277w_v1_sim.fits https://archive.stsci.edu/hlsps/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f277w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f356w_v1_sim.fits https://archive.stsci.edu/hlsps/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f356w_v1_sim.fits"
  "hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f444w_v1_sim.fits https://archive.stsci.edu/hlsps/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f444w_v1_sim.fits"
)

for entry in "${files[@]}"; do
  out="${entry%% *}"
  url="${entry#* }"
  download_one "$url" "$RAW_DIR/$out" &
  while (( "$(jobs -rp | wc -l)" >= MAX_DOWNLOADS )); do
    wait -n
  done
done
wait

echo "[check] downloaded FITS files"
find "$RAW_DIR" -maxdepth 1 -name '*.fits' -printf '%f %s\n' | sort
test "$(find "$RAW_DIR" -maxdepth 1 -name '*.fits' | wc -l)" -eq 13

echo "[analysis] normalize TNG100"
python3 -m tolman.cli mast-normalize \
  --fits-glob "data/raw/illustris/$FIELD/*.fits" \
  --sample-name "$SAMPLE" \
  --out "data/processed_${SAMPLE}.csv" \
  --config-out "configs/${SAMPLE}.toml"

echo "[analysis] apply ASTRODEEP spectroscopic selection model"
python3 -m tolman.cli apply-spec-selection \
  --catalog "data/processed_${SAMPLE}.csv" \
  --model "results/selection/astrodeep_spec_selection.joblib" \
  --mode deterministic \
  --out "data/processed_${SAMPLE}_specmodel.csv"

python3 - <<'PY'
import pandas as pd
d = pd.read_csv("data/processed_tng100_7_6_xyz_hst_jwst_specmodel.csv")
print("[analysis] rows", len(d), "selected", int(d["mock_spec_selected"].sum()), "sum_p", float(d["p_spec"].sum()))
print(d["p_spec"].describe().to_string())
PY

echo "[analysis] measure hard F200W<29 proxy"
python3 -m tolman.cli measure \
  --catalog "data/processed_${SAMPLE}.csv" \
  --config "configs/${SAMPLE}.toml" \
  --result-dir "results/${SAMPLE}" \
  --plot-dir Plots \
  --sample-name "$SAMPLE" \
  --save-measured-catalog

echo "[analysis] measure empirical spectroscopic selection"
python3 -m tolman.cli measure \
  --catalog "data/processed_${SAMPLE}_specmodel.csv" \
  --config "configs/${SAMPLE}.toml" \
  --result-dir "results/${SAMPLE}_specmodel" \
  --plot-dir Plots \
  --sample-name "${SAMPLE}_specmodel" \
  --save-measured-catalog

echo "[analysis] final slope summaries"
python3 - <<'PY'
import pandas as pd
for name in [
    "results/tng100_7_6_xyz_hst_jwst/slope_table.csv",
    "results/tng100_7_6_xyz_hst_jwst_specmodel/slope_table.csv",
]:
    d = pd.read_csv(name)
    print("\n==", name, "==")
    print(d[d["selection"].isin(["mock_spectroscopic", "spectroscopic"])][
        ["observable", "selection", "alpha", "alpha_err", "n", "scatter"]
    ].to_string(index=False))
PY

echo "[done] $(date)"

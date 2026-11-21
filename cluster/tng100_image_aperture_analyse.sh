#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
FIELD=tng100-7-6-xyz_137sqarcmin
SAMPLE=tng100_7_6_xyz_hst_jwst_imageap
N_WORKERS="${N_WORKERS:-4}"

cd "$WORKDIR"
echo "[start] host=$(hostname) date=$(date)"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-tolman-tng}"
mkdir -p "$MPLCONFIGDIR" "results/$SAMPLE" Plots
echo "[env] python=$(command -v python3)"

echo "[analysis] image-aperture normalize TNG100 with $N_WORKERS workers"
python3 -m tolman.cli tng-image-aperture \
  --fits-glob "data/raw/illustris/$FIELD/*.fits" \
  --sample-name "$SAMPLE" \
  --n-workers "$N_WORKERS" \
  --out "data/processed_${SAMPLE}.csv" \
  --config-out "configs/${SAMPLE}.toml" \
  --diagnostics-out "results/${SAMPLE}/image_diagnostics.csv"

echo "[analysis] apply ASTRODEEP spectroscopic selection model"
python3 -m tolman.cli apply-spec-selection \
  --catalog "data/processed_${SAMPLE}.csv" \
  --model "results/selection/astrodeep_spec_selection.joblib" \
  --mode deterministic \
  --out "data/processed_${SAMPLE}_specmodel.csv"

python3 - <<'PY'
import pandas as pd
d = pd.read_csv("data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv")
print("[analysis] rows", len(d), "selected", int(d["mock_spec_selected"].sum()), "sum_p", float(d["p_spec"].sum()))
print(d["p_spec"].describe().to_string())
PY

echo "[analysis] measure empirical spectroscopic image-aperture sample"
python3 -m tolman.cli measure \
  --catalog "data/processed_${SAMPLE}_specmodel.csv" \
  --config "configs/${SAMPLE}.toml" \
  --result-dir "results/${SAMPLE}_specmodel" \
  --plot-dir Plots \
  --sample-name "${SAMPLE}_specmodel" \
  --save-measured-catalog

echo "[analysis] compare to filter-matched ASTRODEEP if available"
if [[ -s "results/astrodeep_cds_tng_common/slope_table.csv" ]]; then
  python3 -m tolman.cli compare-slopes \
    --observed "results/astrodeep_cds_tng_common/slope_table.csv" \
    --mock "results/${SAMPLE}_specmodel/slope_table.csv" \
    --out "results/comparisons/astrodeep_tngcommon_vs_${SAMPLE}_specmodel_slopes.csv" \
    --plot "Plots/astrodeep_tngcommon_vs_${SAMPLE}_specmodel_slopes.png" \
    --report "results/comparisons/astrodeep_tngcommon_vs_${SAMPLE}_specmodel_slopes.md" \
    --observed-label ASTRODEEP_13_common_filters \
    --mock-label TNG100_image_aperture_empirical_spec
fi

echo "[analysis] final slope summary"
python3 - <<'PY'
import pandas as pd
for name in [
    "results/tng100_7_6_xyz_hst_jwst_imageap/image_diagnostics.csv",
    "results/tng100_7_6_xyz_hst_jwst_imageap_specmodel/slope_table.csv",
]:
    print("\n==", name, "==")
    d = pd.read_csv(name)
    if "selection" in d:
        print(d[d["selection"].isin(["mock_spectroscopic", "spectroscopic"])][
            ["observable", "selection", "alpha", "alpha_err", "n", "scatter"]
        ].to_string(index=False))
    else:
        print(d.to_string(index=False))
PY

echo "[done] $(date)"

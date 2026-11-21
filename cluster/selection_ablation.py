"""Selection-model ablation: retrain without log_sb_mean (SB) feature.

Tests whether the +0.25 residual deblended/catalogue ratio slope and the
TNG-ASTRODEEP mean-SB gap depend on whether the empirical spectroscopic-selection
model has access to a SB-correlated feature. The original DEFAULT_FEATURES
include `log_sb_mean`, which is itself derived from $F_\\nu/A_{\\rm ap}$ -- the
slope is fit to this same quantity, raising a circularity concern.

We retrain on ASTRODEEP with the SB feature dropped (and `log_aperture_area`,
which is correlated with SB at fixed flux), reapply to TNG100, and recompute
the deblended SB slope on the new selection.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

WORKDIR = os.environ.get("TOLMAN_WORKDIR", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, WORKDIR)

from tolman.spec_selection import (
    add_spec_features,
    fit_model,
    save_model,
    add_selection_probabilities,
    select_by_probability,
)

ASTRODEEP_CSV = f"{WORKDIR}/results/astrodeep_cds/measured_catalog.csv"
TNG_IMAGEAP_CSV = f"{WORKDIR}/data/processed_tng100_7_6_xyz_hst_jwst_imageap.csv"
PER_FILTER_CSV = f"{WORKDIR}/results/tng100_7_6_xyz_hst_jwst_imageap_deblended/deblending_per_filter.csv"
OUT_DIR = f"{WORKDIR}/results/systematics"
os.makedirs(OUT_DIR, exist_ok=True)

# Reduced feature set: drop log_sb_mean (the directly-circular feature) and
# log_aperture_area (correlated with SB at fixed flux). Retain magnitudes,
# colours, redshift proxy, and number of valid filters.
ABLATED_FEATURES = [
    "mag_F200W",
    "mag_F356W",
    "color_F150W_F356W",
    "color_F200W_F444W",
    "color_F115W_F200W",
    "z_proxy",
    "n_valid_filters",
]


def fit_alpha(z, sb, w=None):
    z = np.asarray(z, dtype=float); sb = np.asarray(sb, dtype=float)
    if w is None:
        w = np.ones_like(z, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
    m = np.isfinite(z) & np.isfinite(sb) & np.isfinite(w) & (z > -1) & (sb > 0) & (w > 0)
    n = int(m.sum())
    if n < 3:
        return np.nan, np.nan, n
    x = np.log10(1.0 + z[m]); y = np.log10(sb[m]); ww = w[m]
    A = np.c_[np.ones(n), x]
    AT_W = A.T * ww
    normal = AT_W @ A
    c = np.linalg.solve(normal, AT_W @ y)
    resid = y - A @ c
    s2 = float(np.sum(ww * resid ** 2) / (n - 2))
    err = float((s2 * np.linalg.inv(normal)[1, 1]) ** 0.5)
    return float(c[1]), err, n


# ── Train ablated model on ASTRODEEP ──────────────────────────────────────────
print("[load] ASTRODEEP measured catalog ...", flush=True)
ast = pd.read_csv(ASTRODEEP_CSV, low_memory=False)
print(f"  rows: {len(ast)}", flush=True)

# Filter to objects that pass the same selection ladder used for the original training
z = pd.to_numeric(ast["z"], errors="coerce")
nv = pd.to_numeric(ast["n_valid_filters"], errors="coerce").fillna(0)
flags = pd.to_numeric(ast["flags"], errors="coerce").fillna(0)
ap = pd.to_numeric(ast["aperture_area_arcsec2_used"], errors="coerce").fillna(0)
snr_cols = [c for c in ast.columns if c.startswith("snr_")]
snr = ast[snr_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
mask = (
    np.isfinite(z) & (z > 0) & (z <= 14)
    & (nv > 0)
    & (np.nanmax(snr, axis=1) >= 5)
    & (flags < 80)
    & (ap > 0)
)
ast_train = ast[mask.to_numpy()].copy()
print(f"  training set after quality cuts: {len(ast_train)} (target: 'has_spec' True/False)")
print(f"  has_spec True: {ast_train['has_spec'].astype(bool).sum()}", flush=True)

# Fit ablated model
print(f"[train] features: {ABLATED_FEATURES}", flush=True)
model, used = fit_model(
    ast_train,
    target_col="has_spec",
    features=ABLATED_FEATURES,
    class_weight=None,  # matches the original "unbalanced" training so selected count is comparable
)
print(f"  used features: {used}", flush=True)

# ROC AUC on the training data (we don't have a held-out split here, but get a number)
feat_df = add_spec_features(ast_train)
y_true = ast_train["has_spec"].astype(bool).astype(int).to_numpy()
y_proba = model.predict_proba(feat_df[used])[:, 1]
auc = roc_auc_score(y_true, y_proba)
print(f"  in-sample ROC AUC: {auc:.3f}", flush=True)

save_model(model, used, f"{OUT_DIR}/astrodeep_spec_selection_ablated.joblib")
print(f"  saved model -> {OUT_DIR}/astrodeep_spec_selection_ablated.joblib")

# ── Apply to TNG100 image-aperture catalogue ──────────────────────────────────
print("[apply] reading TNG100 image-aperture catalog ...", flush=True)
tng = pd.read_csv(TNG_IMAGEAP_CSV, low_memory=False)
print(f"  rows: {len(tng)}", flush=True)

tng_with_p = add_selection_probabilities(tng, model, used)
tng_with_p = select_by_probability(tng_with_p, mode="deterministic")
n_selected = int(tng_with_p["mock_spec_selected"].sum())
sum_p = float(tng_with_p["p_spec"].sum())
print(f"  ablated model: N selected = {n_selected}, sum(p_spec) = {sum_p:.1f}", flush=True)

selected_mask = tng_with_p["mock_spec_selected"].astype(bool).to_numpy()
selected_idx_new = np.flatnonzero(selected_mask)


# ── Compute deblended SB on the new selection ─────────────────────────────────
print("[deblend] reading deblending_per_filter ...", flush=True)
pf = pd.read_csv(PER_FILTER_CSV, low_memory=False)
print(f"  rows: {len(pf)}", flush=True)

# pf has rows for objects in the original selected set (from the original specmodel CSV).
# The `object_index` field is the row index in the *original specmodel* dataframe,
# which is the same as the row index in tng (the underlying image-aperture catalogue).
# So we filter pf rows to those whose object_index is in our new selected set.
new_selected_set = set(selected_idx_new.tolist())
pf_new = pf[pf["object_index"].isin(new_selected_set)].copy()
print(f"  per-filter rows for new selection: {len(pf_new)} ({pf_new['object_index'].nunique()} distinct objects)")

# Pivot to per-object mean SB using deblended fluxes
flux_pivot = pf_new.pivot_table(
    index="object_index", columns="filter", values="deblended_flux_uJy", aggfunc="first"
)
ap_pivot = pf_new.pivot_table(
    index="object_index", columns="filter", values="aperture_flux_uJy", aggfunc="first"
)
flux_pivot = flux_pivot.where(flux_pivot.notna(), ap_pivot)

ap_area = tng.set_index(tng.index)["aperture_area_arcsec2"]
common = flux_pivot.index.intersection(ap_area.index)
flux_pivot = flux_pivot.loc[common]
ap_area = ap_area.loc[common]

sb_per_filter = flux_pivot.div(ap_area, axis=0)
sb_per_filter = sb_per_filter.where(sb_per_filter > 0)

sb_mean = sb_per_filter.mean(axis=1, skipna=True).to_numpy()
sb_max = sb_per_filter.max(axis=1, skipna=True).to_numpy()
z_arr = tng.loc[common, "z"].to_numpy()

a_mean, e_mean, n_mean = fit_alpha(z_arr, sb_mean)
a_max, e_max, _ = fit_alpha(z_arr, sb_max)

print("\n[result] Ablated selection (no log_sb_mean, no log_aperture_area):")
print(f"  N = {n_mean}")
print(f"  mean SB alpha = {a_mean:+.3f} +/- {e_mean:.3f}")
print(f"  max  SB alpha = {a_max:+.3f} +/- {e_max:.3f}")
print(f"  Original Voronoi-deblended (with SB feature): mean = -0.879, max = -0.866")
print(f"  Δα(mean): {a_mean - (-0.879):+.3f}")
print(f"  Δα(max):  {a_max - (-0.866):+.3f}")

# Save summary
pd.DataFrame([
    {"sample": "ablated (no SB feature)", "N": n_mean,
     "alpha_mean_sb": a_mean, "err_mean_sb": e_mean,
     "alpha_max_sb": a_max, "err_max_sb": e_max,
     "auc_train": auc},
    {"sample": "original (with SB feature, reference)", "N": 2901,
     "alpha_mean_sb": -0.879, "err_mean_sb": 0.030,
     "alpha_max_sb": -0.866, "err_max_sb": 0.029,
     "auc_train": np.nan},
]).to_csv(f"{OUT_DIR}/selection_ablation_summary.csv", index=False)
print(f"\nSaved {OUT_DIR}/selection_ablation_summary.csv")
print("[done]", flush=True)

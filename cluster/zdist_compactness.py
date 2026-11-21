"""Redshift-distribution comparison and compactness-bias test.

1. Redshift distribution of the TNG100 empirical-spec sample vs the ASTRODEEP
   spec sample.
2. Does the +0.25 residual deblended/catalogue ratio slope depend on object
   compactness? Compactness proxy: angular aperture area (aperture_area_arcsec2).
   At fixed z, compact = small aperture area (small physical half-light radius).
   If compact objects drive the residual, the +0.25 slope is a
   selection/morphology artefact; if not, something else (e.g. evolution of
   the light concentration) explains it.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKDIR = os.environ.get("TOLMAN_WORKDIR", str(Path(__file__).resolve().parents[1]))
SPECMODEL_CSV = f"{WORKDIR}/data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv"
ASTRODEEP_Z_CSV = f"{WORKDIR}/data/astrodeep_spec_z.csv"
DEBLEND_OBJ_CSV = f"{WORKDIR}/results/tng100_7_6_xyz_hst_jwst_imageap_deblended/deblending_object_summary.csv"
OUT_DIR = f"{WORKDIR}/results/systematics"
PLOT_DIR = f"{WORKDIR}/Plots"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def fit_alpha(z: np.ndarray, ratio: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(z) & np.isfinite(ratio) & (z > -1) & (ratio > 0)
    x = np.log10(1.0 + z[mask])
    y = np.log10(ratio[mask])
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    A = np.c_[np.ones(n), x]
    c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    s2 = float(np.sum((y - A @ c) ** 2) / (n - 2))
    err = float((s2 * np.linalg.inv(A.T @ A)[1, 1]) ** 0.5)
    return float(c[1]), err, n


# ── z-distribution ────────────────────────────────────────────────────

print("[zdist] Loading TNG100 specmodel...", flush=True)
spec = pd.read_csv(SPECMODEL_CSV, usecols=["z", "mock_spec_selected"], low_memory=False)
tng_z = spec.loc[spec["mock_spec_selected"].astype(bool), "z"].dropna().values
print(f"  TNG100 spec-selected: N={len(tng_z)}")

print("[zdist] Loading ASTRODEEP spec z-values...", flush=True)
ast_z = pd.read_csv(ASTRODEEP_Z_CSV)["z"].dropna().values
print(f"  ASTRODEEP spec: N={len(ast_z)}")

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
bins = np.linspace(0, 9, 40)
ax.hist(ast_z, bins=bins, density=True, alpha=0.55, color="#1f77b4", label=f"ASTRODEEP spec (N={len(ast_z):,})")
ax.hist(tng_z, bins=bins, density=True, alpha=0.55, color="#d62728", label=f"TNG100 mock-spec (N={len(tng_z):,})")
ax.set_xlabel("Redshift z")
ax.set_ylabel("Normalised count")
ax.legend(frameon=False, fontsize=9)
path3 = f"{PLOT_DIR}/zdist_astrodeep_vs_tng100_specsel.png"
fig.savefig(path3, dpi=180)
plt.close(fig)
print(f"[zdist] Saved {path3}", flush=True)

# Summary stats
for label, zvals in [("ASTRODEEP", ast_z), ("TNG100", tng_z)]:
    print(f"  {label}: median z={np.median(zvals):.2f}, mean z={np.mean(zvals):.2f}, "
          f"p10={np.percentile(zvals,10):.2f}, p90={np.percentile(zvals,90):.2f}")


# ── Compactness bias ──────────────────────────────────────────────────

print("\n[compactness] Loading specmodel (full) for aperture sizes...", flush=True)
full_spec = pd.read_csv(
    SPECMODEL_CSV,
    usecols=["z", "mock_spec_selected", "aperture_area_arcsec2", "mstar_msun_2rhalf", "app_gmag_rest"],
    low_memory=False,
)
full_spec = full_spec.reset_index()  # index column = object_index in deblending output
full_spec.rename(columns={"index": "object_index"}, inplace=True)
print(f"  Specmodel rows: {len(full_spec)}")

print("[compactness] Loading deblending object summary...", flush=True)
dbl = pd.read_csv(DEBLEND_OBJ_CSV)
print(f"  Deblending summary rows: {len(dbl)}")

# Merge: deblending uses df.index as object_index, which is the row position in specmodel CSV.
merged = dbl.merge(
    full_spec[["object_index", "aperture_area_arcsec2", "mstar_msun_2rhalf", "app_gmag_rest"]],
    on="object_index",
    how="left",
)
print(f"  Merged rows: {len(merged)}, missing aperture_area: {merged['aperture_area_arcsec2'].isna().sum()}")

# At fixed z, split by angular aperture area (compactness proxy)
# We use a z-normalised compactness: residual of log(aperture_area) after removing z trend
z_m = merged["z"].values
ap_m = merged["aperture_area_arcsec2"].values
log_ap = np.log10(np.where(ap_m > 0, ap_m, np.nan))

# Fit linear trend of log_ap vs log(1+z) to remove z dependence
valid_ap = np.isfinite(log_ap) & np.isfinite(z_m) & (z_m > -1)
x_z = np.log10(1.0 + z_m[valid_ap])
y_a = log_ap[valid_ap]
A_ap = np.c_[np.ones_like(x_z), x_z]
c_ap, _, _, _ = np.linalg.lstsq(A_ap, y_a, rcond=None)
log_ap_resid = np.full(len(merged), np.nan)
log_ap_resid[valid_ap] = y_a - A_ap @ c_ap

# Compact = below median residual (smaller aperture at fixed z)
finite_resid = np.isfinite(log_ap_resid)
median_resid = np.nanmedian(log_ap_resid[finite_resid])
compact = finite_resid & (log_ap_resid < median_resid)
extended = finite_resid & (log_ap_resid >= median_resid)

print(f"\n[compactness] Compact: N={compact.sum()}, Extended: N={extended.sum()}")

# Fit deblended_ratio vs z for each half
z_dbl = merged["z"].values
dbl_ratio = merged["median_deblended_ratio"].values
ap_ratio = merged["median_aperture_ratio"].values

alpha_dbl_compact, err_dbl_compact, n_c = fit_alpha(z_dbl[compact], dbl_ratio[compact])
alpha_dbl_extended, err_dbl_extended, n_e = fit_alpha(z_dbl[extended], dbl_ratio[extended])
alpha_dbl_all, err_dbl_all, n_all = fit_alpha(z_dbl[finite_resid], dbl_ratio[finite_resid])

print(f"  All (N={n_all}):      deblended ratio α = {alpha_dbl_all:+.3f} ± {err_dbl_all:.3f}")
print(f"  Compact (N={n_c}):   deblended ratio α = {alpha_dbl_compact:+.3f} ± {err_dbl_compact:.3f}")
print(f"  Extended (N={n_e}):  deblended ratio α = {alpha_dbl_extended:+.3f} ± {err_dbl_extended:.3f}")

# Also check aperture_ratio for consistency
alpha_ap_compact, _, _ = fit_alpha(z_dbl[compact], ap_ratio[compact])
alpha_ap_extended, _, _ = fit_alpha(z_dbl[extended], ap_ratio[extended])
print(f"  Compact:   aperture ratio α = {alpha_ap_compact:+.3f}")
print(f"  Extended:  aperture ratio α = {alpha_ap_extended:+.3f}")

# Plot: deblended ratio vs z, coloured by compact/extended
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

for ax_i, (mask_arr, label, color) in enumerate(
    [(compact, "Compact", "#d62728"), (extended, "Extended", "#1f77b4")]
):
    ax = axes[ax_i]
    sub = merged[mask_arr]
    ax.scatter(sub["z"], sub["median_deblended_ratio"], s=2, alpha=0.3, color=color, rasterized=True)
    z_line = np.linspace(0.1, 8, 100)
    alpha_val = alpha_dbl_compact if ax_i == 0 else alpha_dbl_extended
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Voronoi deblended / catalogue flux ratio")
    ax.set_title(f"{label} (N={mask_arr.sum()}, α={alpha_val:+.3f})", fontsize=10)
    ax.set_xlim(0, 8)
    ax.set_ylim(0.1, 5)
    ax.set_yscale("log")

path7 = f"{PLOT_DIR}/compactness_bias_deblended_ratio.png"
fig.savefig(path7, dpi=180)
plt.close(fig)
print(f"[compactness] Saved {path7}", flush=True)

# Save summary CSV
rows = [
    {"subsample": "all", "N": n_all, "alpha_deblended": alpha_dbl_all, "err_deblended": err_dbl_all},
    {"subsample": "compact", "N": n_c, "alpha_deblended": alpha_dbl_compact, "err_deblended": err_dbl_compact},
    {"subsample": "extended", "N": n_e, "alpha_deblended": alpha_dbl_extended, "err_deblended": err_dbl_extended},
]
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/compactness_bias_summary.csv", index=False)
print(f"[compactness] Saved {OUT_DIR}/compactness_bias_summary.csv")

print("\n[done]", flush=True)

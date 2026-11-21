"""Redshift-distribution reweighting and extended-only TNG SB slopes.

1. Redshift-distribution reweighting:
   Weight each selected TNG object so the TNG z-distribution matches ASTRODEEP's,
   then refit the deblended SB power-law slope. This measures how much of the
   -0.39 TNG-ASTRODEEP gap comes from the transfer of the selection model to TNG
   (the TNG sample is biased to lower z).

2. Extended-only TNG slope:
   The +0.25 residual deblended/catalogue ratio slope is +0.40 in the compact
   half and +0.22 in the extended half (zdist_compactness.py). Refit the
   deblended SB slope on the extended half only, where the compactness bias is
   smallest.

Inputs:
  - TNG specmodel CSV (z, aperture_area_arcsec2, catalog_flux_F*, mock_spec_selected)
  - deblending_per_filter.csv (per-filter Voronoi-deblended fluxes)
  - astrodeep_spec_z.csv (ASTRODEEP spec sample z-values)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKDIR = os.environ.get("TOLMAN_WORKDIR", str(Path(__file__).resolve().parents[1]))
SPECMODEL_CSV = f"{WORKDIR}/data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv"
PER_FILTER_CSV = f"{WORKDIR}/results/tng100_7_6_xyz_hst_jwst_imageap_deblended/deblending_per_filter.csv"
ASTRODEEP_Z_CSV = f"{WORKDIR}/data/astrodeep_spec_z.csv"
OUT_DIR = f"{WORKDIR}/results/systematics"
PLOT_DIR = f"{WORKDIR}/Plots"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Z range where both samples have decent coverage (TNG p90 = 2.86, ASTRODEEP p90 = 4.82)
Z_MIN_OVERLAP = 0.1
Z_MAX_OVERLAP = 4.0


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
    cov = s2 * np.linalg.inv(normal)
    err = float(cov[1, 1] ** 0.5)
    return float(c[1]), err, n


# ── Load deblended SB per object ──────────────────────────────────────────────
print("[load] specmodel...", flush=True)
spec = pd.read_csv(SPECMODEL_CSV, low_memory=False).reset_index().rename(columns={"index": "object_index"})
selected = spec[spec["mock_spec_selected"].astype(bool)].copy()
print(f"  selected: {len(selected)}", flush=True)

print("[load] deblending_per_filter...", flush=True)
pf = pd.read_csv(PER_FILTER_CSV)
print(f"  rows: {len(pf)}, filters: {pf['filter'].nunique()}", flush=True)

# Pivot deblended fluxes: rows = object_index, columns = filter
flux_pivot = pf.pivot_table(
    index="object_index",
    columns="filter",
    values="deblended_flux_uJy",
    aggfunc="first",
)
# Where deblended is NaN, fall back to aperture flux (matches deblended-tng-slopes CLI behaviour)
ap_pivot = pf.pivot_table(
    index="object_index",
    columns="filter",
    values="aperture_flux_uJy",
    aggfunc="first",
)
flux_pivot = flux_pivot.where(flux_pivot.notna(), ap_pivot)

# Aperture area for each selected object (single 'aperture_area_arcsec2' column;
# for TNG fixed-physical-radius aperture this is the same across filters at fixed z)
ap_area = selected.set_index("object_index")["aperture_area_arcsec2"]
common = flux_pivot.index.intersection(ap_area.index)
flux_pivot = flux_pivot.loc[common]
ap_area = ap_area.loc[common]

# Per-filter SB (uJy / arcsec^2)
sb_per_filter = flux_pivot.div(ap_area, axis=0)
sb_per_filter = sb_per_filter.where(sb_per_filter > 0)  # only positive

# Per-object summary observables
sb_mean = sb_per_filter.mean(axis=1, skipna=True)
sb_max = sb_per_filter.max(axis=1, skipna=True)

per_obj = pd.DataFrame({
    "object_index": common.astype(int),
    "z": selected.set_index("object_index").loc[common, "z"].to_numpy(),
    "aperture_area_arcsec2": ap_area.to_numpy(),
    "sb_mean": sb_mean.to_numpy(),
    "sb_max": sb_max.to_numpy(),
})
print(f"  per-object table: {len(per_obj)}", flush=True)


# ── Compactness split (same definition as zdist_compactness.py) ─────────────────────────────
log_ap = np.log10(np.where(per_obj["aperture_area_arcsec2"] > 0, per_obj["aperture_area_arcsec2"], np.nan))
z_arr = per_obj["z"].to_numpy()
valid_ap = np.isfinite(log_ap) & np.isfinite(z_arr) & (z_arr > -1)
x_z = np.log10(1.0 + z_arr[valid_ap]); y_a = log_ap[valid_ap]
A_ap = np.c_[np.ones_like(x_z), x_z]
c_ap, _, _, _ = np.linalg.lstsq(A_ap, y_a, rcond=None)
log_ap_resid = np.full(len(per_obj), np.nan)
log_ap_resid[valid_ap] = y_a - A_ap @ c_ap
median_resid = float(np.nanmedian(log_ap_resid[np.isfinite(log_ap_resid)]))
compact_mask = np.isfinite(log_ap_resid) & (log_ap_resid < median_resid)
extended_mask = np.isfinite(log_ap_resid) & (log_ap_resid >= median_resid)


# ── Extended-only slope ──────────────────────────────────────────────
print("\n[#2 extended-only]", flush=True)
a_full_mean, e_full_mean, n_full = fit_alpha(per_obj["z"], per_obj["sb_mean"])
a_ext_mean, e_ext_mean, n_ext = fit_alpha(per_obj["z"][extended_mask], per_obj["sb_mean"][extended_mask])
a_cmp_mean, e_cmp_mean, n_cmp = fit_alpha(per_obj["z"][compact_mask], per_obj["sb_mean"][compact_mask])

a_full_max, e_full_max, _ = fit_alpha(per_obj["z"], per_obj["sb_max"])
a_ext_max, e_ext_max, _ = fit_alpha(per_obj["z"][extended_mask], per_obj["sb_max"][extended_mask])
a_cmp_max, e_cmp_max, _ = fit_alpha(per_obj["z"][compact_mask], per_obj["sb_max"][compact_mask])

print(f"  full     (N={n_full}): mean α={a_full_mean:+.3f} ± {e_full_mean:.3f}  max α={a_full_max:+.3f} ± {e_full_max:.3f}")
print(f"  compact  (N={n_cmp}): mean α={a_cmp_mean:+.3f} ± {e_cmp_mean:.3f}  max α={a_cmp_max:+.3f} ± {e_cmp_max:.3f}")
print(f"  extended (N={n_ext}): mean α={a_ext_mean:+.3f} ± {e_ext_mean:.3f}  max α={a_ext_max:+.3f} ± {e_ext_max:.3f}")
print(f"  ASTRODEEP reference: mean α=-1.27, max α=-1.10")
print(f"  extended-vs-ASTRODEEP gap (mean SB): {(-1.27) - a_ext_mean:+.3f}")


# ── z-distribution reweighting ───────────────────────────────────────
print("\n[#1 z-reweight to ASTRODEEP]", flush=True)
ast_z_full = pd.read_csv(ASTRODEEP_Z_CSV)["z"].dropna().to_numpy()
print(f"  ASTRODEEP spec z values: {len(ast_z_full)}")

ast_in = ast_z_full[(ast_z_full > Z_MIN_OVERLAP) & (ast_z_full < Z_MAX_OVERLAP)]
tng_z_all = per_obj["z"].to_numpy()
in_range = (tng_z_all > Z_MIN_OVERLAP) & (tng_z_all < Z_MAX_OVERLAP)
print(f"  z range [{Z_MIN_OVERLAP}, {Z_MAX_OVERLAP}]: ASTRODEEP={len(ast_in)}, TNG={int(in_range.sum())}")

# KDE in log(1+z) space (the fitting variable)
log_z_ast = np.log10(1.0 + ast_in)
log_z_tng = np.log10(1.0 + tng_z_all[in_range])
kde_ast = gaussian_kde(log_z_ast, bw_method="scott")
kde_tng = gaussian_kde(log_z_tng, bw_method="scott")

p_ast = kde_ast(log_z_tng)
p_tng = kde_tng(log_z_tng)
weights = p_ast / np.maximum(p_tng, 1e-12)
# Effective sample size diagnostic
n_eff = float((weights.sum() ** 2) / np.sum(weights ** 2))
weights_norm = weights * len(weights) / weights.sum()
print(f"  effective N: {n_eff:.0f}  (max/min weight: {weights_norm.max():.2f}/{weights_norm.min():.2g})")

# Build full-length weight array (only nonzero in-range)
full_weights = np.zeros(len(per_obj))
full_weights[in_range] = weights_norm

# Unweighted slope on the same z-restricted subsample (apples-to-apples)
a_uw_mean, e_uw_mean, n_uw = fit_alpha(per_obj["z"][in_range], per_obj["sb_mean"][in_range])
a_uw_max, e_uw_max, _ = fit_alpha(per_obj["z"][in_range], per_obj["sb_max"][in_range])

# Reweighted slope
a_rw_mean, e_rw_mean, n_rw = fit_alpha(per_obj["z"], per_obj["sb_mean"], full_weights)
a_rw_max, e_rw_max, _ = fit_alpha(per_obj["z"], per_obj["sb_max"], full_weights)

print(f"  full TNG (z<{Z_MAX_OVERLAP}, unweighted, N={n_uw}): mean α={a_uw_mean:+.3f} ± {e_uw_mean:.3f}  max α={a_uw_max:+.3f} ± {e_uw_max:.3f}")
print(f"  reweighted to ASTRODEEP-z (N_eff~{n_eff:.0f}): mean α={a_rw_mean:+.3f} ± {e_rw_mean:.3f}  max α={a_rw_max:+.3f} ± {e_rw_max:.3f}")
print(f"  reweighting Δα: mean {a_rw_mean - a_uw_mean:+.3f}, max {a_rw_max - a_uw_max:+.3f}")

# ASTRODEEP z<4 reference (for an apples-to-apples comparison)
# We need to refit ASTRODEEP itself in z<4 to compare. Use the full ASTRODEEP z list as proxy
# but we don't have ASTRODEEP SB values here. Quote the published full-sample slope.
print(f"  ASTRODEEP full-sample reference: mean α=-1.27 (whole z range)")


# ── Save summary CSV ──────────────────────────────────────────────────────────
rows = [
    {"sample": "TNG full (all z)", "N": n_full, "alpha_mean_sb": a_full_mean, "err_mean_sb": e_full_mean,
     "alpha_max_sb": a_full_max, "err_max_sb": e_full_max},
    {"sample": "TNG compact half", "N": n_cmp, "alpha_mean_sb": a_cmp_mean, "err_mean_sb": e_cmp_mean,
     "alpha_max_sb": a_cmp_max, "err_max_sb": e_cmp_max},
    {"sample": "TNG extended half", "N": n_ext, "alpha_mean_sb": a_ext_mean, "err_mean_sb": e_ext_mean,
     "alpha_max_sb": a_ext_max, "err_max_sb": e_ext_max},
    {"sample": f"TNG z<{Z_MAX_OVERLAP} unweighted", "N": n_uw, "alpha_mean_sb": a_uw_mean, "err_mean_sb": e_uw_mean,
     "alpha_max_sb": a_uw_max, "err_max_sb": e_uw_max},
    {"sample": f"TNG reweighted to ASTRODEEP z (z<{Z_MAX_OVERLAP})", "N": int(n_eff),
     "alpha_mean_sb": a_rw_mean, "err_mean_sb": e_rw_mean,
     "alpha_max_sb": a_rw_max, "err_max_sb": e_rw_max},
]
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/reweight_extended_summary.csv", index=False)
print(f"\nSaved {OUT_DIR}/reweight_extended_summary.csv")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

# Left: bar chart of α values for mean SB
ax0 = axes[0]
labels = ["TNG full", "compact", "extended", f"z<{Z_MAX_OVERLAP}", "reweighted"]
alphas = [a_full_mean, a_cmp_mean, a_ext_mean, a_uw_mean, a_rw_mean]
errs = [e_full_mean, e_cmp_mean, e_ext_mean, e_uw_mean, e_rw_mean]
colors = ["#888888", "#d62728", "#1f77b4", "#888888", "#2ca02c"]
xpos = np.arange(len(labels))
ax0.bar(xpos, alphas, yerr=errs, color=colors, capsize=3)
ax0.axhline(-1.27, color="#ff7f0e", ls="--", lw=1, label="ASTRODEEP α=-1.27")
ax0.axhline(-3.0, color="#9467bd", ls=":", lw=1, label="FRW broadband α=-3")
ax0.set_xticks(xpos); ax0.set_xticklabels(labels, rotation=30, ha="right")
ax0.set_ylabel("Mean SB α (Voronoi-deblended)")
ax0.set_ylim(-2.5, 0.2)
ax0.legend(frameon=False, fontsize=8, loc="lower right")

# Right: histograms of TNG (raw) vs TNG (reweighted) vs ASTRODEEP
ax1 = axes[1]
bins = np.linspace(0, 5, 30)
ax1.hist(ast_in, bins=bins, density=True, alpha=0.5, color="#1f77b4", label=f"ASTRODEEP (N={len(ast_in)})")
ax1.hist(tng_z_all[in_range], bins=bins, density=True, alpha=0.5, color="#d62728", label=f"TNG raw (N={int(in_range.sum())})")
ax1.hist(tng_z_all[in_range], bins=bins, density=True, weights=weights_norm,
         alpha=0.4, color="#2ca02c", label=f"TNG reweighted (N_eff~{n_eff:.0f})")
ax1.set_xlabel("Redshift z")
ax1.set_ylabel("Normalised density")
ax1.legend(frameon=False, fontsize=8)
ax1.set_title(f"reweight to ASTRODEEP, z<{Z_MAX_OVERLAP}", fontsize=9)

path = f"{PLOT_DIR}/reweight_extended_summary.png"
fig.savefig(path, dpi=180)
plt.close(fig)
print(f"Saved {path}")
print("[done]", flush=True)

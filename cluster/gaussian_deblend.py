"""Gaussian-weighted deblending vs Voronoi deblending (F200W, one field).

Compares two deblending strategies for blended TNG100 sources:
  - Voronoi (fiducial): each segment pixel assigned to nearest catalogue position.
  - Gaussian-weighted: weight ∝ exp(-d²/(2σ²)) with σ ~ PSF width in pixels.

For isolated objects both methods are identical. For blended objects, Gaussian
gives smoother boundaries (closer to SExtractor multi-threshold deblending).

Per-segment iteration + KDTree-bounded weights for memory & speed safety.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKDIR = os.environ.get("TOLMAN_WORKDIR", str(Path(__file__).resolve().parents[1]))
FITS_F200W = f"{WORKDIR}/data/raw/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f200w_v1_sim.fits"
SPECMODEL_CSV = f"{WORKDIR}/data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv"
PER_FILTER_CSV = f"{WORKDIR}/results/tng100_7_6_xyz_hst_jwst_imageap_deblended/deblending_per_filter.csv"
OUT_DIR = f"{WORKDIR}/results/systematics"
PLOT_DIR = f"{WORKDIR}/Plots"

SIGMA_PIX = 1.5
THRESHOLD_SIGMA = 3.0
MIN_PIXELS = 8
K_NEAREST = 6  # number of nearest catalogue positions per pixel for weight calc

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def fit_alpha(z, ratio):
    z = np.asarray(z, dtype=float); ratio = np.asarray(ratio, dtype=float)
    m = np.isfinite(z) & np.isfinite(ratio) & (z > -1) & (ratio > 0)
    n = int(m.sum())
    if n < 3:
        return np.nan, np.nan, n
    x = np.log10(1.0 + z[m]); y = np.log10(ratio[m])
    A = np.c_[np.ones(n), x]
    c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    s2 = float(np.sum((y - A @ c) ** 2) / (n - 2))
    err = float((s2 * np.linalg.inv(A.T @ A)[1, 1]) ** 0.5)
    return float(c[1]), err, n


def robust_background(image):
    f = image[np.isfinite(image)].ravel()
    med = float(np.median(f))
    mad = float(np.median(np.abs(f - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(f))
    return med, sigma


# ── Load image ────────────────────────────────────────────────────────────────
print(f"[load] {FITS_F200W}", flush=True)
with fits.open(FITS_F200W, memmap=False) as hdul:
    raw_image = np.asarray(hdul[0].data, dtype=np.float32)
image = raw_image / 1000.0  # nJy → uJy
print(f"  shape={image.shape}  min={float(np.nanmin(image)):.4f}  max={float(np.nanmax(image)):.4f} uJy", flush=True)

bg, sigma = robust_background(image)
threshold = bg + THRESHOLD_SIGMA * sigma
print(f"  bg={bg:.5f} uJy  sigma={sigma:.5f}  threshold={threshold:.5f}", flush=True)


# ── Segmentation ──────────────────────────────────────────────────────────────
print("[seg] connected components...", flush=True)
mask_arr = np.isfinite(image) & (image > threshold)
labels, n_labels = ndimage.label(mask_arr)
counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
weights_image = np.where(labels > 0, image - bg, 0.0)
fluxes = np.bincount(labels.ravel(), weights=weights_image.ravel(), minlength=n_labels + 1)
valid = np.flatnonzero((counts >= MIN_PIXELS) & (fluxes > 0))
valid = valid[valid != 0]
mapping = np.zeros(n_labels + 1, dtype=np.int32)
mapping[valid] = np.arange(1, len(valid) + 1)
seg_labels = mapping[labels].astype(np.int32)
del labels, weights_image
print(f"  {len(valid)} valid segments", flush=True)

print("[seg] computing bboxes...", flush=True)
objects = ndimage.find_objects(seg_labels)  # bbox per remapped sid (sid-1 -> objects[sid-1])


# ── Catalogue positions ───────────────────────────────────────────────────────
print("[cat] reading specmodel...", flush=True)
spec = pd.read_csv(
    SPECMODEL_CSV,
    usecols=["z", "mock_spec_selected", "catalog_flux_F200W_uJy", "x_F200W_pix", "y_F200W_pix"],
    low_memory=False,
).reset_index().rename(columns={"index": "object_index"})
all_x = spec["x_F200W_pix"].to_numpy()
all_y = spec["y_F200W_pix"].to_numpy()
ny, nx = image.shape

# Assign segment ID to each catalogue object
ix_all = np.clip(np.floor(all_x).astype(np.int64), 0, nx - 1)
iy_all = np.clip(np.floor(all_y).astype(np.int64), 0, ny - 1)
seg_id_all = seg_labels[iy_all, ix_all]

# Build seg → list of all catalogue indices (positions in spec)
seg_to_all: dict[int, list[int]] = {}
for idx in range(len(spec)):
    sid = int(seg_id_all[idx])
    if sid > 0:
        seg_to_all.setdefault(sid, []).append(idx)

# Selected objects
selected_mask = spec["mock_spec_selected"].astype(bool).to_numpy()
selected_idx = np.flatnonzero(selected_mask)
print(f"  selected: {len(selected_idx)}", flush=True)

# Build seg → list of (out_i, row_i) for selected only
selected_by_seg: dict[int, list[tuple[int, int]]] = {}
for out_i, row_i in enumerate(selected_idx):
    sid = int(seg_id_all[row_i])
    if sid > 0:
        selected_by_seg.setdefault(sid, []).append((out_i, int(row_i)))
print(f"  selected segments: {len(selected_by_seg)}", flush=True)


# ── Gaussian deblend per segment ──────────────────────────────────────────────
gauss_flux = np.full(len(selected_idx), np.nan)
two_sigma2 = 2.0 * SIGMA_PIX ** 2
n_done = 0
n_isolated = 0
n_blended = 0

print("[deblend] processing segments...", flush=True)
for sid, items in selected_by_seg.items():
    all_in_seg = seg_to_all.get(sid, [])
    n_cat = len(all_in_seg)
    if n_cat == 0:
        continue

    obj = objects[sid - 1]  # bbox for this remapped sid
    if obj is None:
        continue
    sl_y, sl_x = obj
    sub = image[sl_y, sl_x]
    sub_labels = seg_labels[sl_y, sl_x]
    seg_mask = sub_labels == sid
    pix_ys, pix_xs = np.nonzero(seg_mask)
    n_pix = len(pix_ys)
    if n_pix == 0:
        continue
    pix_flux = sub[pix_ys, pix_xs]

    if n_cat == 1:
        # Isolated: trivial sum
        flux = float(np.nansum(pix_flux))
        for out_i, _ in items:
            gauss_flux[out_i] = flux
        n_isolated += len(items)
    else:
        # KDTree-bounded Gaussian weights: contribution from the k nearest catalogue
        # positions captures essentially all weight (exp falls off fast).
        cat_xs_local = all_x[all_in_seg] - sl_x.start
        cat_ys_local = all_y[all_in_seg] - sl_y.start
        cat_xy = np.column_stack([cat_xs_local, cat_ys_local])
        pix_xy = np.column_stack([pix_xs.astype(np.float32), pix_ys.astype(np.float32)])
        k = min(K_NEAREST, n_cat)
        tree = cKDTree(cat_xy)
        dists, inds = tree.query(pix_xy, k=k)  # (n_pix, k)
        if k == 1:
            dists = dists[:, None]; inds = inds[:, None]
        w_topk = np.exp(-(dists ** 2) / two_sigma2).astype(np.float32)
        w_total = w_topk.sum(axis=1)  # (n_pix,)

        # For each selected object in this segment, compute their contribution.
        for out_i, row_i in items:
            local_i = all_in_seg.index(row_i)
            mask_self = (inds == local_i)  # (n_pix, k)
            w_self = (w_topk * mask_self).sum(axis=1)
            norm = np.where(w_total > 0, w_self / w_total, 0.0)
            gauss_flux[out_i] = float(np.nansum(pix_flux * norm))
        n_blended += len(items)

    n_done += 1
    if n_done % 5000 == 0:
        print(f"  ...{n_done} segments processed", flush=True)

print(f"[deblend] done. isolated_assignments={n_isolated} blended_assignments={n_blended} valid={np.isfinite(gauss_flux).sum()}", flush=True)


# ── Voronoi reference fluxes ──────────────────────────────────────────────────
print("[load] reading per_filter Voronoi fluxes (F200W)...", flush=True)
pf = pd.read_csv(PER_FILTER_CSV, low_memory=False)
pf_f200w = pf[pf["filter"] == "F200W"].copy()
voronoi_map = dict(zip(pf_f200w["object_index"].to_numpy(), pf_f200w["deblended_flux_uJy"].to_numpy()))
cat_map = dict(zip(pf_f200w["object_index"].to_numpy(), pf_f200w["catalog_flux_uJy"].to_numpy()))

vor = np.array([voronoi_map.get(int(i), np.nan) for i in selected_idx])
catf = np.array([cat_map.get(int(i), np.nan) for i in selected_idx])
z_sel = spec["z"].to_numpy()[selected_idx]


# ── Slopes ────────────────────────────────────────────────────────────────────
denom = np.where(np.isfinite(catf) & (catf > 0), catf, np.nan)
ratio_vor = vor / denom
ratio_gau = gauss_flux / denom

a_vor, e_vor, n_vor = fit_alpha(z_sel, ratio_vor)
a_gau, e_gau, n_gau = fit_alpha(z_sel, ratio_gau)

print("\n[result] F200W deblending comparison:")
print(f"  Voronoi  / catalogue ratio: alpha = {a_vor:+.3f} +/- {e_vor:.3f}  (N={n_vor})")
print(f"  Gaussian / catalogue ratio: alpha = {a_gau:+.3f} +/- {e_gau:.3f}  (N={n_gau})")
print(f"  Voronoi - Gaussian slope:   delta = {a_vor - a_gau:+.3f}")

both_finite = np.isfinite(vor) & np.isfinite(gauss_flux) & (catf > 0)
print(f"  median Voronoi/cat:  {float(np.nanmedian(ratio_vor[both_finite])):.3f}")
print(f"  median Gaussian/cat: {float(np.nanmedian(ratio_gau[both_finite])):.3f}")

# Save summary
pd.DataFrame([
    {"method": "voronoi", "alpha": a_vor, "alpha_err": e_vor, "N": n_vor,
     "median_ratio": float(np.nanmedian(ratio_vor[both_finite]))},
    {"method": "gaussian", "alpha": a_gau, "alpha_err": e_gau, "N": n_gau,
     "median_ratio": float(np.nanmedian(ratio_gau[both_finite]))},
]).to_csv(f"{OUT_DIR}/deblend_comparison_f200w.csv", index=False)
print(f"  saved {OUT_DIR}/deblend_comparison_f200w.csv")

# Save per-object diagnostic
pd.DataFrame({
    "object_index": selected_idx.astype(int),
    "z": z_sel,
    "catalog_flux_uJy": catf,
    "voronoi_flux_uJy": vor,
    "gaussian_flux_uJy": gauss_flux,
    "voronoi_to_cat": ratio_vor,
    "gaussian_to_cat": ratio_gau,
}).to_csv(f"{OUT_DIR}/deblend_per_object_f200w.csv", index=False)


# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

# Left: Gaussian vs Voronoi flux ratios
ax0 = axes[0]
ax0.scatter(ratio_vor[both_finite], ratio_gau[both_finite], s=3, alpha=0.25, color="#333333", rasterized=True)
hi = float(np.nanpercentile(np.r_[ratio_vor[both_finite], ratio_gau[both_finite]], 98.5))
hi = max(min(hi, 3.0), 1.5)
ax0.plot([0, hi], [0, hi], "r--", lw=1)
ax0.set_xlim(0, hi); ax0.set_ylim(0, hi)
ax0.set_xlabel("Voronoi deblended / catalogue (F200W)")
ax0.set_ylabel("Gaussian deblended / catalogue (F200W)")
ax0.set_title(f"per-object, N={int(both_finite.sum())}", fontsize=10)

# Right: both ratio slopes vs z
ax1 = axes[1]
ax1.scatter(z_sel[both_finite], ratio_vor[both_finite], s=3, alpha=0.18, color="#1f77b4", rasterized=True, label=f"Voronoi α={a_vor:+.3f}")
ax1.scatter(z_sel[both_finite], ratio_gau[both_finite], s=3, alpha=0.18, color="#d62728", rasterized=True, label=f"Gaussian α={a_gau:+.3f}")
ax1.axhline(1.0, color="#555555", lw=0.8, ls="--")
ax1.set_yscale("log")
ax1.set_xlabel("Redshift z")
ax1.set_ylabel("Deblended / catalogue flux ratio (F200W)")
ax1.legend(frameon=False, fontsize=8)

path = f"{PLOT_DIR}/deblend_voronoi_vs_gaussian_f200w.png"
fig.savefig(path, dpi=180)
plt.close(fig)
print(f"  saved {path}")
print("[done]", flush=True)

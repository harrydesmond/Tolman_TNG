"""PSF + sky-noise injection on the TNG100 F200W mock image.

Tests whether the ASTRODEEP-like adaptive-aperture sign-flip ($\\alpha=+0.70$
on the noise-free TNG mock image) reverses toward negative slopes when realistic
imaging effects are added. We convolve the F200W mock image with a Gaussian PSF
matched to JWST NIRCam F200W ($\\sigma\\simeq 1$ pix at the native $\\sim 0.029$
arcsec pixel scale) and add Gaussian sky noise at $\\sigma_{\\rm pix}=0.1$ nJy/pix
(representative of CEERS/JADES F200W depth).

Then we re-measure aperture photometry at the catalogue positions for two
aperture rules: (i) the fixed-physical (TNG nominal) circular aperture and
(ii) the ASTRODEEP-equivalent adaptive aperture. Slopes are fit on the same
empirical mock-spectroscopic sample as the rest of the paper. We compare:

  Noise-free, fixed-physical: alpha_meas (single F200W band) -> baseline
  Noise-free, adaptive:       alpha_meas (single F200W band) -> ~+0.7 expected
  PSF+noise, fixed-physical:  alpha_meas
  PSF+noise, adaptive:        alpha_meas <-- the diagnostic of interest

If the PSF+noise adaptive slope flips toward negative, the
PSF/sky-noise explanation of the adaptive-aperture sign reversal is
validated.
"""
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Defaults to the repository root.  Set the environment variable
# TOLMAN_WORKDIR to override when running on a different cluster or
# workstation, so the staged FITS images and result trees are picked up
# from the correct directory.
WORKDIR = os.environ.get("TOLMAN_WORKDIR", str(Path(__file__).resolve().parents[1]))
FITS_F200W = f"{WORKDIR}/data/raw/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f200w_v1_sim.fits"
SPECMODEL_CSV = f"{WORKDIR}/data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv"
OUT_DIR = f"{WORKDIR}/results/systematics"
PLOT_DIR = f"{WORKDIR}/Plots"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Imaging-chain parameters ──────────────────────────────────────────────────
PSF_SIGMA_PIX = 1.0  # JWST NIRCam F200W FWHM ~0.066 arcsec; native ~0.029 arcsec/pix => sigma ~0.97 pix
SKY_SIGMA_NJY = 0.1  # representative ASTRODEEP F200W depth in nJy/pixel
RNG_SEED = 0

# ASTRODEEP adaptive aperture rule -- radius = apopt/2 (Merlin+24 apopt is a diameter).
def astrodeep_radius_arcsec(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    r = np.full_like(z, np.nan)
    r[(z >= 0) & (z < 2)] = 0.66
    r[(z >= 2) & (z < 3)] = 0.25
    r[(z >= 3) & (z < 4)] = 0.165
    r[z >= 4] = 0.14
    return r


def fit_alpha(z, sb):
    z = np.asarray(z, dtype=float); sb = np.asarray(sb, dtype=float)
    m = np.isfinite(z) & np.isfinite(sb) & (z > -1) & (sb > 0)
    n = int(m.sum())
    if n < 3:
        return np.nan, np.nan, n
    x = np.log10(1.0 + z[m]); y = np.log10(sb[m])
    A = np.c_[np.ones(n), x]
    c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    s2 = float(np.sum((y - A @ c) ** 2) / (n - 2))
    err = float((s2 * np.linalg.inv(A.T @ A)[1, 1]) ** 0.5)
    return float(c[1]), err, n


def aperture_flux(image: np.ndarray, x: np.ndarray, y: np.ndarray, r_pix: np.ndarray) -> np.ndarray:
    """Circular aperture sum with linear-pixel-fraction edge weighting (no subpixel
    interpolation; sufficient for ~1-pix accuracy on r >= 2 pix apertures)."""
    out = np.full(len(x), np.nan)
    ny, nx = image.shape
    for i in range(len(x)):
        if not (np.isfinite(x[i]) and np.isfinite(y[i]) and np.isfinite(r_pix[i])):
            continue
        rp = float(r_pix[i])
        if rp <= 0:
            continue
        xc = float(x[i]); yc = float(y[i])
        x0 = max(0, int(np.floor(xc - rp)))
        x1 = min(nx, int(np.ceil(xc + rp)) + 1)
        y0 = max(0, int(np.floor(yc - rp)))
        y1 = min(ny, int(np.ceil(yc + rp)) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = image[y0:y1, x0:x1]
        py = np.arange(y0, y1) + 0.5  # pixel centres
        px = np.arange(x0, x1) + 0.5
        dx = (px - xc); dy = (py - yc)
        d2 = dy[:, None] ** 2 + dx[None, :] ** 2
        mask = d2 <= rp ** 2
        out[i] = float(np.nansum(sub[mask]))
    return out


# ── Load image and metadata ───────────────────────────────────────────────────
print(f"[load] {FITS_F200W}", flush=True)
with fits.open(FITS_F200W, memmap=False) as hdul:
    raw = np.asarray(hdul[0].data, dtype=np.float32)
    hdr = hdul[0].header

# nJy/pixel native units; we keep nJy for noise injection (which is in nJy/pix)
print(f"  shape={raw.shape}  min={float(np.nanmin(raw)):.3f}  max={float(np.nanmax(raw)):.3f} nJy", flush=True)


# ── Cataloge ──────────────────────────────────────────────────────────────────
print("[load] specmodel CSV ...", flush=True)
spec = pd.read_csv(SPECMODEL_CSV, low_memory=False)
sel = spec["mock_spec_selected"].astype(bool).to_numpy()
selected = spec.iloc[np.flatnonzero(sel)].copy()
print(f"  selected: {len(selected)}", flush=True)

x = selected["x_F200W_pix"].to_numpy()
y = selected["y_F200W_pix"].to_numpy()
z = selected["z"].to_numpy()
kpc_per_arcsec = selected["kpc_per_arcsec"].to_numpy()
photrad_kpc = selected["aperture_radius_arcsec_F200W"].to_numpy() * kpc_per_arcsec  # phys radius
# Pixel scale: try astropy WCS first (correct CD matrix), fall back to area calc.
try:
    from astropy.wcs import WCS
    import astropy.units as u
    wcs = WCS(hdr)
    pixscales = wcs.proj_plane_pixel_scales()
    arcsec_per_pix = float(pixscales[0].to(u.arcsec).value)
    src = "WCS"
except Exception as exc:  # noqa: BLE001
    arcsec_per_pix = np.sqrt(137.0 * 3600.0 / float(raw.shape[0] * raw.shape[1]))
    src = f"area-fallback ({type(exc).__name__})"
if not (0.005 < arcsec_per_pix < 0.5):
    arcsec_per_pix = np.sqrt(137.0 * 3600.0 / float(raw.shape[0] * raw.shape[1]))
    src = f"area-fallback (WCS gave implausible {arcsec_per_pix} arcsec/pix)"
print(f"  arcsec/pix: {arcsec_per_pix:.5f}  [{src}]", flush=True)

# Fixed-physical aperture in pixels (matches the TNG nominal aperture)
r_fixed_pix = (photrad_kpc / kpc_per_arcsec) / arcsec_per_pix
# Adaptive ASTRODEEP-rule aperture in pixels
r_adaptive_arcsec = astrodeep_radius_arcsec(z)
r_adaptive_pix = r_adaptive_arcsec / arcsec_per_pix


# ── Noise-free baseline (single F200W band) ───────────────────────────────────
print("[noise-free] aperture photometry on raw image ...", flush=True)
flux_fixed_clean = aperture_flux(raw, x, y, r_fixed_pix)
flux_adapt_clean = aperture_flux(raw, x, y, r_adaptive_pix)
# Convert nJy -> uJy
flux_fixed_clean_uJy = flux_fixed_clean / 1000.0
flux_adapt_clean_uJy = flux_adapt_clean / 1000.0


# ── PSF + noise version ───────────────────────────────────────────────────────
print(f"[psf+noise] convolving with sigma={PSF_SIGMA_PIX} pix Gaussian PSF ...", flush=True)
img_psf = ndimage.gaussian_filter(raw, sigma=PSF_SIGMA_PIX).astype(np.float32)
print(f"  PSF-convolved: range [{float(img_psf.min()):.3f}, {float(img_psf.max()):.3f}] nJy")

print(f"[psf+noise] adding Gaussian sky sigma={SKY_SIGMA_NJY} nJy/pix ...", flush=True)
rng = np.random.default_rng(RNG_SEED)
img_noisy = img_psf + rng.normal(0.0, SKY_SIGMA_NJY, size=raw.shape).astype(np.float32)
print(f"  noisy: range [{float(img_noisy.min()):.3f}, {float(img_noisy.max()):.3f}] nJy")

# Free PSF-only buffer
del img_psf

print("[psf+noise] aperture photometry on noisy image ...", flush=True)
flux_fixed_noisy = aperture_flux(img_noisy, x, y, r_fixed_pix) / 1000.0
flux_adapt_noisy = aperture_flux(img_noisy, x, y, r_adaptive_pix) / 1000.0


# ── SB and slope fits ─────────────────────────────────────────────────────────
# Aperture areas in arcsec^2
ap_fixed_arcsec2 = np.pi * (r_fixed_pix * arcsec_per_pix) ** 2
ap_adapt_arcsec2 = np.pi * (r_adaptive_arcsec) ** 2

sb_fixed_clean = flux_fixed_clean_uJy / ap_fixed_arcsec2
sb_adapt_clean = flux_adapt_clean_uJy / ap_adapt_arcsec2
sb_fixed_noisy = flux_fixed_noisy / ap_fixed_arcsec2
sb_adapt_noisy = flux_adapt_noisy / ap_adapt_arcsec2

results = []
for label, sb in [
    ("fixed-physical, noise-free (F200W only)", sb_fixed_clean),
    ("fixed-physical, PSF+noise (F200W only)",  sb_fixed_noisy),
    ("adaptive ASTRODEEP, noise-free (F200W only)", sb_adapt_clean),
    ("adaptive ASTRODEEP, PSF+noise (F200W only)",  sb_adapt_noisy),
]:
    a, e, n = fit_alpha(z, sb)
    print(f"  {label}: alpha={a:+.3f}+/-{e:.3f}  N={n}")
    results.append({"sample": label, "alpha": a, "alpha_err": e, "N": n})

print(f"\n[reference]")
print(f"  TNG noise-free Voronoi-deblended (mean over filters): alpha=-0.879 (paper)")
print(f"  TNG noise-free adaptive-aperture (mean over filters): alpha=+0.695 (paper)")
print(f"  ASTRODEEP TNG-common-filter mean: alpha=-1.279 (paper)")

# Save summary
df = pd.DataFrame(results)
df["psf_sigma_pix"] = PSF_SIGMA_PIX
df["sky_sigma_nJy_per_pix"] = SKY_SIGMA_NJY
df.to_csv(f"{OUT_DIR}/psf_noise_injection_f200w.csv", index=False)
print(f"\nSaved {OUT_DIR}/psf_noise_injection_f200w.csv")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

# Left: Adaptive aperture, noise-free vs noisy
ax0 = axes[0]
m_clean = np.isfinite(sb_adapt_clean) & (sb_adapt_clean > 0)
m_noisy = np.isfinite(sb_adapt_noisy) & (sb_adapt_noisy > 0)
ax0.scatter(z[m_clean], sb_adapt_clean[m_clean], s=4, alpha=0.18, color="#999999", rasterized=True, label="noise-free")
ax0.scatter(z[m_noisy], sb_adapt_noisy[m_noisy], s=4, alpha=0.30, color="#d62728", rasterized=True, label="PSF+noise")
ax0.set_yscale("log")
ax0.set_xlabel("Redshift z")
ax0.set_ylabel("ASTRODEEP-adaptive SB (F200W) [μJy/arcsec²]")
ax0.legend(frameon=False, fontsize=9)
ax0.set_title("adaptive aperture", fontsize=10)

# Right: bar chart of alphas
ax1 = axes[1]
labels = ["fixed-phys\nclean", "fixed-phys\nPSF+noise", "adaptive\nclean", "adaptive\nPSF+noise"]
alphas = [r["alpha"] for r in results]
errs = [r["alpha_err"] for r in results]
xpos = np.arange(len(labels))
colors = ["#1f77b4", "#1f77b4", "#d62728", "#d62728"]
hatches = [None, "//", None, "//"]
bars = ax1.bar(xpos, alphas, yerr=errs, color=colors, capsize=3)
for bar, h in zip(bars, hatches):
    if h:
        bar.set_hatch(h)
ax1.axhline(-1.279, color="#ff7f0e", ls="--", lw=1, label="ASTRODEEP TNG-common α=-1.279")
ax1.axhline(0, color="#555555", lw=0.5)
ax1.set_xticks(xpos); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("Single-band F200W SB α")
ax1.legend(frameon=False, fontsize=8, loc="lower right")
ax1.set_title(f"PSF σ={PSF_SIGMA_PIX} pix, sky σ={SKY_SIGMA_NJY} nJy/pix", fontsize=10)

path = f"{PLOT_DIR}/psf_noise_injection_f200w.png"
fig.savefig(path, dpi=180)
plt.close(fig)
print(f"Saved {path}")
print("[done]", flush=True)

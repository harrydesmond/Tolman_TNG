"""Generate the final paper PDFs.

Writes the paper figures to ``$WORKDIR/Plots/final/*.pdf`` and
supporting diagnostics to ``$WORKDIR/Plots/diagnostics/*.pdf``.
``WORKDIR`` defaults to the repository root; set the environment
variable ``TOLMAN_WORKDIR`` to override when running on a different
cluster or workstation.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


WORKDIR = Path(os.environ.get("TOLMAN_WORKDIR", Path(__file__).resolve().parents[1]))
OUT_DIR = WORKDIR / "Plots" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR = WORKDIR / "Plots" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

TNG_COMMON_FILTERS = [
    "F435W", "F606W", "F814W",
    "F125W", "F140W", "F160W",
    "F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W",
]


def style():
    """Larger fonts and serif/mathtext to match LaTeX feel."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "lines.linewidth": 1.6,
    })


# ---------------------------------------------------------------------------
# Redshift-distribution figure (zdist.pdf)
# ---------------------------------------------------------------------------
def build_zdist():
    style()
    spec = pd.read_csv(
        WORKDIR / "data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv",
        usecols=["z", "mock_spec_selected"],
        low_memory=False,
    )
    tng_z = spec.loc[spec["mock_spec_selected"].astype(bool), "z"].dropna().values
    ast_z = pd.read_csv(WORKDIR / "data/astrodeep_spec_z.csv")["z"].dropna().values

    med_ast = float(np.median(ast_z))
    med_tng = float(np.median(tng_z))

    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    bins = np.linspace(0, 9, 40)
    ax.hist(ast_z, bins=bins, density=True, alpha=0.55, color="#1f77b4",
            label=f"ASTRODEEP spec ($N={len(ast_z):,}$)")
    ax.hist(tng_z, bins=bins, density=True, alpha=0.55, color="#d62728",
            label=f"TNG100 mock-spec ($N={len(tng_z):,}$)")
    ax.axvline(med_ast, color="#1f4f86", linestyle="--", linewidth=1.6,
               label=fr"ASTRODEEP median $z={med_ast:.2f}$")
    ax.axvline(med_tng, color="#8b1a1f", linestyle="--", linewidth=1.6,
               label=fr"TNG100 median $z={med_tng:.2f}$")
    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel("Normalised count")
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.legend(frameon=False)
    out = OUT_DIR / "zdist.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}  (medians: ASTRODEEP={med_ast:.3f}, TNG100={med_tng:.3f})")


# ---------------------------------------------------------------------------
# Helpers for per-galaxy SB fits
# ---------------------------------------------------------------------------
def _power_law_fit(x_log, y_log):
    n = len(x_log)
    A = np.c_[np.ones(n), x_log]
    coef, *_ = np.linalg.lstsq(A, y_log, rcond=None)
    beta, alpha = float(coef[0]), float(coef[1])
    resid = y_log - A @ coef
    s2 = float(np.sum(resid ** 2) / (n - 2))
    cov = s2 * np.linalg.inv(A.T @ A)
    alpha_err = float(cov[1, 1] ** 0.5)
    return alpha, alpha_err, beta


def _mean_sb_from_filters(df, flux_cols, area_col):
    """Per-galaxy mean SB across listed filters (NaN-safe)."""
    sb_per_filter = {}
    for c in flux_cols:
        if c not in df.columns:
            continue
        f = df[c].to_numpy(dtype=float)
        a = df[area_col].to_numpy(dtype=float)
        sb = np.where((np.isfinite(f)) & (np.isfinite(a)) & (a > 0) & (f > 0),
                      f / a, np.nan)
        sb_per_filter[c] = sb
    arr = np.column_stack(list(sb_per_filter.values()))
    with np.errstate(invalid="ignore"):
        mean_sb = np.nanmean(arr, axis=1)
    n_valid = np.sum(np.isfinite(arr), axis=1)
    return mean_sb, n_valid


# ---------------------------------------------------------------------------
# Mean-SB power-law fits, ASTRODEEP and TNG (mean_sb_fits.pdf)
# ---------------------------------------------------------------------------
def build_mean_sb_fits():
    style()

    # ---- ASTRODEEP TNG-common spectroscopic ----
    ast = pd.read_csv(
        WORKDIR / "results/astrodeep_cds/measured_catalog.csv",
        low_memory=False,
    )
    # Recompute mean SB over the 13 TNG-common filters
    sb_cols = [f"sb_{f}" for f in TNG_COMMON_FILTERS if f"sb_{f}" in ast.columns]
    sb_mat = ast[sb_cols].to_numpy(dtype=float)
    sb_mat = np.where((sb_mat > 0) & np.isfinite(sb_mat), sb_mat, np.nan)
    with np.errstate(invalid="ignore"):
        ast_sbmean = np.nanmean(sb_mat, axis=1)
    snr_cols = [f"snr_{f}" for f in TNG_COMMON_FILTERS if f"snr_{f}" in ast.columns]
    snr_max = np.nanmax(ast[snr_cols].to_numpy(dtype=float), axis=1)
    z_ast = pd.to_numeric(ast["z"], errors="coerce").to_numpy()
    has_spec = ast["has_spec"].astype(bool).to_numpy() if "has_spec" in ast.columns else np.zeros(len(ast), bool)
    flags = pd.to_numeric(ast.get("flags", pd.Series(0, index=ast.index)), errors="coerce").to_numpy()
    ap_area = pd.to_numeric(ast.get("aperture_area_arcsec2_used", pd.Series(np.nan, index=ast.index)), errors="coerce").to_numpy()
    n_valid_filters = np.sum(np.isfinite(sb_mat), axis=1)
    mask_ast = (
        np.isfinite(z_ast) & (z_ast >= 0.0) & (z_ast <= 14.0)
        & (n_valid_filters > 0)
        & np.isfinite(snr_max) & (snr_max >= 5.0)
        & np.isfinite(ap_area) & (ap_area > 0.0)
        & np.isfinite(flags) & (flags < 80)
        & has_spec
        & np.isfinite(ast_sbmean) & (ast_sbmean > 0)
    )
    z_a = z_ast[mask_ast]
    sb_a = ast_sbmean[mask_ast]
    print(f"ASTRODEEP TNG-common spectroscopic N = {mask_ast.sum()}")

    x_a = np.log10(1.0 + z_a)
    y_a = np.log10(sb_a)
    alpha_a, err_a, beta_a = _power_law_fit(x_a, y_a)

    # ---- TNG100 Voronoi-deblended mock-spec ----
    tng = pd.read_csv(
        WORKDIR / "data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv",
        low_memory=False,
    )
    perfilt = pd.read_csv(
        WORKDIR / "results/tng100_7_6_xyz_hst_jwst_imageap_deblended/deblending_per_filter.csv",
    )
    # Patch flux_<filter>_uJy with deblended_flux_uJy
    for filt in perfilt["filter"].unique():
        flux_col = f"flux_{filt}_uJy"
        if flux_col not in tng.columns:
            continue
        sub = perfilt[perfilt["filter"] == filt][["object_index", "deblended_flux_uJy"]]
        sub = sub.set_index("object_index")["deblended_flux_uJy"]
        valid = sub[sub.notna() & (sub > 0.0)]
        idx = tng.index[tng.index.isin(valid.index)]
        tng.loc[idx, flux_col] = valid.reindex(idx).values
    flux_cols = [f"flux_{f}_uJy" for f in TNG_COMMON_FILTERS if f"flux_{f}_uJy" in tng.columns]
    # area column
    area_col = "aperture_area_arcsec2_F200W" if "aperture_area_arcsec2_F200W" in tng.columns else None
    if area_col is None:
        # Pick any aperture_area_arcsec2_<filter> column
        area_col = next(c for c in tng.columns if c.startswith("aperture_area_arcsec2"))
    sb_mat_t = np.column_stack([
        np.where(
            (tng[c].to_numpy(dtype=float) > 0)
            & (tng[area_col].to_numpy(dtype=float) > 0)
            & np.isfinite(tng[c].to_numpy(dtype=float))
            & np.isfinite(tng[area_col].to_numpy(dtype=float)),
            tng[c].to_numpy(dtype=float) / tng[area_col].to_numpy(dtype=float),
            np.nan,
        )
        for c in flux_cols
    ])
    with np.errstate(invalid="ignore"):
        tng_sbmean = np.nanmean(sb_mat_t, axis=1)
    z_t = tng["z"].to_numpy(dtype=float)
    sel_t = tng["mock_spec_selected"].astype(bool).to_numpy()
    mask_t = sel_t & np.isfinite(z_t) & np.isfinite(tng_sbmean) & (tng_sbmean > 0)
    z_t = z_t[mask_t]; sb_t = tng_sbmean[mask_t]
    print(f"TNG100 deblended mock-spectroscopic N = {mask_t.sum()}")

    x_t = np.log10(1.0 + z_t)
    y_t = np.log10(sb_t)
    alpha_t, err_t, beta_t = _power_law_fit(x_t, y_t)

    # ---- two-panel figure (figure*, full text width)  ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    # --- panel a: ASTRODEEP ---
    ax = axes[0]
    ax.scatter(x_a, y_a, s=7, alpha=0.18, edgecolors="none", color="#3b6ea8",
               rasterized=True)
    xx = np.linspace(np.nanmin(x_a), np.nanmax(x_a), 200)
    yy_fit = beta_a + alpha_a * xx
    ax.plot(xx, yy_fit, color="#111111", linewidth=2.0,
            label=fr"fit $\alpha_{{\rm Tol}}={alpha_a:.2f}$")
    anchor_x = float(np.nanpercentile(x_a, 10.0))
    anchor_y = beta_a + alpha_a * anchor_x
    for slope, lbl in [(-3.3, r"FRW $(1+z)^{-3.3}$"), (-0.3, r"Non-exp. $(1+z)^{-0.3}$")]:
        ax.plot(xx, anchor_y + slope * (xx - anchor_x), linestyle="--",
                linewidth=1.4, label=lbl)
    ax.set_xlabel(r"$\log_{10}(1+z)$")
    ax.set_ylabel(r"$\log_{10}\,\langle I_{\rm SB}^{\rm obs}\rangle$  [$\mu$Jy / arcsec$^{2}$]")
    ax.set_ylim(-3, ax.get_ylim()[1])
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.03, 0.05, "(a) ASTRODEEP", transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="left")

    # --- panel b: TNG ---
    ax = axes[1]
    ax.scatter(x_t, y_t, s=7, alpha=0.18, edgecolors="none", color="#3b6ea8",
               rasterized=True)
    xx = np.linspace(np.nanmin(x_t), np.nanmax(x_t), 200)
    yy_fit = beta_t + alpha_t * xx
    ax.plot(xx, yy_fit, color="#111111", linewidth=2.0,
            label=fr"fit $\alpha_{{\rm Tol}}={alpha_t:.2f}$")
    anchor_x = float(np.nanpercentile(x_t, 10.0))
    anchor_y = beta_t + alpha_t * anchor_x
    for slope, lbl in [(-3.3, r"FRW $(1+z)^{-3.3}$"), (-0.3, r"Non-exp. $(1+z)^{-0.3}$")]:
        ax.plot(xx, anchor_y + slope * (xx - anchor_x), linestyle="--",
                linewidth=1.4, label=lbl)
    ax.set_xlabel(r"$\log_{10}(1+z)$")
    ax.set_ylabel(r"$\log_{10}\,\langle I_{\rm SB}^{\rm obs}\rangle$  [$\mu$Jy / arcsec$^{2}$]")
    ax.set_ylim(-1, ax.get_ylim()[1])
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.03, 0.05, "(b) IllustrisTNG", transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="left")

    out = OUT_DIR / "mean_sb_fits.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}  (alpha_a={alpha_a:.3f}, alpha_t={alpha_t:.3f})")


# ---------------------------------------------------------------------------
# Aperture-ladder slope bar chart (aperture_ladder_bars.pdf)
# ---------------------------------------------------------------------------
def build_aperture_ladder_bars():
    style()
    table = {
        # observable -> per-bar values [(label, alpha, err, color)]
        "Mean SB": [],
        "Max SB": [],
        "Modal mean SB": [],
        "Modal max SB": [],
    }

    src = {
        "TNG fixed aperture": (
            WORKDIR / "results/tng100_7_6_xyz_hst_jwst_imageap_specmodel/slope_table.csv",
            "mock_spectroscopic",
            "#1f77b4",
        ),
        "TNG Voronoi deblended": (
            WORKDIR / "results/tng100_7_6_xyz_hst_jwst_imageap_deblended_sb/slope_table.csv",
            "mock_spectroscopic",
            "#ff7f0e",
        ),
        "TNG adaptive aperture": (
            WORKDIR / "results/tng100_7_6_xyz_hst_jwst_imageap_astrodeep_aperture/slope_table.csv",
            "mock_spectroscopic",
            "#d62728",
        ),
        "ASTRODEEP observed": (
            WORKDIR / "results/astrodeep_cds_tng_common/slope_table.csv",
            "spectroscopic",
            "#2ca02c",
        ),
    }

    obs_keys = {
        "Mean SB": "mean_sb",
        "Max SB": "max_sb",
        "Modal mean SB": "modal_mean_sb",
        "Modal max SB": "modal_max_sb",
    }
    series = []  # [(label, color, [alphas], [errs])]
    for label, (path, sel, color) in src.items():
        df = pd.read_csv(path)
        alphas, errs = [], []
        for ok in obs_keys:
            obs = obs_keys[ok]
            row = df[(df["observable"] == obs) & (df["selection"] == sel)]
            alphas.append(float(row["alpha"].iloc[0]))
            errs.append(float(row["alpha_err"].iloc[0]))
        series.append((label, color, alphas, errs))

    n_groups = 4
    n_bars = len(series)
    bar_w = 0.18
    x_centres = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11.5, 6.0), constrained_layout=True)
    for i, (label, color, alphas, errs) in enumerate(series):
        offset = (i - (n_bars - 1) / 2.0) * bar_w
        ax.bar(x_centres + offset, alphas, bar_w, yerr=errs, color=color,
               capsize=3, label=label, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x_centres)
    ax.set_xticklabels(list(obs_keys.keys()))
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.axhline(-3.3, color="black", linestyle="--", linewidth=1.0,
               label=r"FRW $(1+z)^{-3.3}$")
    ax.axhline(-0.3, color="black", linestyle=":", linewidth=1.0,
               label=r"Non-exp. $(1+z)^{-0.3}$")
    # Block separators
    for k in range(1, n_groups):
        ax.axvline(k - 0.5, color="black", linewidth=0.6)
    ax.set_ylabel(r"$\alpha_{\rm Tol}$  ($I_{\rm SB}\propto(1+z)^{\alpha_{\rm Tol}}$)")
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylim(-3.4, 3.1)  # extend top so modal mean SB red error bar is visible
    # Legend low-left to avoid the dashed -1 line on the right
    ax.legend(frameon=False, loc="lower left", ncol=2)

    out = OUT_DIR / "aperture_ladder_bars.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# PSF + sky-noise injection diagnostic (Plots/diagnostics/psf_noise_injection.pdf)
# ---------------------------------------------------------------------------
def build_psf_noise_injection():
    """Re-run the PSF + sky-noise injection and plot the result."""
    from astropy.io import fits
    from scipy import ndimage

    style()

    PSF_SIGMA_PIX = 1.0
    SKY_SIGMA_NJY = 0.1
    RNG_SEED = 0
    FITS_F200W = WORKDIR / "data/raw/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f200w_v1_sim.fits"
    SPECMODEL_CSV = WORKDIR / "data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv"

    def astrodeep_radius_arcsec(z):
        # apopt is a diameter (Merlin+24); matching aperture radius = apopt/2.
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
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        s2 = float(np.sum((y - A @ c) ** 2) / (n - 2))
        err = float((s2 * np.linalg.inv(A.T @ A)[1, 1]) ** 0.5)
        return float(c[1]), err, n

    def aperture_flux(image, x, y, r_pix):
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
            py = np.arange(y0, y1) + 0.5
            px = np.arange(x0, x1) + 0.5
            dx = (px - xc); dy = (py - yc)
            d2 = dy[:, None] ** 2 + dx[None, :] ** 2
            mask = d2 <= rp ** 2
            out[i] = float(np.nansum(sub[mask]))
        return out

    print("[psf-noise] loading FITS...")
    with fits.open(FITS_F200W, memmap=False) as hdul:
        raw = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header
    print("[psf-noise] loading specmodel csv...")
    spec = pd.read_csv(SPECMODEL_CSV, low_memory=False)
    sel = spec["mock_spec_selected"].astype(bool).to_numpy()
    selected = spec.iloc[np.flatnonzero(sel)].copy()
    x = selected["x_F200W_pix"].to_numpy()
    y = selected["y_F200W_pix"].to_numpy()
    z = selected["z"].to_numpy()
    kpc_per_arcsec = selected["kpc_per_arcsec"].to_numpy()
    photrad_kpc = selected["aperture_radius_arcsec_F200W"].to_numpy() * kpc_per_arcsec
    try:
        from astropy.wcs import WCS
        import astropy.units as u
        wcs = WCS(hdr)
        pixscales = wcs.proj_plane_pixel_scales()
        arcsec_per_pix = float(pixscales[0].to(u.arcsec).value)
    except Exception:
        arcsec_per_pix = float(np.sqrt(137.0 * 3600.0 / float(raw.shape[0] * raw.shape[1])))
    if not (0.005 < arcsec_per_pix < 0.5):
        arcsec_per_pix = float(np.sqrt(137.0 * 3600.0 / float(raw.shape[0] * raw.shape[1])))

    r_fixed_pix = (photrad_kpc / kpc_per_arcsec) / arcsec_per_pix
    r_adaptive_arcsec = astrodeep_radius_arcsec(z)
    r_adaptive_pix = r_adaptive_arcsec / arcsec_per_pix

    print("[psf-noise] noise-free aperture flux...")
    flux_fixed_clean = aperture_flux(raw, x, y, r_fixed_pix) / 1000.0
    flux_adapt_clean = aperture_flux(raw, x, y, r_adaptive_pix) / 1000.0

    print("[psf-noise] PSF + noise...")
    img_psf = ndimage.gaussian_filter(raw, sigma=PSF_SIGMA_PIX).astype(np.float32)
    rng = np.random.default_rng(RNG_SEED)
    img_noisy = img_psf + rng.normal(0.0, SKY_SIGMA_NJY, size=raw.shape).astype(np.float32)
    del img_psf
    flux_fixed_noisy = aperture_flux(img_noisy, x, y, r_fixed_pix) / 1000.0
    flux_adapt_noisy = aperture_flux(img_noisy, x, y, r_adaptive_pix) / 1000.0

    ap_fixed_arcsec2 = np.pi * (r_fixed_pix * arcsec_per_pix) ** 2
    ap_adapt_arcsec2 = np.pi * r_adaptive_arcsec ** 2
    sb_fixed_clean = flux_fixed_clean / ap_fixed_arcsec2
    sb_adapt_clean = flux_adapt_clean / ap_adapt_arcsec2
    sb_fixed_noisy = flux_fixed_noisy / ap_fixed_arcsec2
    sb_adapt_noisy = flux_adapt_noisy / ap_adapt_arcsec2

    sb_arrays = [
        ("fixed-physical, noise-free", sb_fixed_clean),
        ("fixed-physical, PSF+noise",  sb_fixed_noisy),
        ("adaptive ASTRODEEP, noise-free", sb_adapt_clean),
        ("adaptive ASTRODEEP, PSF+noise",  sb_adapt_noisy),
    ]
    alphas, errs = [], []
    for label, sb in sb_arrays:
        a, e, n = fit_alpha(z, sb)
        print(f"  {label}: alpha={a:+.3f}+/-{e:.3f} (N={n})")
        alphas.append(a); errs.append(e)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)

    # left: scatter of adaptive aperture, clean vs noisy
    ax0 = axes[0]
    m_clean = np.isfinite(sb_adapt_clean) & (sb_adapt_clean > 0)
    m_noisy = np.isfinite(sb_adapt_noisy) & (sb_adapt_noisy > 0)
    ax0.scatter(z[m_clean], sb_adapt_clean[m_clean], s=8, alpha=0.20,
                color="#999999", rasterized=True, label="noise-free")
    ax0.scatter(z[m_noisy], sb_adapt_noisy[m_noisy], s=8, alpha=0.30,
                color="#d62728", rasterized=True, label="PSF + noise")
    ax0.set_yscale("log")
    ax0.set_xlabel(r"Redshift $z$")
    ax0.set_ylabel(r"ASTRODEEP-adaptive $I_{\rm SB}^{\rm F200W}$  [$\mu$Jy / arcsec$^{2}$]")
    ax0.legend(frameon=False, loc="lower left")
    ax0.set_title(r"Adaptive aperture", fontsize=15)

    # right: bar chart of alphas
    ax1 = axes[1]
    labels = ["fixed-phys.\nclean", "fixed-phys.\nPSF+noise",
              "adaptive\nclean", "adaptive\nPSF+noise"]
    xpos = np.arange(len(labels))
    colors = ["#1f77b4", "#1f77b4", "#d62728", "#d62728"]
    hatches = [None, "//", None, "//"]
    bars = ax1.bar(xpos, alphas, yerr=errs, color=colors, capsize=4,
                   edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    ax1.axhline(-1.279, color="#ff7f0e", linestyle="--", linewidth=1.4,
                label=r"ASTRODEEP TNG-common $\alpha=-1.279$")
    ax1.axhline(0.0, color="black", linewidth=0.6)
    ax1.set_xticks(xpos); ax1.set_xticklabels(labels)
    ax1.set_ylabel(r"Single-band F200W SB $\alpha$")
    ax1.legend(frameon=False, loc="lower right")
    ax1.set_title(fr"PSF $\sigma={PSF_SIGMA_PIX:g}$ pix,  sky $\sigma={SKY_SIGMA_NJY:g}$ nJy/pix",
                  fontsize=15)

    out = DIAG_DIR / "psf_noise_injection.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Li-DDR three-panel bridge figure (ddr_bridge.pdf)
# ---------------------------------------------------------------------------
def build_ddr_bridge():
    style()

    radio = pd.read_csv(WORKDIR / "results/li_ddr/radio_per_source_ddr.csv")
    tng = pd.read_csv(WORKDIR / "results/li_ddr/tng_per_source_ddr.csv")
    radio_z = radio["z"].to_numpy(dtype=float)
    radio_r = radio["ddr_ratio"].to_numpy(dtype=float)
    tng_z = tng["z"].to_numpy(dtype=float)
    tng_r = tng["ddr_ratio"].to_numpy(dtype=float)

    def equal_pop_bins(z, y, n_bins):
        m = np.isfinite(z) & np.isfinite(y)
        z = z[m]; y = y[m]
        order = np.argsort(z)
        z = z[order]; y = y[order]
        edges = np.linspace(0, len(z), n_bins + 1).astype(int)
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi - lo < 4:
                continue
            zb = z[lo:hi]; yb = y[lo:hi]
            med = float(np.median(yb))
            lo_y = float(np.percentile(yb, 16))
            hi_y = float(np.percentile(yb, 84))
            sigma_med = 1.2533 * np.std(yb, ddof=1) / np.sqrt(len(yb))
            out.append((float(np.median(zb)), med, lo_y, hi_y, float(sigma_med)))
        a = np.array(out)
        return dict(x=a[:, 0], med=a[:, 1], lo=a[:, 2], hi=a[:, 3], sigma_med=a[:, 4])

    rb = equal_pop_bins(radio_z, radio_r, 27)
    tb = equal_pop_bins(tng_z, tng_r, 27)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.fill_between(tb["x"], tb["lo"], tb["hi"], color="C2", alpha=0.18, lw=0,
                    label=fr"TNG100 F200W ({len(tng_z)}) 16--84\,\%")
    z_grid = np.linspace(0.01, 5.0, 400)
    ax.plot(z_grid, (1 + z_grid) ** 2, "--", color="C0",
            label=r"FRW  $D_L=(1+z)^{2} D_A$", zorder=2)
    ax.plot(z_grid, (1 + z_grid), "-", color="C3",
            label=r"Non-exp.  $D_L=(1+z) D_A$", zorder=2)
    ax.errorbar(rb["x"], rb["med"], yerr=rb["sigma_med"], fmt="o", color="black",
                ms=4.5, lw=0.8, capsize=2, zorder=4,
                label=fr"JJ2.29 + GKF5.0 ({len(radio_z)})")
    ax.errorbar(tb["x"], tb["med"], yerr=tb["sigma_med"], fmt="s", color="C2",
                ms=4.5, lw=0.8, capsize=2, mfc="white", mec="C2", zorder=3,
                label=fr"TNG100 F200W ({len(tng_z)}) median")
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"$D_L / D_A$")
    ax.set_xlim(0, 5.0)
    y_top = max(np.nanpercentile(rb["med"], 99),
                np.nanpercentile(tb["med"], 99),
                (1 + 5.0)) * 1.4
    ax.set_ylim(0, y_top)
    ax.legend(frameon=False, loc="upper left")
    out = OUT_DIR / "ddr_bridge.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    build_zdist()
    build_mean_sb_fits()
    build_aperture_ladder_bars()
    build_psf_noise_injection()
    build_ddr_bridge()
    print("All figures done.")


if __name__ == "__main__":
    main()

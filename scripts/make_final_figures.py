"""Build the zdist, mean-SB-fit and aperture-ladder paper figures locally.

Inputs (must be present locally):
    data/astrodeep_spec_z.csv                                # full ASTRODEEP spec z list
    data/astrodeep_tng_common_spec_z_sbmean.csv              # ASTRODEEP TNG-common spec sample (z, sb_mean)
    data/tng100_mockspec_z_sbmean_deblended.csv              # TNG100 Voronoi-deblended mock-spec sample (z, sb_mean)
    data/tng100_mockspec_z.csv                               # TNG100 mock-spec z values (zdist.pdf)
    data/psf_noise_injection_per_galaxy.csv                  # per-galaxy z, sb_*_clean/noisy diagnostic
    results/<...>/slope_table.csv                            # aperture_ladder_bars.pdf
    results/li_ddr/{radio,tng}_per_source_ddr.csv            # DDR diagnostics

Writes the paper figures to Plots/final/*.pdf and supporting diagnostics
to Plots/diagnostics/*.pdf.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parents[1]
OUT_DIR = HERE / "Plots" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR = HERE / "Plots" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def style():
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
        "lines.linewidth": 1.6,
    })


def _power_law_fit(x_log, y_log):
    n = len(x_log)
    A = np.c_[np.ones(n), x_log]
    coef, *_ = np.linalg.lstsq(A, y_log, rcond=None)
    beta, alpha = float(coef[0]), float(coef[1])
    resid = y_log - A @ coef
    s2 = float(np.sum(resid ** 2) / (n - 2))
    cov = s2 * np.linalg.inv(A.T @ A)
    return alpha, float(cov[1, 1] ** 0.5), beta


# ---------------------------------------------------------------------------
def build_zdist():
    style()
    ast_z = pd.read_csv(HERE / "data/astrodeep_spec_z.csv")["z"].dropna().values
    tng_z = pd.read_csv(HERE / "data/tng100_mockspec_z.csv")["z"].dropna().values
    med_ast = float(np.median(ast_z))
    med_tng = float(np.median(tng_z))

    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    bins = np.linspace(0, 9, 40)
    ax.hist(ast_z, bins=bins, density=True, alpha=0.55, color="#1f77b4",
            label=fr"ASTRODEEP spec ($N={len(ast_z):,}$)")
    ax.hist(tng_z, bins=bins, density=True, alpha=0.55, color="#d62728",
            label=fr"TNG100 mock-spec ($N={len(tng_z):,}$)")
    ax.axvline(med_ast, color="#1f4f86", linestyle="--", linewidth=1.8,
               label=fr"ASTRODEEP median $z={med_ast:.2f}$")
    ax.axvline(med_tng, color="#8b1a1f", linestyle="--", linewidth=1.8,
               label=fr"TNG100 median $z={med_tng:.2f}$")
    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel("Normalised count")
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.legend(frameon=False)
    out = OUT_DIR / "zdist.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
def build_mean_sb_fits():
    style()
    a = pd.read_csv(HERE / "data/astrodeep_tng_common_spec_z_sbmean.csv")
    t = pd.read_csv(HERE / "data/tng100_mockspec_z_sbmean_deblended.csv")
    z_a = a["z"].to_numpy(dtype=float); sb_a = a["sb_mean"].to_numpy(dtype=float)
    z_t = t["z"].to_numpy(dtype=float); sb_t = t["sb_mean"].to_numpy(dtype=float)

    x_a = np.log10(1.0 + z_a); y_a = np.log10(sb_a)
    x_t = np.log10(1.0 + z_t); y_t = np.log10(sb_t)
    alpha_a, _, beta_a = _power_law_fit(x_a, y_a)
    alpha_t, _, beta_t = _power_law_fit(x_t, y_t)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.scatter(x_a, y_a, s=7, alpha=0.25, edgecolors="none",
               color="black", rasterized=True)
    xx = np.linspace(np.nanmin(x_a), np.nanmax(x_a), 200)
    ax.plot(xx, beta_a + alpha_a * xx, color="black", linewidth=2.6,
            label=fr"fit $\alpha_{{\rm Tol}}={alpha_a:.2f}$")
    anchor_x = float(np.nanpercentile(x_a, 10.0))
    anchor_y = beta_a + alpha_a * anchor_x
    for slope, lbl, col in [(-3.3, r"FRW ($\alpha_{\rm SED}=0.3$): $(1+z)^{-3.3}$", "C0"),
                            (-0.3, r"Non-exp. ($\alpha_{\rm SED}=0.3$): $(1+z)^{-0.3}$", "C3")]:
        ax.plot(xx, anchor_y + slope * (xx - anchor_x), linestyle="--",
                linewidth=1.4, color=col, label=lbl)
    ax.set_xlabel(r"$\log_{10}(1+z)$")
    ax.set_ylabel(r"$\log_{10}\,\langle I_{\rm SB}^{\rm obs}\rangle\ [\mu\mathrm{Jy}\,/\,\mathrm{arcsec}^{2}]$")
    ax.set_xlim(np.nanmin(x_a), np.nanmax(x_a))
    ax.set_ylim(np.nanmin(y_a), np.nanmax(y_a))
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.03, 0.05, "(a) ASTRODEEP", transform=ax.transAxes,
            fontsize=15, fontweight="bold")

    ax = axes[1]
    tng_green = "#2ca02c"
    ax.scatter(x_t, y_t, s=7, alpha=0.25, edgecolors="none",
               color=tng_green, rasterized=True)
    xx = np.linspace(np.nanmin(x_t), np.nanmax(x_t), 200)
    ax.plot(xx, beta_t + alpha_t * xx, color=tng_green, linewidth=2.6,
            label=fr"fit $\alpha_{{\rm Tol}}={alpha_t:.2f}$")
    anchor_x = float(np.nanpercentile(x_t, 10.0))
    anchor_y = beta_t + alpha_t * anchor_x
    for slope, lbl, col in [(-3.3, r"FRW ($\alpha_{\rm SED}=0.3$): $(1+z)^{-3.3}$", "C0"),
                            (-0.3, r"Non-exp. ($\alpha_{\rm SED}=0.3$): $(1+z)^{-0.3}$", "C3")]:
        ax.plot(xx, anchor_y + slope * (xx - anchor_x), linestyle="--",
                linewidth=1.4, color=col, label=lbl)
    ax.set_xlabel(r"$\log_{10}(1+z)$")
    ax.set_ylabel(r"$\log_{10}\,\langle I_{\rm SB}^{\rm obs}\rangle\ [\mu\mathrm{Jy}\,/\,\mathrm{arcsec}^{2}]$")
    ax.set_xlim(np.nanmin(x_t), np.nanmax(x_t))
    ax.set_ylim(np.nanmin(y_t), np.nanmax(y_t))
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.03, 0.05, "(b) IllustrisTNG", transform=ax.transAxes,
            fontsize=15, fontweight="bold")

    out = OUT_DIR / "mean_sb_fits.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}  alpha_a={alpha_a:.3f}  alpha_t={alpha_t:.3f}")


# ---------------------------------------------------------------------------
def build_aperture_ladder_bars():
    style()
    src = {
        "TNG catalogue flux": (
            HERE / "results/tng100_7_6_xyz_hst_jwst_specmodel/slope_table.csv",
            "mock_spectroscopic", "#9467bd"),
        "TNG fixed aperture": (
            HERE / "results/tng100_7_6_xyz_hst_jwst_imageap_specmodel/slope_table.csv",
            "mock_spectroscopic", "#1f77b4"),
        "TNG Voronoi deblended": (
            HERE / "results/tng100_7_6_xyz_hst_jwst_imageap_deblended_sb/slope_table.csv",
            "mock_spectroscopic", "#ff7f0e"),
        "TNG adaptive aperture": (
            HERE / "results/tng100_7_6_xyz_hst_jwst_imageap_astrodeep_aperture/slope_table.csv",
            "mock_spectroscopic", "#d62728"),
        "ASTRODEEP observed": (
            HERE / "results/astrodeep_cds_tng_common/slope_table.csv",
            "spectroscopic", "#2ca02c"),
    }
    obs_keys = {"Mean SB": "mean_sb", "Max SB": "max_sb",
                "Modal mean SB": "modal_mean_sb", "Modal max SB": "modal_max_sb"}

    series = []
    for label, (path, sel, color) in src.items():
        df = pd.read_csv(path)
        alphas, errs = [], []
        for ok, oname in obs_keys.items():
            row = df[(df["observable"] == oname) & (df["selection"] == sel)]
            alphas.append(float(row["alpha"].iloc[0]))
            errs.append(float(row["alpha_err"].iloc[0]))
        series.append((label, color, alphas, errs))

    n_groups = 4
    n_bars = len(series)
    bar_w = 0.15
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
    for k in range(1, n_groups):
        ax.axvline(k - 0.5, color="black", linewidth=0.6)
    ax.set_ylabel(r"$\alpha_{\rm Tol}$  ($I_{\rm SB}\propto(1+z)^{\alpha_{\rm Tol}}$)")
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylim(-3.4, 3.1)
    # Legend above the axes so that no entry crosses the vertical dividers
    # between the four observable panels.
    handles, labels = ax.get_legend_handles_labels()
    hl = dict(zip(labels, handles))
    order = [r"FRW $(1+z)^{-3.3}$", r"Non-exp. $(1+z)^{-0.3}$",
             "ASTRODEEP observed", "TNG catalogue flux", "TNG fixed aperture",
             "TNG Voronoi deblended", "TNG adaptive aperture"]
    ax.legend([hl[l] for l in order], order, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              columnspacing=1.5, handletextpad=0.5)

    out = OUT_DIR / "aperture_ladder_bars.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
def build_psf_noise_injection():
    """Plot from the per-galaxy CSV produced on the cluster."""
    style()
    df = pd.read_csv(HERE / "data/psf_noise_injection_per_galaxy.csv")
    z = df["z"].to_numpy(dtype=float)
    sb_adapt_clean = df["sb_adapt_clean"].to_numpy(dtype=float)
    sb_adapt_noisy = df["sb_adapt_noisy"].to_numpy(dtype=float)
    sb_fixed_clean = df["sb_fixed_clean"].to_numpy(dtype=float)
    sb_fixed_noisy = df["sb_fixed_noisy"].to_numpy(dtype=float)

    def fit_alpha(z, sb):
        m = np.isfinite(z) & np.isfinite(sb) & (z > -1) & (sb > 0)
        n = int(m.sum())
        x = np.log10(1.0 + z[m]); y = np.log10(sb[m])
        A = np.c_[np.ones(n), x]
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        s2 = float(np.sum((y - A @ c) ** 2) / (n - 2))
        err = float((s2 * np.linalg.inv(A.T @ A)[1, 1]) ** 0.5)
        return float(c[1]), err, n

    pairs = [("fixed-physical, noise-free", sb_fixed_clean),
             ("fixed-physical, PSF+noise",  sb_fixed_noisy),
             ("adaptive ASTRODEEP, noise-free", sb_adapt_clean),
             ("adaptive ASTRODEEP, PSF+noise", sb_adapt_noisy)]
    alphas, errs = [], []
    for label, sb in pairs:
        a, e, n = fit_alpha(z, sb)
        alphas.append(a); errs.append(e)
        print(f"  {label}: alpha={a:+.3f}+/-{e:.3f} (N={n})")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    ax0 = axes[0]
    m_clean = np.isfinite(sb_adapt_clean) & (sb_adapt_clean > 0)
    m_noisy = np.isfinite(sb_adapt_noisy) & (sb_adapt_noisy > 0)
    ax0.scatter(z[m_clean], sb_adapt_clean[m_clean], s=8, alpha=0.20,
                color="#999999", rasterized=True, label="noise-free")
    ax0.scatter(z[m_noisy], sb_adapt_noisy[m_noisy], s=8, alpha=0.30,
                color="#d62728", rasterized=True, label="PSF + noise")
    ax0.set_yscale("log")
    ax0.set_xlabel(r"Redshift $z$")
    ax0.set_ylabel(r"ASTRODEEP-adaptive $I_{\rm SB}^{\rm F200W}\ [\mu\mathrm{Jy}\,/\,\mathrm{arcsec}^{2}]$")
    ax0.legend(frameon=False, loc="lower left")
    ax0.set_title("Adaptive aperture", fontsize=15)

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
    ax1.set_title(r"PSF $\sigma=1$ pix,  sky $\sigma=0.1$ nJy/pix", fontsize=15)

    out = DIAG_DIR / "psf_noise_injection.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
def build_ddr_bridge():
    print("Skipping ddr_bridge.pdf here; use scripts/fig_ddr_normalisation_compare.py for the paper version.")


def main():
    build_zdist()
    build_mean_sb_fits()
    build_aperture_ladder_bars()
    build_psf_noise_injection()
    build_ddr_bridge()
    print("All figures done.")


if __name__ == "__main__":
    main()

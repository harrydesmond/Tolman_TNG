"""Diagnostic plots for the Li 2023 reproduction and TNG analogue."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

from .li_reproduction import COSMO_LI, RAD_PER_MAS, bin_median


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_luminosity_redshift(
    df: pd.DataFrame,
    L_cgs: np.ndarray,
    fit_log_L_at_z,
    *,
    sample_label: str,
    out_path: Path,
) -> Path:
    """Li 2023 Fig. 2: log10 L_t vs redshift with the parameterised best fit."""
    z = df["z"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.scatter(z, np.log10(L_cgs), s=4, color="0.4", alpha=0.7, label=sample_label)
    z_grid = np.linspace(max(z.min(), 1e-3), z.max(), 200)
    ax.plot(z_grid, fit_log_L_at_z(z_grid), color="C3", lw=2, label="Li eq. (5) fit")
    ax.set_xlabel("redshift z")
    ax.set_ylabel(r"$\log_{10} (L_t / [\mathrm{erg\,s^{-1}\,Hz^{-1}}])$")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=180)
    plt.close(fig)
    return out_path


def plot_calibrated_theta_z(
    z: np.ndarray,
    theta_calib_mas: np.ndarray,
    *,
    sample_label: str,
    n_bins: int,
    cosmo: FlatLambdaCDM = COSMO_LI,
    out_path: Path,
) -> Path:
    """Li 2023 Fig. 3: binned median theta_calib vs z, with the two predictions."""
    binned = bin_median(z, theta_calib_mas, n_bins=n_bins)
    z_grid = np.linspace(max(z.min(), 1e-3), z.max(), 400)
    D_A_pc = cosmo.angular_diameter_distance(z_grid).to("pc").value

    # d_0 for each line: median per_source_d0 under each hypothesis.
    d0_exp = float(np.nanmedian(theta_calib_mas * RAD_PER_MAS * cosmo.angular_diameter_distance(z).to("pc").value))
    d0_non = float(np.nanmedian(theta_calib_mas * RAD_PER_MAS * cosmo.angular_diameter_distance(z).to("pc").value * (1 + z)))

    theta_exp = (d0_exp / D_A_pc) / RAD_PER_MAS
    theta_non = (d0_non / ((1 + z_grid) * D_A_pc)) / RAD_PER_MAS

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.errorbar(
        binned["x_median"], np.log10(binned["y_median"]),
        yerr=binned["y_sigma_med"] / (np.log(10.0) * binned["y_median"]),
        fmt="o", color="black", ms=4, lw=0.7, capsize=2, label=f"{sample_label} (binned)",
    )
    ax.plot(z_grid, np.log10(theta_exp), "--", color="C0", label=r"FRW $\theta = d_0/D_A$")
    ax.plot(z_grid, np.log10(theta_non), "-", color="C3", label=r"non-exp. $\theta = d_0/[(1+z)D_A]$")
    ax.set_xlabel("redshift z")
    ax.set_ylabel(r"$\log_{10} \theta$ [mas]")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=180)
    plt.close(fig)
    return out_path


def plot_distance_duality(
    z: np.ndarray,
    ddr_ratio: np.ndarray,
    *,
    sample_label: str,
    n_bins: int,
    cosmo: FlatLambdaCDM = COSMO_LI,
    out_path: Path,
) -> Path:
    """Li 2023 Fig. 4: D_L/D_A vs z, with FRW and non-expanding lines."""
    binned = bin_median(z, ddr_ratio, n_bins=n_bins)
    z_grid = np.linspace(max(z.min(), 1e-3), z.max(), 400)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.errorbar(
        binned["x_median"], binned["y_median"],
        yerr=binned["y_sigma_med"], fmt="o", color="black", ms=4, lw=0.7,
        capsize=2, label=f"{sample_label} (binned)",
    )
    ax.plot(z_grid, (1 + z_grid), "-", color="C3", label=r"non-exp. $D_L = (1+z)D_A$")
    ax.plot(z_grid, (1 + z_grid) ** 2, "--", color="C0", label=r"FRW $D_L = (1+z)^2 D_A$")
    ax.set_xlabel("redshift z")
    ax.set_ylabel(r"$D_L / D_A$")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=180)
    plt.close(fig)
    return out_path


def _bin_quantiles(x: np.ndarray, y: np.ndarray, *, n_bins: int):
    """Equal-population binning with median + 16/84 percentiles per bin."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    order = np.argsort(x)
    x, y = x[order], y[order]
    edges = np.linspace(0, len(x), n_bins + 1, dtype=int)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo < 3:
            continue
        xb, yb = x[lo:hi], y[lo:hi]
        rows.append(
            dict(
                x_median=float(np.median(xb)),
                y_median=float(np.median(yb)),
                y_lo=float(np.percentile(yb, 16.0)),
                y_hi=float(np.percentile(yb, 84.0)),
                y_sigma_med=float(1.2533 * np.std(yb, ddof=1) / np.sqrt(len(yb))),
                n=int(len(yb)),
            )
        )
    return pd.DataFrame(rows)


def plot_raw_ddr(
    *,
    z: np.ndarray,
    ratio: np.ndarray,
    sample_label: str,
    out_path: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    point_color: str = "C2",
    point_alpha: float = 0.35,
    point_size: float = 6.0,
) -> Path:
    """Per-source D_L/D_A vs z (no binning), with both reference lines."""
    z = np.asarray(z, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    if xlim is None:
        xlim = (0.0, float(np.nanmax(z)) * 1.02)
    z_grid = np.linspace(max(min(xlim[0], 1e-3), 1e-3), xlim[1], 400)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.scatter(z, ratio, s=point_size, color=point_color, alpha=point_alpha,
               edgecolor="none", zorder=2, label=sample_label)
    ax.plot(z_grid, (1 + z_grid) ** 2, "--", color="C0", lw=1.6,
            label=r"FRW $D_L = (1+z)^2 D_A$", zorder=3)
    ax.plot(z_grid, (1 + z_grid), "-", color="C3", lw=1.6,
            label=r"non-exp. $D_L = (1+z) D_A$", zorder=3)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"$D_L / D_A$")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=180)
    plt.close(fig)
    return out_path


def plot_sb_vs_z_raw(
    *,
    z: np.ndarray,
    sb: np.ndarray,
    sample_label: str,
    out_path: Path,
    sb_unit_label: str = r"SB [$\mu$Jy / arcsec$^2$]",
    z_anchor_max: float = 0.5,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    point_color: str = "0.4",
    point_alpha: float = 0.35,
    point_size: float = 6.0,
) -> Path:
    """Per-galaxy SB vs (1+z) on log-log axes, with broadband reference lines.

    Tsymbal+ 2026 Fig. 2 overplots the classical bolometric/tired-light
    -4/-1 powers on a broadband catalogue-flux quantity. For the alpha_SED=0
    specific-flux convention used here, the corresponding reference powers
    are -3 (FRW) and 0 (non-expanding/tired-light).
    """
    z = np.asarray(z, dtype=float)
    sb = np.asarray(sb, dtype=float)
    finite = np.isfinite(z) & np.isfinite(sb) & (sb > 0)
    z, sb = z[finite], sb[finite]
    one_plus_z = 1.0 + z

    anchor_mask = z < z_anchor_max
    if anchor_mask.sum() < 5:  # not enough low-z anchors; use the lowest 10%
        order = np.argsort(z)
        anchor_mask = np.zeros_like(z, dtype=bool)
        anchor_mask[order[: max(int(0.1 * len(z)), 5)]] = True
    sb_anchor = float(np.median(sb[anchor_mask]))
    z_anchor = float(np.median(one_plus_z[anchor_mask]))
    norm_lcdm = sb_anchor * z_anchor ** 3
    norm_tl = sb_anchor

    if xlim is None:
        xlim = (1.0, float(np.nanmax(one_plus_z)) * 1.05)
    z_grid = np.linspace(xlim[0], xlim[1], 400)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.scatter(one_plus_z, sb, s=point_size, color=point_color,
               alpha=point_alpha, edgecolor="none", zorder=2,
               label=sample_label)
    ax.plot(z_grid, norm_lcdm * z_grid ** -3, "-", color="C3", lw=1.6,
            label=r"FRW $\propto (1+z)^{-3}$", zorder=3)
    ax.plot(z_grid, norm_tl * np.ones_like(z_grid), "-", color="C0", lw=1.6,
            label=r"non-exp. $\propto (1+z)^0$", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(r"$1 + z$")
    ax.set_ylabel(sb_unit_label)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=180)
    plt.close(fig)
    return out_path


def plot_combined_ddr(
    *,
    radio_z: np.ndarray,
    radio_ratio: np.ndarray,
    tng_z: np.ndarray,
    tng_ratio: np.ndarray,
    radio_label: str,
    tng_label: str,
    out_path: Path,
    radio_n_bins: int = 27,
    tng_n_bins: int = 27,
) -> Path:
    """Reproduce Li 2023 Fig. 4 with TNG100 overlaid as binned points + band.

    Both samples are binned with equal population per bin (Li's convention,
    27 bins for the combined radio sample of 943 sources).  TNG band shows
    the 16-84 percentile spread to communicate the per-galaxy scatter; the
    overlaid TNG points are the per-bin medians with sigma_med error bars.
    """
    radio_bin = _bin_quantiles(radio_z, radio_ratio, n_bins=radio_n_bins)
    tng_bin = _bin_quantiles(tng_z, tng_ratio, n_bins=tng_n_bins)

    z_lo = max(min(radio_z.min(), tng_z.min()), 1e-3)
    z_hi = max(radio_z.max(), tng_z.max())
    z_grid = np.linspace(z_lo, z_hi, 400)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    # TNG band first (under everything else)
    ax.fill_between(
        tng_bin["x_median"], tng_bin["y_lo"], tng_bin["y_hi"],
        color="C2", alpha=0.18, lw=0, label=f"{tng_label} 16-84%",
    )
    # Reference lines
    ax.plot(z_grid, (1 + z_grid) ** 2, "--", color="C0",
            label=r"FRW $D_L = (1+z)^2 D_A$", zorder=2)
    ax.plot(z_grid, (1 + z_grid), "-", color="C3",
            label=r"non-exp. $D_L = (1+z) D_A$", zorder=2)
    # Radio binned (Li's points)
    ax.errorbar(
        radio_bin["x_median"], radio_bin["y_median"],
        yerr=radio_bin["y_sigma_med"], fmt="o", color="black",
        ms=4, lw=0.7, capsize=2, zorder=4, label=radio_label,
    )
    # TNG binned medians
    ax.errorbar(
        tng_bin["x_median"], tng_bin["y_median"],
        yerr=tng_bin["y_sigma_med"], fmt="s", color="C2",
        ms=4, lw=0.7, capsize=2, mfc="white", mec="C2", zorder=3,
        label=f"{tng_label} median",
    )
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"$D_L / D_A$")
    ax.set_xlim(0, z_hi * 1.02)
    ax.set_ylim(0, max(np.percentile(radio_bin["y_median"], 99),
                        np.percentile(tng_bin["y_median"], 99),
                        (1 + z_hi)) * 1.4)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(_ensure_dir(out_path), dpi=180)
    plt.close(fig)
    return out_path

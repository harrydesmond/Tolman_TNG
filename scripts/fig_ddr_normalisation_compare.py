#!/usr/bin/env python3
"""Build the three-panel DDR normalisation comparison used as Fig. 4.

Panel (a) uses the non-expanding d_0 convention of Li (2023), panel (b)
uses the FRW d_0 convention, and panel (c) anchors each sample to its
lowest-redshift bin so that d_0 cancels.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "results" / "li_ddr"
OUT = REPO / "Plots" / "Fig6_alt_normalisations.png"
OUT_P1684 = REPO / "Plots" / "Fig6_alt_normalisations_p1684.png"
OUT_FINAL = REPO / "Plots" / "final" / "ddr_bridge.pdf"
WRITE_DIAGNOSTICS = "--write-diagnostics" in sys.argv

radio = pd.read_csv(DATA / "radio_per_source_ddr.csv")
tng = pd.read_csv(DATA / "tng_per_source_ddr.csv")
radio_d0 = pd.read_csv(DATA / "li_radio_table1.csv").set_index("sample")
tng_d0 = pd.read_csv(DATA / "tng_li_analogue.csv")
tng_d0_row = tng_d0[tng_d0["power"] == 3].iloc[0]

# Cached ratios use the non-expanding normalisation. Moving to the FRW
# convention is a constant d0_non / d0_exp scaling for each sample.
factor_tng = float(tng_d0_row["d0_nonexpanding_pc"] / tng_d0_row["d0_expanding_pc"])
factor_jj = float(
    radio_d0.loc["JJ2.29", "d0_nonexpanding_pc"]
    / radio_d0.loc["JJ2.29", "d0_expanding_pc"]
)
factor_gkf = float(
    radio_d0.loc["GKF5.0", "d0_nonexpanding_pc"]
    / radio_d0.loc["GKF5.0", "d0_expanding_pc"]
)

radio_non = radio["ddr_ratio"].to_numpy(float)
tng_non = tng["ddr_ratio"].to_numpy(float)
radio_exp = radio_non.copy()
m_jj = radio["sample"].to_numpy() == "JJ2.29"
m_gkf = radio["sample"].to_numpy() == "GKF5.0"
radio_exp[m_jj] = radio_non[m_jj] * factor_jj
radio_exp[m_gkf] = radio_non[m_gkf] * factor_gkf
tng_exp = tng_non * factor_tng

print(
    f"d0_non/d0_exp factors -- TNG: {factor_tng:.3f}, "
    f"JJ: {factor_jj:.3f}, GKF: {factor_gkf:.3f}"
)


def equal_pop_bins(z, y, n_bins):
    finite = np.isfinite(z) & np.isfinite(y)
    z = z[finite]
    y = y[finite]
    order = np.argsort(z)
    z = z[order]
    y = y[order]
    edges = np.linspace(0, len(z), n_bins + 1).astype(int)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < 4:
            continue
        zb = z[lo:hi]
        yb = y[lo:hi]
        med = float(np.median(yb))
        sigma_med = 1.2533 * float(np.std(yb, ddof=1)) / np.sqrt(len(yb))
        out.append(
            (
                float(np.median(zb)),
                med,
                sigma_med,
                float(np.percentile(yb, 16)),
                float(np.percentile(yb, 84)),
            )
        )
    values = np.array(out)
    return {
        "x": values[:, 0],
        "med": values[:, 1],
        "sigma": values[:, 2],
        "lo": values[:, 3],
        "hi": values[:, 4],
    }


radio_z = radio["z"].to_numpy(float)
tng_z = tng["z"].to_numpy(float)
rb_non = equal_pop_bins(radio_z, radio_non, 27)
rb_exp = equal_pop_bins(radio_z, radio_exp, 27)
tb_non = equal_pop_bins(tng_z, tng_non, 27)
tb_exp = equal_pop_bins(tng_z, tng_exp, 27)

z_grid = np.linspace(0.01, 5.0, 400)
frw = (1 + z_grid) ** 2
non = 1 + z_grid
r_anchor_r = rb_non["med"][0]
r_anchor_t = tb_non["med"][0]
z_a_ref = rb_non["x"][0]
frw_anchored = (1 + z_grid) ** 2 / (1 + z_a_ref) ** 2
non_anchored = (1 + z_grid) / (1 + z_a_ref)


def fit_power_law_slope(binned):
    x = np.log(1.0 + np.asarray(binned["x"], dtype=float))
    y = np.log(np.asarray(binned["med"], dtype=float))
    sigma = np.asarray(binned["sigma"], dtype=float) / np.asarray(binned["med"], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0.0)
    beta, _ = np.polyfit(x[finite], y[finite], 1, w=1.0 / sigma[finite])
    return float(beta)


def fit_power_law_norm(z, y, beta):
    x = np.log(1.0 + np.asarray(z, dtype=float))
    log_y = np.log(np.asarray(y, dtype=float))
    finite = np.isfinite(x) & np.isfinite(log_y)
    return float(np.exp(np.mean(log_y[finite] - beta * x[finite])))


radio_beta = fit_power_law_slope(rb_non)
AXIS_LABEL_FS = 15
TICK_LABEL_FS = 13


def errorbars(binned, style, scale=1.0):
    if style == "sem":
        return binned["sigma"] / scale
    lo = (binned["med"] - binned["lo"]) / scale
    hi = (binned["hi"] - binned["med"]) / scale
    return np.vstack([lo, hi])


def draw_absolute(ax, radio_binned, tng_binned, *, err_style):
    ax.fill_between(
        tng_binned["x"],
        tng_binned["lo"],
        tng_binned["hi"],
        color="C2",
        alpha=0.18,
        lw=0,
        label=fr"TNG100 F200W ({len(tng_z)}) 16 to 84%",
    )
    ax.plot(z_grid, frw, "--", color="C0", label=r"FRW $D_L=(1+z)^{2} D_A$", zorder=2)
    ax.plot(z_grid, non, "--", color="C3", label=r"Non-exp.  $D_L=(1+z) D_A$", zorder=2)
    ax.errorbar(
        radio_binned["x"],
        radio_binned["med"],
        yerr=errorbars(radio_binned, err_style),
        fmt="o",
        color="black",
        ms=4.5,
        lw=0.8,
        capsize=2,
        zorder=4,
        label=fr"JJ+GKF ({len(radio_z)})",
    )
    if err_style == "sem":
        ax.errorbar(
            tng_binned["x"],
            tng_binned["med"],
            yerr=tng_binned["sigma"],
            fmt="s",
            color="C2",
            ms=4.5,
            lw=0.8,
            capsize=2,
            mfc="white",
            mec="C2",
            zorder=3,
            label=fr"TNG100 F200W ({len(tng_z)}) median",
        )
    else:
        ax.plot(
            tng_binned["x"],
            tng_binned["med"],
            "s",
            color="C2",
            ms=4.5,
            mfc="white",
            mec="C2",
            zorder=6,
            label=fr"TNG100 F200W ({len(tng_z)}) median",
        )
    fit_norm = fit_power_law_norm(radio_binned["x"], radio_binned["med"], radio_beta)
    ax.plot(z_grid, fit_norm * (1.0 + z_grid) ** radio_beta, color="black", lw=1.0, alpha=0.85)
    ax.set_xlabel(r"redshift $z$", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel(r"$D_L\,/\,D_A^{\rm Li}$", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FS)
    ax.set_xlim(0, 5.0)
    if err_style == "sem":
        y_top = max(np.nanpercentile(radio_binned["med"], 99), np.nanpercentile(tng_binned["med"], 99), 6.0) * 1.4
    else:
        y_top = max(np.nanpercentile(radio_binned["hi"], 95), np.nanpercentile(tng_binned["hi"], 95), 6.0) * 1.2
    ax.set_ylim(0, y_top)


def draw_anchored(ax, *, err_style):
    ax.fill_between(
        tb_non["x"], tb_non["lo"] / r_anchor_t, tb_non["hi"] / r_anchor_t,
        color="C2", alpha=0.18, lw=0,
    )
    ax.plot(z_grid, frw_anchored, "--", color="C0", zorder=2)
    ax.plot(z_grid, non_anchored, "--", color="C3", zorder=2)
    ax.errorbar(
        rb_non["x"], rb_non["med"] / r_anchor_r,
        yerr=errorbars(rb_non, err_style, scale=r_anchor_r),
        fmt="o", color="black", ms=4.5, lw=0.8, capsize=2, zorder=4,
    )
    if err_style == "sem":
        ax.errorbar(
            tb_non["x"], tb_non["med"] / r_anchor_t,
            yerr=tb_non["sigma"] / r_anchor_t, fmt="s", color="C2",
            ms=4.5, lw=0.8, capsize=2, mfc="white", mec="C2", zorder=3,
        )
    else:
        ax.plot(
            tb_non["x"], tb_non["med"] / r_anchor_t, "s", color="C2",
            ms=4.5, mfc="white", mec="C2", zorder=6,
        )
    fit_norm = fit_power_law_norm(rb_non["x"], rb_non["med"] / r_anchor_r, radio_beta)
    ax.plot(z_grid, fit_norm * (1.0 + z_grid) ** radio_beta, color="black", lw=1.0, alpha=0.85)
    ax.set_xlabel(r"redshift $z$", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel(r"$(D_L/D_A^{\rm Li})(z)\,/\,(D_L/D_A^{\rm Li})(z_{\rm anchor})$", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FS)
    ax.set_xlim(0, 5.0)
    if err_style == "sem":
        ax.set_ylim(0, 11.0)
    else:
        ymax = max(np.nanpercentile(rb_non["hi"] / r_anchor_r, 95), np.nanpercentile(tb_non["hi"] / r_anchor_t, 95))
        ax.set_ylim(0, ymax * 1.15)


def build(out_path, *, err_style):
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.5), sharey=False)
    draw_absolute(axes[0], rb_non, tb_non, err_style=err_style)
    draw_absolute(axes[1], rb_exp, tb_exp, err_style=err_style)
    draw_anchored(axes[2], err_style=err_style)
    labels = [r"(a) non-expanding $d_0$", r"(b) FRW $d_0$", r"(c) self-anchored"]
    for ax, label in zip(axes, labels):
        ax.text(0.97, 0.03, label, transform=ax.transAxes, fontsize=15, fontweight="bold", ha="right", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper center", ncol=len(handles), fontsize=14,
        bbox_to_anchor=(0.03, 0.92, 0.94, 0.06), mode="expand",
        frameon=True, columnspacing=2.0, handletextpad=0.6, borderpad=0.7,
    )
    legend.get_frame().set_edgecolor("0.5")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_facecolor("white")
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.12, top=0.86, wspace=0.16)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if WRITE_DIAGNOSTICS:
    build(OUT, err_style="sem")
    build(OUT_P1684, err_style="p1684")
build(OUT_FINAL, err_style="p1684")

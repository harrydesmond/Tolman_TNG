#!/usr/bin/env python3
"""Two alternative views of the DDR figure that avoid the d_0 normalisation.

Panel A (slope-only / self-anchored, linear y-scale):
    Each sample's D_L/D_A^Li values are divided by the median ratio of its
    own lowest-z bin. d_0 cancels in this ratio, so the curves pass through
    1 at the anchor redshift and the only remaining content is the slope.
    Reference lines (1+z)^2/(1+z_anchor)^2 and (1+z)/(1+z_anchor) are
    anchored to the same point.

Panel B (simulation-physical rho_L coordinates, with per-galaxy scatter band):
    y-axis is rho_L(z)/rho_L(z->0) = (1+z)^gamma. The TNG forward model
    delivers this directly from L/V in cgs/kpc with NO d_0 assumption; we
    show per-galaxy 16-84 spread and binned medians, the aperture-envelope
    band, and the data-implied gamma band. Reference lines:
    gamma=0 (FRW DDR) and gamma=3 (non-expanding limit).
"""
from __future__ import annotations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

REPO = Path(__file__).resolve().parent.parent
PUB = REPO / "Tolman_TNG"
DATA = REPO / "results" / "li_ddr"
PUB_DATA = PUB / "results" / "li_ddr"

OUT_A = REPO / "Plots" / "Fig6_alt_A_slope_only.png"
OUT_B = REPO / "Plots" / "Fig6_alt_B_rho_L.png"
# Plot B is the rho_L_evolution.pdf paper figure; plot A is a diagnostic.
OUT_B_FINAL = REPO / "Plots" / "final" / "rho_L_evolution.pdf"
WRITE_DIAGNOSTICS = "--write-diagnostics" in sys.argv


def result_path(name: str) -> Path:
    local = DATA / name
    if local.exists():
        return local
    return PUB_DATA / name


radio = pd.read_csv(result_path("radio_per_source_ddr.csv"))
tng = pd.read_csv(result_path("tng_per_source_ddr.csv"))


def equal_pop_bins(z, y, n_bins):
    m = np.isfinite(z) & np.isfinite(y)
    z = z[m]; y = y[m]
    order = np.argsort(z); z = z[order]; y = y[order]
    edges = np.linspace(0, len(z), n_bins + 1).astype(int)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < 4:
            continue
        zb = z[lo:hi]; yb = y[lo:hi]
        out.append((
            float(np.median(zb)), float(np.median(yb)),
            1.2533 * float(np.std(yb, ddof=1)) / np.sqrt(len(yb)),
            float(np.percentile(yb, 16)), float(np.percentile(yb, 84)),
        ))
    a = np.array(out)
    return dict(x=a[:, 0], med=a[:, 1], sigma=a[:, 2], lo=a[:, 3], hi=a[:, 4])


# ====================================================================
# Plot A: slope-only / self-anchored, LINEAR y-axis
# ====================================================================
radio_z = radio["z"].to_numpy(float); radio_r = radio["ddr_ratio"].to_numpy(float)
tng_z = tng["z"].to_numpy(float); tng_r = tng["ddr_ratio"].to_numpy(float)

rb = equal_pop_bins(radio_z, radio_r, 27)
tb = equal_pop_bins(tng_z, tng_r, 27)

r_anchor_r = rb["med"][0]
r_anchor_t = tb["med"][0]
z_a_ref = rb["x"][0]
z_grid = np.linspace(0.005, 5.0, 600)
frw_anchored = (1 + z_grid) ** 2 / (1 + z_a_ref) ** 2
non_anchored = (1 + z_grid) / (1 + z_a_ref)

fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
ax.fill_between(
    tb["x"], tb["lo"] / r_anchor_t, tb["hi"] / r_anchor_t,
    color="C2", alpha=0.18, lw=0,
    label=fr"TNG100 F200W ({len(tng_z)}) 16 to 84%",
)
ax.plot(z_grid, frw_anchored, "--", color="C0",
        label=r"FRW slope $(1+z)^{2}$")
ax.plot(z_grid, non_anchored, "-", color="C3",
        label=r"non-exp. slope $(1+z)$")
ax.errorbar(
    rb["x"], rb["med"] / r_anchor_r, yerr=rb["sigma"] / r_anchor_r,
    fmt="o", color="black", ms=4.5, lw=0.8, capsize=2, zorder=4,
    label=fr"JJ+GKF ({len(radio_z)})",
)
ax.errorbar(
    tb["x"], tb["med"] / r_anchor_t, yerr=tb["sigma"] / r_anchor_t,
    fmt="s", color="C2", ms=4.5, lw=0.8, capsize=2,
    mfc="white", mec="C2", zorder=3,
    label=fr"TNG100 F200W ({len(tng_z)}) median",
)
ax.set_xlabel(r"Redshift $z$")
ax.set_ylabel(r"$(D_L/D_A^{\rm Li})(z)\,/\,(D_L/D_A^{\rm Li})(z_{\rm anchor})$")
ax.set_xlim(0, 5.0)
ax.set_ylim(0, 11.0)
ax.axhline(1.0, color="grey", lw=0.5, ls=":")
if WRITE_DIAGNOSTICS:
    fig.savefig(OUT_A, dpi=150)
    print(f"Saved {OUT_A}")
plt.close(fig)


# ====================================================================
# Plot B: simulation-physical rho_L with per-galaxy scatter band
# ====================================================================
rho = pd.read_csv(result_path("tng_rho_L_per_galaxy_deblended.csv"))
ap = pd.read_csv(result_path("tng_gamma_aperture_table.csv"))

# Sanity: gamma should be ~2.30 on the Voronoi-deblended sample (fiducial).
log1pz = np.log10(1 + rho["z"].to_numpy(float))
log_rho = np.log10(rho["rho_L_cgs_per_cm3"].to_numpy(float))
g_fit, c_fit = np.polyfit(log1pz, log_rho, 1)
# Anchor curves at the predicted value at z=0 from the fit.
rho_anchor = 10 ** c_fit

# Per-galaxy normalised rho_L for plotting.
y_per_gal = (rho["rho_L_cgs_per_cm3"].to_numpy(float) / rho_anchor)
z_per_gal = rho["z"].to_numpy(float)
rho_bins = equal_pop_bins(z_per_gal, y_per_gal, 14)

# Aperture variants as in tng_gamma_aperture_table.csv:
# variant-specific flux, but intrinsic photrad_kpc volume for the adaptive row.
# OLS standard errors from tng_gamma_aperture_table.csv.
rows = [
    ("specmodel (catalogue flux)", "variant_volume", "catalogue flux", "#1b9e77", "-"),
    ("imageap_specmodel (canonical)", "variant_volume", "fixed aperture", "#238b45", "-"),
    ("imageap_deblended (Voronoi)", "variant_volume", "Voronoi deblended", "#66a61e", "--"),
    ("imageap_astrodeep_ap", "intrinsic_photrad_volume", "adaptive flux", "#006d77", "-."),
]
aperture_curves = []
for variant, volume_defn, label, color, linestyle in rows:
    match = ap[(ap["variant"] == variant) & (ap["volume_defn"] == volume_defn)]
    if len(match) != 1:
        raise RuntimeError(f"Missing unique gamma row for {variant} / {volume_defn}")
    row_ap = match.iloc[0]
    aperture_curves.append({
        "label": label,
        "gamma": float(row_ap["gamma"]),
        "sigma": float(row_ap["gamma_ols_stderr"]),
        "color": color,
        "linestyle": linestyle,
    })

data_ddr = pd.read_csv(result_path("ddr_radio_gamma_eff.csv"))
row = data_ddr[
    (data_ddr["sample"] == "JJ+GKF Li-style concat")
    & (data_ddr["power"] == 3)
].iloc[0]
gamma_data = float(row["gamma_eff"]); gamma_data_err = float(row["gamma_eff_err"])

print(f"TNG canonical (fit): gamma = {g_fit:.3f}")
for curve in aperture_curves:
    print(f"TNG {curve['label']}: gamma = {curve['gamma']:.3f} "
          f"+/- {curve['sigma']:.3f}")
print(f"Data (JJ+GKF Li-style concat p=3) gamma = {gamma_data:.3f} +/- {gamma_data_err:.3f}")


def power_law(z, g):
    return (1 + z) ** g


FS_LABEL = 18
FS_TICK = 15
FS_LEGEND = 14
fig, ax = plt.subplots(figsize=(8.2, 5.5), constrained_layout=True)
legend_handles = {}

# Reference lines.
legend_handles["No evolution"] = ax.plot(
    z_grid, (1 + z_grid) ** 0, "--", color="C0",
    label=r"No evolution", zorder=2,
)[0]
legend_handles["Apparent non-exp."] = ax.plot(
    z_grid, (1 + z_grid) ** 3, "--", color="C3",
    label=r"Apparent non-exp.", zorder=2,
)[0]

# TNG aperture variants, with OLS statistical gamma bands.
voronoi_color = "#66a61e"
for curve in aperture_curves:
    g = curve["gamma"]
    sigma = curve["sigma"]
    color = curve["color"]
    ax.fill_between(
        z_grid, power_law(z_grid, g - sigma), power_law(z_grid, g + sigma),
        color=color, alpha=0.13, lw=0, zorder=2,
    )
    line = ax.plot(
        z_grid, power_law(z_grid, g),
        color=color, lw=1.5, ls=curve["linestyle"], zorder=4,
        label=fr"TNG {curve['label']}",
    )[0]
    legend_handles[curve["label"]] = (
        Patch(facecolor=color, edgecolor="none", alpha=0.13),
        Line2D([0], [0], color=color, lw=1.5, ls=curve["linestyle"]),
    )

# TNG per-galaxy median points. The per-galaxy rho_L distribution spans many
# decades, so the linear-space standard-error-on-median can exceed the median
# itself in bins dominated by a few high-rho_L outliers (the lower error bar
# then can't render on a log axis). We therefore quote the standard error on
# the median *in log10 rho_L* and render the bars asymmetrically in linear
# units.
z_per_gal_sorted = np.sort(z_per_gal)
edges = np.linspace(0, len(z_per_gal), 15).astype(int)
order = np.argsort(z_per_gal)
y_sorted = y_per_gal[order]
z_sorted = z_per_gal[order]
zc, ymed, ylo, yhi = [], [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    if hi - lo < 4:
        continue
    yb = y_sorted[lo:hi]
    log_yb = np.log10(yb[np.isfinite(yb) & (yb > 0)])
    sigma_log = 1.2533 * float(np.std(log_yb, ddof=1)) / np.sqrt(len(log_yb))
    med = float(np.median(yb))
    zc.append(float(np.median(z_sorted[lo:hi])))
    ymed.append(med)
    ylo.append(med * (1 - 10 ** (-sigma_log)))
    yhi.append(med * (10 ** sigma_log - 1))
points = ax.errorbar(
    zc, ymed, yerr=[ylo, yhi],
    fmt="s", color=voronoi_color, ms=4.5, lw=0.8, capsize=2,
    mfc="white", mec=voronoi_color, zorder=5,
    label=fr"TNG Voronoi bin medians",
)
legend_handles["voronoi medians"] = points

# Data-implied gamma band.
lo_d = power_law(z_grid, gamma_data - gamma_data_err)
hi_d = power_law(z_grid, gamma_data + gamma_data_err)
data_band = ax.fill_between(
    z_grid, lo_d, hi_d, color="black", alpha=0.22, lw=0, zorder=1,
    label=fr"Radio data",
)
data_line = ax.plot(
    z_grid, power_law(z_grid, gamma_data),
    color="0.35", lw=1.4, zorder=3,
    label=fr"Radio data",
)[0]
legend_handles["Radio data"] = (
    Patch(facecolor="black", edgecolor="none", alpha=0.22),
    Line2D([0], [0], color="0.35", lw=1.4),
)

ax.set_yscale("log")
ax.set_xlabel(r"Redshift $z$", fontsize=FS_LABEL)
ax.set_ylabel(r"$\rho_L(z)\,/\,\rho_L(z\!\to\!0)\;=\;(1+z)^{\gamma}$", fontsize=FS_LABEL)
ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
ax.tick_params(axis="both", which="minor", labelsize=FS_TICK)
ax.set_xlim(0, 5.0)
ax.set_ylim(0.8, 300.0)
legend_order = [
    "Apparent non-exp.",
    "No evolution",
    "voronoi medians",
    "fixed aperture",
    "catalogue flux",
    "Voronoi deblended",
    "adaptive flux",
    "Radio data",
]
legend_labels = {
    "Apparent non-exp.": "Apparent non-exp.",
    "No evolution": "No evolution",
    "voronoi medians": "TNG Voronoi bin medians",
    "fixed aperture": "TNG fixed aperture",
    "catalogue flux": "TNG catalogue flux",
    "Voronoi deblended": "TNG Voronoi deblended",
    "adaptive flux": "TNG adaptive flux",
    "Radio data": "Radio data",
}
ax.legend(
    [legend_handles[key] for key in legend_order],
    [legend_labels[key] for key in legend_order],
    frameon=False, loc="upper left", fontsize=FS_LEGEND,
    handler_map={tuple: HandlerTuple(ndivide=1)},
)
if WRITE_DIAGNOSTICS:
    fig.savefig(OUT_B, dpi=150)
    print(f"Saved {OUT_B}")
fig.savefig(OUT_B_FINAL)
plt.close(fig)
print(f"Saved {OUT_B_FINAL}")

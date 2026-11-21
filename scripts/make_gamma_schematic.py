"""Build the gamma schematic (schematic illustration figure of the paper).

Two-panel illustration placing the two mappings of the empirically common
effective evolution exponent on one shared gamma axis:

    Tolman:  I_SB ∝ (1 + z)^[alpha_Tol],
             alpha_Tol = -(3 + alpha_SED) + gamma
    DDR:     D_L / D_A ∝ (1 + z)^[2 - gamma/3]

Vertical bands mark the TNG forward-model prediction (green), the Tolman-data
inference (purple), and the DDR-data inference (orange):

    gamma (TNG)         = 2.23 +/- 0.20  (alpha_SED = 0.3)
    gamma (DDR data)    = 2.50 +/- 0.26
    gamma (Tolman data) = 2.02 +/- 0.05  (alpha_SED = 0.3; OLS and cut-to-cut in quadrature)

Writes Plots/final/gamma_schematic.pdf.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
OUT_PRIMARY = REPO / "Plots" / "final" / "gamma_schematic.pdf"

GAMMA_TNG = 2.23
GAMMA_TNG_SIG = 0.20
GAMMA_DDR = 2.50
GAMMA_DDR_SIG = 0.26
GAMMA_TOLMAN = 2.02
GAMMA_TOLMAN_SIG = 0.05
GAMMA_MAX = 3.1
ALPHA_SED = 0.3

BLUE = "#1f77b4"
RED = "#d62728"
TNG_GREEN = "#2ca02c"
DDR_ORANGE = "#ff7f0e"
TOLMAN_PURPLE = "#9467bd"
REF_LS = (0, (5, 3))
GAMMA_LS = (0, (1.2, 2.2))


def add_gamma_bands(ax, *, tolman=False, ddr=False):
    ax.axvspan(
        GAMMA_TNG - GAMMA_TNG_SIG,
        GAMMA_TNG + GAMMA_TNG_SIG,
        color=TNG_GREEN,
        alpha=0.28,
        lw=0,
    )
    if tolman:
        ax.axvspan(
            GAMMA_TOLMAN - GAMMA_TOLMAN_SIG,
            GAMMA_TOLMAN + GAMMA_TOLMAN_SIG,
            color=TOLMAN_PURPLE,
            alpha=0.32,
            lw=0,
        )
    if ddr:
        ax.axvspan(
            GAMMA_DDR - GAMMA_DDR_SIG,
            GAMMA_DDR + GAMMA_DDR_SIG,
            color=DDR_ORANGE,
            alpha=0.25,
            lw=0,
        )
    ax.axvline(3.0, color="black", ls=GAMMA_LS, lw=1.2, alpha=0.85)


def main():
    gamma = np.linspace(0.0, GAMMA_MAX, 400)
    tolman_power = -(3.0 + ALPHA_SED) + gamma
    ddr_power = 2.0 - gamma / 3.0

    fig = plt.figure(figsize=(7.6, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.15, 1.0, 1.0])
    ax_leg = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1])
    ax_d = fig.add_subplot(gs[2], sharex=ax_t)
    axes = (ax_t, ax_d)
    ax_leg.axis("off")

    add_gamma_bands(ax_t, tolman=True)
    add_gamma_bands(ax_d, ddr=True)
    for ax in axes:
        ax.set_xlim(0, GAMMA_MAX)
        ax.tick_params(labelsize=11)
    ax_t.tick_params(labelbottom=False)

    ax_t.axhspan(
        -(3.0 + ALPHA_SED) + GAMMA_TOLMAN - GAMMA_TOLMAN_SIG,
        -(3.0 + ALPHA_SED) + GAMMA_TOLMAN + GAMMA_TOLMAN_SIG,
        color=TOLMAN_PURPLE,
        alpha=0.18,
        lw=0,
    )
    ax_d.axhspan(
        2.0 - (GAMMA_DDR + GAMMA_DDR_SIG) / 3.0,
        2.0 - (GAMMA_DDR - GAMMA_DDR_SIG) / 3.0,
        color=DDR_ORANGE,
        alpha=0.18,
        lw=0,
    )

    ax_t.plot(gamma, tolman_power, color="black", lw=1.8)
    ax_t.axhline(-(3.0 + ALPHA_SED), color="black", ls=REF_LS, lw=1.2, alpha=0.75)
    ax_t.axhline(-ALPHA_SED, color="black", ls=REF_LS, lw=1.2, alpha=0.75)
    ax_t.set_ylim(-3.58, 0.05)
    ax_t.set_ylabel(
        r"$\alpha_{\rm Tol}$ ($I_{\rm SB}\propto(1+z)^{\alpha_{\rm Tol}}$)",
        fontsize=12,
    )
    ax_t.text(0.06, -(3.0 + ALPHA_SED) - 0.06, "FRW", color="black", fontsize=10, va="top")
    ax_t.text(0.06, -ALPHA_SED + 0.08, "non-expanding", color="black", fontsize=10)

    ax_d.plot(gamma, ddr_power, color="black", lw=1.8)
    ax_d.axhline(2.0, color="black", ls=REF_LS, lw=1.2, alpha=0.75)
    ax_d.axhline(1.0, color="black", ls=REF_LS, lw=1.2, alpha=0.75)
    ax_d.set_ylim(0.88, 2.12)
    ax_d.set_ylabel(
        r"$\alpha_{\rm DDR}$ ($D_L/D_A\propto(1+z)^{\alpha_{\rm DDR}}$)",
        fontsize=12,
    )
    ax_d.set_xlabel(
        r"Equivalent luminosity-density evolution exponent $\gamma$",
        fontsize=13,
    )
    ax_d.text(0.06, 2.03, "FRW", color="black", fontsize=10)
    ax_d.text(0.06, 0.98, "non-expanding", color="black", fontsize=10, va="top")

    shared_handles = [
        Patch(facecolor=TNG_GREEN, alpha=0.28, label=r"TNG bright galaxies"),
        Patch(facecolor=TOLMAN_PURPLE, alpha=0.32, label=r"Tolman data"),
        Patch(facecolor=DDR_ORANGE, alpha=0.25, label=r"DDR radio data"),
    ]
    ax_leg.legend(
        handles=shared_handles,
        loc="center",
        mode="expand",
        ncol=3,
        fontsize=12,
        frameon=True,
        framealpha=0.95,
        bbox_to_anchor=(0.0, -0.08, 1.0, 1.0),
        borderaxespad=0.0,
    )

    OUT_PRIMARY.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PRIMARY)
    print(f"Wrote {OUT_PRIMARY}")
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the Li 2023 DDR test on the radio samples and the TNG analogue.

Usage:
    python scripts/li_ddr_run.py [--tng-catalog PATH] [--filter F200W]
                                  [--out-dir results/li_ddr]
                                  [--plot-dir Plots/li_ddr]

Outputs:
    results/li_ddr/li_radio_table1.csv      -- reproduction of Li 2023 Table 1
    results/li_ddr/tng_li_analogue.csv      -- TNG analogue Li-style fits
    results/li_ddr/tng_intrinsic_density.csv -- direct rho_L(z), Sigma_L(z) gammas
    Plots/li_ddr/*.png                       -- diagnostic plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tolman.li_ddr import io as li_io
from tolman.li_ddr.li_reproduction import (
    COSMO_LI,
    bin_median,
    calibrated_angular_size,
    distance_duality_ratio,
    fit_d0_expanding,
    fit_d0_nonexpanding,
    fit_size_redshift_relation,
    luminosity_at_band,
)
from tolman.li_ddr.plots import (
    plot_calibrated_theta_z,
    plot_combined_ddr,
    plot_distance_duality,
    plot_luminosity_redshift,
)
from tolman.li_ddr.tng_analogue import (
    intrinsic_luminosity_density_evolution,
    run_li_on_tng,
    to_summary_table,
)


def _li_table_row(df: pd.DataFrame, sample_label: str) -> dict:
    z = df["z"].to_numpy(dtype=float)
    S = df["S_jy"].to_numpy(dtype=float)
    alpha = df["alpha_spec"].to_numpy(dtype=float)
    theta = df["theta_mas"].to_numpy(dtype=float)
    L = luminosity_at_band(S, z, alpha)
    fit = fit_size_redshift_relation(z, L)
    theta_calib = calibrated_angular_size(theta, z, fit.a, fit.b)
    d0_exp, d0_exp_err, _ = fit_d0_expanding(theta_calib, z)
    d0_non, d0_non_err, _ = fit_d0_nonexpanding(theta_calib, z)
    return dict(
        sample=sample_label,
        n=fit.n,
        a=fit.a, a_err=fit.a_err,
        b=fit.b, b_err=fit.b_err,
        log_L0=fit.log_L0, log_L0_err=fit.log_L0_err,
        d0_expanding_pc=d0_exp, d0_expanding_err_pc=d0_exp_err,
        d0_nonexpanding_pc=d0_non, d0_nonexpanding_err_pc=d0_non_err,
    )


def _radio_per_source_ddr(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-source (z, D_L/D_A) for one radio sample, using its own (a, b, d_0)."""
    z = df["z"].to_numpy(dtype=float)
    S = df["S_jy"].to_numpy(dtype=float)
    alpha = df["alpha_spec"].to_numpy(dtype=float)
    theta = df["theta_mas"].to_numpy(dtype=float)
    L = luminosity_at_band(S, z, alpha)
    fit = fit_size_redshift_relation(z, L)
    theta_calib = calibrated_angular_size(theta, z, fit.a, fit.b)
    d0_non, _, _ = fit_d0_nonexpanding(theta_calib, z)
    ratio = distance_duality_ratio(theta_calib, z, d0_pc=d0_non)
    return z, ratio


def _make_radio_plots(df: pd.DataFrame, label: str, plot_dir: Path) -> dict[str, Path]:
    z = df["z"].to_numpy(dtype=float)
    S = df["S_jy"].to_numpy(dtype=float)
    alpha = df["alpha_spec"].to_numpy(dtype=float)
    theta = df["theta_mas"].to_numpy(dtype=float)
    L = luminosity_at_band(S, z, alpha)
    fit = fit_size_redshift_relation(z, L)
    theta_calib = calibrated_angular_size(theta, z, fit.a, fit.b)
    d0_non, _, _ = fit_d0_nonexpanding(theta_calib, z)
    ratio = distance_duality_ratio(theta_calib, z, d0_pc=d0_non)
    safe = label.replace(".", "p").replace(" ", "_")
    paths = {}
    paths["L_z"] = plot_luminosity_redshift(
        df, L, fit.predict_log_L,
        sample_label=label,
        out_path=plot_dir / f"li_{safe}_L_vs_z.png",
    )
    paths["theta_z"] = plot_calibrated_theta_z(
        z, theta_calib, sample_label=label, n_bins=max(20, len(z) // 25),
        out_path=plot_dir / f"li_{safe}_theta_vs_z.png",
    )
    paths["DDR"] = plot_distance_duality(
        z, ratio, sample_label=label, n_bins=max(20, len(z) // 25),
        out_path=plot_dir / f"li_{safe}_DDR.png",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tng-catalog", type=Path,
                        default=Path("results/tng50_12_11_xyz_hst_jwst_specmodel/measured_catalog.csv"),
                        help="Per-galaxy TNG measured-catalog CSV (use the TNG100 "
                             "spec-selected catalogue for the canonical run).")
    parser.add_argument("--filter", default="F200W")
    parser.add_argument(
        "--alpha-spec",
        type=float,
        default=-0.3,
        help=("Li/code K-correction spectral index in F_nu propto nu^alpha_spec; "
              "paper alpha_SED = -alpha_spec, and default -0.3 matches the "
              "paper-wide alpha_SED=0.3 convention."),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/li_ddr"))
    parser.add_argument("--plot-dir", type=Path, default=Path("Plots/li_ddr"))
    parser.add_argument("--no-tng", action="store_true",
                        help="Skip TNG analogue (e.g. when only validating reproduction).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase A: reproduction of Li 2023 on radio samples ----
    jj = li_io.load_jj229()
    gkf = li_io.load_gkf50()
    radio_rows = [
        _li_table_row(jj, "JJ2.29"),
        _li_table_row(gkf, "GKF5.0"),
    ]
    radio_table = pd.DataFrame(radio_rows)
    radio_table.to_csv(args.out_dir / "li_radio_table1.csv", index=False)
    print(f"radio Table 1 reproduction -> {args.out_dir/'li_radio_table1.csv'}")
    print(radio_table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    radio_plot_paths = {
        "JJ2.29": _make_radio_plots(jj, "JJ2.29", args.plot_dir),
        "GKF5.0": _make_radio_plots(gkf, "GKF5.0", args.plot_dir),
    }
    for label, paths in radio_plot_paths.items():
        for kind, p in paths.items():
            print(f"plot {label} {kind:8s} -> {p}")

    # Per-source D_L/D_A for radio samples (combined for the Li-Fig4 overlay)
    z_jj, r_jj = _radio_per_source_ddr(jj)
    z_gkf, r_gkf = _radio_per_source_ddr(gkf)
    radio_per_source = pd.DataFrame({
        "z": np.concatenate([z_jj, z_gkf]),
        "ddr_ratio": np.concatenate([r_jj, r_gkf]),
        "sample": ["JJ2.29"] * len(z_jj) + ["GKF5.0"] * len(z_gkf),
    })
    radio_per_source.to_csv(args.out_dir / "radio_per_source_ddr.csv", index=False)

    # ---- Phase B: TNG analogue (apply Li to TNG) ----
    if args.no_tng:
        return
    if not args.tng_catalog.exists():
        print(f"\nNote: TNG catalog {args.tng_catalog} not found locally; skipping TNG analogue.\n"
              f"      Point --tng-catalog at the TNG100 spec-selected catalogue "
              f"(produced on a cluster) for the canonical run.")
        return
    tng = li_io.load_tng_galaxy_sample(args.tng_catalog, filter_name=args.filter)
    print(f"\nTNG sample: {tng.n} galaxies, filter {tng.filter_name}, "
          f"z range [{tng.df['z'].min():.3f}, {tng.df['z'].max():.3f}]")

    tng_summaries = []
    tng_results: dict[int, "object"] = {}
    for power in (3, 2):  # 3 = exact Li, 2 = 2D galaxy variant
        result = run_li_on_tng(tng, alpha_spec=args.alpha_spec, power=power)
        tng_results[power] = result
        label = f"TNG {tng.filter_name} L^d{power}"
        tng_summaries.append(to_summary_table(result, label))
        # plots
        safe = label.replace(" ", "_").replace("^", "")
        plot_calibrated_theta_z(
            result.z, result.theta_calib_mas, sample_label=label,
            n_bins=max(15, result.fit.n // 100),
            out_path=args.plot_dir / f"tng_{safe}_theta_vs_z.png",
        )
        plot_distance_duality(
            result.z, result.ddr_ratio, sample_label=label,
            n_bins=max(15, result.fit.n // 100),
            out_path=args.plot_dir / f"tng_{safe}_DDR.png",
        )
    tng_table = pd.concat(tng_summaries, ignore_index=True)
    tng_table.to_csv(args.out_dir / "tng_li_analogue.csv", index=False)
    print(f"\nTNG analogue -> {args.out_dir/'tng_li_analogue.csv'}")
    print(tng_table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # Per-source TNG DDR (power=3, the exact-Li version) and combined Fig.4 overlay
    tng_p3 = tng_results[3]
    tng_per_source = pd.DataFrame({
        "z": tng_p3.z,
        "ddr_ratio": tng_p3.ddr_ratio,
    })
    tng_per_source.to_csv(args.out_dir / "tng_per_source_ddr.csv", index=False)

    combined_path = plot_combined_ddr(
        radio_z=radio_per_source["z"].to_numpy(),
        radio_ratio=radio_per_source["ddr_ratio"].to_numpy(),
        tng_z=tng_p3.z,
        tng_ratio=tng_p3.ddr_ratio,
        radio_label="JJ2.29 + GKF5.0 (943)",
        tng_label=f"TNG100 {tng.filter_name} ({tng_p3.fit.n})",
        out_path=args.plot_dir / "li_fig4_radio_vs_tng100.png",
    )
    print(f"\ncombined Li Fig.4 with TNG overlay -> {combined_path}")

    # ---- Phase C: direct intrinsic rho_L(z) for TNG ----
    rho = intrinsic_luminosity_density_evolution(tng, alpha_spec=args.alpha_spec)
    rho_summary = pd.DataFrame([
        dict(
            filter=tng.filter_name,
            alpha_spec=args.alpha_spec,
            n=rho["n"],
            gamma_rho_volume=rho["gamma_rho_volume"],
            log10_rho_L_z0=rho["log10_rho_L_z0"],
            gamma_sigma_area=rho["gamma_sigma_area"],
            log10_sigma_L_z0=rho["log10_sigma_L_z0"],
            li_required_gamma=3.0,
        )
    ])
    rho_summary.to_csv(args.out_dir / "tng_intrinsic_density.csv", index=False)
    print(f"\nIntrinsic rho_L evolution -> {args.out_dir/'tng_intrinsic_density.csv'}")
    print(rho_summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Data-side DDR gamma_eff from the radio samples.

Reproduces the JJ2.29 (Jackson & Jannetta 2006) and GKF5.0 (Gurvits,
Kellermann & Frey 1999) DDR reduction of Li (2023) and converts the
binned-median power-law slope D_L/D_A_Li ~ (1+z)^beta into the
luminosity-density evolution exponent gamma_eff implied by the radio data.

The fiducial paper convention follows Li's Fig. 4 construction: fit the
JJ2.29 and GKF5.0 calibrations separately, convert each source to
D_L/D_A_Li using its own sample calibration, then concatenate the derived
ratios into the 943-source sample and fit 27 equal-population redshift bins.
A diagnostic joint-raw-refit row concatenates the raw catalogues first and
forces one shared size-luminosity calibration.

Outputs:
  results/li_ddr/ddr_radio_gamma_eff.csv
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tolman.li_ddr import io as li_io
from tolman.li_ddr.li_reproduction import (
    COSMO_LI,
    bin_median,
    calibrated_angular_size,
    distance_duality_ratio,
    fit_d0_nonexpanding,
    luminosity_at_band,
)
from tolman.li_ddr.tng_analogue import _fit_with_power

ROOT = Path(os.environ.get("TOLMAN_WORKDIR", Path(__file__).resolve().parents[1]))


def _ratio_with_power(df: pd.DataFrame, power: int):
    z = df["z"].to_numpy(dtype=float)
    flux_jy = df["S_jy"].to_numpy(dtype=float)
    alpha = df["alpha_spec"].to_numpy(dtype=float)
    theta = df["theta_mas"].to_numpy(dtype=float)

    luminosity = luminosity_at_band(flux_jy, z, alpha, cosmo=COSMO_LI)
    fit = _fit_with_power(z, luminosity, power=power)
    theta_calib = calibrated_angular_size(theta, z, fit.a, fit.b)
    d0_non, d0_non_err, _ = fit_d0_nonexpanding(theta_calib, z, cosmo=COSMO_LI)
    ratio = distance_duality_ratio(theta_calib, z, d0_pc=d0_non, cosmo=COSMO_LI)
    return z, ratio, fit, d0_non, d0_non_err


def _fit_ratio_power_law(z: np.ndarray, ratio: np.ndarray, *, power: int, n_bins: int) -> dict:
    finite = np.isfinite(z) & np.isfinite(ratio) & (z > 0) & (ratio > 0)
    z_ok = z[finite]
    ratio_ok = ratio[finite]
    binned = bin_median(z_ok, ratio_ok, n_bins=n_bins)

    log1pz = np.log10(1.0 + binned["x_median"].to_numpy())
    log_ratio = np.log10(binned["y_median"].to_numpy())
    sigma = (
        binned["y_sigma_med"].to_numpy()
        / (binned["y_median"].to_numpy() * np.log(10.0))
    )
    weights = 1.0 / np.where(sigma > 0.0, sigma, 1.0)
    coef, cov = np.polyfit(log1pz, log_ratio, 1, w=weights, cov=True)
    beta = float(coef[0])
    beta_err = float(np.sqrt(cov[0, 0]))
    if power == 3:
        gamma_eff = 3.0 * (2.0 - beta)
        gamma_eff_err = 3.0 * beta_err
    elif power == 2:
        gamma_eff = 2.0 * (2.0 - beta)
        gamma_eff_err = 2.0 * beta_err
    else:
        raise ValueError(f"unsupported power={power}")

    return dict(
        power=power,
        n=int(finite.sum()),
        n_bins=int(len(binned)),
        beta=beta,
        beta_err=beta_err,
        gamma_eff=gamma_eff,
        gamma_eff_err=gamma_eff_err,
    )


def _single_sample(df: pd.DataFrame, power: int) -> dict:
    z, ratio, fit, d0_non, d0_non_err = _ratio_with_power(df, power)
    row = _fit_ratio_power_law(z, ratio, power=power, n_bins=max(20, len(z) // 25))
    row.update(
        reduction="single_sample",
        a=fit.a,
        a_err=fit.a_err,
        b=fit.b,
        b_err=fit.b_err,
        log_L0=fit.log_L0,
        log_L0_err=fit.log_L0_err,
        d0_nonexpanding_pc=d0_non,
        d0_nonexpanding_err_pc=d0_non_err,
    )
    return row


def _separate_then_concat(dfs: list[pd.DataFrame], power: int) -> dict:
    z_parts = []
    ratio_parts = []
    for df in dfs:
        z, ratio, _, _, _ = _ratio_with_power(df, power)
        z_parts.append(z)
        ratio_parts.append(ratio)
    z = np.concatenate(z_parts)
    ratio = np.concatenate(ratio_parts)
    row = _fit_ratio_power_law(z, ratio, power=power, n_bins=27)
    row.update(
        reduction="separate_then_concat",
        a=np.nan,
        a_err=np.nan,
        b=np.nan,
        b_err=np.nan,
        log_L0=np.nan,
        log_L0_err=np.nan,
        d0_nonexpanding_pc=np.nan,
        d0_nonexpanding_err_pc=np.nan,
    )
    return row


def _joint_raw_refit(df: pd.DataFrame, power: int) -> dict:
    z, ratio, fit, d0_non, d0_non_err = _ratio_with_power(df, power)
    row = _fit_ratio_power_law(z, ratio, power=power, n_bins=max(20, len(z) // 25))
    row.update(
        reduction="joint_raw_refit",
        a=fit.a,
        a_err=fit.a_err,
        b=fit.b,
        b_err=fit.b_err,
        log_L0=fit.log_L0,
        log_L0_err=fit.log_L0_err,
        d0_nonexpanding_pc=d0_non,
        d0_nonexpanding_err_pc=d0_non_err,
    )
    return row


def main() -> None:
    jj = li_io.load_jj229()
    gkf = li_io.load_gkf50()
    combined = pd.concat([jj, gkf], ignore_index=True)

    rows = []
    for label, reducer in (
        ("JJ2.29", lambda p: _single_sample(jj, p)),
        ("GKF5.0", lambda p: _single_sample(gkf, p)),
        ("JJ+GKF Li-style concat", lambda p: _separate_then_concat([jj, gkf], p)),
        ("JJ+GKF joint raw refit", lambda p: _joint_raw_refit(combined, p)),
    ):
        for power in (3, 2):
            row = reducer(power)
            row["sample"] = label
            rows.append(row)
            print(
                f"{label:24s} power={power} n={row['n']:4d} "
                f"beta={row['beta']:+.3f}+/-{row['beta_err']:.3f} "
                f"=> gamma_eff={row['gamma_eff']:+.3f}+/-{row['gamma_eff_err']:.3f}"
            )

    table = pd.DataFrame(rows)
    cols_first = [
        "sample",
        "reduction",
        "power",
        "n",
        "n_bins",
        "a",
        "a_err",
        "b",
        "b_err",
        "log_L0",
        "log_L0_err",
        "d0_nonexpanding_pc",
        "d0_nonexpanding_err_pc",
        "beta",
        "beta_err",
        "gamma_eff",
        "gamma_eff_err",
    ]
    table = table[cols_first]
    out_dir = ROOT / "results" / "li_ddr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ddr_radio_gamma_eff.csv"
    table.to_csv(out_path, index=False)
    print(f"\nWrote -> {out_path}")

    fid = table[table["sample"] == "JJ+GKF Li-style concat"].set_index("power")
    joint = table[table["sample"] == "JJ+GKF joint raw refit"].set_index("power")
    print(
        "\n[fiducial Li-style concatenated data DDR] "
        f"gamma_eff(p=3)={fid.loc[3, 'gamma_eff']:+.3f}+/-{fid.loc[3, 'gamma_eff_err']:.3f}; "
        f"gamma_eff(p=2)={fid.loc[2, 'gamma_eff']:+.3f}+/-{fid.loc[2, 'gamma_eff_err']:.3f}"
    )
    print(
        "[diagnostic joint raw refit] "
        f"gamma_eff(p=3)={joint.loc[3, 'gamma_eff']:+.3f}+/-{joint.loc[3, 'gamma_eff_err']:.3f}"
    )


if __name__ == "__main__":
    main()

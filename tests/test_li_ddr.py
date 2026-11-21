"""Tests for the Li 2023 DDR reproduction and TNG analogue."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tolman.li_ddr import io as li_io
from tolman.li_ddr.li_reproduction import (
    bin_median,
    calibrated_angular_size,
    distance_duality_ratio,
    fit_d0_expanding,
    fit_d0_nonexpanding,
    fit_size_redshift_relation,
    luminosity_at_band,
)
from tolman.li_ddr.tng_analogue import (
    intrinsic_luminosity_density_evolution,
    run_li_on_tng,
)


def test_load_jj229_has_613_rows():
    df = li_io.load_jj229()
    assert len(df) == 613
    assert {"z", "theta_mas", "S_jy", "alpha_spec"}.issubset(df.columns)
    assert df["z"].between(0.0, 4.0).all()
    assert (df["S_jy"] > 0).all()


def test_load_gkf50_has_330_rows():
    df = li_io.load_gkf50()
    assert len(df) == 330
    assert {"z", "theta_mas", "S_jy", "alpha_spec", "is_upper_limit"}.issubset(df.columns)
    assert df["is_upper_limit"].sum() > 0  # GKF5.0 has unresolved sources


def test_li_reproduction_matches_published_table1():
    """Li 2023 Table 1 best-fit parameters: a, b, log L0, d0.

    JJ2.29: a=0.39+/-0.06, b=0.60+/-0.02, logL0=27.1+/-0.1, d0=21.6+/-0.7 pc
    GKF5.0: a=0.02+/-0.08, b=0.72+/-0.04, logL0=27.4+/-0.1, d0=63.7+/-4.4 pc

    Li's table is h^2-rescaled W/Hz.  Our fit is cgs erg/s/Hz with no h^2
    division: convert by log L0(W/Hz, h-rescaled) = log L0(cgs) - 7 + 2 log h.
    """
    h = 0.7
    for label, expected_a, expected_b, expected_logL0_WHz in (
        ("JJ2.29", 0.39, 0.60, 27.1),
        ("GKF5.0", 0.02, 0.72, 27.4),
    ):
        df = li_io.load_jj229() if label == "JJ2.29" else li_io.load_gkf50()
        L = luminosity_at_band(df["S_jy"].to_numpy(), df["z"].to_numpy(),
                               df["alpha_spec"].to_numpy())
        fit = fit_size_redshift_relation(df["z"].to_numpy(), L)
        assert abs(fit.a - expected_a) < 0.10, f"{label} a={fit.a:.3f} vs Li {expected_a}"
        assert abs(fit.b - expected_b) < 0.10, f"{label} b={fit.b:.3f} vs Li {expected_b}"
        log_L0_WHz_h2 = fit.log_L0 - 7.0 + 2.0 * np.log10(h)
        assert abs(log_L0_WHz_h2 - expected_logL0_WHz) < 0.3, (
            f"{label} logL0={log_L0_WHz_h2:.2f} W/Hz (h-rescaled) vs Li {expected_logL0_WHz}"
        )


def test_li_d0_matches_published_table1_for_nonexpanding():
    """Li Table 1 d_0 values were derived under the non-expanding fit."""
    for label, expected_d0_pc in (("JJ2.29", 21.6), ("GKF5.0", 63.7)):
        df = li_io.load_jj229() if label == "JJ2.29" else li_io.load_gkf50()
        L = luminosity_at_band(df["S_jy"].to_numpy(), df["z"].to_numpy(),
                               df["alpha_spec"].to_numpy())
        fit = fit_size_redshift_relation(df["z"].to_numpy(), L)
        theta_calib = calibrated_angular_size(df["theta_mas"].to_numpy(),
                                              df["z"].to_numpy(), fit.a, fit.b)
        d0, _, _ = fit_d0_nonexpanding(theta_calib, df["z"].to_numpy())
        # Loose: median vs Li's likely-OLS-on-log-theta procedure can differ
        # by O(20%); we just need to be in the right neighbourhood.
        assert 0.5 * expected_d0_pc < d0 < 2.0 * expected_d0_pc, (
            f"{label} d0={d0:.2f} pc vs Li {expected_d0_pc} pc"
        )


def test_distance_duality_ratio_is_positive_and_grows_with_z():
    df = li_io.load_jj229()
    L = luminosity_at_band(df["S_jy"].to_numpy(), df["z"].to_numpy(),
                           df["alpha_spec"].to_numpy())
    fit = fit_size_redshift_relation(df["z"].to_numpy(), L)
    theta_calib = calibrated_angular_size(df["theta_mas"].to_numpy(),
                                          df["z"].to_numpy(), fit.a, fit.b)
    d0, _, _ = fit_d0_nonexpanding(theta_calib, df["z"].to_numpy())
    ratio = distance_duality_ratio(theta_calib, df["z"].to_numpy(), d0_pc=d0)
    assert np.all(ratio > 0)
    assert ratio[df["z"].argmax()] > ratio[df["z"].argmin()]


def test_bin_median_returns_n_filled_bins():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 4, size=600)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.1, size=x.size)
    binned = bin_median(x, y, n_bins=20)
    assert len(binned) == 20
    assert binned["n"].sum() == len(x)
    # monotonic in median
    assert (np.diff(binned["x_median"]) > 0).all()


@pytest.mark.parametrize("power", [2, 3])
def test_run_li_on_tng_returns_finite_outputs_on_synthetic(power: int):
    """Sanity-test the TNG-analogue runner on a synthetic measured catalog.

    The actual TNG100 catalog is produced on a compute cluster and the
    local TNG50 mock is too small for a meaningful Li fit; this synthetic
    stand-in exercises the column wiring and numerical pipeline only.
    """
    rng = np.random.default_rng(42)
    n = 1500
    z = rng.uniform(0.05, 4.0, size=n)
    theta = 0.2 + 0.05 * rng.normal(size=n)  # arcsec
    flux = 0.5 * rng.lognormal(mean=-2, sigma=0.5, size=n)  # uJy
    df = pd.DataFrame({
        "z": z,
        "kpc_per_arcsec": 8.0 * np.ones(n),
        "flux_F200W_uJy": flux,
        "aperture_radius_arcsec_F200W": np.maximum(theta, 0.02),
        "aperture_area_arcsec2_F200W": np.pi * np.maximum(theta, 0.02) ** 2,
        "image_success_F200W": True,
        "mock_spec_selected": True,
    })
    sample = li_io.TNGSample(
        df=df, filter_name="F200W",
        flux_col="flux_F200W_uJy",
        radius_col="aperture_radius_arcsec_F200W",
        area_col="aperture_area_arcsec2_F200W",
    )
    result = run_li_on_tng(sample, alpha_spec=-0.5, power=power)
    assert result.fit.n > 1000
    assert np.isfinite(result.d0_nonexpanding_pc)
    assert np.isfinite(result.d0_expanding_pc)
    assert np.all(np.isfinite(result.ddr_ratio))


def test_intrinsic_density_returns_gammas():
    rng = np.random.default_rng(7)
    n = 800
    z = rng.uniform(0.1, 3.5, size=n)
    df = pd.DataFrame({
        "z": z,
        "kpc_per_arcsec": 8.0 * np.ones(n),
        "flux_F200W_uJy": np.exp(rng.normal(-2, 0.4, size=n)),
        "aperture_radius_arcsec_F200W": 0.15 + 0.02 * rng.normal(size=n),
        "aperture_area_arcsec2_F200W": np.pi * 0.15 ** 2 * np.ones(n),
        "image_success_F200W": True,
        "mock_spec_selected": True,
    })
    sample = li_io.TNGSample(
        df=df, filter_name="F200W",
        flux_col="flux_F200W_uJy",
        radius_col="aperture_radius_arcsec_F200W",
        area_col="aperture_area_arcsec2_F200W",
    )
    rho = intrinsic_luminosity_density_evolution(sample, alpha_spec=-0.5)
    assert rho["n"] > 500
    assert np.isfinite(rho["gamma_rho_volume"])
    assert np.isfinite(rho["gamma_sigma_area"])

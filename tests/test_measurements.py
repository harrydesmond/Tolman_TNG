import numpy as np
import pandas as pd

from tolman.filters import CatalogSchema, FilterSpec
from tolman.fitting import fit_power_law
from tolman.measurements import (
    add_all_observables,
    modal_typical_surface_brightness,
)


def schema():
    return CatalogSchema(
        redshift_col="z",
        area_col="area",
        filters=(
            FilterSpec("F150W", 15000.0, "flux_F150W", "err_F150W"),
            FilterSpec("F200W", 20000.0, "flux_F200W", "err_F200W"),
            FilterSpec("F356W", 35600.0, "flux_F356W", "err_F356W"),
        ),
    )


def test_surface_brightness_and_rest_uv_selection():
    df = pd.DataFrame(
        {
            "z": [7.0, 2.0],
            "area": [2.0, 4.0],
            "flux_F150W": [10.0, 4.0],
            "flux_F200W": [20.0, 8.0],
            "flux_F356W": [36.0, 12.0],
            "err_F150W": [1.0, 1.0],
            "err_F200W": [1.0, 1.0],
            "err_F356W": [1.0, 1.0],
        }
    )
    out = add_all_observables(df, schema())
    assert np.isclose(out.loc[0, "sb_F150W"], 5.0)
    assert np.isclose(out.loc[0, "sb_mean"], np.mean([5.0, 10.0, 18.0]))
    assert np.isclose(out.loc[0, "sb_max"], 18.0)
    assert np.isclose(out.loc[0, "sb_rest_uv_1500_2000"], 5.0)
    assert np.isnan(out.loc[1, "sb_rest_uv_1500_2000"])


def test_power_law_fit_recovers_alpha():
    z = np.linspace(0.1, 8.0, 200)
    alpha = -4.0
    sb = 7.5 * (1.0 + z) ** alpha
    fit = fit_power_law(z, sb)
    assert np.isclose(fit.alpha, alpha, atol=1.0e-10)
    assert fit.n == 200


def test_modal_typical_surface_brightness_returns_power_law_bins():
    rng = np.random.default_rng(0)
    z = rng.uniform(0.1, 8.5, 3000)
    sb = 4.0 * (1.0 + z) ** -1.2 * rng.lognormal(0.0, 0.08, len(z))
    modal = modal_typical_surface_brightness(z, sb, n_z_bins=12, n_sb_bins=20)
    assert len(modal) >= 8
    fit = fit_power_law(modal["z_center"], modal["sb"], y_sigma=modal["log_sb_err"])
    assert np.isclose(fit.alpha, -1.2, atol=0.25)


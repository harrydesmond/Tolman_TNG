from pathlib import Path

import numpy as np
import pandas as pd

from tolman.imageap_diagnostics import (
    aperture_physics_bins,
    aperture_physics_models,
    object_flux_ratio_table,
    slope_decomposition,
    write_image_aperture_diagnostics,
)


def _mock_catalogues(n: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = np.linspace(0.2, 7.0, n)
    x = 1.0 + z
    catalogue_sb = 20.0 * x**-1.0
    aperture_ratio = 0.45 * x**0.35
    image_sb = catalogue_sb * aperture_ratio
    rows = {
        "object_key": [f"obj-{idx}" for idx in range(n)],
        "row_id": np.arange(n),
        "z": z,
        "mock_spec_selected": np.arange(n) % 2 == 0,
        "p_spec": np.linspace(0.0, 1.0, n),
        "n_valid_filters": 3,
        "aperture_area_arcsec2_used": 1.0,
        "sb_mean": catalogue_sb,
        "sb_max": 1.4 * catalogue_sb,
        "sb_rest_uv_1500_2000": 0.6 * catalogue_sb,
    }
    catalogue = pd.DataFrame(rows)
    image = pd.DataFrame(rows).copy()
    image["mock_spec_selected"] = np.arange(n) % 3 == 0
    image["sb_mean"] = image_sb
    image["sb_max"] = 1.4 * image_sb
    image["sb_rest_uv_1500_2000"] = 0.6 * image_sb
    image["flux_F200W_uJy"] = image_sb
    image["catalog_flux_F200W_uJy"] = catalogue_sb
    image["aperture_radius_arcsec_F200W"] = 0.2 * x**-0.2
    image["kpc_per_arcsec"] = 6.0
    image["mstar_msun_2rhalf"] = 1.0e9 * x
    image["sfr_msun_per_yr_2rhalf"] = 0.5 * x
    image["app_gmag_rest"] = 25.0 + z
    return catalogue, image


def test_slope_decomposition_recovers_aperture_ratio_slope():
    catalogue, image = _mock_catalogues()
    merged = catalogue.merge(image, on="object_key", suffixes=("_catalogue", "_image"))
    merged["z"] = merged["z_image"]

    _, summary = slope_decomposition(merged)
    mean = summary[summary["observable"] == "mean_sb"].iloc[0]

    np.testing.assert_allclose(mean["alpha_catalogue_original_selection"], -1.0, atol=1.0e-10)
    assert abs(mean["delta_aperture_on_image_selection"] - 0.35) < 1.0e-10
    assert abs(mean["alpha_image_to_catalogue_ratio"] - 0.35) < 1.0e-10


def test_write_image_aperture_diagnostics_outputs(tmp_path: Path):
    catalogue, image = _mock_catalogues()
    cat_path = tmp_path / "catalogue.csv"
    image_path = tmp_path / "image.csv"
    out_dir = tmp_path / "out"
    plot_dir = tmp_path / "plots"
    catalogue.to_csv(cat_path, index=False)
    image.to_csv(image_path, index=False)

    tables = write_image_aperture_diagnostics(
        cat_path,
        image_path,
        out_dir=out_dir,
        plot_dir=plot_dir,
        sample_name="test_imageap",
    )

    assert set(tables) == {
        "detailed",
        "summary",
        "flux",
        "binned",
        "objects",
        "physics_univariate",
        "physics_conditional",
        "physics_bins",
    }
    assert (out_dir / "slope_decomposition_summary.csv").exists()
    assert (out_dir / "filter_flux_ratio_slopes.csv").exists()
    assert (out_dir / "aperture_physics_conditional_redshift_models.csv").exists()
    assert (out_dir / "summary.md").exists()
    assert (plot_dir / "test_imageap_slope_decomposition.png").exists()


def test_aperture_physics_models_keep_redshift_alpha_in_raw_log1pz():
    _, image = _mock_catalogues()
    objects = object_flux_ratio_table(image)
    univariate, conditional = aperture_physics_models(objects)
    bins = aperture_physics_bins(objects, n_bins=3)

    redshift = univariate[univariate["predictors"] == "log1pz"].iloc[0]
    no_control = conditional[conditional["controls"] == ""].iloc[0]

    assert abs(redshift["coef_log1pz"] - 0.35) < 1.0e-10
    assert abs(no_control["coef_log1pz"] - 0.35) < 1.0e-10
    assert len(bins) == 3

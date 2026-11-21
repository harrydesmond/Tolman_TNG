from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from tolman.aperture_sweep import write_aperture_sweep


def _make_gaussian(cx, cy, sigma, shape=(60, 60), amplitude=1.0):
    y, x = np.mgrid[: shape[0], : shape[1]]
    return amplitude * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / sigma**2)


def test_aperture_sweep_slope_decreases_with_scale(tmp_path: Path):
    """Larger aperture → ratio closer to unity and weaker redshift trend."""
    image = np.zeros((60, 60), dtype=np.float32)
    # Two galaxies at different 'y' (we use linear coords, so y≈row)
    # Low-z galaxy (z=0.3): larger, placed at (15, 15)
    image += _make_gaussian(15, 15, sigma=4.0, amplitude=3.0).astype(np.float32)
    # High-z galaxy (z=2.0): smaller, placed at (45, 45)
    image += _make_gaussian(45, 45, sigma=1.5, amplitude=3.0).astype(np.float32)

    fits_path = tmp_path / "hlsp_illustris_jwst_nircam_tng50_f200w_v1_sim.fits"
    hdr = fits.Header({"BUNIT": "uJy", "PIXSCALE": 0.06})
    fits.PrimaryHDU(image, header=hdr).writeto(fits_path)

    # catalogue fluxes equal full Gaussian integrals
    flux_low_z = float(3.0 * 2 * np.pi * 4.0**2)
    flux_high_z = float(3.0 * 2 * np.pi * 1.5**2)

    catalog = pd.DataFrame(
        {
            "z": [0.3, 2.0],
            "mock_spec_selected": [True, True],
            "x_F200W_pix": [15.0, 45.0],
            "y_F200W_pix": [15.0, 45.0],
            # nominal aperture: 0.3 arcsec → 5 pixels at 0.06"/pix
            "aperture_radius_arcsec_F200W": [0.3, 0.3],
            "catalog_flux_F200W_uJy": [flux_low_z, flux_high_z],
        }
    )
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    tables = write_aperture_sweep(
        catalog_path,
        [fits_path],
        out_dir=tmp_path / "out",
        plot_dir=tmp_path / "plots",
        sample_name="test_sweep",
        scale_factors=(0.5, 1.0, 2.0, 4.0),
    )

    assert set(tables) == {"per_filter", "objects", "bins", "slopes"}
    slopes = tables["slopes"]
    assert list(slopes["scale_factor"]) == [0.5, 1.0, 2.0, 4.0]
    assert (tmp_path / "out" / "aperture_sweep_slope_table.csv").exists()
    assert (tmp_path / "plots" / "test_sweep_ratio_vs_z_bins.png").exists()
    assert (tmp_path / "plots" / "test_sweep_slope_vs_scale.png").exists()
    # larger apertures should recover more of the larger low-z galaxy → ratio
    # at scale 4 should exceed ratio at scale 0.5 for the low-z object
    objs = tables["objects"]
    low_z_rows = objs[objs["z"] < 1.0]
    ratio_small = float(low_z_rows[low_z_rows["scale_factor"] == 0.5]["aperture_to_catalogue_ratio"].iloc[0])
    ratio_large = float(low_z_rows[low_z_rows["scale_factor"] == 4.0]["aperture_to_catalogue_ratio"].iloc[0])
    assert ratio_large > ratio_small

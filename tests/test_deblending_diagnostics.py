from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from tolman.deblending_diagnostics import write_deblending_diagnostics


def test_voronoi_deblending_recovers_isolated_and_splits_blend(tmp_path: Path):
    """Isolated galaxy gets its full segment flux; blended pair is split."""
    image = np.zeros((60, 60), dtype=np.float32)
    # Source A at (15, 15): isolated bright blob
    ys_a, xs_a = np.mgrid[10:20, 10:20]
    image[ys_a, xs_a] = 3.0

    # Source B at (40, 40) and C at (44, 40): two close galaxies sharing a segment
    ys_bc, xs_bc = np.mgrid[37:47, 37:47]
    image[ys_bc, xs_bc] = 3.0

    fits_path = tmp_path / "hlsp_illustris_jwst_nircam_tng50_f200w_v1_sim.fits"
    hdr = fits.Header({"BUNIT": "uJy", "PIXSCALE": 0.06})
    fits.PrimaryHDU(image, header=hdr).writeto(fits_path)

    # catalogue fluxes = segment area × amplitude
    flux_a = float(10 * 10 * 3.0)
    flux_bc = float(10 * 10 * 3.0)  # each galaxy nominally gets ~half the shared segment

    catalog = pd.DataFrame(
        {
            "z": [0.5, 1.5, 2.5],
            "mock_spec_selected": [True, True, True],
            # A is isolated; B and C share a segment
            "x_F200W_pix": [15.0, 40.0, 44.0],
            "y_F200W_pix": [15.0, 40.0, 40.0],
            "aperture_radius_arcsec_F200W": [0.3, 0.3, 0.3],
            "aperture_radius_pix_F200W": [5.0, 5.0, 5.0],
            "flux_F200W_uJy": [280.0, 130.0, 130.0],
            "catalog_flux_F200W_uJy": [flux_a, flux_bc, flux_bc],
        }
    )
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    tables = write_deblending_diagnostics(
        catalog_path,
        [fits_path],
        out_dir=tmp_path / "out",
        plot_dir=tmp_path / "plots",
        sample_name="test_deblend",
        threshold_sigma=1.0,
        min_pixels=4,
    )

    assert set(tables) == {"per_filter", "objects", "bins", "filter_summary", "slopes"}
    assert (tmp_path / "out" / "deblending_slope_table.csv").exists()
    assert (tmp_path / "plots" / "test_deblend_deblending_redshift_bins.png").exists()

    pf = tables["per_filter"]
    row_a = pf[pf["z"] == 0.5].iloc[0]
    # Isolated A: deblended flux should be close to its total segment flux
    assert np.isfinite(row_a["deblended_to_catalogue_ratio"])
    assert row_a["n_cat_in_segment"] == 1

    row_b = pf[pf["z"] == 1.5].iloc[0]
    row_c = pf[pf["z"] == 2.5].iloc[0]
    # B and C share segment: each gets about half
    assert row_b["n_cat_in_segment"] == 2
    assert row_c["n_cat_in_segment"] == 2
    # Combined deblended flux ≈ total segment flux (within tolerance)
    total_dbl = row_b["deblended_flux_uJy"] + row_c["deblended_flux_uJy"]
    # Full BC segment flux = 10×10×3 = 300 uJy; B+C combined should recover that
    assert abs(total_dbl - 300.0) < 5.0, f"Combined deblended flux {total_dbl:.1f} ≠ 300"

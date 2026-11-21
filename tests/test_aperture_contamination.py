from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from tolman.aperture_contamination import write_annulus_contamination_diagnostics


def test_annulus_contamination_outputs(tmp_path: Path):
    image = np.ones((40, 40), dtype=np.float32)
    image[18:23, 18:23] += 5.0
    path = tmp_path / "hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f200w_v1_sim.fits"
    fits.PrimaryHDU(image, header=fits.Header({"BUNIT": "uJy"})).writeto(path)

    catalog = pd.DataFrame(
        {
            "z": np.linspace(0.2, 2.0, 6),
            "mock_spec_selected": [True] * 6,
            "x_F200W_pix": [20, 12, 28, 20, 14, 26],
            "y_F200W_pix": [20, 12, 28, 14, 26, 20],
            "aperture_radius_pix_F200W": [2.0] * 6,
            "flux_F200W_uJy": [70, 14, 14, 14, 14, 14],
            "catalog_flux_F200W_uJy": [60, 20, 20, 20, 20, 20],
        }
    )
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    out = write_annulus_contamination_diagnostics(
        catalog_path,
        [path],
        out_dir=tmp_path / "out",
        plot_dir=tmp_path / "plots",
        sample_name="test_annulus",
    )

    assert set(out) == {"per_filter", "summary", "objects", "bins", "fits"}
    assert (tmp_path / "out" / "annulus_filter_summary.csv").exists()
    assert (tmp_path / "out" / "annulus_ratio_fits.csv").exists()
    assert (tmp_path / "plots" / "test_annulus_annulus_redshift_bins.png").exists()
    assert out["summary"]["median_local_annulus_to_catalogue_ratio"].iloc[0] > 0.0

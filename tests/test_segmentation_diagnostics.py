from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from tolman.segmentation_diagnostics import write_segmentation_diagnostics


def test_segmentation_diagnostics_flags_blends_and_isolated_sources(tmp_path: Path):
    image = np.zeros((40, 40), dtype=np.float32)
    image[9:14, 9:14] = 5.0
    image[25:29, 25:29] = 4.0
    path = tmp_path / "hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f200w_v1_sim.fits"
    fits.PrimaryHDU(image, header=fits.Header({"BUNIT": "uJy"})).writeto(path)

    catalog = pd.DataFrame(
        {
            "z": [0.3, 0.6, 1.0],
            "mock_spec_selected": [True, True, True],
            "x_F200W_pix": [11.0, 12.0, 26.0],
            "y_F200W_pix": [11.0, 12.0, 26.0],
            "aperture_radius_pix_F200W": [2.0, 2.0, 2.0],
            "flux_F200W_uJy": [50.0, 50.0, 40.0],
            "catalog_flux_F200W_uJy": [100.0, 100.0, 80.0],
        }
    )
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    out = write_segmentation_diagnostics(
        catalog_path,
        [path],
        out_dir=tmp_path / "out",
        plot_dir=tmp_path / "plots",
        sample_name="test_segmentation",
        threshold_sigma=1.0,
        min_pixels=4,
    )

    assert set(out) == {"per_filter", "detections", "summary", "objects", "bins", "fits"}
    assert (tmp_path / "out" / "segmentation_ratio_fits.csv").exists()
    assert (tmp_path / "plots" / "test_segmentation_segmentation_redshift_bins.png").exists()
    assert len(out["detections"]) == 2
    assert out["per_filter"]["n_catalogue_objects_in_segment"].tolist() == [2, 2, 1]
    assert out["per_filter"]["is_isolated_catalogue_segment"].tolist() == [False, False, True]

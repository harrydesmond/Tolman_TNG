import numpy as np
import pandas as pd
from astropy.io import fits

from tolman.image_photometry import add_circular_aperture_fluxes, circular_aperture_sum


def test_circular_aperture_sum_counts_pixels():
    image = np.ones((9, 9), dtype=float)
    flux = circular_aperture_sum(image, x=4.5, y=4.5, radius_pix=1.5)
    assert flux > 4.0
    assert flux < 10.0


def test_add_circular_aperture_fluxes(tmp_path):
    path = tmp_path / "f200w.fits"
    fits.writeto(path, np.ones((9, 9), dtype=float), overwrite=True)
    catalog = pd.DataFrame({"x": [4.0], "y": [4.0], "r": [1.5]})
    out = add_circular_aperture_fluxes(
        catalog,
        image_paths={"F200W": path},
        x_col="x",
        y_col="y",
        radius_pix_col="r",
        pixel_scale_arcsec=0.03,
    )
    assert "flux_F200W_uJy" in out
    assert "aperture_area_arcsec2" in out
    assert out.loc[0, "flux_F200W_uJy"] > 0

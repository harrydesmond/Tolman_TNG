import numpy as np
from astropy.io import fits
from astropy.table import Table

from tolman.tng_image_aperture import normalize_tng_image_aperture_fits


def test_tng_image_aperture_normalizes_njy_image(tmp_path):
    image = np.zeros((21, 21), dtype=np.float32)
    image[10, 10] = 1000.0

    output = Table()
    output["image_success"] = [1]
    output["primary_flag"] = [1]
    output["photrad_kpc"] = [0.03]
    output["cutoutfov_kpc"] = [1.0]
    output["cutout_size"] = [1]
    output["n_arcmin"] = [0.0]
    output["total_quant"] = [1000.0]
    output["Snapshot number"] = [90]
    output["Subhalo index"] = [7]
    output["RA degree"] = [0.0]
    output["DEC degree"] = [0.0]
    output["RA true z"] = [0.0]
    output["DEC true z"] = [0.0]
    output["RA inferred z"] = [0.0]
    output["DEC inferred z"] = [0.0]
    output["True z"] = [1.0]
    output["Inferred z"] = [1.0]
    output["Peculiar z"] = [0.0]
    output["True scale"] = [1.0]
    output["Comoving X"] = [0.0]
    output["Comoving Y"] = [0.0]
    output["Comoving Z"] = [0.0]
    output["True angular distance"] = [1.0]
    output["Inferred angular distance"] = [1.0]
    output["Snapshot z"] = [1.0]
    output["Geometric z"] = [1.0]
    output["Lightcone number"] = [1]
    output["Stellar mass w2sr"] = [1.0]
    output["Total gas mass w2sr"] = [1.0]
    output["Total subhalo mass"] = [1.0]
    output["Total BH mass w2sr"] = [1.0]
    output["Total baryon mass w2sr"] = [1.0]
    output["SFR w2sr"] = [1.0]
    output["Total BH accretion rate"] = [1.0]
    output["Camera X"] = [0.0]
    output["Camera Y"] = [0.0]
    output["Camera Z"] = [0.0]
    output["Intrinsic g mag"] = [1.0]
    output["Intrinsic r mag"] = [1.0]
    output["Intrinsic i mag"] = [1.0]
    output["Intrinsic z mag"] = [1.0]
    output["Galaxy motion X"] = [0.0]
    output["Galaxy motion Y"] = [0.0]
    output["Galaxy motion Z/Peculiar"] = [0.0]
    output["Cosmological expansion"] = [1.0]
    output["Apparent total gmag"] = [1.0]

    path = tmp_path / "hlsp_illustris_jwst_nircam_tng50-12-11-xyz_f200w_v1_sim.fits"
    primary = fits.PrimaryHDU(image)
    primary.header["BUNIT"] = "nanoJanskies"
    primary.header["PIXSCALE"] = 0.03
    fits.HDUList([primary, fits.BinTableHDU(output, name="OutputData")]).writeto(path)

    out, diag = normalize_tng_image_aperture_fits([path], sample_name="test", coordinate_mode="linear")
    assert np.isclose(out.loc[0, "flux_F200W_uJy"], 1.0)
    assert np.isclose(diag.loc[0, "median_aperture_to_catalog_flux"], 1.0)

from tolman.mast_illustris import filter_name_from_path


def test_filter_name_from_tng_fits_path():
    name = filter_name_from_path("hlsp_illustris_jwst_nircam_tng50-12-11-xyz_f200w_v1_sim.fits")
    assert name == "F200W"


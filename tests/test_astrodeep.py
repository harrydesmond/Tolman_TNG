import numpy as np
import pandas as pd

from tolman.astrodeep import normalize_astrodeep_catalogs, write_astrodeep_schema
from tolman.filters import schema_from_toml
from tolman.io import read_catalog


def test_normalize_astrodeep_catalogs(tmp_path):
    phot = pd.DataFrame(
        {
            "ID": [1, 2],
            "RA": [10.0, 11.0],
            "DEC": [-1.0, -2.0],
            "optap": [0.4, 0.6],
            "F200W": [1.0, 2.0],
            "eF200W": [0.1, 0.2],
            "flag": [0, 99],
        }
    )
    red = pd.DataFrame({"ID": [1, 2], "zspec": [3.0, -1.0], "zphot": [3.1, 4.2]})
    phot_path = tmp_path / "phot.csv"
    red_path = tmp_path / "red.csv"
    phot.to_csv(phot_path, index=False)
    red.to_csv(red_path, index=False)

    out = normalize_astrodeep_catalogs(
        phot_path,
        red_path,
        field_name="test",
        aperture_kind="diameter",
    )
    assert np.isclose(out.loc[0, "aperture_area_arcsec2"], np.pi * 0.2**2)
    assert out.loc[0, "has_spec"]
    assert not out.loc[1, "has_spec"]
    assert np.isclose(out.loc[0, "z"], 3.0)
    assert np.isclose(out.loc[1, "z"], 4.2)
    assert "flux_F200W_uJy" in out
    assert "fluxerr_F200W_uJy" in out

    out_radius = normalize_astrodeep_catalogs(
        phot_path, red_path, field_name="test", aperture_kind="radius"
    )
    assert np.isclose(out_radius.loc[0, "aperture_area_arcsec2"], np.pi * 0.4**2)


def test_write_astrodeep_schema(tmp_path):
    path = tmp_path / "schema.toml"
    write_astrodeep_schema(path)
    schema = schema_from_toml(path)
    assert schema.redshift_col == "z"
    assert schema.selection.spec_selection_col == "has_spec"
    names = [spec.name for spec in schema.filters]
    assert "F140W" in names
    assert "F850LP" not in names


def test_read_vizier_asu_tsv(tmp_path):
    path = tmp_path / "vizier.tsv"
    path.write_text(
        "\n".join(
            [
                "#   VizieR Astronomical Server vizier.cds.unistra.fr",
                "#Table\tJ_A_A_691_A240_a2744o:",
                "#Column\trecno\t(I8)\tRecord number",
                "recno\tID\tapopt\tFF140W\te_FF140W",
                " \t \tarcsec\tuJy\tuJy",
                "--------\t------\t-----\t------------\t------------",
                "       1\t     7\t 0.28\t 1.10000e-02\t 2.00000e-03",
                "       2\t     8\t 0.50\t 2.20000e-02\t 3.00000e-03",
            ]
        ),
        encoding="utf-8",
    )

    table = read_catalog(path)
    assert list(table["ID"]) == [7, 8]
    assert np.isclose(table.loc[0, "FF140W"], 0.011)

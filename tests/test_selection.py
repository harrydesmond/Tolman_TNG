import numpy as np
import pandas as pd

from tolman.filters import CatalogSchema, FilterSpec, SelectionConfig
from tolman.measurements import add_all_observables
from tolman.selection import flux_ujy_to_abmag, selection_ladder


def test_flux_to_abmag():
    assert np.isclose(flux_ujy_to_abmag([1.0])[0], 23.9)


def test_selection_ladder_applies_quality_and_spec_proxy():
    schema = CatalogSchema(
        redshift_col="z",
        area_col="area",
        filters=(FilterSpec("F200W", 20000.0, "flux", "err"),),
        flags_col="flags",
        bad_flag_min=80,
        blend_col="blend",
        selection=SelectionConfig(min_snr=5, spec_mag_filter="F200W", spec_mag_limit=25),
    )
    df = pd.DataFrame(
        {
            "z": [1.0, 2.0, 3.0, 4.0],
            "area": [1.0, 1.0, 1.0, 1.0],
            "flux": [10.0, 10.0, 0.01, 10.0],
            "err": [1.0, 1.0, 1.0, 1.0],
            "flags": [0, 99, 0, 0],
            "blend": [False, False, False, True],
        }
    )
    measured = add_all_observables(df, schema)
    stages = {stage.name: stage.df for stage in selection_ladder(measured, schema)}
    assert len(stages["detected_catalogue"]) == 4
    assert len(stages["snr_limited"]) == 3
    assert len(stages["aperture_quality"]) == 2
    assert len(stages["blend_edge_point_clean"]) == 1
    assert len(stages["mock_spectroscopic"]) == 1


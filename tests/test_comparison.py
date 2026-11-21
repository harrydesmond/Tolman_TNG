import numpy as np
import pandas as pd

from tolman.comparison import compare_slope_tables


def test_compare_slope_tables_matches_common_observables(tmp_path):
    obs = pd.DataFrame(
        {
            "observable": ["mean_sb", "max_sb"],
            "selection": ["spectroscopic", "spectroscopic"],
            "alpha": [-1.2, -1.0],
            "alpha_err": [0.1, 0.2],
            "n": [10, 10],
            "x_min": [0.0, 0.0],
            "x_max": [1.0, 1.0],
        }
    )
    mock = pd.DataFrame(
        {
            "observable": ["mean_sb"],
            "selection": ["mock_spectroscopic"],
            "alpha": [-1.1],
            "alpha_err": [0.2],
            "n": [12],
            "x_min": [0.1],
            "x_max": [0.9],
        }
    )
    obs_path = tmp_path / "obs.csv"
    mock_path = tmp_path / "mock.csv"
    obs.to_csv(obs_path, index=False)
    mock.to_csv(mock_path, index=False)
    out = compare_slope_tables(obs_path, mock_path)
    assert list(out["observable"]) == ["mean_sb"]
    assert np.isclose(out.loc[0, "delta_alpha_mock_minus_observed"], 0.1)

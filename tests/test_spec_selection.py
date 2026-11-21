import numpy as np
import pandas as pd

from tolman.spec_selection import (
    add_spec_features,
    add_selection_probabilities,
    fit_model,
    select_by_probability,
)


def test_spec_selection_fit_and_apply():
    rng = np.random.default_rng(0)
    n = 200
    flux = 10 ** rng.uniform(-2.5, 1.0, n)
    df = pd.DataFrame(
        {
            "flux_F200W_uJy": flux,
            "flux_F356W_uJy": flux * rng.lognormal(0.0, 0.1, n),
            "flux_F150W_uJy": flux * rng.lognormal(0.0, 0.1, n),
            "flux_F444W_uJy": flux * rng.lognormal(0.0, 0.1, n),
            "sb_mean": flux / 0.02,
            "aperture_area_arcsec2": 0.02,
            "z": rng.uniform(0.1, 5.0, n),
            "n_valid_filters": 4,
        }
    )
    df["has_spec"] = df["flux_F200W_uJy"] > np.median(df["flux_F200W_uJy"])
    model, features = fit_model(df)
    out = add_selection_probabilities(df, model, features)
    assert "p_spec" in out
    assert out["p_spec"].between(0.0, 1.0).all()
    selected = select_by_probability(out, mode="deterministic", target_count=50)
    assert selected["mock_spec_selected"].sum() == 50


def test_spec_features_compute_sb_mean_when_absent():
    df = pd.DataFrame(
        {
            "flux_F200W_uJy": [2.0, -1.0],
            "flux_F356W_uJy": [4.0, 6.0],
            "aperture_area_arcsec2": [2.0, 3.0],
            "z": [1.0, 2.0],
        }
    )
    out = add_spec_features(df)
    assert np.isclose(out.loc[0, "sb_mean"], 1.5)
    assert np.isclose(out.loc[1, "sb_mean"], 2.0)
    assert list(out["n_valid_filters"]) == [2, 1]

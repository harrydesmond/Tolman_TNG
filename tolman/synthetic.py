"""Synthetic catalogue generator for pipeline validation and demos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .filters import ASTRODEEP_JWST_FILTERS


def generate_synthetic_catalog(n: int = 6000, seed: int = 1) -> pd.DataFrame:
    """Generate a mock-observed catalogue with an apparent alpha near -1.

    The generator is simple and is not a galaxy-formation model.
    It provides a deterministic integration test for the measurement pipeline.
    """

    rng = np.random.default_rng(seed)
    log1pz = rng.uniform(np.log10(1.03), np.log10(9.5), n)
    z = 10.0**log1pz - 1.0
    field = rng.choice(["tng100_xyz", "tng100_yxz", "tng300_xyz"], size=n, p=[0.4, 0.4, 0.2])
    aperture_radius = 0.14 * (1.0 + z) ** -0.45 * rng.lognormal(0.0, 0.35, n) + 0.025
    area = np.pi * aperture_radius**2

    intrinsic_surface_boost = (1.0 + z) ** 2.0
    tolman_dimming = (1.0 + z) ** -3.0
    base_sb = 2.5 * intrinsic_surface_boost * tolman_dimming
    base_sb *= rng.lognormal(0.0, 0.55, n)

    data: dict[str, object] = {
        "id": np.arange(n),
        "z": z,
        "field": field,
        "aperture_area_arcsec2": area,
        "flags": rng.choice([0, 1, 2, 32, 96], size=n, p=[0.72, 0.12, 0.08, 0.06, 0.02]),
        "is_blended": rng.random(n) < 0.04,
        "is_edge": rng.random(n) < 0.02,
        "is_point_source": rng.random(n) < 0.03,
    }

    for name, lam in ASTRODEEP_JWST_FILTERS.items():
        rest_lam = lam / (1.0 + z)
        uv_bump = np.exp(-0.5 * ((np.log(rest_lam) - np.log(1750.0)) / 0.48) ** 2)
        optical_bump = 0.45 * np.exp(-0.5 * ((np.log(rest_lam) - np.log(4500.0)) / 0.55) ** 2)
        sed = 0.25 + uv_bump + optical_bump
        sb = base_sb * sed * rng.lognormal(0.0, 0.12, n)
        flux = sb * area
        depth = 0.0025 + 0.0007 * (lam / 20000.0)
        err = np.sqrt(depth**2 + (0.06 * flux) ** 2)
        observed_flux = flux + rng.normal(0.0, err)
        nondetected = observed_flux / err < 1.0
        observed_flux[nondetected] = np.nan
        data[f"flux_{name}_uJy"] = observed_flux
        data[f"fluxerr_{name}_uJy"] = err
    return pd.DataFrame(data)

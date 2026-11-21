#!/usr/bin/env python3
"""Selection-importance check: full TNG image-aperture-success pool gamma.

Produces the selection-stability numbers cited in the paper:

  * full image-aperture-success pool (N ~ 70,142): gamma_full
  * z < 4 subset of the same pool:                gamma_z<4
  * brightest-K F200W generic bright-galaxy selection (K = mock-spec sample size):
                                                  gamma_brightK
  * canonical mock-spec sample (sanity check vs tng_gamma_aperture_table):
                                                  gamma_canon

Each gamma is fitted by OLS of log10(rho_L) on log10(1+z), with rho_L = L_nu / V,
L_nu the K-corrected rest-frame F200W luminosity (assumed power-law SED slope
ALPHA_SPEC; alpha_SED = -ALPHA_SPEC in paper convention), and V the intrinsic
photrad_kpc volume. ALPHA_SPEC = -0.3 matches the paper's alpha_SED = 0.3 convention;
gamma shifts linearly with that choice (Delta gamma = +Delta alpha_SED).

Bootstrap statistical uncertainties on each gamma are computed by per-galaxy
resampling.

Output: results/li_ddr/full_pool_gamma.csv

Run via cluster/full_pool_gamma.sh (~2-3 minutes on a single core; the
catalogue is ~150k rows on disk).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tolman.li_ddr.li_reproduction import COSMO_LI, luminosity_at_band

# ROOT defaults to the repository root. Set the
# environment variable TOLMAN_WORKDIR to override when running on a different
# cluster / workstation where the data and result trees live elsewhere.
import os
ROOT = Path(os.environ.get("TOLMAN_WORKDIR", Path(__file__).resolve().parents[1]))
CANONICAL_CAT = ROOT / "results/tng100_7_6_xyz_hst_jwst_imageap_specmodel/measured_catalog.csv"
OUT_CSV = ROOT / "results/li_ddr/full_pool_gamma.csv"

FILTER_NAME = "F200W"
ALPHA_SPEC = -0.3   # Li convention; equivalent to paper alpha_SED = 0.3.
N_BOOT = 200
SEED = 20260515


def _intrinsic_photrad_kpc(df: pd.DataFrame) -> np.ndarray:
    """photrad_kpc = aperture_radius_arcsec * kpc_per_arcsec (MAST mock convention)."""
    return (
        df[f"aperture_radius_arcsec_{FILTER_NAME}"].to_numpy(dtype=float)
        * df["kpc_per_arcsec"].to_numpy(dtype=float)
    )


def _pool_mask(df: pd.DataFrame) -> np.ndarray:
    """The image-aperture-success pool: usable F200W flux + finite aperture."""
    keep = np.isfinite(df[f"flux_{FILTER_NAME}_uJy"]) & (df[f"flux_{FILTER_NAME}_uJy"] > 0)
    keep &= df[f"aperture_radius_arcsec_{FILTER_NAME}"] > 0
    if "image_success_F200W" in df.columns:
        keep &= df["image_success_F200W"].astype(bool)
    return keep.to_numpy()


def _gamma_ols(z: np.ndarray, log_rho_L: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(z) & np.isfinite(log_rho_L) & (z > 0)
    if mask.sum() < 10:
        return float("nan"), int(mask.sum())
    g, _ = np.polyfit(np.log10(1.0 + z[mask]), log_rho_L[mask], 1)
    return float(g), int(mask.sum())


def _gamma_bootstrap(z: np.ndarray, log_rho_L: np.ndarray, n_boot: int,
                      rng: np.random.Generator) -> tuple[float, float]:
    mask = np.isfinite(z) & np.isfinite(log_rho_L) & (z > 0)
    z = z[mask]; log_rho_L = log_rho_L[mask]
    n = len(z)
    if n < 10:
        return float("nan"), float("nan")
    gammas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        gammas[k], _ = np.polyfit(np.log10(1.0 + z[idx]), log_rho_L[idx], 1)
    return float(np.mean(gammas)), float(np.std(gammas, ddof=1))


def _compute_one_row(label: str, df: pd.DataFrame,
                      rng: np.random.Generator) -> dict:
    z = df["z"].to_numpy(dtype=float)
    S_jy = df[f"flux_{FILTER_NAME}_uJy"].to_numpy(dtype=float) * 1e-6
    L_cgs = luminosity_at_band(S_jy, z, np.full_like(z, ALPHA_SPEC), cosmo=COSMO_LI)
    photrad_kpc = _intrinsic_photrad_kpc(df)
    V_kpc3 = (4.0 / 3.0) * np.pi * photrad_kpc**3
    rho_L = L_cgs / V_kpc3
    log_rho = np.log10(np.where(rho_L > 0, rho_L, np.nan))
    g_ols, n = _gamma_ols(z, log_rho)
    g_boot_mean, g_boot_std = _gamma_bootstrap(z, log_rho, N_BOOT, rng)
    print(f"  {label:32s}  n={n:6d}  gamma={g_ols:+.3f}  "
          f"(boot {g_boot_mean:+.3f} +/- {g_boot_std:.3f})")
    return {
        "selection": label,
        "n": n,
        "gamma": g_ols,
        "gamma_boot_mean": g_boot_mean,
        "gamma_boot_std": g_boot_std,
        "alpha_spec_li": ALPHA_SPEC,
        "alpha_sed_paper": -ALPHA_SPEC,
    }


def main() -> None:
    print(f"[load] {CANONICAL_CAT}")
    df = pd.read_csv(CANONICAL_CAT, low_memory=False)
    print(f"  rows in catalogue: {len(df)}")

    pool = df.loc[_pool_mask(df)].copy()
    print(f"  image-aperture-success pool: {len(pool)}")

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    # Full image-aperture-success pool
    rows.append(_compute_one_row("full_pool", pool, rng))

    # z < 4 subset of the same pool
    rows.append(_compute_one_row("full_pool_z<4",
                                  pool.loc[pool["z"] < 4.0].copy(), rng))

    # Canonical mock-spec sample (sanity check)
    canon = pool.loc[pool["mock_spec_selected"].astype(bool)].copy() \
        if "mock_spec_selected" in pool.columns else pool.iloc[:0]
    rows.append(_compute_one_row("mock_spec_canonical", canon, rng))

    # Brightest-K F200W generic bright-galaxy selection (K = canonical sample size)
    K = len(canon) if len(canon) else 2901
    bright_idx = np.argsort(pool[f"flux_{FILTER_NAME}_uJy"].to_numpy(dtype=float))[::-1][:K]
    bright = pool.iloc[bright_idx].copy()
    rows.append(_compute_one_row(f"brightest_K={K}", bright, rng))

    table = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print(f"\n[wrote] {OUT_CSV}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

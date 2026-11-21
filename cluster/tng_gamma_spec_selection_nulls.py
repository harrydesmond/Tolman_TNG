#!/usr/bin/env python3
"""Spec-selection null tests on the TNG gamma_rho.

Tests whether the TNG gamma_rho ~= 2 is driven by a
Malmquist-style truncation imposed by the empirical spec-selection
classifier --- i.e.\ whether the classifier preferentially keeps brighter
mocks at high z and dimmer ones at low z, producing a positive
d log L / dz unrelated to the physical evolution of the galaxies.

Two complementary nulls are evaluated against the canonical mock-spec
selected sample:

  (1) Permuted-label classifier null: shuffle the ``has_spec`` target in
      ASTRODEEP, refit logistic regression with balanced class weights,
      apply to TNG, take the top-K by p_spec_null at the canonical
      K=2901, recompute gamma_rho on the volume-fixed observable.
  (2) Uniform-random subsample null: pick K=2901 TNG mocks uniformly at
      random from the image-aperture-success pool, recompute gamma_rho.
      This is the simplest test: if (1) and (2) give similar gammas,
      the classifier-permutation has effectively collapsed to a uniform
      sample.

Both nulls are repeated over N_SEEDS seeds for a distribution of gamma_null.

Outputs:
  results/li_ddr/tng_gamma_spec_selection_nulls.csv          -- per-seed gamma
  results/li_ddr/tng_gamma_spec_selection_nulls_summary.csv  -- summary stats
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from tolman.li_ddr.li_reproduction import COSMO_LI, luminosity_at_band
from tolman.spec_selection import (
    DEFAULT_FEATURES,
    add_selection_probabilities,
    add_spec_features,
    available_features,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get("TOLMAN_WORKDIR", Path(__file__).resolve().parents[1]))
FILTER_NAME = "F200W"
ALPHA_SPEC = -0.3  # Li convention; equivalent to paper alpha_SED = 0.3.
N_SEEDS = 30
TARGET_N = 2901  # canonical image-aperture spec-selected count

ASTRODEEP_CAT = ROOT / "results/astrodeep_cds/measured_catalog.csv"
TNG_CAT = ROOT / "results/tng100_7_6_xyz_hst_jwst_imageap_specmodel/measured_catalog.csv"


def _intrinsic_photrad_kpc(df: pd.DataFrame) -> np.ndarray:
    return (
        df["aperture_radius_arcsec_F200W"].to_numpy(dtype=float)
        * df["kpc_per_arcsec"].to_numpy(dtype=float)
    )


def _gamma_rho_volume_controlled(df: pd.DataFrame) -> tuple[float, int]:
    """Volume-controlled gamma_rho on a TNG selected subset.

    Uses photrad_kpc (= 2 r_half_phys, fixed in physical units) as V to avoid
    the aperture-volume artefact discussed in tng_gamma_aperture_table.py.
    """
    z = df["z"].to_numpy(dtype=float)
    flux_uJy = df[f"flux_{FILTER_NAME}_uJy"].to_numpy(dtype=float)
    photrad_kpc = _intrinsic_photrad_kpc(df)
    flux_jy = flux_uJy * 1e-6
    L = luminosity_at_band(flux_jy, z, np.full_like(z, ALPHA_SPEC), cosmo=COSMO_LI)
    r_cm = photrad_kpc * 3.0857e21
    rho_L = L / ((4.0 / 3.0) * np.pi * r_cm**3)
    finite = np.isfinite(rho_L) & (rho_L > 0) & np.isfinite(z) & (z > 0)
    if finite.sum() < 10:
        return float("nan"), int(finite.sum())
    g, _ = np.polyfit(np.log10(1.0 + z[finite]), np.log10(rho_L[finite]), 1)
    return float(g), int(finite.sum())


def _train_permuted_classifier(
    astrodeep_feats: pd.DataFrame, features: list[str], rng: np.random.Generator
):
    """Fit logistic regression with shuffled has_spec labels."""
    target = pd.to_numeric(astrodeep_feats["has_spec"], errors="coerce").fillna(0).astype(int).to_numpy()
    permuted = rng.permutation(target)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    model.fit(astrodeep_feats[features], permuted)
    return model


def _topk_select(probs: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-probs)
    sel = np.zeros(len(probs), dtype=bool)
    sel[order[: max(0, min(k, len(probs)))]] = True
    return sel


def main() -> None:
    print(f"[load] ASTRODEEP measured catalog -> {ASTRODEEP_CAT}")
    astrodeep = pd.read_csv(ASTRODEEP_CAT, low_memory=False)
    print(f"  rows={len(astrodeep)}  has_spec={astrodeep['has_spec'].sum()}")
    astrodeep_feats = add_spec_features(astrodeep)
    feats = available_features(astrodeep_feats, list(DEFAULT_FEATURES))
    print(f"[features] using {feats}")

    print(f"[load] TNG measured catalog -> {TNG_CAT}")
    tng = pd.read_csv(TNG_CAT, low_memory=False)
    # Image-aperture-success pool (the population the canonical selector picks K=2901 from):
    pool_mask = (
        tng["image_success_F200W"].astype(bool)
        & np.isfinite(tng[f"flux_{FILTER_NAME}_uJy"])
        & (tng[f"flux_{FILTER_NAME}_uJy"] > 0)
        & (tng[f"aperture_radius_arcsec_{FILTER_NAME}"] > 0)
    )
    tng_pool = tng.loc[pool_mask].copy()
    print(f"  rows={len(tng)}  pool(image-aperture-success)={len(tng_pool)}")

    # Reference: the canonical selected sample's volume-controlled gamma
    canon_mask = pool_mask & tng["mock_spec_selected"].astype(bool)
    canon = tng.loc[canon_mask].copy()
    g_canon, n_canon = _gamma_rho_volume_controlled(canon)
    print(
        f"\n[reference] canonical mock-spec-selected: "
        f"n={n_canon}  gamma_rho_intrinsic_volume={g_canon:+.3f}"
    )

    # Add features once on the TNG side
    tng_pool_feats = add_spec_features(tng_pool)

    rng = np.random.default_rng(20260509)

    rows = []
    for seed in range(N_SEEDS):
        seed_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))

        # ----- (b1) permuted-label classifier null -----
        model_perm = _train_permuted_classifier(astrodeep_feats, feats, seed_rng)
        probs_perm = model_perm.predict_proba(tng_pool_feats[feats])[:, 1]
        sel_perm = _topk_select(probs_perm, TARGET_N)
        df_perm = tng_pool.iloc[sel_perm].copy()
        g_perm, n_perm = _gamma_rho_volume_controlled(df_perm)

        # ----- (b2) uniform-random null -----
        idx = seed_rng.choice(len(tng_pool), size=min(TARGET_N, len(tng_pool)), replace=False)
        df_unif = tng_pool.iloc[idx].copy()
        g_unif, n_unif = _gamma_rho_volume_controlled(df_unif)

        rows.append(dict(
            seed=seed, n_perm=n_perm, gamma_perm=g_perm,
            n_unif=n_unif, gamma_unif=g_unif,
        ))
        if seed % 5 == 0:
            print(
                f"  seed={seed:3d}  "
                f"gamma_perm={g_perm:+.3f}  gamma_unif={g_unif:+.3f}"
            )

    table = pd.DataFrame(rows)
    out_dir = ROOT / "results" / "li_ddr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_seeds = out_dir / "tng_gamma_spec_selection_nulls.csv"
    table.to_csv(out_seeds, index=False)

    summary = pd.DataFrame([
        dict(
            null_kind="permuted_label_classifier",
            gamma_mean=float(table["gamma_perm"].mean()),
            gamma_std=float(table["gamma_perm"].std(ddof=1)),
            gamma_p16=float(np.percentile(table["gamma_perm"], 16)),
            gamma_p84=float(np.percentile(table["gamma_perm"], 84)),
            n_seeds=N_SEEDS,
        ),
        dict(
            null_kind="uniform_random_subsample",
            gamma_mean=float(table["gamma_unif"].mean()),
            gamma_std=float(table["gamma_unif"].std(ddof=1)),
            gamma_p16=float(np.percentile(table["gamma_unif"], 16)),
            gamma_p84=float(np.percentile(table["gamma_unif"], 84)),
            n_seeds=N_SEEDS,
        ),
        dict(
            null_kind="canonical_reference",
            gamma_mean=g_canon,
            gamma_std=float("nan"),
            gamma_p16=float("nan"),
            gamma_p84=float("nan"),
            n_seeds=1,
        ),
    ])
    out_summary = out_dir / "tng_gamma_spec_selection_nulls_summary.csv"
    summary.to_csv(out_summary, index=False)

    print(f"\nWrote -> {out_seeds}")
    print(f"Wrote -> {out_summary}")
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    print(
        f"\nInterpretation: gamma_null >= 1 would indicate Malmquist-style truncation drives "
        f"the fiducial gamma; gamma_null <= 0.3 says it does not. "
        f"Canonical reference gamma = {g_canon:+.3f}."
    )


if __name__ == "__main__":
    main()

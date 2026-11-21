#!/usr/bin/env python3
"""Spatial sub-block jackknife on the canonical xyz TNG100 lightcone.

Tile the 137 arcmin^2 footprint into n_tiles_ra x n_tiles_dec sub-fields by
the ra_deg, dec_deg columns of the measured catalogue. For each sub-block,
recompute the volume-controlled gamma_rho on the canonical (image-aperture)
spec-selected sample. The std across sub-blocks is the within-lightcone
component of cosmic variance -- it captures clustering at scales smaller
than the tile size but does NOT capture super-tile modes. It is therefore
a lower bound on the true cosmic-variance contribution.

We run a 2x2 (4-tile) split, a 3x3 (9-tile) split, and a 4x4 (16-tile)
split, and report the std at each scale plus a Poisson-resampling
sanity check.

Outputs:
  results/li_ddr/cosmic_variance/xyz_subblocks.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from tolman.li_ddr.li_reproduction import COSMO_LI, luminosity_at_band

ROOT = Path(os.environ.get("TOLMAN_WORKDIR", Path(__file__).resolve().parents[1]))
FILTER_NAME = "F200W"
ALPHA_SPEC = -0.3  # Li convention; equivalent to paper alpha_SED = 0.3.
CANONICAL_CAT = ROOT / "results/tng100_7_6_xyz_hst_jwst_imageap_specmodel/measured_catalog.csv"


def _spec_select(df: pd.DataFrame) -> pd.DataFrame:
    keep = df["mock_spec_selected"].astype(bool)
    if "image_success_F200W" in df.columns:
        keep &= df["image_success_F200W"].astype(bool)
    keep &= df[f"flux_{FILTER_NAME}_uJy"] > 0
    keep &= df[f"aperture_radius_arcsec_{FILTER_NAME}"] > 0
    return df.loc[keep].copy()


def _gamma(z: np.ndarray, log_rho: np.ndarray) -> float:
    finite = np.isfinite(z) & np.isfinite(log_rho) & (z > 0)
    if finite.sum() < 30:
        return float("nan")
    g, _ = np.polyfit(np.log10(1.0 + z[finite]), log_rho[finite], 1)
    return float(g)


def _gamma_for_subset(df_sub: pd.DataFrame) -> tuple[float, int]:
    z = df_sub["z"].to_numpy(dtype=float)
    flux_jy = df_sub[f"flux_{FILTER_NAME}_uJy"].to_numpy(dtype=float) * 1e-6
    photrad_kpc = (df_sub[f"aperture_radius_arcsec_{FILTER_NAME}"].to_numpy(dtype=float)
                   * df_sub["kpc_per_arcsec"].to_numpy(dtype=float))
    L = luminosity_at_band(flux_jy, z, np.full_like(z, ALPHA_SPEC), cosmo=COSMO_LI)
    r_cm = photrad_kpc * 3.0857e21
    rho_L = L / ((4.0 / 3.0) * np.pi * r_cm**3)
    finite = np.isfinite(rho_L) & (rho_L > 0) & np.isfinite(z) & (z > 0)
    return _gamma(z[finite], np.log10(rho_L[finite])), int(finite.sum())


def main() -> None:
    print(f"[load] {CANONICAL_CAT}")
    df = pd.read_csv(CANONICAL_CAT, low_memory=False)
    df = _spec_select(df)
    print(f"  selected={len(df)}; ra range={df['ra_deg'].min():.4f}--{df['ra_deg'].max():.4f}, "
          f"dec range={df['dec_deg'].min():.4f}--{df['dec_deg'].max():.4f}")

    g_full, n_full = _gamma_for_subset(df)
    print(f"\n[reference] full canonical xyz: n={n_full}  gamma={g_full:+.3f}")

    rows: list[dict] = []
    rows.append(dict(scheme="full", n_tiles=1, tile_index=0, n=n_full, gamma=g_full,
                     alpha_spec_li=ALPHA_SPEC, alpha_sed_paper=-ALPHA_SPEC))

    rng = np.random.default_rng(20260509)

    for n_per_side in (2, 3, 4):
        ra_edges = np.linspace(df["ra_deg"].min(), df["ra_deg"].max() + 1e-9, n_per_side + 1)
        dec_edges = np.linspace(df["dec_deg"].min(), df["dec_deg"].max() + 1e-9, n_per_side + 1)
        ra_idx = np.digitize(df["ra_deg"].to_numpy(), ra_edges) - 1
        dec_idx = np.digitize(df["dec_deg"].to_numpy(), dec_edges) - 1
        ra_idx = np.clip(ra_idx, 0, n_per_side - 1)
        dec_idx = np.clip(dec_idx, 0, n_per_side - 1)
        tile_id = ra_idx * n_per_side + dec_idx
        gammas = []
        ns = []
        for t in range(n_per_side * n_per_side):
            mask = tile_id == t
            if mask.sum() < 30:
                continue
            g, n = _gamma_for_subset(df.iloc[mask.nonzero()[0]])
            if not np.isfinite(g):
                continue
            gammas.append(g); ns.append(n)
            rows.append(dict(scheme=f"{n_per_side}x{n_per_side}", n_tiles=n_per_side**2,
                             tile_index=int(t), n=n, gamma=float(g),
                             alpha_spec_li=ALPHA_SPEC, alpha_sed_paper=-ALPHA_SPEC))
        gammas = np.asarray(gammas)
        if len(gammas) >= 2:
            print(f"\n[scheme {n_per_side}x{n_per_side}] non-empty tiles={len(gammas)} "
                  f"gamma=[{gammas.min():+.3f}, {gammas.max():+.3f}]  "
                  f"mean={gammas.mean():+.3f}  std={gammas.std(ddof=1):.3f}  "
                  f"sem={gammas.std(ddof=1)/np.sqrt(len(gammas)):.3f}")

    # Poisson-resampling sanity check: if gamma scatter were purely shot noise
    # (Poisson on object counts in z bins), how big would the spread be?
    n_resamp = 200
    n_per_block = int(n_full / 16)  # 16-tile equivalent
    boots = []
    for _ in range(n_resamp):
        idx = rng.choice(len(df), size=n_per_block, replace=False)
        g, _ = _gamma_for_subset(df.iloc[idx])
        if np.isfinite(g):
            boots.append(g)
    boots = np.asarray(boots)
    print(f"\n[poisson-equivalent] random-{n_per_block}-subsample over {len(boots)} resamples: "
          f"mean={boots.mean():+.3f}  std={boots.std(ddof=1):.3f}")

    table = pd.DataFrame(rows)
    out = ROOT / "results/li_ddr/cosmic_variance/xyz_subblocks.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nWrote -> {out}")

    summary = []
    for scheme, sub in table.groupby("scheme"):
        g = sub["gamma"].dropna().to_numpy()
        if len(g) < 2:
            continue
        summary.append(dict(scheme=scheme, n_tiles=int(sub["n_tiles"].iloc[0]),
                            n_used=len(g), gamma_mean=float(g.mean()),
                            gamma_std=float(g.std(ddof=1)),
                            gamma_sem=float(g.std(ddof=1) / np.sqrt(len(g))),
                            alpha_spec_li=ALPHA_SPEC, alpha_sed_paper=-ALPHA_SPEC))
    summary.append(dict(scheme="poisson_equivalent_random", n_tiles=16,
                        n_used=len(boots), gamma_mean=float(boots.mean()),
                        gamma_std=float(boots.std(ddof=1)),
                        gamma_sem=float(boots.std(ddof=1)/np.sqrt(len(boots))),
                        alpha_spec_li=ALPHA_SPEC, alpha_sed_paper=-ALPHA_SPEC))
    pd.DataFrame(summary).to_csv(
        ROOT / "results/li_ddr/cosmic_variance/xyz_subblocks_summary.csv", index=False)


if __name__ == "__main__":
    main()

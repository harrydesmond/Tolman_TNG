#!/usr/bin/env python3
"""SB normalisation across TNG aperture variants vs ASTRODEEP.

Computes median F200W SB in three redshift bins (z<0.5, 0.5<=z<1.5, 1.5<=z<3)
for each TNG variant + ASTRODEEP, then reports the TNG/ASTRODEEP ratio
per bin per variant.  Shows how much of the absolute SB offset survives
into each z bin -- if the ratio is z-independent, the systematic is just
a normalisation; if it grows with z, it eats into the slope as well.

Expects the TNG and ASTRODEEP measured catalogues to be present under
``$TOLMAN_WORKDIR`` (default: repository root); run from the repo with
``PYTHONPATH=.``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("TOLMAN_WORKDIR", Path(__file__).resolve().parents[1]))
F = "F200W"
Z_BINS = [(0.0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 7.0)]

CANONICAL_CAT = ROOT / "results/tng100_7_6_xyz_hst_jwst_imageap_specmodel/measured_catalog.csv"
CATALOGUE_CAT = ROOT / "results/tng100_7_6_xyz_hst_jwst_specmodel/measured_catalog.csv"
DEBLENDED_PER_OBJ = ROOT / "results/systematics/deblend_per_object_f200w.csv"

AD_CAT = ROOT / "data/astrodeep_cds_all.csv"
AD_MEASURED = ROOT / "results/astrodeep_cds/measured_catalog.csv"


def _bin_medians(z: np.ndarray, sb: np.ndarray) -> dict[str, float]:
    out = {}
    for lo, hi in Z_BINS:
        mask = (z >= lo) & (z < hi) & np.isfinite(sb) & (sb > 0)
        out[f"sb_z{lo}-{hi}"] = float(np.median(sb[mask])) if mask.sum() > 0 else np.nan
        out[f"n_z{lo}-{hi}"] = int(mask.sum())
    return out


def _astrodeep() -> dict:
    df = pd.read_csv(AD_MEASURED, low_memory=False)
    snr_cols = [c for c in df.columns if c.startswith("snr_")]
    max_snr = df[snr_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
    keep = (
        pd.to_numeric(df["z"], errors="coerce").between(0.0, 14.0)
        & pd.to_numeric(df["flags"], errors="coerce").lt(80)
        & pd.to_numeric(df["aperture_area_arcsec2"], errors="coerce").gt(0)
        & pd.to_numeric(df["n_valid_filters"], errors="coerce").gt(0)
        & df["has_spec"].fillna(False).astype(bool)
        & max_snr.ge(5.0)
    )
    sub = df.loc[keep]
    # Per-filter SB columns aren't always present; compute from flux/area.
    flux = pd.to_numeric(sub[f"flux_{F}_uJy"], errors="coerce").to_numpy(dtype=float)
    area = pd.to_numeric(sub["aperture_area_arcsec2"], errors="coerce").to_numpy(dtype=float)
    sb = flux / area
    return {"n_total": int(len(sub)), **_bin_medians(sub["z"].to_numpy(), sb)}


def _spec_keep(df: pd.DataFrame) -> pd.DataFrame:
    keep = df["mock_spec_selected"].astype(bool)
    if "image_success_F200W" in df.columns:
        keep &= df["image_success_F200W"].astype(bool)
    keep &= df[f"flux_{F}_uJy"] > 0
    keep &= df[f"aperture_radius_arcsec_{F}"] > 0
    return df.loc[keep].copy()


def _from_canonical(flux_col_override: str | None = None) -> dict:
    df = pd.read_csv(CANONICAL_CAT, low_memory=False)
    sub = _spec_keep(df)
    flux = sub[flux_col_override or f"flux_{F}_uJy"].to_numpy(dtype=float)
    area = sub[f"aperture_area_arcsec2_{F}"].to_numpy(dtype=float)
    sb = flux / area
    return _bin_medians(sub["z"].to_numpy(), sb) | {"n_total": int(len(sub))}


def _deblended() -> dict:
    """deblend_per_object_f200w.csv `object_index` matches the FULL canonical
    catalogue's positional row index (not the spec-selected subset)."""
    canon = pd.read_csv(CANONICAL_CAT, low_memory=False)  # keep full DataFrame
    deb = pd.read_csv(DEBLENDED_PER_OBJ).set_index("object_index")
    canon["voronoi_flux_uJy"] = canon.index.map(deb["voronoi_flux_uJy"])
    sub = _spec_keep(canon)
    flux = sub["voronoi_flux_uJy"].to_numpy(dtype=float)
    area = sub[f"aperture_area_arcsec2_{F}"].to_numpy(dtype=float)
    sb = flux / area
    return _bin_medians(sub["z"].to_numpy(), sb) | {"n_total": int(np.isfinite(flux).sum())}


def _adaptive() -> dict:
    """Compute F200W flux at ASTRODEEP-equivalent z-dependent radii."""
    from astropy.io import fits
    from tolman.adaptive_aperture import astrodeep_aperture_radius
    from tolman.image_photometry import circular_aperture_sum, image_unit_to_ujy_factor
    from tolman.tng_image_aperture import _pixel_scale_arcsec

    fits_dir = ROOT / "data/raw/illustris/tng100-7-6-xyz_137sqarcmin"
    paths = sorted(list(fits_dir.glob("*F200W*.fits")) + list(fits_dir.glob("*f200w*.fits")))
    if not paths:
        return {"n_total": 0, "error": "no F200W FITS"}
    fits_path = paths[0]

    canon = _spec_keep(pd.read_csv(CANONICAL_CAT, low_memory=False))
    with fits.open(fits_path, memmap=True) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header
        ny, nx = image.shape
        pixel_scale = _pixel_scale_arcsec(header, None)
        unit_factor = image_unit_to_ujy_factor(header.get("BUNIT"))

    z = canon["z"].to_numpy(dtype=float)
    x = canon["x_F200W_pix"].to_numpy(dtype=float)
    y = canon["y_F200W_pix"].to_numpy(dtype=float)
    r_arcsec = astrodeep_aperture_radius(z)
    r_pix = r_arcsec / pixel_scale
    flux = np.array(
        [circular_aperture_sum(image, xi, yi, ri) for xi, yi, ri in zip(x, y, r_pix)],
        dtype=float,
    ) * unit_factor
    edge = (x - r_pix < 0) | (y - r_pix < 0) | (x + r_pix >= nx) | (y + r_pix >= ny)
    flux = np.where(np.isfinite(flux) & (flux > 0) & ~edge, flux, np.nan)
    area = np.pi * r_arcsec ** 2
    sb = flux / area
    return _bin_medians(z, sb) | {"n_total": int(np.isfinite(sb).sum())}


def main() -> None:
    # Catalogue-flux variant: read directly
    df = pd.read_csv(CATALOGUE_CAT, low_memory=False)
    sub = _spec_keep(df)
    flux_cat = sub[f"flux_{F}_uJy"].to_numpy(dtype=float)
    area_cat = sub[f"aperture_area_arcsec2_{F}"].to_numpy(dtype=float)
    sb_cat = flux_cat / area_cat
    rows = [
        {"variant": "ASTRODEEP", **_astrodeep()},
        {"variant": "TNG specmodel (catalogue flux)",
         **_bin_medians(sub["z"].to_numpy(), sb_cat), "n_total": int(len(sub))},
        {"variant": "TNG imageap_specmodel (canonical)", **_from_canonical()},
        {"variant": "TNG imageap_deblended (Voronoi)", **_deblended()},
        {"variant": "TNG imageap_astrodeep_ap", **_adaptive()},
    ]

    table = pd.DataFrame(rows)
    out = ROOT / "results/li_ddr/sb_normalisation_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    # Pretty print
    print(f"\nMedian {F} SB [µJy/arcsec²] in z bins:\n")
    cols = [f"sb_z{lo}-{hi}" for (lo, hi) in Z_BINS]
    hdr = "{:38s}".format("variant") + "".join(f"  {c.replace('sb_z','z'):>18s}" for c in cols) + "  N"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = "{:38s}".format(r["variant"])
        for c in cols:
            v = r.get(c, np.nan)
            line += f"  {v:>18.3g}"
        line += f"  {r.get('n_total', 0)}"
        print(line)

    # TNG/ASTRODEEP ratio per z bin per variant
    ad = rows[0]
    print(f"\nTNG / ASTRODEEP ratio of median {F} SB:\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows[1:]:
        line = "{:38s}".format(r["variant"])
        for c in cols:
            v = r.get(c, np.nan)
            v_ad = ad.get(c, np.nan)
            ratio = v / v_ad if (np.isfinite(v) and np.isfinite(v_ad) and v_ad > 0) else np.nan
            line += f"  {ratio:>18.2f}"
        line += f"  {r.get('n_total', 0)}"
        print(line)
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()

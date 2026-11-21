"""Re-measure TNG image apertures with ASTRODEEP-equivalent z-dependent radii.

Uses the same pixel positions from the specmodel catalogue but replaces the
fixed physical-kpc aperture with the matched ASTRODEEP aperture radius at each
object's redshift. The ASTRODEEP ``apopt`` column values are diameters per
Merlin et al.\\ 2024 (the catalogue's aperture set is {0.20, 0.28, 0.33, 0.50,
0.66, 0.70, 1.32, 2.65, 5.30}\\,arcsec in diameter), so the radius matching
each median ``apopt`` z-bin is half the diameter:

    z < 2.0      → diameter 1.32  → radius 0.66 arcsec
    2 ≤ z < 3    → diameter 0.50  → radius 0.25 arcsec
    3 ≤ z < 4    → diameter 0.33  → radius 0.165 arcsec
    z ≥ 4        → diameter 0.28  → radius 0.14 arcsec

This quantifies how much of the TNG–ASTRODEEP SB slope gap is explained by the
different aperture conventions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from .image_photometry import circular_aperture_sum, image_unit_to_ujy_factor
from .io import write_table
from .mast_illustris import filter_name_from_path
from .tng_image_aperture import _pixel_scale_arcsec


# ASTRODEEP apopt median *radius* by z bin: half of the median apopt diameter
# (Merlin et al. 2024 -- the catalogue's apopt is a diameter, not a radius).
_ASTRODEEP_BREAKS = [(0.0, 2.0, 0.66), (2.0, 3.0, 0.25), (3.0, 4.0, 0.165), (4.0, np.inf, 0.14)]


def astrodeep_aperture_radius(z: np.ndarray) -> np.ndarray:
    """Return ASTRODEEP-equivalent aperture radius (arcsec) for each redshift.

    The ASTRODEEP ``apopt`` column values are diameters (Merlin et al.\\ 2024);
    this routine returns the matching *radius* (apopt/2) needed to size a
    circular aperture on a TNG mock image.
    """
    r = np.full_like(z, np.nan, dtype=float)
    for lo, hi, rad in _ASTRODEEP_BREAKS:
        mask = (z >= lo) & (z < hi)
        r[mask] = rad
    return r


def _patch_one_filter(
    df: pd.DataFrame,
    path_str: str,
    selected_col: str,
) -> pd.DataFrame | None:
    """Remeasure one filter with ASTRODEEP-equivalent aperture radii."""
    path = Path(path_str)
    filter_name = filter_name_from_path(path)

    x_col = f"x_{filter_name}_pix"
    y_col = f"y_{filter_name}_pix"
    if x_col not in df.columns:
        return None

    with fits.open(path, memmap=True) as hdul:
        image = np.array(hdul[0].data, dtype=np.float32, copy=True)
        header = hdul[0].header
        ny, nx = image.shape
        pixel_scale = _pixel_scale_arcsec(header, None)
        unit_factor = image_unit_to_ujy_factor(header.get("BUNIT"))

    selected = df[df[selected_col].astype(bool)].copy()
    x = pd.to_numeric(selected[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(selected[y_col], errors="coerce").to_numpy(dtype=float)
    z_vals = pd.to_numeric(selected["z"], errors="coerce").to_numpy(dtype=float)

    r_arcsec = astrodeep_aperture_radius(z_vals)
    r_pix = r_arcsec / pixel_scale

    flux = np.array(
        [circular_aperture_sum(image, xi, yi, ri) for xi, yi, ri in zip(x, y, r_pix)],
        dtype=float,
    ) * unit_factor

    edge = (
        (x - r_pix < 0.0) | (y - r_pix < 0.0)
        | (x + r_pix >= nx) | (y + r_pix >= ny)
    )
    flux = np.where(np.isfinite(flux) & (flux > 0.0) & ~edge, flux, np.nan)

    out = selected[["z"]].copy()
    out.index = selected.index
    out[f"flux_{filter_name}_uJy"] = flux
    out[f"aperture_area_arcsec2_{filter_name}"] = np.pi * r_arcsec**2
    return out


def write_adaptive_aperture_sb(
    catalog: str | Path,
    fits_paths: list[str | Path],
    config: str | Path,
    *,
    out_dir: str | Path,
    plot_dir: str | Path,
    sample_name: str = "tng100_imageap_astrodeep_aperture",
    selected_col: str = "mock_spec_selected",
) -> pd.DataFrame:
    """Compute TNG SB slopes with ASTRODEEP-equivalent z-dependent apertures.

    Patches *catalog*'s ``flux_{filter}_uJy`` and ``aperture_area_arcsec2``
    columns with values measured at the ASTRODEEP apopt radius for each
    object's redshift, then runs the standard SB pipeline.

    Returns the slope table DataFrame.
    """
    from .filters import schema_from_toml
    from .io import read_catalog
    from .pipeline import run_catalog_pipeline

    df = read_catalog(catalog)
    schema = schema_from_toml(config)
    out_dir = Path(out_dir)
    plot_dir = Path(plot_dir)

    area_columns: list[str] = []
    for path in fits_paths:
        patch = _patch_one_filter(df, str(path), selected_col)
        if patch is None:
            continue
        filter_name = filter_name_from_path(Path(path))
        flux_col = f"flux_{filter_name}_uJy"
        area_col = f"aperture_area_arcsec2_{filter_name}"
        area_columns.append(area_col)
        # Only overwrite for selected objects; leave non-selected as-is
        df.loc[patch.index, flux_col] = patch[flux_col].values
        df.loc[patch.index, area_col] = patch[area_col].values

    if area_columns:
        sel_mask = df[selected_col].astype(bool)
        df.loc[sel_mask, "aperture_area_arcsec2"] = (
            df.loc[sel_mask, area_columns].median(axis=1, skipna=True)
        )

    _, slope_table = run_catalog_pipeline(
        df,
        schema,
        result_dir=out_dir,
        plot_dir=plot_dir,
        sample_name=sample_name,
    )
    write_table(df[df[selected_col].astype(bool)][["z", "aperture_area_arcsec2"]].assign(
        aperture_r_arcsec=lambda d: np.sqrt(d["aperture_area_arcsec2"] / np.pi)
    ), out_dir / "adaptive_aperture_radii.csv")
    return slope_table

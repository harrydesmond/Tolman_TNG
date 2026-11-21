"""Small FITS aperture-photometry helpers.

This is narrower than a full source-detection pipeline: it re-measures fluxes
at known mock-catalogue positions and feeds them into the same Tolman
measurements.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


def circular_aperture_sum(image: np.ndarray, x: float, y: float, radius_pix: float) -> float:
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(radius_pix) or radius_pix <= 0:
        return np.nan

    ny, nx = image.shape
    x0 = max(int(np.floor(x - radius_pix)), 0)
    x1 = min(int(np.ceil(x + radius_pix)) + 1, nx)
    y0 = max(int(np.floor(y - radius_pix)), 0)
    y1 = min(int(np.ceil(y + radius_pix)) + 1, ny)
    if x0 >= x1 or y0 >= y1:
        return np.nan

    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= radius_pix**2
    if not np.any(mask):
        return np.nan
    values = np.asarray(image[y0:y1, x0:x1], dtype=float)
    return float(np.nansum(values[mask]))


def image_unit_to_ujy_factor(unit: str | None) -> float:
    """Return the multiplicative factor converting image pixels to microJy."""

    if unit is None:
        return 1.0
    compact = str(unit).strip().lower().replace(" ", "")
    if compact in {"ujy", "microjy", "microjanskies", "microjansky"}:
        return 1.0
    if compact in {"njy", "nanojy", "nanojanskies", "nanojansky"}:
        return 1.0e-3
    if compact in {"jy", "janskies", "jansky"}:
        return 1.0e6
    raise ValueError(f"Unsupported image flux unit {unit!r}; pass a converted image or add a converter")


def world_to_pixel_positions(header, ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert ICRS sky positions to FITS zero-based pixel coordinates."""

    wcs = WCS(header)
    x, y = wcs.all_world2pix(np.asarray(ra_deg, dtype=float), np.asarray(dec_deg, dtype=float), 0)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def add_circular_aperture_fluxes(
    catalog: pd.DataFrame,
    *,
    image_paths: dict[str, str | Path],
    x_col: str,
    y_col: str,
    radius_pix_col: str,
    pixel_scale_arcsec: float,
    flux_prefix: str = "flux",
    flux_suffix: str = "uJy",
) -> pd.DataFrame:
    """Add per-filter aperture fluxes from FITS images.

    The image pixel values are assumed to be in flux per pixel, for example
    microJy/pixel. If the mock image uses a different calibration, convert it
    before this step or adjust the resulting columns consistently.
    """

    out = catalog.copy()
    x = pd.to_numeric(out[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(out[y_col], errors="coerce").to_numpy(dtype=float)
    radius = pd.to_numeric(out[radius_pix_col], errors="coerce").to_numpy(dtype=float)
    out["aperture_area_arcsec2"] = np.pi * (radius * pixel_scale_arcsec) ** 2

    for filter_name, path in image_paths.items():
        with fits.open(path, memmap=True) as hdul:
            image = np.asarray(hdul[0].data, dtype=float)
        fluxes = [
            circular_aperture_sum(image, xi, yi, ri)
            for xi, yi, ri in zip(x, y, radius, strict=True)
        ]
        out[f"{flux_prefix}_{filter_name}_{flux_suffix}"] = fluxes
    return out


def add_wcs_circular_aperture_fluxes(
    catalog: pd.DataFrame,
    *,
    image_paths: dict[str, str | Path],
    ra_col: str = "ra_deg",
    dec_col: str = "dec_deg",
    radius_arcsec_col: str | None = None,
    filter_radius_col_template: str = "aperture_radius_arcsec_{filter}",
    output_flux_col_template: str = "flux_{filter}_uJy",
    replace_existing_flux: bool = True,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Add aperture fluxes from WCS-calibrated FITS images.

    This is intended for the MAST ILLUSTRIS/TNG mock images. It measures the
    same sky-position circular aperture in the image plane and converts common
    FITS flux units to microJy.
    """

    out = catalog.copy()
    if max_rows is not None:
        out = out.iloc[:max_rows].copy()
    if ra_col not in out or dec_col not in out:
        raise KeyError(f"Catalogue must contain {ra_col!r} and {dec_col!r}")

    ra = pd.to_numeric(out[ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(out[dec_col], errors="coerce").to_numpy(dtype=float)
    area_columns: list[str] = []

    for filter_name, path in image_paths.items():
        with fits.open(path, memmap=True) as hdul:
            image = np.asarray(hdul[0].data, dtype=float)
            header = hdul[0].header
            x, y = world_to_pixel_positions(header, ra, dec)
            pixel_scale = float(header.get("PIXSCALE", np.nan))
            if not np.isfinite(pixel_scale) or pixel_scale <= 0.0:
                cdelt = abs(float(header.get("PC1_1", header.get("CDELT1", np.nan)))) * 3600.0
                pixel_scale = cdelt
            unit_factor = image_unit_to_ujy_factor(header.get("BUNIT"))

        if not np.isfinite(pixel_scale) or pixel_scale <= 0.0:
            raise ValueError(f"Could not infer positive pixel scale for {path}")

        filter_radius_col = filter_radius_col_template.format(filter=filter_name)
        if filter_radius_col in out:
            radius_arcsec = pd.to_numeric(out[filter_radius_col], errors="coerce").to_numpy(dtype=float)
        elif radius_arcsec_col and radius_arcsec_col in out:
            radius_arcsec = pd.to_numeric(out[radius_arcsec_col], errors="coerce").to_numpy(dtype=float)
        else:
            raise KeyError(
                f"Missing aperture radius column {filter_radius_col!r}"
                + (f" or {radius_arcsec_col!r}" if radius_arcsec_col else "")
            )
        radius_pix = radius_arcsec / pixel_scale
        fluxes = np.array(
            [circular_aperture_sum(image, xi, yi, ri) for xi, yi, ri in zip(x, y, radius_pix, strict=True)],
            dtype=float,
        )
        fluxes *= unit_factor
        out[f"x_{filter_name}_pix"] = x
        out[f"y_{filter_name}_pix"] = y
        out[f"image_aperture_radius_pix_{filter_name}"] = radius_pix
        out[f"image_aperture_area_arcsec2_{filter_name}"] = np.pi * radius_arcsec * radius_arcsec
        area_columns.append(f"image_aperture_area_arcsec2_{filter_name}")

        flux_col = output_flux_col_template.format(filter=filter_name)
        if replace_existing_flux or flux_col not in out:
            out[flux_col] = fluxes
        else:
            out[f"image_{flux_col}"] = fluxes

    if area_columns:
        out["aperture_area_arcsec2"] = out[area_columns].median(axis=1, skipna=True)
    return out

"""Image-aperture photometry for MAST ILLUSTRIS/TNG mock FITS products."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from .image_photometry import circular_aperture_sum, image_unit_to_ujy_factor, world_to_pixel_positions
from .mast_illustris import _base_frame, _output_table, filter_name_from_path, write_schema_for_normalized_catalog


def _pixel_scale_arcsec(header, override: float | None) -> float:
    if override is not None:
        return float(override)
    value = header.get("PIXSCALE")
    if value is not None:
        return float(value)
    pc = header.get("PC1_1")
    cdelt = header.get("CDELT1", 1.0)
    if pc is not None:
        return abs(float(pc) * float(cdelt)) * 3600.0
    if cdelt is not None:
        return abs(float(cdelt)) * 3600.0
    raise ValueError("Could not infer pixel scale from FITS header")


def _linear_tng_pixels(
    *,
    nx: int,
    ny: int,
    pixel_scale_arcsec: float,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """TNG mock images are centred on RA=DEC=0 with small-angle coordinates."""

    x = 0.5 * nx + np.asarray(ra_deg, dtype=float) * 3600.0 / pixel_scale_arcsec
    y = 0.5 * ny + np.asarray(dec_deg, dtype=float) * 3600.0 / pixel_scale_arcsec
    return x, y


def _measure_one_filter(
    item: tuple[str, str, int | None, float | None, str],
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    path_str, sample_name, max_rows, pixel_scale_override, coordinate_mode = item
    path = Path(path_str)
    filter_name = filter_name_from_path(path)
    table = _output_table(path)
    if max_rows is not None:
        table = table.iloc[:max_rows].copy()
    base = _base_frame(table)

    with fits.open(path, memmap=True) as hdul:
        image = np.array(hdul[0].data, dtype=np.float32, copy=True)
        header = hdul[0].header
        ny, nx = image.shape
        pixel_scale = _pixel_scale_arcsec(header, pixel_scale_override)
        unit_factor = image_unit_to_ujy_factor(header.get("BUNIT"))
        if coordinate_mode == "wcs":
            x, y = world_to_pixel_positions(header, base["ra_deg"].to_numpy(), base["dec_deg"].to_numpy())
        elif coordinate_mode == "linear":
            x, y = _linear_tng_pixels(
                nx=nx,
                ny=ny,
                pixel_scale_arcsec=pixel_scale,
                ra_deg=base["ra_deg"].to_numpy(),
                dec_deg=base["dec_deg"].to_numpy(),
            )
        else:
            raise ValueError("coordinate_mode must be 'wcs' or 'linear'")

        radius_arcsec = base["aperture_radius_arcsec"].to_numpy(dtype=float)
        radius_pix = radius_arcsec / pixel_scale
        image_flux_ujy = np.array(
            [circular_aperture_sum(image, xi, yi, ri) for xi, yi, ri in zip(x, y, radius_pix, strict=True)],
            dtype=float,
        )
        image_flux_ujy *= unit_factor
        image_sum_ujy = float(np.nansum(image) * unit_factor)

    success = pd.to_numeric(table["image_success"], errors="coerce").fillna(0).astype(bool).to_numpy()
    catalog_flux_ujy = pd.to_numeric(table["total_quant"], errors="coerce").to_numpy(dtype=float) / 1000.0
    edge = (
        (x - radius_pix < 0.0)
        | (y - radius_pix < 0.0)
        | (x + radius_pix >= nx)
        | (y + radius_pix >= ny)
    )
    image_flux_ujy = np.where(success & np.isfinite(image_flux_ujy) & (image_flux_ujy > 0.0), image_flux_ujy, np.nan)
    ratio = image_flux_ujy / catalog_flux_ujy
    finite_ratio = np.isfinite(ratio) & (catalog_flux_ujy > 0.0)

    per_filter = base[
        [
            "row_id",
            "object_key",
            "snapshot",
            "subhalo_index",
            "lightcone_number",
            "ra_deg",
            "dec_deg",
            "z",
            "z_inferred",
            "kpc_per_arcsec",
            "mstar_msun_2rhalf",
            "sfr_msun_per_yr_2rhalf",
            "app_gmag_rest",
            "primary_flag",
            "aperture_radius_arcsec",
            "aperture_area_arcsec2",
        ]
    ].copy()
    per_filter[f"flux_{filter_name}_uJy"] = image_flux_ujy
    per_filter[f"catalog_flux_{filter_name}_uJy"] = catalog_flux_ujy
    per_filter[f"image_success_{filter_name}"] = success
    per_filter[f"image_edge_{filter_name}"] = edge
    per_filter[f"x_{filter_name}_pix"] = x
    per_filter[f"y_{filter_name}_pix"] = y
    per_filter[f"aperture_radius_pix_{filter_name}"] = radius_pix
    per_filter = per_filter.rename(
        columns={
            "aperture_radius_arcsec": f"aperture_radius_arcsec_{filter_name}",
            "aperture_area_arcsec2": f"aperture_area_arcsec2_{filter_name}",
        }
    )

    diag: dict[str, float | int | str] = {
        "filter": filter_name,
        "path": str(path),
        "n_rows": int(len(per_filter)),
        "n_success": int(np.sum(success)),
        "n_positive_image_flux": int(np.sum(np.isfinite(image_flux_ujy) & (image_flux_ujy > 0.0))),
        "n_edge": int(np.sum(edge)),
        "pixel_scale_arcsec": float(pixel_scale),
        "image_sum_uJy": image_sum_ujy,
        "catalog_success_flux_sum_uJy": float(np.nansum(np.where(success, catalog_flux_ujy, np.nan))),
        "aperture_flux_sum_uJy": float(np.nansum(image_flux_ujy)),
        "median_aperture_to_catalog_flux": float(np.nanmedian(ratio[finite_ratio])) if np.any(finite_ratio) else np.nan,
        "p16_aperture_to_catalog_flux": float(np.nanpercentile(ratio[finite_ratio], 16.0)) if np.any(finite_ratio) else np.nan,
        "p84_aperture_to_catalog_flux": float(np.nanpercentile(ratio[finite_ratio], 84.0)) if np.any(finite_ratio) else np.nan,
        "coordinate_mode": coordinate_mode,
    }
    return per_filter, diag


def normalize_tng_image_aperture_fits(
    paths: list[str | Path],
    *,
    sample_name: str,
    max_rows: int | None = None,
    pixel_scale_arcsec: float | None = None,
    coordinate_mode: str = "linear",
    n_workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure image-aperture fluxes and merge per-filter TNG products."""

    if not paths:
        raise ValueError("At least one FITS path is required")
    items = [(str(path), sample_name, max_rows, pixel_scale_arcsec, coordinate_mode) for path in paths]
    if n_workers > 1 and len(items) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            measured = list(pool.map(_measure_one_filter, items))
    else:
        measured = [_measure_one_filter(item) for item in items]

    merged: pd.DataFrame | None = None
    diagnostics: list[dict[str, float | int | str]] = []
    area_columns: list[str] = []
    edge_columns: list[str] = []
    for per_filter, diag in measured:
        filter_name = str(diag["filter"])
        diagnostics.append(diag)
        area_columns.append(f"aperture_area_arcsec2_{filter_name}")
        edge_columns.append(f"image_edge_{filter_name}")
        if merged is None:
            merged = per_filter
        else:
            keep = [
                "row_id",
                "object_key",
                f"flux_{filter_name}_uJy",
                f"catalog_flux_{filter_name}_uJy",
                f"image_success_{filter_name}",
                f"image_edge_{filter_name}",
                f"x_{filter_name}_pix",
                f"y_{filter_name}_pix",
                f"aperture_radius_pix_{filter_name}",
                f"aperture_radius_arcsec_{filter_name}",
                f"aperture_area_arcsec2_{filter_name}",
            ]
            merged = merged.merge(per_filter[keep], on=["row_id", "object_key"], how="outer")

    assert merged is not None
    merged["field"] = sample_name
    merged["id"] = np.arange(len(merged))
    merged["aperture_area_arcsec2"] = merged[area_columns].median(axis=1, skipna=True)
    success_cols = [col for col in merged if col.startswith("image_success_")]
    merged["any_image_success"] = merged[success_cols].any(axis=1)
    merged["is_edge"] = merged[edge_columns].any(axis=1)
    return merged, pd.DataFrame(diagnostics).sort_values("filter").reset_index(drop=True)


def write_tng_image_aperture_schema(filters: list[str], path: str | Path) -> None:
    write_schema_for_normalized_catalog(filters, path, edge_col="is_edge")

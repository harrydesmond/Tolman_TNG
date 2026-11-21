"""Normalize MAST ILLUSTRIS/TNG mock-survey FITS products."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from .filters import TNG_MAST_FILTERS


FILTER_RE = re.compile(r"_([a-z]\d{3}[a-z]?)_v\d+_sim\.fits$", re.IGNORECASE)


BASE_COLUMNS = {
    "Snapshot number": "snapshot",
    "Subhalo index": "subhalo_index",
    "RA degree": "ra_deg",
    "DEC degree": "dec_deg",
    "True z": "z",
    "Inferred z": "z_inferred",
    "True scale": "kpc_per_arcsec",
    "Stellar mass w2sr": "mstar_msun_2rhalf",
    "SFR w2sr": "sfr_msun_per_yr_2rhalf",
    "Apparent total gmag": "app_gmag_rest",
    "Lightcone number": "lightcone_number",
    "primary_flag": "primary_flag",
}


def filter_name_from_path(path: str | Path) -> str:
    match = FILTER_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Could not infer filter name from {path}")
    return match.group(1).upper()


def _output_table(path: str | Path) -> pd.DataFrame:
    with fits.open(path, memmap=True) as hdul:
        if "OutputData" not in hdul:
            raise ValueError(f"{path} has no OutputData HDU")
        table = Table(hdul["OutputData"].data).to_pandas()
    return table


def _base_frame(table: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in BASE_COLUMNS if col not in table]
    if missing:
        raise KeyError(f"OutputData table is missing required columns: {missing}")
    out = table[list(BASE_COLUMNS)].rename(columns=BASE_COLUMNS).copy()
    out["row_id"] = np.arange(len(out))
    out["object_key"] = (
        out["snapshot"].astype(str)
        + "_"
        + out["subhalo_index"].astype(str)
        + "_"
        + out["lightcone_number"].astype(str)
        + "_"
        + out["row_id"].astype(str)
    )
    out["aperture_radius_arcsec"] = pd.to_numeric(table["photrad_kpc"], errors="coerce") / pd.to_numeric(
        table["True scale"], errors="coerce"
    )
    out["aperture_area_arcsec2"] = np.pi * out["aperture_radius_arcsec"] ** 2
    return out


def normalize_tng_filter_fits(paths: list[str | Path], *, sample_name: str = "tng") -> pd.DataFrame:
    """Merge per-filter TNG FITS OutputData tables into pipeline-ready columns."""

    if not paths:
        raise ValueError("At least one FITS path is required")

    merged: pd.DataFrame | None = None
    area_columns: list[str] = []
    for path in paths:
        filter_name = filter_name_from_path(path)
        table = _output_table(path)
        base = _base_frame(table)
        flux_njy = pd.to_numeric(table["total_quant"], errors="coerce")
        success = pd.to_numeric(table["image_success"], errors="coerce").fillna(0).astype(bool)
        flux_ujy = (flux_njy / 1000.0).where(success & np.isfinite(flux_njy) & (flux_njy > 0.0))
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
        per_filter[f"flux_{filter_name}_uJy"] = flux_ujy
        per_filter[f"image_success_{filter_name}"] = success
        per_filter = per_filter.rename(
            columns={
                "aperture_radius_arcsec": f"aperture_radius_arcsec_{filter_name}",
                "aperture_area_arcsec2": f"aperture_area_arcsec2_{filter_name}",
            }
        )
        area_columns.append(f"aperture_area_arcsec2_{filter_name}")

        if merged is None:
            merged = per_filter
        else:
            keep = [
                "row_id",
                "object_key",
                f"flux_{filter_name}_uJy",
                f"image_success_{filter_name}",
                f"aperture_radius_arcsec_{filter_name}",
                f"aperture_area_arcsec2_{filter_name}",
            ]
            merged = merged.merge(per_filter[keep], on=["row_id", "object_key"], how="outer")

    assert merged is not None
    merged["field"] = sample_name
    merged["id"] = np.arange(len(merged))
    merged["aperture_area_arcsec2"] = merged[area_columns].median(axis=1, skipna=True)
    merged["any_image_success"] = merged[[col for col in merged if col.startswith("image_success_")]].any(axis=1)
    return merged


def write_schema_for_normalized_catalog(
    filters: list[str],
    path: str | Path,
    *,
    redshift_col: str = "z",
    spec_mag_filter: str | None = "F200W",
    spec_mag_limit: float = 29.0,
    edge_col: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "[catalog]",
        f'redshift_col = "{redshift_col}"',
        'area_col = "aperture_area_arcsec2"',
        'object_id_col = "id"',
        'field_col = "field"',
    ]
    if edge_col:
        lines.append(f'edge_col = "{edge_col}"')
    lines.extend(
        [
            "",
            "[selection]",
            "min_snr = 5",
            f'spec_mag_filter = "{spec_mag_filter if spec_mag_filter in filters else filters[0]}"',
            f"spec_mag_limit = {spec_mag_limit}",
            "z_min = 0.0",
            "z_max = 14.0",
            "",
        ]
    )
    for filter_name in filters:
        if filter_name not in TNG_MAST_FILTERS:
            raise KeyError(f"No effective wavelength known for {filter_name}")
        lines.extend(
            [
                "[[filters]]",
                f'name = "{filter_name}"',
                f"lambda_eff_angstrom = {TNG_MAST_FILTERS[filter_name]:.0f}",
                f'flux_col = "flux_{filter_name}_uJy"',
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")

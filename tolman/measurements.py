"""Surface-brightness measurements matching the Tolman-test paper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .filters import CatalogSchema


def aperture_area_arcsec2(df: pd.DataFrame, schema: CatalogSchema) -> pd.Series:
    if schema.area_col:
        if schema.area_col not in df:
            raise KeyError(f"Missing aperture area column: {schema.area_col}")
        area = pd.to_numeric(df[schema.area_col], errors="coerce")
    elif schema.radius_col:
        if schema.radius_col not in df:
            raise KeyError(f"Missing aperture radius column: {schema.radius_col}")
        if schema.radius_unit != "arcsec":
            raise ValueError(f"Unsupported radius_unit={schema.radius_unit!r}; use arcsec")
        radius = pd.to_numeric(df[schema.radius_col], errors="coerce")
        area = np.pi * radius * radius
    else:
        raise ValueError("Schema must define area_col or radius_col")
    return area.where(np.isfinite(area) & (area > 0.0))


def add_surface_brightness_columns(df: pd.DataFrame, schema: CatalogSchema) -> pd.DataFrame:
    out = df.copy()
    area = aperture_area_arcsec2(out, schema)
    out["aperture_area_arcsec2_used"] = area
    for spec in schema.filters:
        if spec.flux_col not in out:
            raise KeyError(f"Missing flux column for {spec.name}: {spec.flux_col}")
        flux = pd.to_numeric(out[spec.flux_col], errors="coerce")
        sb = flux / area
        out[spec.sb_col] = sb.where(np.isfinite(sb) & (sb > 0.0))
        if spec.flux_err_col and spec.flux_err_col in out:
            err = pd.to_numeric(out[spec.flux_err_col], errors="coerce")
            snr = flux / err
            out[f"snr_{spec.name}"] = snr.where(np.isfinite(snr) & (err > 0.0))
    return out


def add_summary_observables(df: pd.DataFrame, schema: CatalogSchema) -> pd.DataFrame:
    out = df.copy()
    sb = out[schema.sb_cols].to_numpy(dtype=float)
    counts = np.sum(np.isfinite(sb), axis=1)
    sums = np.nansum(sb, axis=1)
    max_values = np.max(np.where(np.isfinite(sb), sb, -np.inf), axis=1)
    out["n_valid_filters"] = counts
    out["sb_mean"] = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    out["sb_max"] = np.where(counts > 0, max_values, np.nan)
    return out


def add_rest_uv_observable(
    df: pd.DataFrame,
    schema: CatalogSchema,
    *,
    lambda_min_angstrom: float = 1500.0,
    lambda_max_angstrom: float = 2000.0,
    z_min: float = 3.0,
    z_max: float = 14.0,
    min_filters: int = 1,
) -> pd.DataFrame:
    out = df.copy()
    z = pd.to_numeric(out[schema.redshift_col], errors="coerce").to_numpy(dtype=float)
    values = []
    for spec in schema.filters:
        rest_lambda = spec.lambda_eff_angstrom / (1.0 + z)
        sb = pd.to_numeric(out[spec.sb_col], errors="coerce").to_numpy(dtype=float)
        use = (
            np.isfinite(rest_lambda)
            & (rest_lambda >= lambda_min_angstrom)
            & (rest_lambda <= lambda_max_angstrom)
            & np.isfinite(sb)
        )
        values.append(np.where(use, sb, np.nan))
    matrix = np.column_stack(values)
    counts = np.sum(np.isfinite(matrix), axis=1)
    sums = np.nansum(matrix, axis=1)
    rest_uv = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    in_z = np.isfinite(z) & (z > z_min) & (z < z_max) & (counts >= min_filters)
    out["sb_rest_uv_1500_2000"] = np.where(in_z, rest_uv, np.nan)
    out["n_rest_uv_filters"] = np.where(in_z, counts, 0)
    return out


def modal_typical_surface_brightness(
    z: np.ndarray,
    sb: np.ndarray,
    *,
    n_z_bins: int = 25,
    n_sb_bins: int = 25,
    z_min: float | None = None,
    z_max: float | None = None,
    min_count: int = 3,
) -> pd.DataFrame:
    z = np.asarray(z, dtype=float)
    sb = np.asarray(sb, dtype=float)
    mask = np.isfinite(z) & np.isfinite(sb) & (z > -1.0) & (sb > 0.0)
    if z_min is not None:
        mask &= z >= z_min
    if z_max is not None:
        mask &= z <= z_max
    if mask.sum() < min_count:
        return pd.DataFrame()

    x = np.log10(1.0 + z[mask])
    y = np.log10(sb[mask])
    if z_min is not None and z_max is not None:
        x_edges = np.linspace(np.log10(1.0 + z_min), np.log10(1.0 + z_max), n_z_bins + 1)
    else:
        x_edges = np.linspace(np.min(x), np.max(x), n_z_bins + 1)
    y_edges = np.linspace(np.min(y), np.max(y), n_sb_bins + 1)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    rows: list[dict[str, float | int]] = []

    for bin_idx in range(n_z_bins):
        if bin_idx == n_z_bins - 1:
            in_bin = (x >= x_edges[bin_idx]) & (x <= x_edges[bin_idx + 1])
        else:
            in_bin = (x >= x_edges[bin_idx]) & (x < x_edges[bin_idx + 1])
        n = int(np.sum(in_bin))
        if n < min_count:
            continue
        counts, _ = np.histogram(y[in_bin], bins=y_edges)
        mode_count = int(np.max(counts))
        if mode_count <= 0:
            continue
        near_mode = counts >= (mode_count - np.sqrt(mode_count))
        near_mode &= counts > 0
        if not np.any(near_mode):
            continue
        weights = counts[near_mode].astype(float)
        centers = y_centers[near_mode]
        log_sb = float(np.average(centers, weights=weights))
        low = float(np.min(centers))
        high = float(np.max(centers))
        rows.append(
            {
                "bin": bin_idx,
                "n": n,
                "n_mode": mode_count,
                "log1pz": float(0.5 * (x_edges[bin_idx] + x_edges[bin_idx + 1])),
                "z_center": float(10.0 ** (0.5 * (x_edges[bin_idx] + x_edges[bin_idx + 1])) - 1.0),
                "log_sb": log_sb,
                "sb": float(10.0**log_sb),
                "log_sb_err": float(max((high - low) / 2.0, 1.0e-6)),
                "log_sb_low": low,
                "log_sb_high": high,
            }
        )
    return pd.DataFrame(rows)


def add_all_observables(df: pd.DataFrame, schema: CatalogSchema) -> pd.DataFrame:
    out = add_surface_brightness_columns(df, schema)
    out = add_summary_observables(out, schema)
    return add_rest_uv_observable(out, schema)

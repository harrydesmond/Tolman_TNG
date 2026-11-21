"""Selection functions for the Tolman mock-observation ladder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .filters import CatalogSchema


@dataclass(frozen=True)
class SelectionStage:
    name: str
    description: str
    df: pd.DataFrame


def flux_ujy_to_abmag(flux_ujy: pd.Series | np.ndarray) -> np.ndarray:
    flux = np.asarray(flux_ujy, dtype=float)
    return np.where(flux > 0.0, 23.9 - 2.5 * np.log10(flux), np.nan)


def _finite_redshift_mask(df: pd.DataFrame, schema: CatalogSchema) -> pd.Series:
    z = pd.to_numeric(df[schema.redshift_col], errors="coerce")
    sel = schema.selection
    return np.isfinite(z) & (z >= sel.z_min) & (z <= sel.z_max)


def _flag_mask(df: pd.DataFrame, schema: CatalogSchema) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if schema.flags_col and schema.flags_col in df and schema.bad_flag_min is not None:
        flags = pd.to_numeric(df[schema.flags_col], errors="coerce")
        mask &= np.isfinite(flags) & (flags < schema.bad_flag_min)
    return mask


def _boolean_clean_mask(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col or col not in df:
        return pd.Series(True, index=df.index)
    values = df[col]
    if values.dtype == bool:
        return ~values
    numeric = pd.to_numeric(values, errors="coerce")
    return ~(numeric.fillna(0).astype(bool))


def _snr_mask(df: pd.DataFrame, schema: CatalogSchema) -> pd.Series:
    snr_cols = [f"snr_{spec.name}" for spec in schema.filters if f"snr_{spec.name}" in df]
    if not snr_cols or schema.selection.min_snr <= 0.0:
        return pd.Series(True, index=df.index)
    snr = df[snr_cols].to_numpy(dtype=float)
    return pd.Series(np.nanmax(snr, axis=1) >= schema.selection.min_snr, index=df.index)


def _mock_spec_mask(df: pd.DataFrame, schema: CatalogSchema) -> pd.Series:
    sel = schema.selection
    if sel.spec_selection_col and sel.spec_selection_col in df:
        values = df[sel.spec_selection_col]
        if values.dtype == bool:
            return values.fillna(False)
        numeric = pd.to_numeric(values, errors="coerce")
        return pd.Series(numeric.fillna(0).astype(bool), index=df.index)
    if "mock_spec_selected" in df:
        values = df["mock_spec_selected"]
        if values.dtype == bool:
            return values.fillna(False)
        numeric = pd.to_numeric(values, errors="coerce")
        return pd.Series(numeric.fillna(0).astype(bool), index=df.index)
    if not sel.spec_mag_filter or sel.spec_mag_limit is None:
        return pd.Series(True, index=df.index)
    spec = schema.filter_by_name.get(sel.spec_mag_filter)
    if spec is None or spec.flux_col not in df:
        return pd.Series(True, index=df.index)
    mag = flux_ujy_to_abmag(df[spec.flux_col])
    return pd.Series(np.isfinite(mag) & (mag <= sel.spec_mag_limit), index=df.index)


def _spec_stage_metadata(df: pd.DataFrame, schema: CatalogSchema) -> tuple[str, str]:
    sel = schema.selection
    if sel.spec_selection_col and sel.spec_selection_col in df:
        if sel.spec_selection_col == "has_spec":
            return "spectroscopic", "Objects with a positive spectroscopic redshift."
        return "mock_spectroscopic", "Configured mock spectroscopic-selection column."
    if "mock_spec_selected" in df or (sel.spec_mag_filter and sel.spec_mag_limit is not None):
        return "mock_spectroscopic", "Proxy for spectroscopic confirmation."
    return "spectroscopic_selection", "No configured spectroscopic selection; stage is a no-op."


def selection_ladder(df: pd.DataFrame, schema: CatalogSchema) -> list[SelectionStage]:
    stages: list[SelectionStage] = []

    intrinsic = df[_finite_redshift_mask(df, schema)].copy()
    stages.append(
        SelectionStage(
            "intrinsic_redshift_range",
            "Finite redshift inside configured z range.",
            intrinsic,
        )
    )

    detected = intrinsic[intrinsic["n_valid_filters"] > 0].copy()
    stages.append(
        SelectionStage(
            "detected_catalogue",
            "At least one positive finite filter surface brightness.",
            detected,
        )
    )

    snr = detected[_snr_mask(detected, schema)].copy()
    stages.append(
        SelectionStage(
            "snr_limited",
            "At least one configured filter exceeds the S/N threshold; no-op if errors are absent.",
            snr,
        )
    )

    aperture = snr[np.isfinite(snr["aperture_area_arcsec2_used"]) & (snr["aperture_area_arcsec2_used"] > 0.0)].copy()
    aperture = aperture[_flag_mask(aperture, schema)].copy()
    stages.append(
        SelectionStage(
            "aperture_quality",
            "Positive aperture area plus configured flag cut, e.g. flags < 80.",
            aperture,
        )
    )

    clean_mask = (
        _boolean_clean_mask(aperture, schema.blend_col)
        & _boolean_clean_mask(aperture, schema.edge_col)
        & _boolean_clean_mask(aperture, schema.point_source_col)
    )
    clean = aperture[clean_mask].copy()
    stages.append(
        SelectionStage(
            "blend_edge_point_clean",
            "Configured blend, edge, and point-source rejection.",
            clean,
        )
    )

    spec = clean[_mock_spec_mask(clean, schema)].copy()
    spec_name, spec_description = _spec_stage_metadata(clean, schema)
    stages.append(
        SelectionStage(
            spec_name,
            spec_description,
            spec,
        )
    )
    return stages

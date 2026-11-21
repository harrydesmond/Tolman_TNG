"""Normalize ASTRODEEP-JWST photometry and redshift catalogues."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .filters import ASTRODEEP_JWST_FILTERS
from .io import read_catalog


FILTERS = tuple(ASTRODEEP_JWST_FILTERS)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _find_col(columns: list[str], candidates: list[str], *, required: bool = True) -> str | None:
    norm_map = {_norm(col): col for col in columns}
    for cand in candidates:
        if _norm(cand) in norm_map:
            return norm_map[_norm(cand)]
    if required:
        raise KeyError(f"Could not find any of columns: {candidates}")
    return None


def _find_filter_col(columns: list[str], filter_name: str, *, error: bool) -> str | None:
    target = filter_name.lower()
    patterns: list[re.Pattern[str]]
    if error:
        patterns = [
            re.compile(rf"^(e|err|error|df|dflux|fluxerr|fluxerror).*{target}$", re.I),
            re.compile(rf"^{target}_(e|err|error|df|dflux|fluxerr|fluxerror)$", re.I),
            re.compile(rf"^(e|err|error|df|dflux|fluxerr|fluxerror)_{target}$", re.I),
        ]
    else:
        patterns = [
            re.compile(rf"^(f|flux|flux_)?{target}$", re.I),
            re.compile(rf"^{target}_(f|flux)$", re.I),
        ]
    for col in columns:
        compact = _norm(col)
        for pat in patterns:
            if pat.match(compact):
                return col
    fallback_terms = ("err", "error", "dflux", "fluxerr", "fluxerror", "ef")
    for col in columns:
        compact = _norm(col)
        has_filter = target in compact
        has_error = any(term in compact for term in fallback_terms)
        if has_filter and has_error == error:
            return col
    return None


def normalize_astrodeep_catalogs(
    phot_catalog: str | Path,
    redshift_catalog: str | Path | None,
    *,
    field_name: str,
    id_col: str | None = None,
    redshift_id_col: str | None = None,
    zspec_col: str | None = None,
    zphot_col: str | None = None,
    aperture_col: str | None = None,
    aperture_kind: str = "diameter",
    flag_col: str | None = None,
) -> pd.DataFrame:
    """Merge ASTRODEEP photometry and redshift catalogues into local schema."""

    phot = read_catalog(phot_catalog)
    phot_columns = list(phot.columns)
    id_col = id_col or _find_col(phot_columns, ["id", "ID", "NUMBER", "num", "object_id"])
    aperture_col = aperture_col or _find_col(
        phot_columns,
        ["apopt", "optap", "opt_ap", "optimal_aperture", "aperture", "aperture_arcsec", "optap_arcsec"],
    )
    flag_col = flag_col or _find_col(phot_columns, ["flag", "flags", "FLAG"], required=False)
    ra_col = _find_col(phot_columns, ["ra", "RA", "RAJ2000", "RAdeg", "alpha"], required=False)
    dec_col = _find_col(phot_columns, ["dec", "DEC", "DEJ2000", "DEdeg", "delta"], required=False)
    optional_phot_cols = {
        "FlagsSE": ["FlagsSE", "flags_se", "sextractor_flags"],
        "ClassStarSE": ["ClassStarSE", "class_star", "classstar"],
        "IsoAreaSE": ["IsoAreaSE", "isoarea", "iso_area"],
        "r50SE": ["r50SE", "flux_radius", "r50"],
        "FautoSE": ["FautoSE", "flux_auto", "fauto"],
        "e_FautoSE": ["e_FautoSE", "fluxerr_auto", "efauto"],
        "aAP": ["aAP", "a_ap"],
        "ellAP": ["ellAP", "ell_ap", "ellipticity"],
        "rKronAP": ["rKronAP", "rkron", "kron_radius"],
    }

    out = pd.DataFrame(
        {
            "id": phot[id_col],
            "field": field_name,
            "aperture_value_arcsec": pd.to_numeric(phot[aperture_col], errors="coerce"),
        }
    )
    if aperture_kind == "diameter":
        out["aperture_diameter_arcsec"] = out["aperture_value_arcsec"]
        out["aperture_radius_arcsec"] = out["aperture_value_arcsec"] / 2.0
    elif aperture_kind == "radius":
        out["aperture_radius_arcsec"] = out["aperture_value_arcsec"]
        out["aperture_diameter_arcsec"] = 2.0 * out["aperture_value_arcsec"]
    else:
        raise ValueError("aperture_kind must be 'diameter' or 'radius'")
    out["aperture_area_arcsec2"] = np.pi * out["aperture_radius_arcsec"] ** 2
    if ra_col:
        out["ra_deg"] = phot[ra_col]
    if dec_col:
        out["dec_deg"] = phot[dec_col]
    if flag_col:
        out["flags"] = phot[flag_col]
    for out_col, candidates in optional_phot_cols.items():
        col = _find_col(phot_columns, candidates, required=False)
        if col:
            out[out_col] = phot[col]

    for filter_name in FILTERS:
        flux_col = _find_filter_col(phot_columns, filter_name, error=False)
        err_col = _find_filter_col(phot_columns, filter_name, error=True)
        if flux_col:
            out[f"flux_{filter_name}_uJy"] = pd.to_numeric(phot[flux_col], errors="coerce")
        if err_col:
            out[f"fluxerr_{filter_name}_uJy"] = pd.to_numeric(phot[err_col], errors="coerce")

    if redshift_catalog is not None:
        red = read_catalog(redshift_catalog)
        red_columns = list(red.columns)
        redshift_id_col = redshift_id_col or _find_col(red_columns, [id_col, "id", "ID", "NUMBER", "num"])
        zspec_col = zspec_col or _find_col(red_columns, ["zspec", "z_spec", "specz", "zsp", "z_spectroscopic"], required=False)
        zphot_col = zphot_col or _find_col(red_columns, ["zphot", "z_phot", "photoz", "zbest", "z_best"], required=False)
        keep = [redshift_id_col]
        rename = {redshift_id_col: "id"}
        if zspec_col:
            keep.append(zspec_col)
            rename[zspec_col] = "z_spec"
        if zphot_col:
            keep.append(zphot_col)
            rename[zphot_col] = "z_phot"
        red_flag_col = _find_col(red_columns, ["Flag", "flags", "redshift_flag"], required=False)
        if red_flag_col:
            keep.append(red_flag_col)
            rename[red_flag_col] = "redshift_flags"
        red_small = red[keep].rename(columns=rename)
        out = out.merge(red_small, on="id", how="left")
    else:
        out["z_spec"] = np.nan
        out["z_phot"] = np.nan

    if "z_spec" not in out:
        out["z_spec"] = np.nan
    if "z_phot" not in out:
        out["z_phot"] = np.nan
    z_spec = pd.to_numeric(out["z_spec"], errors="coerce")
    z_phot = pd.to_numeric(out["z_phot"], errors="coerce")
    out["has_spec"] = np.isfinite(z_spec) & (z_spec > 0.0)
    out["z"] = np.where(out["has_spec"], z_spec, z_phot)
    return out


def write_astrodeep_schema(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "[catalog]",
        'redshift_col = "z"',
        'area_col = "aperture_area_arcsec2"',
        'object_id_col = "id"',
        'field_col = "field"',
        'flags_col = "flags"',
        "bad_flag_min = 80",
        "",
        "[selection]",
        "min_snr = 5",
        'spec_selection_col = "has_spec"',
        "z_min = 0.0",
        "z_max = 14.0",
        "",
    ]
    for name, wavelength in ASTRODEEP_JWST_FILTERS.items():
        lines.extend(
            [
                "[[filters]]",
                f'name = "{name}"',
                f"lambda_eff_angstrom = {wavelength:.0f}",
                f'flux_col = "flux_{name}_uJy"',
                f'flux_err_col = "fluxerr_{name}_uJy"',
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")

"""Catalogue schema and filter metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 system installations.
    import tomli as tomllib


@dataclass(frozen=True)
class FilterSpec:
    name: str
    lambda_eff_angstrom: float
    flux_col: str
    flux_err_col: str | None = None

    @property
    def sb_col(self) -> str:
        return f"sb_{self.name}"


@dataclass(frozen=True)
class SelectionConfig:
    min_snr: float = 5.0
    spec_mag_filter: str | None = None
    spec_mag_limit: float | None = None
    spec_selection_col: str | None = None
    z_min: float = 0.0
    z_max: float = 14.0


@dataclass(frozen=True)
class CatalogSchema:
    redshift_col: str
    filters: tuple[FilterSpec, ...]
    area_col: str | None = None
    radius_col: str | None = None
    radius_unit: str = "arcsec"
    object_id_col: str | None = None
    field_col: str | None = None
    flags_col: str | None = None
    bad_flag_min: float | None = None
    blend_col: str | None = None
    edge_col: str | None = None
    point_source_col: str | None = None
    selection: SelectionConfig = SelectionConfig()

    @property
    def sb_cols(self) -> list[str]:
        return [spec.sb_col for spec in self.filters]

    @property
    def filter_by_name(self) -> dict[str, FilterSpec]:
        return {spec.name: spec for spec in self.filters}


def schema_from_toml(path: str | Path) -> CatalogSchema:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)

    cat = raw.get("catalog", {})
    if "redshift_col" not in cat:
        raise ValueError("Config must define [catalog].redshift_col")
    if not raw.get("filters"):
        raise ValueError("Config must define at least one [[filters]] entry")

    filters = tuple(
        FilterSpec(
            name=str(item["name"]),
            lambda_eff_angstrom=float(item["lambda_eff_angstrom"]),
            flux_col=str(item["flux_col"]),
            flux_err_col=str(item["flux_err_col"]) if item.get("flux_err_col") else None,
        )
        for item in raw["filters"]
    )

    sel_raw = raw.get("selection", {})
    selection = SelectionConfig(
        min_snr=float(sel_raw.get("min_snr", 5.0)),
        spec_mag_filter=sel_raw.get("spec_mag_filter"),
        spec_mag_limit=(
            float(sel_raw["spec_mag_limit"])
            if sel_raw.get("spec_mag_limit") is not None
            else None
        ),
        spec_selection_col=sel_raw.get("spec_selection_col"),
        z_min=float(sel_raw.get("z_min", 0.0)),
        z_max=float(sel_raw.get("z_max", 14.0)),
    )

    return CatalogSchema(
        redshift_col=str(cat["redshift_col"]),
        filters=filters,
        area_col=cat.get("area_col"),
        radius_col=cat.get("radius_col"),
        radius_unit=str(cat.get("radius_unit", "arcsec")),
        object_id_col=cat.get("object_id_col"),
        field_col=cat.get("field_col"),
        flags_col=cat.get("flags_col"),
        bad_flag_min=(
            float(cat["bad_flag_min"]) if cat.get("bad_flag_min") is not None else None
        ),
        blend_col=cat.get("blend_col"),
        edge_col=cat.get("edge_col"),
        point_source_col=cat.get("point_source_col"),
        selection=selection,
    )


ASTRODEEP_JWST_FILTERS = {
    "F435W": 4325.0,
    "F606W": 5921.0,
    "F775W": 7693.0,
    "F814W": 8047.0,
    "F105W": 10550.0,
    "F125W": 12486.0,
    "F140W": 13923.0,
    "F160W": 15369.0,
    "F090W": 9020.0,
    "F115W": 11540.0,
    "F150W": 15010.0,
    "F200W": 19890.0,
    "F277W": 27620.0,
    "F356W": 35680.0,
    "F410M": 40820.0,
    "F444W": 44040.0,
}


TNG_MAST_FILTERS = {
    "F435W": 4325.0,
    "F606W": 5921.0,
    "F814W": 8047.0,
    "F125W": 12486.0,
    "F140W": 13923.0,
    "F160W": 15369.0,
    "F090W": 9020.0,
    "F115W": 11540.0,
    "F150W": 15010.0,
    "F200W": 19890.0,
    "F277W": 27620.0,
    "F356W": 35680.0,
    "F444W": 44040.0,
}

"""Diagnostics for reproducing the Tolman paper's ASTRODEEP sample."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def astrodeep_sample_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return summary counts and field counts for candidate paper cuts."""

    z = pd.to_numeric(df["z"], errors="coerce")
    zspec = pd.to_numeric(df.get("z_spec", df.get("z")), errors="coerce")
    flags = pd.to_numeric(df.get("flags", np.nan), errors="coerce")
    redshift_flags = pd.to_numeric(df.get("redshift_flags", flags), errors="coerce")
    area = pd.to_numeric(df.get("aperture_area_arcsec2_used", df.get("aperture_area_arcsec2")), errors="coerce")
    n_valid = pd.to_numeric(df.get("n_valid_filters", np.nan), errors="coerce")
    snr_cols = [col for col in df.columns if col.startswith("snr_")]
    if snr_cols:
        snr = df[snr_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        any_snr = np.nanmax(snr, axis=1)
    else:
        any_snr = np.full(len(df), np.nan)

    base = np.isfinite(zspec) & (zspec > 0.0) & np.isfinite(z) & (z <= 14.0) & np.isfinite(area) & (area > 0.0)
    flag_cut = base & np.isfinite(flags) & (flags < 80.0)
    redshift_flag_cut = base & np.isfinite(redshift_flags) & (redshift_flags < 80.0)
    rows: list[dict[str, int | str]] = [
        {"cut": "all", "n": int(len(df))},
        {"cut": "positive_zspec", "n": int(np.sum(np.isfinite(zspec) & (zspec > 0.0)))},
        {"cut": "positive_zspec_z_le_14_positive_aperture", "n": int(np.sum(base))},
        {"cut": "base_plus_phot_flag_lt_80", "n": int(np.sum(flag_cut))},
        {"cut": "base_plus_redshift_flag_lt_80", "n": int(np.sum(redshift_flag_cut))},
        {"cut": "base_plus_both_flags_lt_80", "n": int(np.sum(flag_cut & redshift_flag_cut))},
    ]
    for threshold in (5, 6, 7, 8, 9, 10, 12, 15, 20):
        rows.append({"cut": f"phot_flag_lt_80_and_any_snr_ge_{threshold}", "n": int(np.sum(flag_cut & (any_snr >= threshold)))})
    for minimum in range(1, 17):
        count = int(np.sum(flag_cut & (n_valid >= minimum)))
        if minimum in {1, 4, 8, 12, 16} or 6800 <= count <= 6920:
            rows.append({"cut": f"phot_flag_lt_80_and_n_valid_filters_ge_{minimum}", "n": count})

    if "field" in df:
        field_counts = (
            df.loc[flag_cut, "field"]
            .value_counts()
            .rename_axis("field")
            .reset_index(name="n_base_plus_phot_flag_lt_80")
        )
    else:
        field_counts = pd.DataFrame()
    return pd.DataFrame(rows), field_counts


def write_astrodeep_sample_audit(df: pd.DataFrame, out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, fields = astrodeep_sample_audit(df)
    counts.to_csv(out_dir / "sample_cut_counts.csv", index=False)
    fields.to_csv(out_dir / "field_counts.csv", index=False)
    lines = ["# ASTRODEEP Sample Audit", ""]
    lines.extend(f"- {row.cut}: {row.n}" for row in counts.itertuples(index=False))
    if len(fields):
        lines.extend(["", "## Field Counts", ""])
        lines.extend(f"- {row.field}: {row.n_base_plus_phot_flag_lt_80}" for row in fields.itertuples(index=False))
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts, fields

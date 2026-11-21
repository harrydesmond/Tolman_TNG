"""Input/output helpers for catalogues and result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype


def _looks_like_vizier_asu(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in range(80):
            line = handle.readline()
            if not line:
                break
            if line.startswith("#") and ("VizieR" in line or line.startswith("#Table")):
                return True
    return False


def _read_vizier_asu_tsv(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", comment="#", engine="python")
    if "recno" in table.columns:
        recno = pd.to_numeric(table["recno"], errors="coerce")
        table = table.loc[recno.notna()].copy()
        table["recno"] = recno.loc[recno.notna()].astype(int).to_numpy()
    for col in table.columns:
        if is_numeric_dtype(table[col]):
            continue
        original_notna = table[col].notna()
        numeric = pd.to_numeric(table[col], errors="coerce")
        if numeric.notna().equals(original_notna):
            table[col] = numeric
    return table.reset_index(drop=True)


def read_catalog(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv"):
        return pd.read_csv(path)
    if suffixes.endswith(".tsv") or suffixes.endswith(".txt"):
        if _looks_like_vizier_asu(path):
            return _read_vizier_asu_tsv(path)
        return pd.read_csv(path, sep=None, engine="python")
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if any(suffixes.endswith(ext) for ext in (".fits", ".fit", ".fits.gz", ".fit.gz")):
        from astropy.table import Table

        return Table.read(path).to_pandas()
    raise ValueError(f"Unsupported catalogue format: {path}")


def write_table(table: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)

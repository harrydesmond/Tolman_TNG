"""Load Li 2023 input samples and TNG galaxy catalogues for the analogue."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
JJ229_DEFAULT = REPO_ROOT / "data" / "li2023" / "jj229.csv"
GKF50_DEFAULT = REPO_ROOT / "data" / "li2023" / "gkf5p0.csv"

# Li 2023 sets the median spectral index for sources with no measured value to -0.1
ALPHA_SPEC_MEDIAN = -0.1


@dataclass
class TNGSample:
    """Per-galaxy TNG sample wrapped for the Li-analogue."""

    df: pd.DataFrame
    filter_name: str
    flux_col: str
    radius_col: str
    area_col: str

    @property
    def n(self) -> int:
        return len(self.df)


def load_jj229(path: str | Path | None = None) -> pd.DataFrame:
    """Load Jackson & Jannetta (2006) 613-source 2.29 GHz catalogue.

    Source: northumbria-eprint 13109 (Jackson & Jannetta 2013 PDF deposit).
    Columns: redshift `z`, observed angular size `theta_mas` (mas) and its
    uncertainty `theta_err_mas`, total radio flux density `S_jy` (Jy)
    at 2.29 GHz.  The PDF does not list per-source spectral indices, so
    `alpha_spec` is filled with the Li-2023 median value (-0.1).  This is a
    small approximation: Preston et al. (1985) Table VI lists per-source
    alpha for ~94% of objects, but is not in machine-readable form here.
    """
    path = Path(path) if path else JJ229_DEFAULT
    df = pd.read_csv(path)
    df = df.rename(columns={"flux_jy": "S_jy"})
    df["frequency_ghz"] = 2.29
    df["alpha_spec"] = ALPHA_SPEC_MEDIAN
    df["sample"] = "JJ2.29"
    df["is_upper_limit"] = False
    return df


def load_gkf50(path: str | Path | None = None) -> pd.DataFrame:
    """Load Gurvits, Kellermann & Frey (1999) 330-source 5 GHz catalogue.

    Source: VizieR J/A+A/342/378 table1.  Columns include redshift `z`,
    angular size `theta_mas` (mas), structure flag, optical type, total
    flux at 6 cm `S6cm_jy` and 20 cm `S20cm_jy`, and per-source spectral
    index `spectral_index`.  `theta_limit == '<'` flags upper limits;
    Li 2023 keeps these entries when computing the binned median theta(z).
    """
    path = Path(path) if path else GKF50_DEFAULT
    df = pd.read_csv(path)
    df["is_upper_limit"] = df.get("theta_limit", "").fillna("").astype(str).str.strip() == "<"
    df["S_jy"] = df["S6cm_jy"]
    df["frequency_ghz"] = 5.0
    # Li 2023 fills missing alpha with the median (-0.1).
    df["alpha_spec"] = df["spectral_index"].fillna(ALPHA_SPEC_MEDIAN)
    df["sample"] = "GKF5.0"
    return df


def load_tng_galaxy_sample(
    measured_catalog: str | Path,
    filter_name: str = "F200W",
    *,
    require_spec_selected: bool = True,
    require_image_success: bool = True,
) -> TNGSample:
    """Load a per-galaxy TNG measured catalogue and select usable rows.

    The catalogue is the per-galaxy CSV produced by `run_catalog_pipeline`
    after `tng-image-aperture` and `apply-spec-selection`.  We require the
    selected filter to have a successful image-aperture measurement and a
    finite flux + angular radius.  By default we restrict to the
    spec-selected sample (`mock_spec_selected == True`) to mirror the
    flux-limited selection of Li's radio samples.
    """
    path = Path(measured_catalog)
    df = pd.read_csv(path)
    flux_col = f"flux_{filter_name}_uJy"
    radius_col = f"aperture_radius_arcsec_{filter_name}"
    area_col = f"aperture_area_arcsec2_{filter_name}"
    success_col = f"image_success_{filter_name}"

    needed = [flux_col, radius_col, area_col, "z", "kpc_per_arcsec"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns in {path}: {missing}")

    keep = np.isfinite(df[flux_col]) & np.isfinite(df[radius_col]) & (df[radius_col] > 0)
    if require_image_success and success_col in df.columns:
        keep &= df[success_col].astype(bool)
    if require_spec_selected and "mock_spec_selected" in df.columns:
        keep &= df["mock_spec_selected"].astype(bool)

    sub = df.loc[keep, :].copy()
    return TNGSample(
        df=sub,
        filter_name=filter_name,
        flux_col=flux_col,
        radius_col=radius_col,
        area_col=area_col,
    )

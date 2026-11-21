#!/usr/bin/env python3
"""ASTRODEEP slope stability under additional quality cuts.

Tests:
  A) n_valid_filters >= 8  => ~6,867 objects
  B) any S/N >= 10         => ~6,850 objects

Reports mean SB and max SB alpha for each vs the 7,059-object baseline.
"""
import numpy as np
import pandas as pd

CATALOG = "results/astrodeep_cds/measured_catalog.csv"


def fit_alpha(z: np.ndarray, sb: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(z) & np.isfinite(sb) & (z > -1) & (sb > 0)
    x = np.log10(1.0 + z[mask])
    y = np.log10(sb[mask])
    n = int(mask.sum())
    A = np.c_[np.ones(n), x]
    c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ c
    s2 = float(np.sum(resid**2) / (n - 2))
    err = float((s2 * np.linalg.inv(A.T @ A)[1, 1]) ** 0.5)
    return float(c[1]), err, n


def main() -> None:
    print(f"Loading {CATALOG} ...", flush=True)
    df = pd.read_csv(CATALOG, low_memory=False)
    print(f"  {len(df):,} total rows")

    z = pd.to_numeric(df["z"], errors="coerce").values
    n_valid = pd.to_numeric(df["n_valid_filters"], errors="coerce").fillna(0).values
    flags = pd.to_numeric(df["flags"], errors="coerce").fillna(0).values
    has_spec = df["has_spec"].astype(bool).values
    ap_area = pd.to_numeric(df["aperture_area_arcsec2_used"], errors="coerce").fillna(0).values
    snr_cols = [c for c in df.columns if c.startswith("snr_")]
    snr = df[snr_cols].apply(pd.to_numeric, errors="coerce").values

    base_mask = (
        np.isfinite(z) & (z > 0) & (z <= 14)
        & (n_valid > 0)
        & (np.nanmax(snr, axis=1) >= 5)
        & (flags < 80)
        & (ap_area > 0)
        & has_spec
    )
    z_b = z[base_mask]
    snr_b = snr[base_mask]
    nv_b = n_valid[base_mask]
    sb_mean_b = pd.to_numeric(df.loc[base_mask, "sb_mean"], errors="coerce").values
    sb_max_b = pd.to_numeric(df.loc[base_mask, "sb_max"], errors="coerce").values

    a_m, e_m, n_m = fit_alpha(z_b, sb_mean_b)
    a_x, e_x, _ = fit_alpha(z_b, sb_max_b)
    print(f"\nBaseline:    N={n_m:,}  mean α={a_m:+.3f} ± {e_m:.3f}  max α={a_x:+.3f} ± {e_x:.3f}")

    # Cut A: n_valid_filters >= 8
    cut_a = nv_b >= 8
    aa_m, ae_m, na = fit_alpha(z_b[cut_a], sb_mean_b[cut_a])
    aa_x, ae_x, _ = fit_alpha(z_b[cut_a], sb_max_b[cut_a])
    print(f"n_valid>=8:  N={na:,}  mean α={aa_m:+.3f} ± {ae_m:.3f} (Δ={aa_m-a_m:+.3f})  max α={aa_x:+.3f} ± {ae_x:.3f} (Δ={aa_x-a_x:+.3f})")

    # Cut B: any S/N >= 10
    cut_b = np.nanmax(snr_b, axis=1) >= 10
    ab_m, abe_m, nb = fit_alpha(z_b[cut_b], sb_mean_b[cut_b])
    ab_x, abe_x, _ = fit_alpha(z_b[cut_b], sb_max_b[cut_b])
    print(f"any S/N>=10: N={nb:,}  mean α={ab_m:+.3f} ± {abe_m:.3f} (Δ={ab_m-a_m:+.3f})  max α={ab_x:+.3f} ± {abe_x:.3f} (Δ={ab_x-a_x:+.3f})")


if __name__ == "__main__":
    main()

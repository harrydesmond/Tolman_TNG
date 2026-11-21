#!/usr/bin/env python3
"""Per-aperture TNG gamma table (volume-controlled).

Computes the intrinsic luminosity-density evolution exponent
``gamma = d log10(rho_L) / d log10(1+z)`` for the four TNG aperture
variants reported in the paper's gamma aperture-envelope table:

    (i)   specmodel catalogue flux + photrad_kpc aperture
    (ii)  fixed-physical image aperture (canonical)
    (iii) Voronoi-deblended image flux (fiducial)
    (iv)  adaptive flux at ASTRODEEP-equivalent z-binned aperture radii

In every variant the luminosity L uses the variant-specific flux, but the
volume V is held FIXED at the intrinsic ``photrad_kpc`` volume
``V = (4/3) pi (photrad_kpc)^3``.  This decouples gamma from the imposed
redshift dependence of the adaptive aperture radius, which would otherwise
contribute mechanically to the adaptive-variant gamma.

A per-variant statistical floor is provided by the OLS standard error on the
log-linear gamma fit. Bootstrap resampling of the individual galaxies is also
reported as a robustness check.

Outputs:
  results/li_ddr/tng_gamma_aperture_table.csv -- gamma per variant under
       (a) the variant aperture's own volume
       (b) the volume-controlled (intrinsic photrad_kpc) definition
       plus OLS/bootstrap stats and median radii.

The K-correction assumes the paper-wide convention alpha_SED = 0.3 (Li/code
convention ALPHA_SPEC = -0.3, with ALPHA_SPEC = -alpha_SED); gamma shifts
linearly by +Delta alpha_SED if this choice is changed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

from tolman.li_ddr.io import TNGSample
from tolman.li_ddr.li_reproduction import COSMO_LI, luminosity_at_band

ROOT = Path(os.environ.get("TOLMAN_WORKDIR", Path(__file__).resolve().parents[1]))
FILTER_NAME = "F200W"
ALPHA_SPEC = -0.3  # Li convention; equivalent to paper alpha_SED = 0.3.
N_BOOT = 200
RNG = np.random.default_rng(20260509)

CANONICAL_CAT = ROOT / "results/tng100_7_6_xyz_hst_jwst_imageap_specmodel/measured_catalog.csv"
CATALOGUE_CAT = ROOT / "results/tng100_7_6_xyz_hst_jwst_specmodel/measured_catalog.csv"
DEBLENDED_PER_OBJ = ROOT / "results/systematics/deblend_per_object_f200w.csv"
_FITS_DIR = ROOT / "data/raw/illustris/tng100-7-6-xyz_137sqarcmin"


def _intrinsic_photrad_kpc(df: pd.DataFrame) -> np.ndarray:
    """Reconstruct photrad_kpc = 2 r_half_phys from canonical catalog.

    For the MAST TNG100 mocks, aperture_radius_arcsec_F200W is the same as
    aperture_radius_arcsec at any other filter (both = photrad_kpc / kpc_per_arcsec),
    so multiplying by kpc_per_arcsec recovers photrad_kpc exactly.
    """
    return (
        df["aperture_radius_arcsec_F200W"].to_numpy(dtype=float)
        * df["kpc_per_arcsec"].to_numpy(dtype=float)
    )


def _spec_select(df: pd.DataFrame) -> pd.DataFrame:
    keep = df["mock_spec_selected"].astype(bool)
    if "image_success_F200W" in df.columns:
        keep &= df["image_success_F200W"].astype(bool)
    keep &= df[f"flux_{FILTER_NAME}_uJy"] > 0
    keep &= df[f"aperture_radius_arcsec_{FILTER_NAME}"] > 0
    return df.loc[keep].copy()


def _gamma_from_arrays(z: np.ndarray, log_rho_L: np.ndarray) -> float:
    finite = np.isfinite(z) & np.isfinite(log_rho_L) & (z > 0)
    if finite.sum() < 10:
        return float("nan")
    g, _ = np.polyfit(np.log10(1.0 + z[finite]), log_rho_L[finite], 1)
    return float(g)


def _ols_gamma(z: np.ndarray, log_rho_L: np.ndarray) -> dict:
    finite = np.isfinite(z) & np.isfinite(log_rho_L) & (z > 0)
    n = int(finite.sum())
    if n < 3:
        return dict(gamma=float("nan"), gamma_ols_stderr=float("nan"))

    x = np.log10(1.0 + z[finite])
    y = log_rho_L[finite]
    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = n - design.shape[1]
    if dof <= 0:
        stderr = float("nan")
    else:
        sigma2 = float(np.sum(resid**2) / dof)
        cov = sigma2 * np.linalg.inv(design.T @ design)
        stderr = float(np.sqrt(cov[0, 0]))
    return dict(gamma=float(coef[0]), gamma_ols_stderr=stderr)


def _bootstrap_gamma(z: np.ndarray, log_rho_L: np.ndarray) -> dict:
    g0 = _gamma_from_arrays(z, log_rho_L)
    n = len(z)
    if n < 10:
        return dict(gamma=g0, gamma_boot_mean=float("nan"), gamma_boot_std=float("nan"),
                    gamma_boot_p16=float("nan"), gamma_boot_p84=float("nan"))
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        boots[i] = _gamma_from_arrays(z[idx], log_rho_L[idx])
    return dict(
        gamma=g0,
        gamma_boot_mean=float(np.mean(boots)),
        gamma_boot_std=float(np.std(boots, ddof=1)),
        gamma_boot_p16=float(np.percentile(boots, 16)),
        gamma_boot_p84=float(np.percentile(boots, 84)),
    )


def _compute_one_variant(
    label: str,
    df: pd.DataFrame,
    flux_uJy: np.ndarray,
    radius_arcsec_var: np.ndarray,
    photrad_kpc_intrinsic: np.ndarray,
    cosmo: FlatLambdaCDM = COSMO_LI,
) -> list[dict]:
    """Return two rows: original (variant volume) and volume-controlled."""
    z = df["z"].to_numpy(dtype=float)
    kpc_per_arcsec = df["kpc_per_arcsec"].to_numpy(dtype=float)
    flux_jy = flux_uJy * 1e-6
    L = luminosity_at_band(flux_jy, z, np.full_like(z, ALPHA_SPEC), cosmo=cosmo)

    # Original (variant-defined) volume:
    r_kpc_var = radius_arcsec_var * kpc_per_arcsec
    r_cm_var = r_kpc_var * 3.0857e21
    rho_L_var = L / ((4.0 / 3.0) * np.pi * r_cm_var**3)

    # Volume-controlled (intrinsic photrad_kpc):
    r_cm_int = photrad_kpc_intrinsic * 3.0857e21
    rho_L_int = L / ((4.0 / 3.0) * np.pi * r_cm_int**3)

    rows = []
    for v_label, rho_L, radius_arcsec in (
        ("variant_volume", rho_L_var, radius_arcsec_var),
        ("intrinsic_photrad_volume", rho_L_int, radius_arcsec_var),
    ):
        finite = np.isfinite(rho_L) & (rho_L > 0) & np.isfinite(z) & (z > 0)
        log_rho = np.where(finite, np.log10(np.where(finite, rho_L, 1.0)), np.nan)
        ols = _ols_gamma(z[finite], log_rho[finite])
        boot = _bootstrap_gamma(z[finite], log_rho[finite])
        boot.pop("gamma", None)
        rows.append(
            dict(
                variant=label,
                volume_defn=v_label,
                n=int(finite.sum()),
                median_aperture_radius_arcsec=float(np.median(radius_arcsec[finite])),
                median_volume_kpc3=float(
                    np.median(((4.0 / 3.0) * np.pi * (
                        (radius_arcsec[finite] * kpc_per_arcsec[finite])
                        if v_label == "variant_volume"
                        else photrad_kpc_intrinsic[finite]
                    ) ** 3))
                ),
                **ols,
                **boot,
            )
        )
    return rows


def variant_specmodel():
    df = pd.read_csv(CATALOGUE_CAT, low_memory=False)
    df = _spec_select(df)
    flux = df[f"flux_{FILTER_NAME}_uJy"].to_numpy(dtype=float)
    radius_arcsec = df[f"aperture_radius_arcsec_{FILTER_NAME}"].to_numpy(dtype=float)
    photrad_kpc = _intrinsic_photrad_kpc(df)
    return df, flux, radius_arcsec, photrad_kpc


def variant_imageap_specmodel():
    df = pd.read_csv(CANONICAL_CAT, low_memory=False)
    df = _spec_select(df)
    flux = df[f"flux_{FILTER_NAME}_uJy"].to_numpy(dtype=float)
    radius_arcsec = df[f"aperture_radius_arcsec_{FILTER_NAME}"].to_numpy(dtype=float)
    photrad_kpc = _intrinsic_photrad_kpc(df)
    return df, flux, radius_arcsec, photrad_kpc


def variant_imageap_deblended():
    canon = pd.read_csv(CANONICAL_CAT, low_memory=False)
    deb = pd.read_csv(DEBLENDED_PER_OBJ)
    deb_f200 = deb.set_index("object_index")[["voronoi_flux_uJy"]]
    deb_f200.columns = [f"flux_{FILTER_NAME}_uJy"]
    if "object_index" in canon.columns:
        merged = canon.set_index("object_index").copy()
    else:
        merged = canon.copy()
        merged.index.name = "object_index"
    merged.update(deb_f200)
    merged = merged.reset_index()
    merged = _spec_select(merged)
    flux = merged[f"flux_{FILTER_NAME}_uJy"].to_numpy(dtype=float)
    radius_arcsec = merged[f"aperture_radius_arcsec_{FILTER_NAME}"].to_numpy(dtype=float)
    photrad_kpc = _intrinsic_photrad_kpc(merged)
    return merged, flux, radius_arcsec, photrad_kpc


def variant_imageap_astrodeep_ap():
    """Re-photometer F200W at ASTRODEEP-equivalent z-dependent radii."""
    from astropy.io import fits
    from tolman.adaptive_aperture import astrodeep_aperture_radius
    from tolman.image_photometry import circular_aperture_sum, image_unit_to_ujy_factor
    from tolman.tng_image_aperture import _pixel_scale_arcsec

    fits_paths = sorted(
        list(_FITS_DIR.glob("*F200W*.fits")) + list(_FITS_DIR.glob("*f200w*.fits"))
    )
    if not fits_paths:
        raise FileNotFoundError(f"No F200W FITS under {_FITS_DIR}")
    fits_path = fits_paths[0]

    canon = pd.read_csv(CANONICAL_CAT, low_memory=False)
    canon = _spec_select(canon)

    with fits.open(fits_path, memmap=True) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header
        ny, nx = image.shape
        pixel_scale = _pixel_scale_arcsec(header, None)
        unit_factor = image_unit_to_ujy_factor(header.get("BUNIT"))

    z = canon["z"].to_numpy(dtype=float)
    x = canon["x_F200W_pix"].to_numpy(dtype=float)
    y = canon["y_F200W_pix"].to_numpy(dtype=float)
    r_arcsec = astrodeep_aperture_radius(z)
    r_pix = r_arcsec / pixel_scale

    flux = np.array(
        [circular_aperture_sum(image, xi, yi, ri) for xi, yi, ri in zip(x, y, r_pix)],
        dtype=float,
    ) * unit_factor
    edge = (
        (x - r_pix < 0.0) | (y - r_pix < 0.0)
        | (x + r_pix >= nx) | (y + r_pix >= ny)
    )
    flux = np.where(np.isfinite(flux) & (flux > 0.0) & ~edge, flux, np.nan)
    keep = np.isfinite(flux) & (flux > 0)
    canon = canon.loc[keep].copy()
    flux = flux[keep]
    r_arcsec_kept = r_arcsec[keep]
    photrad_kpc = _intrinsic_photrad_kpc(canon)
    return canon, flux, r_arcsec_kept, photrad_kpc


def main() -> None:
    builders = [
        ("specmodel (catalogue flux)", variant_specmodel),
        ("imageap_specmodel (canonical)", variant_imageap_specmodel),
        ("imageap_deblended (Voronoi)", variant_imageap_deblended),
        ("imageap_astrodeep_ap", variant_imageap_astrodeep_ap),
    ]
    rows: list[dict] = []
    for label, build in builders:
        try:
            df, flux, radius_arcsec, photrad_kpc = build()
        except Exception as exc:  # noqa: BLE001
            print(f"!! {label} skipped: {exc}")
            continue
        var_rows = _compute_one_variant(label, df, flux, radius_arcsec, photrad_kpc)
        rows.extend(var_rows)
        for r in var_rows:
            print(
                f"{label:35s} [{r['volume_defn']:25s}] n={r['n']:5d} "
                f"gamma={r['gamma']:+.3f} "
                f"(OLS={r['gamma_ols_stderr']:.3f}, "
                f"boot={r['gamma_boot_mean']:+.3f}+/-{r['gamma_boot_std']:.3f}) "
                f"<r>={r['median_aperture_radius_arcsec']:.3f}\" "
                f"<V>={r['median_volume_kpc3']:.1f} kpc^3"
            )

    table = pd.DataFrame(rows)
    out_dir = ROOT / "results" / "li_ddr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tng_gamma_aperture_table.csv"
    table.to_csv(out_path, index=False)
    print(f"\nWrote -> {out_path}")

    # Summary: gamma spread across variants for each volume defn.
    for v_label, sub in table.groupby("volume_defn"):
        g = sub["gamma"].to_numpy()
        go = sub["gamma_ols_stderr"].to_numpy()
        gb = sub["gamma_boot_std"].to_numpy()
        print(
            f"\n[{v_label}] across {len(sub)} variants: "
            f"mean={g.mean():+.3f}  min={g.min():+.3f}  max={g.max():+.3f}  "
            f"range={g.max()-g.min():.3f}  "
            f"median per-variant ols_sigma={np.median(go):.3f}  "
            f"median per-variant boot_sigma={np.median(gb):.3f}"
        )

    # Sanity check: the three non-adaptive variants under intrinsic volume
    # are expected to collapse to within ~0.1 in gamma; the adaptive variant
    # drifts only by a small residual once the volume term is fixed.
    real_only = table[
        (table["volume_defn"] == "intrinsic_photrad_volume")
        & (table["variant"] != "imageap_astrodeep_ap")
    ]["gamma"].to_numpy()
    if len(real_only) >= 2:
        spread_real = float(real_only.max() - real_only.min())
        print(
            f"\n[volume-controlled check] intrinsic-volume gamma should collapse to within +/-0.10:"
            f"\n  3 non-adaptive variants under intrinsic volume: range={spread_real:.3f}"
        )
        adaptive_int = table[
            (table["volume_defn"] == "intrinsic_photrad_volume")
            & (table["variant"] == "imageap_astrodeep_ap")
        ]["gamma"]
        adaptive_var = table[
            (table["volume_defn"] == "variant_volume")
            & (table["variant"] == "imageap_astrodeep_ap")
        ]["gamma"]
        if len(adaptive_int) and len(adaptive_var):
            print(
                f"  imageap_astrodeep_ap: gamma_variant={adaptive_var.iloc[0]:+.3f}, "
                f"gamma_intrinsic={adaptive_int.iloc[0]:+.3f}, "
                f"shift={adaptive_int.iloc[0]-adaptive_var.iloc[0]:+.3f}"
            )


if __name__ == "__main__":
    main()

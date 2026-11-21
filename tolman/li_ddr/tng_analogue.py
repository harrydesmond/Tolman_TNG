r"""Apply Li 2023 procedure to TNG mock galaxies (single-band analogue).

Li's framework was built for spherical compact radio sources (L \propto d^3).
Galaxies are nearly 2D in projection, so the strict L--d^3 relationship is
not the right physical model.  We still apply the *exact* Li procedure to
TNG mocks, on the grounds that the test answers a precise question:

    "If you fed Li's procedure a population of objects evolved by FRW
     hydrodynamics, would you recover the FRW DDR or Li's nonexpanding
     DDR?"

The result quantifies how much a FRW-driven luminosity-density evolution
mimics the Li 'conspiracy'.  A complementary 2D-aware variant (replace the
^3 with ^2) is exposed via `power=2` for galaxies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import curve_fit

from .io import TNGSample
from .li_reproduction import (
    COSMO_LI,
    SizeRedshiftFit,
    calibrated_angular_size,
    distance_duality_ratio,
    fit_d0_expanding,
    fit_d0_nonexpanding,
    luminosity_at_band,
)


@dataclass
class TNGLiAnalogue:
    """Outputs of running Li's procedure on a TNG sample at a single band."""

    z: np.ndarray
    flux_jy: np.ndarray
    theta_mas: np.ndarray
    L_band_cgs: np.ndarray
    fit: SizeRedshiftFit
    theta_calib_mas: np.ndarray
    d0_expanding_pc: float
    d0_expanding_err_pc: float
    d0_nonexpanding_pc: float
    d0_nonexpanding_err_pc: float
    ddr_ratio: np.ndarray
    filter_name: str
    alpha_spec: float
    power: int


def _log_L_model_p(z: np.ndarray, log_L0: float, a: float, b: float, p: int) -> np.ndarray:
    """Same as Li eq. (5) but with adjustable exponent p (default 3, galaxies use 2)."""
    one_plus_z = 1.0 + z
    return log_L0 + p * (a * np.log10(one_plus_z) + b * np.log10(np.log(one_plus_z)))


def _fit_with_power(z: np.ndarray, L: np.ndarray, *, power: int, p0=None) -> SizeRedshiftFit:
    z = np.asarray(z, dtype=float)
    L = np.asarray(L, dtype=float)
    mask = np.isfinite(z) & np.isfinite(L) & (z > 0) & (L > 0)
    z_fit, L_fit = z[mask], L[mask]
    if p0 is None:
        p0 = (float(np.log10(np.median(L_fit))), 0.3, 0.6)

    def model(zz, log_L0, a, b):
        return _log_L_model_p(zz, log_L0, a, b, power)

    p, cov = curve_fit(model, z_fit, np.log10(L_fit), p0=p0, maxfev=20000)
    err = np.sqrt(np.diag(cov))
    return SizeRedshiftFit(
        log_L0=float(p[0]),
        log_L0_err=float(err[0]),
        a=float(p[1]),
        a_err=float(err[1]),
        b=float(p[2]),
        b_err=float(err[2]),
        n=int(mask.sum()),
    )


def run_li_on_tng(
    sample: TNGSample,
    *,
    alpha_spec: float = 0.0,
    power: int = 3,
    cosmo: FlatLambdaCDM = COSMO_LI,
) -> TNGLiAnalogue:
    """Apply Li's procedure to a TNG single-band sample.

    Inputs:
        sample      -- TNGSample (per-galaxy fluxes, angular radii, redshift).
        alpha_spec  -- effective spectral index for the K-correction in
                       Li eq. (2), using Li's convention F_nu propto
                       nu^alpha_spec, so alpha_spec = -alpha_SED.  The
                       default 0.0 is a flat SED; the paper uses
                       alpha_spec = -0.3.
        power       -- 3 reproduces Li exactly (rho_L constant in 3D);
                       2 is the galaxy-flux/area analogue (constant SB).
    """
    df = sample.df
    z = df["z"].to_numpy(dtype=float)
    flux_uJy = df[sample.flux_col].to_numpy(dtype=float)
    theta_arcsec = df[sample.radius_col].to_numpy(dtype=float)

    flux_jy = flux_uJy * 1e-6
    theta_mas = theta_arcsec * 1e3
    L = luminosity_at_band(flux_jy, z, np.full_like(z, alpha_spec), cosmo=cosmo)
    fit = _fit_with_power(z, L, power=power)

    theta_calib = calibrated_angular_size(theta_mas, z, fit.a, fit.b)
    d0_exp, d0_exp_err, _ = fit_d0_expanding(theta_calib, z, cosmo=cosmo)
    d0_non, d0_non_err, _ = fit_d0_nonexpanding(theta_calib, z, cosmo=cosmo)
    # Use the non-expanding d_0 for the DDR ratio plot, mirroring Li Fig. 4.
    ratio = distance_duality_ratio(theta_calib, z, d0_pc=d0_non, cosmo=cosmo)

    return TNGLiAnalogue(
        z=z,
        flux_jy=flux_jy,
        theta_mas=theta_mas,
        L_band_cgs=L,
        fit=fit,
        theta_calib_mas=theta_calib,
        d0_expanding_pc=d0_exp,
        d0_expanding_err_pc=d0_exp_err,
        d0_nonexpanding_pc=d0_non,
        d0_nonexpanding_err_pc=d0_non_err,
        ddr_ratio=ratio,
        filter_name=sample.filter_name,
        alpha_spec=alpha_spec,
        power=power,
    )


def intrinsic_luminosity_density_evolution(
    sample: TNGSample,
    *,
    alpha_spec: float = 0.0,
    cosmo: FlatLambdaCDM = COSMO_LI,
    volume_kpc3: float | None = None,
) -> dict:
    """Direct rho_L(z) for TNG galaxies, ignoring Li's parameterisation.

    ``alpha_spec`` follows Li's convention F_nu propto nu^alpha_spec; in
    the paper convention alpha_spec = -alpha_SED.

    For each galaxy compute:
        L_nu  -- rest-frame at the band, eq. (2)
        V_phys = (4/3) pi r_phys^3 with r_phys = (theta_arcsec * kpc_per_arcsec)
        rho_L = L / V
        Sigma_L = L / (pi r_phys^2)  (the SB analogue, per area)

    Then fit log10 rho_L = c0 + gamma_vol * log10(1+z) and likewise for
    Sigma_L.  Li's required gamma_vol = 3 to keep FRW; a positive deviation
    closer to 3 means TNG already contains the conspiracy Li reports for
    radio sources.

    If ``volume_kpc3`` is given, it overrides the per-galaxy V_phys with a
    single fixed volume (in kpc^3). This is the fixed-V control used to
    isolate the contribution of the z-varying aperture schedule from a
    intrinsic luminosity-density evolution.
    """
    df = sample.df
    z = df["z"].to_numpy(dtype=float)
    flux_jy = df[sample.flux_col].to_numpy(dtype=float) * 1e-6
    theta_arcsec = df[sample.radius_col].to_numpy(dtype=float)
    kpc_per_arcsec = df["kpc_per_arcsec"].to_numpy(dtype=float)

    L_cgs = luminosity_at_band(flux_jy, z, np.full_like(z, alpha_spec), cosmo=cosmo)
    r_kpc = theta_arcsec * kpc_per_arcsec
    r_cm = r_kpc * 3.0857e21
    if volume_kpc3 is None:
        V_cm3 = (4.0 / 3.0) * np.pi * r_cm**3
    else:
        V_cm3 = np.full_like(L_cgs, float(volume_kpc3) * (3.0857e21) ** 3)
    rho_L = L_cgs / V_cm3  # erg/s/Hz / cm^3
    sigma_L = L_cgs / (np.pi * r_cm**2)  # erg/s/Hz / cm^2

    finite = np.isfinite(rho_L) & np.isfinite(sigma_L) & (rho_L > 0) & (sigma_L > 0) & (z > 0)
    z = z[finite]
    rho_L = rho_L[finite]
    sigma_L = sigma_L[finite]

    log_one_plus_z = np.log10(1.0 + z)
    gamma_rho, c_rho = np.polyfit(log_one_plus_z, np.log10(rho_L), 1)
    gamma_sig, c_sig = np.polyfit(log_one_plus_z, np.log10(sigma_L), 1)

    return dict(
        n=int(finite.sum()),
        gamma_rho_volume=float(gamma_rho),
        log10_rho_L_z0=float(c_rho),
        gamma_sigma_area=float(gamma_sig),
        log10_sigma_L_z0=float(c_sig),
        z=z,
        rho_L=rho_L,
        sigma_L=sigma_L,
    )


def to_summary_table(result: TNGLiAnalogue, sample_label: str) -> pd.DataFrame:
    """Compact one-row summary mirroring Li 2023 Table 1."""
    return pd.DataFrame(
        [
            dict(
                sample=sample_label,
                filter=result.filter_name,
                alpha_spec=result.alpha_spec,
                power=result.power,
                a=result.fit.a,
                a_err=result.fit.a_err,
                b=result.fit.b,
                b_err=result.fit.b_err,
                log_L0=result.fit.log_L0,
                log_L0_err=result.fit.log_L0_err,
                d0_expanding_pc=result.d0_expanding_pc,
                d0_expanding_err_pc=result.d0_expanding_err_pc,
                d0_nonexpanding_pc=result.d0_nonexpanding_pc,
                d0_nonexpanding_err_pc=result.d0_nonexpanding_err_pc,
                n=result.fit.n,
            )
        ]
    )

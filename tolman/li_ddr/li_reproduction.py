r"""Implementation of Li 2023 procedure (eqs. 2-9, table 1).

The procedure is faithful to the paper:

* eq. (2): per-source rest-frame luminosity at the observing band
  L_t = 4 pi D_L^2 S_t / (1+z)^(1+alpha) [erg/s/Hz]
* eq. (4): linear-size parameterisation
  d(z) = d_0 (1+z)^a [ln(1+z)]^b
* eq. (5): under L \propto d^3 at constant rho_L,
  L_t h^2 = L_0 [(1+z)^a [ln(1+z)]^b]^3
  fitted in log-space by `curve_fit`
* eq. (7): calibrated angular size
  theta = theta_obs (1+z)^(-a) [ln(1+z)]^(-b)
* eq. (8) / (9): single-parameter d_0 fits for expanding / non-expanding
* and the DDR ratio plot D_L/D_A vs z

The cosmology defaults to the Suzuki et al. 2012 SNe Ia best fit Li uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import curve_fit

# Li 2023 default: H0 = 70.0, Om = 0.277, OmL = 0.723 (Suzuki+ 2012)
COSMO_LI = FlatLambdaCDM(H0=70.0, Om0=0.277)

JY_TO_CGS = 1e-23  # erg/s/cm^2/Hz per Jy
RAD_PER_MAS = (np.pi / 180.0) / (3600.0 * 1000.0)


def luminosity_at_band(
    S_jy: np.ndarray,
    z: np.ndarray,
    alpha_spec: np.ndarray,
    cosmo: FlatLambdaCDM = COSMO_LI,
) -> np.ndarray:
    """Li 2023 eq. (2): rest-frame Lnu at the observing band [erg/s/Hz]."""
    z = np.asarray(z, dtype=float)
    S_jy = np.asarray(S_jy, dtype=float)
    alpha_spec = np.asarray(alpha_spec, dtype=float)
    D_L_cm = cosmo.luminosity_distance(z).to("cm").value
    return 4.0 * np.pi * D_L_cm**2 * S_jy * JY_TO_CGS / (1.0 + z) ** (1.0 + alpha_spec)


def _log_L_model(z: np.ndarray, log_L0: float, a: float, b: float) -> np.ndarray:
    """log10 L_t h^2 = log10 L_0 + 3 [a log10(1+z) + b log10(ln(1+z))]."""
    one_plus_z = 1.0 + z
    return log_L0 + 3.0 * (a * np.log10(one_plus_z) + b * np.log10(np.log(one_plus_z)))


@dataclass
class SizeRedshiftFit:
    log_L0: float
    log_L0_err: float
    a: float
    a_err: float
    b: float
    b_err: float
    n: int

    def predict_log_L(self, z: np.ndarray) -> np.ndarray:
        return _log_L_model(np.asarray(z, dtype=float), self.log_L0, self.a, self.b)


def fit_size_redshift_relation(
    z: np.ndarray,
    L: np.ndarray,
    *,
    p0: tuple[float, float, float] | None = None,
) -> SizeRedshiftFit:
    """Fit L_t = L_0 [(1+z)^a [ln(1+z)]^b]^3 in log-space (Li eq. 5)."""
    z = np.asarray(z, dtype=float)
    L = np.asarray(L, dtype=float)
    mask = np.isfinite(z) & np.isfinite(L) & (z > 0) & (L > 0)
    z_fit, L_fit = z[mask], L[mask]
    if p0 is None:
        p0 = (float(np.log10(np.median(L_fit))), 0.3, 0.6)
    p, cov = curve_fit(_log_L_model, z_fit, np.log10(L_fit), p0=p0, maxfev=20000)
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


def calibrated_angular_size(
    theta_mas: np.ndarray,
    z: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Li eq. (7): theta_calib = theta_obs (1+z)^(-a) [ln(1+z)]^(-b)."""
    z = np.asarray(z, dtype=float)
    theta = np.asarray(theta_mas, dtype=float)
    return theta * (1.0 + z) ** (-a) * np.log(1.0 + z) ** (-b)


def _per_source_d0(
    theta_calib_mas: np.ndarray,
    z: np.ndarray,
    *,
    expanding: bool,
    cosmo: FlatLambdaCDM = COSMO_LI,
) -> np.ndarray:
    """Per-source d_0 implied by Li eq. (8) [expanding] or (9) [non-expanding]."""
    D_A_pc = cosmo.angular_diameter_distance(z).to("pc").value
    base = theta_calib_mas * RAD_PER_MAS * D_A_pc
    if expanding:
        return base
    return base * (1.0 + z)


def fit_d0_expanding(
    theta_calib_mas: np.ndarray,
    z: np.ndarray,
    cosmo: FlatLambdaCDM = COSMO_LI,
) -> tuple[float, float, np.ndarray]:
    """Li eq. (8) single-parameter fit; returns (d0_pc, d0_err_pc, per_source)."""
    per_source = _per_source_d0(theta_calib_mas, z, expanding=True, cosmo=cosmo)
    d0 = float(np.nanmedian(per_source))
    se = float(1.2533 * np.nanstd(per_source) / np.sqrt(np.isfinite(per_source).sum()))
    return d0, se, per_source


def fit_d0_nonexpanding(
    theta_calib_mas: np.ndarray,
    z: np.ndarray,
    cosmo: FlatLambdaCDM = COSMO_LI,
) -> tuple[float, float, np.ndarray]:
    """Li eq. (9) single-parameter fit; returns (d0_pc, d0_err_pc, per_source)."""
    per_source = _per_source_d0(theta_calib_mas, z, expanding=False, cosmo=cosmo)
    d0 = float(np.nanmedian(per_source))
    se = float(1.2533 * np.nanstd(per_source) / np.sqrt(np.isfinite(per_source).sum()))
    return d0, se, per_source


def distance_duality_ratio(
    theta_calib_mas: np.ndarray,
    z: np.ndarray,
    d0_pc: float,
    cosmo: FlatLambdaCDM = COSMO_LI,
) -> np.ndarray:
    """Li 2023 Fig. 4 ordinate: D_L/D_A computed entirely from the data.

    D_A = d0 / theta_calib (geometric inversion); D_L from SNe Ia cosmology.
    """
    z = np.asarray(z, dtype=float)
    theta_calib_rad = np.asarray(theta_calib_mas, dtype=float) * RAD_PER_MAS
    D_A_pc = d0_pc / theta_calib_rad
    D_L_pc = cosmo.luminosity_distance(z).to("pc").value
    return D_L_pc / D_A_pc


def bin_median(x: np.ndarray, y: np.ndarray, *, n_bins: int) -> pd.DataFrame:
    """Equal-population binning of y vs x and the per-bin median.

    Mirrors Li's binning approach for Fig. 3 and 4: nearly equal population
    per bin and per-bin median values with sigma_med = 1.2533 * sigma / sqrt(N).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    order = np.argsort(x)
    x, y = x[order], y[order]
    edges_idx = np.linspace(0, len(x), n_bins + 1, dtype=int)
    rows = []
    for i in range(n_bins):
        lo, hi = edges_idx[i], edges_idx[i + 1]
        if hi - lo < 2:
            continue
        xb, yb = x[lo:hi], y[lo:hi]
        rows.append(
            dict(
                x_median=float(np.median(xb)),
                y_median=float(np.median(yb)),
                y_sigma_med=float(1.2533 * np.std(yb, ddof=1) / np.sqrt(len(yb))),
                n=int(len(yb)),
            )
        )
    return pd.DataFrame(rows)

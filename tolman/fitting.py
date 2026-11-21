"""Power-law fitting for surface-brightness evolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FitResult:
    alpha: float
    beta: float
    alpha_err: float
    beta_err: float
    n: int
    scatter: float
    x_min: float
    x_max: float
    observable: str = ""
    sample: str = ""
    selection: str = ""

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def finite_power_law_inputs(z: np.ndarray, sb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float)
    sb = np.asarray(sb, dtype=float)
    mask = np.isfinite(z) & np.isfinite(sb) & (z > -1.0) & (sb > 0.0)
    return np.log10(1.0 + z[mask]), np.log10(sb[mask])


def fit_power_law(
    z: np.ndarray,
    sb: np.ndarray,
    *,
    y_sigma: np.ndarray | None = None,
    observable: str = "",
    sample: str = "",
    selection: str = "",
) -> FitResult:
    """Fit log10(SB) = beta + alpha log10(1 + z)."""

    z_arr = np.asarray(z, dtype=float)
    sb_arr = np.asarray(sb, dtype=float)
    mask = np.isfinite(z_arr) & np.isfinite(sb_arr) & (z_arr > -1.0) & (sb_arr > 0.0)
    if y_sigma is not None:
        sigma_arr = np.asarray(y_sigma, dtype=float)
        mask &= np.isfinite(sigma_arr) & (sigma_arr > 0.0)
    if mask.sum() < 3:
        return FitResult(
            alpha=np.nan,
            beta=np.nan,
            alpha_err=np.nan,
            beta_err=np.nan,
            n=int(mask.sum()),
            scatter=np.nan,
            x_min=np.nan,
            x_max=np.nan,
            observable=observable,
            sample=sample,
            selection=selection,
        )

    x = np.log10(1.0 + z_arr[mask])
    y = np.log10(sb_arr[mask])
    design = np.column_stack([np.ones_like(x), x])

    if y_sigma is None:
        weights = np.ones_like(y)
    else:
        sigma = np.asarray(y_sigma, dtype=float)[mask]
        weights = 1.0 / np.square(sigma)

    xtw = design.T * weights
    normal = xtw @ design
    beta_alpha = np.linalg.solve(normal, xtw @ y)
    beta, alpha = beta_alpha
    resid = y - design @ beta_alpha
    dof = max(len(y) - 2, 1)
    chi2 = float(np.sum(weights * resid * resid))
    scale = chi2 / dof
    cov = np.linalg.inv(normal) * scale
    beta_err, alpha_err = np.sqrt(np.diag(cov))

    return FitResult(
        alpha=float(alpha),
        beta=float(beta),
        alpha_err=float(alpha_err),
        beta_err=float(beta_err),
        n=int(len(y)),
        scatter=float(np.std(resid, ddof=2)) if len(y) > 2 else np.nan,
        x_min=float(np.min(x)),
        x_max=float(np.max(x)),
        observable=observable,
        sample=sample,
        selection=selection,
    )


def bootstrap_alpha(
    z: np.ndarray,
    sb: np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    x, y = finite_power_law_inputs(z, sb)
    if len(y) < 3:
        return {"alpha_boot_mean": np.nan, "alpha_boot_std": np.nan, "alpha_p16": np.nan, "alpha_p84": np.nan}

    rng = np.random.default_rng(seed)
    alphas = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        take = rng.integers(0, len(y), len(y))
        fit = fit_power_law(10.0 ** x[take] - 1.0, 10.0 ** y[take])
        alphas[idx] = fit.alpha
    return {
        "alpha_boot_mean": float(np.nanmean(alphas)),
        "alpha_boot_std": float(np.nanstd(alphas, ddof=1)),
        "alpha_p16": float(np.nanpercentile(alphas, 16.0)),
        "alpha_p84": float(np.nanpercentile(alphas, 84.0)),
    }


def fit_results_table(results: list[FitResult]) -> pd.DataFrame:
    return pd.DataFrame([item.as_dict() for item in results])


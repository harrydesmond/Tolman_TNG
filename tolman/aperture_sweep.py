"""Aperture radius sweep diagnostic for TNG image apertures.

Measures circular aperture fluxes at multiple scale factors of the nominal
photrad_kpc aperture radius to quantify how the aperture/catalogue-flux ratio
depends on aperture size and redshift.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from .fitting import fit_power_law
from .image_photometry import circular_aperture_sum, image_unit_to_ujy_factor
from .io import write_table
from .mast_illustris import filter_name_from_path
from .plots import archive_existing
from .tng_image_aperture import _pixel_scale_arcsec


_DEFAULT_SCALE_FACTORS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _sweep_task(args):
    return _sweep_one_filter(*args)


def _sweep_one_filter(
    df: pd.DataFrame,
    path_str: str,
    selected_col: str,
    scale_factors: tuple[float, ...],
) -> pd.DataFrame | None:
    path = Path(path_str)
    filter_name = filter_name_from_path(path)

    x_col = f"x_{filter_name}_pix"
    y_col = f"y_{filter_name}_pix"
    cat_col = f"catalog_flux_{filter_name}_uJy"
    r_col = f"aperture_radius_arcsec_{filter_name}"

    if any(c not in df.columns for c in (x_col, y_col, cat_col, r_col)):
        return None

    with fits.open(path, memmap=True) as hdul:
        image = np.array(hdul[0].data, dtype=np.float32, copy=True)
        header = hdul[0].header
        pixel_scale = _pixel_scale_arcsec(header, None)
        unit_factor = image_unit_to_ujy_factor(header.get("BUNIT"))

    selected = df[df[selected_col].astype(bool)].copy()
    x = pd.to_numeric(selected[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(selected[y_col], errors="coerce").to_numpy(dtype=float)
    nominal_r = pd.to_numeric(selected[r_col], errors="coerce").to_numpy(dtype=float)
    cat_flux = pd.to_numeric(selected[cat_col], errors="coerce").to_numpy(dtype=float)
    z_vals = pd.to_numeric(selected["z"], errors="coerce").to_numpy(dtype=float)
    obj_idx = selected.index.to_numpy()

    rows: list[dict] = []
    for sf in scale_factors:
        r_arcsec = nominal_r * sf
        r_pix = r_arcsec / pixel_scale
        flux = (
            np.array(
                [circular_aperture_sum(image, xi, yi, ri) for xi, yi, ri in zip(x, y, r_pix)],
                dtype=float,
            )
            * unit_factor
        )
        flux = np.where(np.isfinite(flux) & (flux > 0.0), flux, np.nan)
        valid_cat = np.where(cat_flux > 0.0, cat_flux, np.nan)
        ratio = flux / valid_cat

        for i in range(len(selected)):
            rows.append(
                {
                    "object_index": int(obj_idx[i]),
                    "filter": filter_name,
                    "z": float(z_vals[i]) if np.isfinite(z_vals[i]) else np.nan,
                    "scale_factor": float(sf),
                    "nominal_r_arcsec": float(nominal_r[i]) if np.isfinite(nominal_r[i]) else np.nan,
                    "r_arcsec": float(r_arcsec[i]) if np.isfinite(r_arcsec[i]) else np.nan,
                    "catalog_flux_uJy": float(cat_flux[i]) if np.isfinite(cat_flux[i]) else np.nan,
                    "aperture_flux_uJy": float(flux[i]) if np.isfinite(flux[i]) else np.nan,
                    "aperture_to_catalogue_ratio": float(ratio[i]) if np.isfinite(ratio[i]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _object_medians(per_filter: pd.DataFrame) -> pd.DataFrame:
    """Median ratio per object per scale factor."""
    grouped = per_filter.groupby(["object_index", "scale_factor"])
    med = grouped["aperture_to_catalogue_ratio"].median().reset_index()
    z_med = per_filter.groupby("object_index")["z"].median().reset_index()
    return med.merge(z_med, on="object_index")


def _redshift_bins_by_scale(
    objects: pd.DataFrame,
    *,
    n_bins: int = 8,
) -> pd.DataFrame:
    z = pd.to_numeric(objects["z"], errors="coerce")
    x = np.log10(1.0 + z)
    finite_x = np.isfinite(x)
    if not finite_x.any():
        return pd.DataFrame()
    edges = np.linspace(float(np.nanmin(x[finite_x])), float(np.nanmax(x[finite_x])), n_bins + 1)

    rows: list[dict] = []
    for scale in objects["scale_factor"].unique():
        sub = objects[objects["scale_factor"] == scale].copy()
        sub_x = np.log10(1.0 + pd.to_numeric(sub["z"], errors="coerce"))
        for idx in range(n_bins):
            lo, hi = edges[idx], edges[idx + 1]
            in_bin = (sub_x >= lo) & (sub_x < hi if idx < n_bins - 1 else sub_x <= hi)
            if in_bin.sum() < 1:
                continue
            ratio_vals = sub.loc[in_bin.values, "aperture_to_catalogue_ratio"]
            finite = np.isfinite(ratio_vals)
            if not finite.any():
                continue
            rows.append(
                {
                    "scale_factor": float(scale),
                    "bin": idx,
                    "z_mid": float(10.0 ** (0.5 * (lo + hi)) - 1.0),
                    "n": int(in_bin.sum()),
                    "median_ratio": float(np.nanmedian(ratio_vals[finite])),
                    "p16_ratio": float(np.nanpercentile(ratio_vals[finite], 16.0)),
                    "p84_ratio": float(np.nanpercentile(ratio_vals[finite], 84.0)),
                }
            )
    return pd.DataFrame(rows)


def _slopes_by_scale(objects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for scale in sorted(objects["scale_factor"].unique()):
        sub = objects[objects["scale_factor"] == scale].copy()
        fit = fit_power_law(sub["z"].to_numpy(), sub["aperture_to_catalogue_ratio"].to_numpy())
        rows.append(
            {
                "scale_factor": float(scale),
                "alpha": fit.alpha,
                "alpha_err": fit.alpha_err,
                "beta": fit.beta,
                "n": fit.n,
            }
        )
    return pd.DataFrame(rows)


def _plot_ratio_vs_z_bins(bins: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = archive_existing(path)
    scales = sorted(bins["scale_factor"].unique())
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(scales) - 1, 1)) for i in range(len(scales))]

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    for scale, color in zip(scales, colors):
        sub = bins[bins["scale_factor"] == scale].sort_values("z_mid")
        ax.plot(
            sub["z_mid"],
            sub["median_ratio"],
            marker="o",
            ms=4,
            color=color,
            label=f"×{scale:.1f}",
        )
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Aperture flux / catalogue flux")
    ax.legend(
        frameon=False, fontsize=7, title="Aperture scale", title_fontsize=7,
        ncols=2, loc="best",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_slope_vs_scale(slopes: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = archive_existing(path)
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.errorbar(
        slopes["scale_factor"],
        slopes["alpha"],
        yerr=slopes["alpha_err"],
        fmt="o-",
        color="#1f77b4",
        capsize=3,
    )
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Aperture scale factor (× nominal photrad_kpc radius)")
    ax.set_ylabel("Redshift slope alpha of aperture/catalogue ratio")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_aperture_sweep(
    catalog: str | Path,
    fits_paths: list[str | Path],
    *,
    out_dir: str | Path,
    plot_dir: str | Path,
    sample_name: str = "tng100_imageap_sweep",
    selected_col: str = "mock_spec_selected",
    scale_factors: tuple[float, ...] = _DEFAULT_SCALE_FACTORS,
    n_workers: int = 1,
) -> dict[str, pd.DataFrame]:
    from .io import read_catalog

    df = read_catalog(catalog)
    out_dir = Path(out_dir)
    plot_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(df, str(p), selected_col, scale_factors) for p in fits_paths]

    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_sweep_task, tasks))
    else:
        results = [_sweep_task(t) for t in tasks]

    per_filter_parts = [r for r in results if r is not None]
    if not per_filter_parts:
        raise RuntimeError("No valid filter data returned from aperture sweep")

    per_filter = pd.concat(per_filter_parts, ignore_index=True)
    objects = _object_medians(per_filter)
    bins = _redshift_bins_by_scale(objects)
    slopes = _slopes_by_scale(objects)

    write_table(per_filter, out_dir / "aperture_sweep_per_filter.csv")
    write_table(objects, out_dir / "aperture_sweep_object_summary.csv")
    write_table(bins, out_dir / "aperture_sweep_redshift_bins.csv")
    write_table(slopes, out_dir / "aperture_sweep_slope_table.csv")

    _plot_ratio_vs_z_bins(bins, plot_dir / f"{sample_name}_ratio_vs_z_bins.png")
    _plot_slope_vs_scale(slopes, plot_dir / f"{sample_name}_slope_vs_scale.png")

    lines = [
        "# Aperture Radius Sweep Diagnostic",
        "",
        f"Scale factors tested: {list(scale_factors)}",
        f"Selected objects: {objects['object_index'].nunique():,}",
        "",
        "| Scale factor | alpha | N |",
        "|---:|---:|---:|",
    ]
    for row in slopes.itertuples(index=False):
        lines.append(f"| ×{row.scale_factor:.1f} | {row.alpha:.3f} ± {row.alpha_err:.3f} | {row.n:d} |")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "per_filter": per_filter,
        "objects": objects,
        "bins": bins,
        "slopes": slopes,
    }

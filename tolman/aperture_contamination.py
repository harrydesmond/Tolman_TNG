"""Local-annulus contamination diagnostics for TNG image apertures."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from .fitting import fit_power_law
from .image_photometry import circular_aperture_sum, image_unit_to_ujy_factor
from .io import read_catalog, write_table
from .mast_illustris import filter_name_from_path
from .plots import archive_existing


def _annulus_task(args):
    df, path, inner_factor, outer_factor = args
    return _annulus_correct_one_filter(df, path, inner_factor=inner_factor, outer_factor=outer_factor)


def _annulus_correct_one_filter(
    df: pd.DataFrame,
    image_path: str | Path,
    *,
    inner_factor: float,
    outer_factor: float,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    path = Path(image_path)
    filter_name = filter_name_from_path(path)
    x_col = f"x_{filter_name}_pix"
    y_col = f"y_{filter_name}_pix"
    r_col = f"aperture_radius_pix_{filter_name}"
    flux_col = f"flux_{filter_name}_uJy"
    cat_col = f"catalog_flux_{filter_name}_uJy"
    needed = [x_col, y_col, r_col, flux_col, cat_col, "z", "mock_spec_selected"]
    missing = [col for col in needed if col not in df]
    if missing:
        raise KeyError(f"Missing columns for {filter_name}: {missing}")

    selected = df["mock_spec_selected"].astype(bool)
    work = df.loc[selected, needed].copy()
    x = pd.to_numeric(work[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(work[y_col], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(work[r_col], errors="coerce").to_numpy(dtype=float)

    with fits.open(path, memmap=True) as hdul:
        image = np.array(hdul[0].data, dtype=np.float32, copy=True)
        unit_factor = image_unit_to_ujy_factor(hdul[0].header.get("BUNIT"))

    inner = inner_factor * r
    outer = outer_factor * r
    inner_sum = np.array(
        [circular_aperture_sum(image, xi, yi, ri) for xi, yi, ri in zip(x, y, inner, strict=True)],
        dtype=float,
    )
    outer_sum = np.array(
        [circular_aperture_sum(image, xi, yi, ro) for xi, yi, ro in zip(x, y, outer, strict=True)],
        dtype=float,
    )
    annulus_flux = (outer_sum - inner_sum) * unit_factor
    annulus_area_pix = np.pi * (outer * outer - inner * inner)
    aperture_area_pix = np.pi * r * r
    local_flux = annulus_flux * aperture_area_pix / annulus_area_pix

    aperture_flux = pd.to_numeric(work[flux_col], errors="coerce").to_numpy(dtype=float)
    catalog_flux = pd.to_numeric(work[cat_col], errors="coerce").to_numpy(dtype=float)
    corrected = aperture_flux - local_flux
    ratio = aperture_flux / catalog_flux
    local_ratio = local_flux / catalog_flux
    corrected_ratio = corrected / catalog_flux

    out = pd.DataFrame(
        {
            "filter": filter_name,
            "z": pd.to_numeric(work["z"], errors="coerce").to_numpy(dtype=float),
            "catalog_flux_uJy": catalog_flux,
            "aperture_flux_uJy": aperture_flux,
            "local_annulus_flux_in_aperture_uJy": local_flux,
            "annulus_corrected_flux_uJy": corrected,
            "aperture_to_catalogue_ratio": ratio,
            "local_annulus_to_catalogue_ratio": local_ratio,
            "annulus_corrected_to_catalogue_ratio": corrected_ratio,
        }
    )
    finite = np.isfinite(ratio) & (ratio > 0.0)
    finite_corr = np.isfinite(corrected_ratio) & (corrected_ratio > 0.0)
    fit_raw = fit_power_law(out.loc[finite, "z"].to_numpy(), out.loc[finite, "aperture_to_catalogue_ratio"].to_numpy())
    fit_corr = fit_power_law(
        out.loc[finite_corr, "z"].to_numpy(),
        out.loc[finite_corr, "annulus_corrected_to_catalogue_ratio"].to_numpy(),
    )
    summary = {
        "filter": filter_name,
        "n": int(len(out)),
        "n_positive_corrected": int(finite_corr.sum()),
        "median_aperture_to_catalogue_ratio": float(np.nanmedian(ratio[finite])) if finite.any() else np.nan,
        "median_local_annulus_to_catalogue_ratio": float(np.nanmedian(local_ratio[np.isfinite(local_ratio)])),
        "median_annulus_corrected_to_catalogue_ratio": float(np.nanmedian(corrected_ratio[finite_corr]))
        if finite_corr.any()
        else np.nan,
        "frac_aperture_ratio_gt_1": float(np.mean(ratio[finite] > 1.0)) if finite.any() else np.nan,
        "frac_corrected_ratio_gt_1": float(np.mean(corrected_ratio[finite_corr] > 1.0)) if finite_corr.any() else np.nan,
        "alpha_raw_ratio": fit_raw.alpha,
        "alpha_raw_ratio_err": fit_raw.alpha_err,
        "alpha_corrected_ratio": fit_corr.alpha,
        "alpha_corrected_ratio_err": fit_corr.alpha_err,
    }
    return out, summary


def _object_summary(per_filter: pd.DataFrame) -> pd.DataFrame:
    grouped = per_filter.groupby(per_filter.index)
    out = pd.DataFrame(
        {
            "z": grouped["z"].median(),
            "median_aperture_to_catalogue_ratio": grouped["aperture_to_catalogue_ratio"].median(),
            "median_local_annulus_to_catalogue_ratio": grouped["local_annulus_to_catalogue_ratio"].median(),
            "median_annulus_corrected_to_catalogue_ratio": grouped["annulus_corrected_to_catalogue_ratio"].median(),
            "n_filters": grouped["annulus_corrected_to_catalogue_ratio"].count(),
        }
    )
    return out.reset_index(drop=True)


def _redshift_bins(objects: pd.DataFrame, *, n_bins: int = 8) -> pd.DataFrame:
    z = pd.to_numeric(objects["z"], errors="coerce")
    x = np.log10(1.0 + z)
    finite_x = np.isfinite(x)
    edges = np.linspace(float(np.nanmin(x[finite_x])), float(np.nanmax(x[finite_x])), n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for idx in range(n_bins):
        if idx == n_bins - 1:
            in_bin = (x >= edges[idx]) & (x <= edges[idx + 1])
        else:
            in_bin = (x >= edges[idx]) & (x < edges[idx + 1])
        if in_bin.sum() < 1:
            continue
        sub = objects.loc[in_bin]
        rows.append(
            {
                "bin": idx,
                "z_mid": float(10.0 ** (0.5 * (edges[idx] + edges[idx + 1])) - 1.0),
                "n": int(len(sub)),
                "median_raw_ratio": float(np.nanmedian(sub["median_aperture_to_catalogue_ratio"])),
                "median_local_annulus_ratio": float(np.nanmedian(sub["median_local_annulus_to_catalogue_ratio"])),
                "median_corrected_ratio": float(np.nanmedian(sub["median_annulus_corrected_to_catalogue_ratio"])),
                "frac_raw_gt_1": float(np.nanmean(sub["median_aperture_to_catalogue_ratio"] > 1.0)),
                "frac_corrected_gt_1": float(np.nanmean(sub["median_annulus_corrected_to_catalogue_ratio"] > 1.0)),
            }
        )
    return pd.DataFrame(rows)


def _plot_contamination_bins(bins: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = archive_existing(path)
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    ax.plot(bins["z_mid"], bins["median_raw_ratio"], marker="o", label="raw aperture")
    ax.plot(bins["z_mid"], bins["median_corrected_ratio"], marker="s", label="annulus-subtracted")
    ax.plot(bins["z_mid"], bins["median_local_annulus_ratio"], marker="^", label="annulus estimate")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Flux / catalogue flux")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_annulus_contamination_diagnostics(
    catalog: str | Path,
    fits_paths: list[str | Path],
    *,
    out_dir: str | Path,
    plot_dir: str | Path,
    sample_name: str = "tng100_imageap_annulus",
    inner_factor: float = 2.0,
    outer_factor: float = 3.0,
    n_workers: int = 1,
) -> dict[str, pd.DataFrame]:
    df = read_catalog(catalog)
    out_dir = Path(out_dir)
    plot_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(df, path, inner_factor, outer_factor) for path in fits_paths]
    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            measured = list(pool.map(_annulus_task, tasks))
    else:
        measured = [_annulus_task(task) for task in tasks]
    per_filter_rows = [rows for rows, _ in measured]
    summaries = [summary for _, summary in measured]
    per_filter = pd.concat(per_filter_rows, ignore_index=False)
    summary_table = pd.DataFrame(summaries).sort_values("filter")
    objects = _object_summary(per_filter)
    bins = _redshift_bins(objects)

    raw_fit = fit_power_law(objects["z"].to_numpy(), objects["median_aperture_to_catalogue_ratio"].to_numpy())
    corrected_fit = fit_power_law(
        objects["z"].to_numpy(),
        objects["median_annulus_corrected_to_catalogue_ratio"].to_numpy(),
    )
    fit_table = pd.DataFrame(
        [
            {"quantity": "median_raw_ratio", **raw_fit.as_dict()},
            {"quantity": "median_annulus_corrected_ratio", **corrected_fit.as_dict()},
        ]
    )

    write_table(per_filter.reset_index(drop=True), out_dir / "annulus_per_filter_objects.csv")
    write_table(summary_table, out_dir / "annulus_filter_summary.csv")
    write_table(objects, out_dir / "annulus_object_summary.csv")
    write_table(bins, out_dir / "annulus_redshift_bins.csv")
    write_table(fit_table, out_dir / "annulus_ratio_fits.csv")
    _plot_contamination_bins(bins, plot_dir / f"{sample_name}_annulus_redshift_bins.png")

    lines = [
        "# TNG Image-Aperture Annulus Contamination Diagnostic",
        "",
        f"Annulus: `{inner_factor:.1f} r` to `{outer_factor:.1f} r` around each selected object.",
        f"Selected objects with object-level summaries: {len(objects):,}.",
        "",
        "| Quantity | alpha | N |",
        "|---|---:|---:|",
    ]
    for row in fit_table.itertuples(index=False):
        lines.append(f"| {row.quantity} | {row.alpha:.3f} +/- {row.alpha_err:.3f} | {row.n:d} |")
    lines.extend(
        [
            "",
            "Caveat: this is a local-annulus subtraction diagnostic, not a replacement for segmentation/deblending.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "per_filter": per_filter,
        "summary": summary_table,
        "objects": objects,
        "bins": bins,
        "fits": fit_table,
    }

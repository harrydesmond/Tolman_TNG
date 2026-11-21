"""Threshold-segmentation diagnostics for TNG image apertures."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from .fitting import fit_power_law
from .image_detection import segment_sources
from .image_photometry import image_unit_to_ujy_factor
from .io import read_catalog, write_table
from .mast_illustris import filter_name_from_path
from .plots import archive_existing


def _aperture_radius_col(df: pd.DataFrame, filter_name: str) -> str:
    candidates = [
        f"aperture_radius_pix_{filter_name}",
        f"image_aperture_radius_pix_{filter_name}",
    ]
    for col in candidates:
        if col in df:
            return col
    raise KeyError(f"Missing aperture-radius pixel column for {filter_name}: tried {candidates}")


def _assign_source_ids(labels: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.full(len(x), -1, dtype=np.int64)
    finite = np.isfinite(x) & np.isfinite(y)
    ix = np.floor(x[finite]).astype(np.int64)
    iy = np.floor(y[finite]).astype(np.int64)
    inside = (ix >= 0) & (iy >= 0) & (ix < labels.shape[1]) & (iy < labels.shape[0])
    finite_idx = np.flatnonzero(finite)
    assigned = labels[iy[inside], ix[inside]].astype(np.int64) - 1
    out[finite_idx[inside]] = assigned
    out[out < 0] = -1
    return out


def _take_counts(source_ids: np.ndarray, counts: np.ndarray) -> np.ndarray:
    out = np.zeros(len(source_ids), dtype=np.int64)
    valid = source_ids >= 0
    out[valid] = counts[source_ids[valid]]
    return out


def _positive_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(len(numerator), np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0.0)
    np.divide(numerator, denominator, out=out, where=valid)
    return out


def _nanmedian(values) -> float:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.nan
    return float(np.nanmedian(arr[finite]))


def _segmentation_task(args):
    return _segment_one_filter(*args)


def _segment_one_filter(
    df: pd.DataFrame,
    image_path: str | Path,
    selected_col: str,
    threshold_sigma: float,
    min_pixels: int,
    edge_buffer_pix: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    path = Path(image_path)
    filter_name = filter_name_from_path(path)
    x_col = f"x_{filter_name}_pix"
    y_col = f"y_{filter_name}_pix"
    flux_col = f"flux_{filter_name}_uJy"
    cat_col = f"catalog_flux_{filter_name}_uJy"
    r_col = _aperture_radius_col(df, filter_name)
    needed = [x_col, y_col, r_col, flux_col, cat_col, "z", selected_col]
    missing = [col for col in needed if col not in df]
    if missing:
        raise KeyError(f"Missing columns for {filter_name}: {missing}")

    with fits.open(path, memmap=True) as hdul:
        image = np.array(hdul[0].data, dtype=np.float32, copy=True)
        unit_factor = image_unit_to_ujy_factor(hdul[0].header.get("BUNIT"))
    detections, labels, bg, sigma = segment_sources(
        image,
        threshold_sigma=threshold_sigma,
        min_pixels=min_pixels,
        edge_buffer_pix=edge_buffer_pix,
    )
    detections = detections.copy()
    if len(detections):
        detections["filter"] = filter_name
        detections["background"] = bg
        detections["background_sigma"] = sigma
        detections["detection_flux_uJy"] = detections["detection_flux"].to_numpy(dtype=float) * unit_factor
        source_flux = detections["detection_flux_uJy"].to_numpy(dtype=float)
        source_edge = detections["is_edge"].to_numpy(dtype=bool)
        source_area = detections["seg_area_pix"].to_numpy(dtype=float)
    else:
        source_flux = np.array([], dtype=float)
        source_edge = np.array([], dtype=bool)
        source_area = np.array([], dtype=float)

    all_x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    all_y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    all_source_ids = _assign_source_ids(labels, all_x, all_y)
    n_sources = len(source_flux)
    all_counts = np.bincount(all_source_ids[all_source_ids >= 0], minlength=n_sources)
    selected_mask = df[selected_col].astype(bool).to_numpy()
    selected_source_ids = all_source_ids[selected_mask]
    selected_counts = np.bincount(selected_source_ids[selected_source_ids >= 0], minlength=n_sources)

    work = df.loc[selected_mask, needed].copy()
    object_index = work.index.to_numpy(dtype=np.int64)
    source_ids = selected_source_ids
    has_segment = source_ids >= 0
    segment_flux = np.full(len(work), np.nan, dtype=float)
    is_edge = np.full(len(work), False, dtype=bool)
    seg_area = np.full(len(work), np.nan, dtype=float)
    if has_segment.any():
        segment_flux[has_segment] = source_flux[source_ids[has_segment]]
        is_edge[has_segment] = source_edge[source_ids[has_segment]]
        seg_area[has_segment] = source_area[source_ids[has_segment]]

    aperture_flux = pd.to_numeric(work[flux_col], errors="coerce").to_numpy(dtype=float)
    catalog_flux = pd.to_numeric(work[cat_col], errors="coerce").to_numpy(dtype=float)
    aperture_ratio = _positive_ratio(aperture_flux, catalog_flux)
    segment_ratio = _positive_ratio(segment_flux, catalog_flux)
    n_all = _take_counts(source_ids, all_counts)
    n_selected = _take_counts(source_ids, selected_counts)
    isolated_all = has_segment & (n_all == 1)
    isolated_selected = has_segment & (n_selected == 1)
    isolated_ratio = np.where(isolated_all, segment_ratio, np.nan)

    per_object_filter = pd.DataFrame(
        {
            "object_index": object_index,
            "filter": filter_name,
            "z": pd.to_numeric(work["z"], errors="coerce").to_numpy(dtype=float),
            "catalog_flux_uJy": catalog_flux,
            "aperture_flux_uJy": aperture_flux,
            "segment_flux_uJy": segment_flux,
            "aperture_to_catalogue_ratio": aperture_ratio,
            "segment_to_catalogue_ratio": segment_ratio,
            "isolated_segment_to_catalogue_ratio": isolated_ratio,
            "assigned_source_id": source_ids,
            "has_segment": has_segment,
            "is_edge_segment": is_edge,
            "seg_area_pix": seg_area,
            "n_catalogue_objects_in_segment": n_all,
            "n_selected_objects_in_segment": n_selected,
            "is_isolated_catalogue_segment": isolated_all,
            "is_isolated_selected_segment": isolated_selected,
        }
    )

    finite_segment = np.isfinite(segment_ratio) & (segment_ratio > 0.0)
    finite_isolated = np.isfinite(isolated_ratio) & (isolated_ratio > 0.0)
    finite_aperture = np.isfinite(aperture_ratio) & (aperture_ratio > 0.0)
    fit_ap = fit_power_law(per_object_filter.loc[finite_aperture, "z"], aperture_ratio[finite_aperture])
    fit_seg = fit_power_law(per_object_filter.loc[finite_segment, "z"], segment_ratio[finite_segment])
    fit_iso = fit_power_law(per_object_filter.loc[finite_isolated, "z"], isolated_ratio[finite_isolated])
    summary = {
        "filter": filter_name,
        "n_detections": int(len(detections)),
        "background": float(bg),
        "background_sigma": float(sigma),
        "n_selected": int(len(work)),
        "n_assigned_segment": int(has_segment.sum()),
        "frac_assigned_segment": float(np.mean(has_segment)) if len(work) else np.nan,
        "frac_isolated_catalogue_segment": float(np.mean(isolated_all)) if len(work) else np.nan,
        "frac_isolated_selected_segment": float(np.mean(isolated_selected)) if len(work) else np.nan,
        "median_aperture_to_catalogue_ratio": _nanmedian(aperture_ratio[finite_aperture]),
        "median_segment_to_catalogue_ratio": _nanmedian(segment_ratio[finite_segment]),
        "median_isolated_segment_to_catalogue_ratio": _nanmedian(isolated_ratio[finite_isolated]),
        "alpha_aperture_ratio": fit_ap.alpha,
        "alpha_aperture_ratio_err": fit_ap.alpha_err,
        "alpha_segment_ratio": fit_seg.alpha,
        "alpha_segment_ratio_err": fit_seg.alpha_err,
        "alpha_isolated_segment_ratio": fit_iso.alpha,
        "alpha_isolated_segment_ratio_err": fit_iso.alpha_err,
    }
    return per_object_filter, detections, summary


def _object_summary(per_filter: pd.DataFrame) -> pd.DataFrame:
    grouped = per_filter.groupby("object_index")
    objects = pd.DataFrame(
        {
            "z": grouped["z"].median(),
            "median_aperture_to_catalogue_ratio": grouped["aperture_to_catalogue_ratio"].median(),
            "median_segment_to_catalogue_ratio": grouped["segment_to_catalogue_ratio"].median(),
            "median_isolated_segment_to_catalogue_ratio": grouped["isolated_segment_to_catalogue_ratio"].median(),
            "n_filters": grouped["filter"].count(),
            "n_segment_filters": grouped["segment_to_catalogue_ratio"].count(),
            "n_isolated_segment_filters": grouped["isolated_segment_to_catalogue_ratio"].count(),
            "frac_segment_filters": grouped["has_segment"].mean(),
            "frac_isolated_catalogue_segment_filters": grouped["is_isolated_catalogue_segment"].mean(),
            "frac_edge_segment_filters": grouped["is_edge_segment"].mean(),
        }
    )
    return objects.reset_index()


def _redshift_bins(objects: pd.DataFrame, *, n_bins: int = 8) -> pd.DataFrame:
    z = pd.to_numeric(objects["z"], errors="coerce")
    x = np.log10(1.0 + z)
    finite_x = np.isfinite(x)
    edges = np.linspace(float(np.nanmin(x[finite_x])), float(np.nanmax(x[finite_x])), n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for idx in range(n_bins):
        in_bin = (x >= edges[idx]) & (x <= edges[idx + 1] if idx == n_bins - 1 else x < edges[idx + 1])
        if in_bin.sum() < 1:
            continue
        sub = objects.loc[in_bin]
        rows.append(
            {
                "bin": idx,
                "z_mid": float(10.0 ** (0.5 * (edges[idx] + edges[idx + 1])) - 1.0),
                "n": int(len(sub)),
                "median_aperture_ratio": _nanmedian(sub["median_aperture_to_catalogue_ratio"]),
                "median_segment_ratio": _nanmedian(sub["median_segment_to_catalogue_ratio"]),
                "median_isolated_segment_ratio": _nanmedian(sub["median_isolated_segment_to_catalogue_ratio"]),
                "median_frac_segment_filters": _nanmedian(sub["frac_segment_filters"]),
                "median_frac_isolated_catalogue_segment_filters": _nanmedian(
                    sub["frac_isolated_catalogue_segment_filters"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_segmentation_bins(bins: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = archive_existing(path)
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    ax.plot(bins["z_mid"], bins["median_aperture_ratio"], marker="o", label="circular aperture")
    ax.plot(bins["z_mid"], bins["median_segment_ratio"], marker="s", label="threshold segment")
    ax.plot(bins["z_mid"], bins["median_isolated_segment_ratio"], marker="^", label="isolated segment")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Flux / catalogue flux")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_segmentation_diagnostics(
    catalog: str | Path,
    fits_paths: list[str | Path],
    *,
    out_dir: str | Path,
    plot_dir: str | Path,
    sample_name: str = "tng100_imageap_segmentation",
    selected_col: str = "mock_spec_selected",
    threshold_sigma: float = 3.0,
    min_pixels: int = 8,
    edge_buffer_pix: int = 3,
    n_workers: int = 1,
) -> dict[str, pd.DataFrame]:
    df = read_catalog(catalog)
    out_dir = Path(out_dir)
    plot_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(df, path, selected_col, threshold_sigma, min_pixels, edge_buffer_pix) for path in fits_paths]
    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            measured = list(pool.map(_segmentation_task, tasks))
    else:
        measured = [_segmentation_task(task) for task in tasks]

    per_filter = pd.concat([rows for rows, _, _ in measured], ignore_index=True)
    detections = pd.concat([det for _, det, _ in measured], ignore_index=True)
    summary_table = pd.DataFrame([summary for _, _, summary in measured]).sort_values("filter")
    objects = _object_summary(per_filter)
    bins = _redshift_bins(objects)

    ratio_columns = [
        ("median_aperture_ratio", "median_aperture_to_catalogue_ratio"),
        ("median_segment_ratio", "median_segment_to_catalogue_ratio"),
        ("median_isolated_segment_ratio", "median_isolated_segment_to_catalogue_ratio"),
    ]
    fit_rows = []
    for quantity, col in ratio_columns:
        fit = fit_power_law(objects["z"].to_numpy(), objects[col].to_numpy())
        fit_rows.append({"quantity": quantity, **fit.as_dict()})
    fit_table = pd.DataFrame(fit_rows)

    write_table(per_filter, out_dir / "segmentation_per_filter_objects.csv")
    write_table(detections, out_dir / "segmentation_detections.csv")
    write_table(summary_table, out_dir / "segmentation_filter_summary.csv")
    write_table(objects, out_dir / "segmentation_object_summary.csv")
    write_table(bins, out_dir / "segmentation_redshift_bins.csv")
    write_table(fit_table, out_dir / "segmentation_ratio_fits.csv")
    _plot_segmentation_bins(bins, plot_dir / f"{sample_name}_segmentation_redshift_bins.png")

    lines = [
        "# TNG Image Segmentation Diagnostic",
        "",
        f"Threshold: `background + {threshold_sigma:.1f} sigma`; minimum island size: `{min_pixels}` pixels.",
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
            "Caveat: this is threshold segmentation with catalogue-position label assignment, not a full ASTRODEEP deblender.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "per_filter": per_filter,
        "detections": detections,
        "summary": summary_table,
        "objects": objects,
        "bins": bins,
        "fits": fit_table,
    }

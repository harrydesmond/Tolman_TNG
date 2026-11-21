"""Voronoi deblending diagnostic for TNG image segments.

For each detected segment containing multiple catalogue objects, pixels are
assigned to their nearest catalogue position (Voronoi tessellation).  This
gives a per-object deblended flux without requiring photutils.

The main entry point is ``write_deblending_diagnostics``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import find_objects

from .fitting import fit_power_law
from .image_detection import segment_sources
from .image_photometry import image_unit_to_ujy_factor
from .io import write_table
from .mast_illustris import filter_name_from_path
from .plots import archive_existing


def _pixel_floor(x: np.ndarray, y: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return ix, iy, inside


def _deblend_one_filter(
    df: pd.DataFrame,
    image_path: str | Path,
    *,
    all_positions_col_x: str,
    all_positions_col_y: str,
    selected_col: str,
    cat_flux_col: str,
    aperture_flux_col: str,
    threshold_sigma: float,
    min_pixels: int,
    edge_buffer_pix: int,
) -> tuple[pd.DataFrame, dict]:
    path = Path(image_path)
    filter_name = filter_name_from_path(path)

    with fits.open(path, memmap=True) as hdul:
        image = np.array(hdul[0].data, dtype=np.float32, copy=True)
        unit_factor = image_unit_to_ujy_factor(hdul[0].header.get("BUNIT"))

    ny, nx = image.shape
    _, labels, bg, sigma = segment_sources(
        image,
        threshold_sigma=threshold_sigma,
        min_pixels=min_pixels,
        edge_buffer_pix=edge_buffer_pix,
    )

    # Pixel positions for ALL catalogue objects (for Voronoi partitioning)
    all_x = pd.to_numeric(df[all_positions_col_x], errors="coerce").to_numpy(dtype=float)
    all_y = pd.to_numeric(df[all_positions_col_y], errors="coerce").to_numpy(dtype=float)
    ix_all, iy_all, inside_all = _pixel_floor(all_x, all_y, (ny, nx))

    # Segment label per catalogue object (-1 if not in any segment)
    seg_id_all = np.full(len(df), -1, dtype=np.int64)
    seg_id_all[inside_all] = labels[iy_all[inside_all], ix_all[inside_all]] - 1  # 0-based

    # Pre-build: mapping from segment_id → list of all catalogue row indices in it
    seg_to_all: dict[int, list[int]] = {}
    for row_i, sid in enumerate(seg_id_all):
        if sid >= 0:
            seg_to_all.setdefault(int(sid), []).append(row_i)

    # Pre-compute bounding boxes from labels array
    bboxes = find_objects(labels)  # index i → slices for label i+1

    # Compute deblended flux — process each unique segment once to avoid O(n_selected²) work
    selected_mask = df[selected_col].astype(bool).to_numpy()
    selected_idx = np.flatnonzero(selected_mask)
    n_selected = len(selected_idx)

    cat_flux = pd.to_numeric(df[cat_flux_col], errors="coerce").to_numpy(dtype=float)
    ap_flux = pd.to_numeric(df.get(aperture_flux_col, pd.Series(np.nan, index=df.index)), errors="coerce").to_numpy(
        dtype=float
    )
    z_vals = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)

    deblended_flux = np.full(n_selected, np.nan, dtype=float)
    n_cat_in_seg = np.zeros(n_selected, dtype=np.int64)
    has_segment = np.zeros(n_selected, dtype=bool)

    # Group selected objects by their segment ID so we compute Voronoi once per segment
    from collections import defaultdict

    selected_by_seg: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for out_i, row_i in enumerate(selected_idx):
        sid = int(seg_id_all[row_i])
        if sid >= 0:
            has_segment[out_i] = True
            selected_by_seg[sid].append((out_i, int(row_i)))

    for sid, items in selected_by_seg.items():
        all_in_seg = seg_to_all.get(sid, [])
        n_cat = len(all_in_seg)
        for out_i, _ in items:
            n_cat_in_seg[out_i] = n_cat

        bbox = bboxes[sid]
        if bbox is None:
            continue
        sl_y, sl_x = bbox
        sub = image[sl_y, sl_x]
        sub_labels = labels[sl_y, sl_x]
        seg_mask = sub_labels == (sid + 1)
        pix_ys, pix_xs = np.nonzero(seg_mask)

        if n_cat == 1:
            total_flux = float(np.nansum(sub[pix_ys, pix_xs])) * unit_factor
            for out_i, _ in items:
                deblended_flux[out_i] = total_flux
        else:
            # Voronoi via KDTree: O(n_pixels) memory instead of O(n_pixels * n_cat)
            from scipy.spatial import KDTree

            cat_xs = all_x[all_in_seg] - sl_x.start
            cat_ys = all_y[all_in_seg] - sl_y.start
            cat_xy = np.column_stack([cat_xs, cat_ys])
            pix_xy = np.column_stack([pix_xs.astype(float), pix_ys.astype(float)])
            _, nearest = KDTree(cat_xy).query(pix_xy, workers=1)

            # Build per-cat-row deblended flux for the selected rows we need
            needed_row_ids = {row_i for _, row_i in items}
            for local_i, cat_row in enumerate(all_in_seg):
                if cat_row not in needed_row_ids:
                    continue
                my_pixels = nearest == local_i
                flux = float(np.nansum(sub[pix_ys[my_pixels], pix_xs[my_pixels]])) * unit_factor
                for out_i, row_i in items:
                    if row_i == cat_row:
                        deblended_flux[out_i] = flux

    # Build output frame for selected objects
    ap_flux_sel = np.where(
        np.isfinite(ap_flux[selected_idx]) & (ap_flux[selected_idx] > 0.0),
        ap_flux[selected_idx],
        np.nan,
    )
    cat_flux_sel = cat_flux[selected_idx]
    dbl_flux_clean = np.where(np.isfinite(deblended_flux) & (deblended_flux > 0.0), deblended_flux, np.nan)
    valid_cat = np.where(cat_flux_sel > 0.0, cat_flux_sel, np.nan)

    aperture_ratio = ap_flux_sel / valid_cat
    deblended_ratio = dbl_flux_clean / valid_cat

    per_object = pd.DataFrame(
        {
            "object_index": df.index[selected_idx],
            "filter": filter_name,
            "z": z_vals[selected_idx],
            "catalog_flux_uJy": cat_flux_sel,
            "aperture_flux_uJy": ap_flux_sel,
            "deblended_flux_uJy": dbl_flux_clean,
            "aperture_to_catalogue_ratio": aperture_ratio,
            "deblended_to_catalogue_ratio": deblended_ratio,
            "n_cat_in_segment": n_cat_in_seg,
            "has_segment": has_segment,
            "is_isolated": n_cat_in_seg == 1,
        }
    )

    def _nm(arr):
        finite = np.isfinite(arr)
        return float(np.nanmedian(arr[finite])) if finite.any() else np.nan

    fit_ap = fit_power_law(per_object["z"].to_numpy(), aperture_ratio)
    fit_dbl = fit_power_law(per_object["z"].to_numpy(), deblended_ratio)

    summary = {
        "filter": filter_name,
        "n_selected": n_selected,
        "n_has_segment": int(has_segment.sum()),
        "n_isolated": int((n_cat_in_seg == 1).sum()),
        "frac_isolated": float(np.mean(n_cat_in_seg[has_segment] == 1)) if has_segment.any() else np.nan,
        "median_aperture_ratio": _nm(aperture_ratio),
        "median_deblended_ratio": _nm(deblended_ratio),
        "alpha_aperture": fit_ap.alpha,
        "alpha_aperture_err": fit_ap.alpha_err,
        "alpha_deblended": fit_dbl.alpha,
        "alpha_deblended_err": fit_dbl.alpha_err,
    }
    return per_object, summary


def _object_summary(per_filter: pd.DataFrame) -> pd.DataFrame:
    grp = per_filter.groupby("object_index")
    return pd.DataFrame(
        {
            "z": grp["z"].median(),
            "median_aperture_ratio": grp["aperture_to_catalogue_ratio"].median(),
            "median_deblended_ratio": grp["deblended_to_catalogue_ratio"].median(),
            "n_filters": grp["filter"].count(),
            "mean_n_cat_in_segment": grp["n_cat_in_segment"].mean(),
            "frac_isolated_filters": grp["is_isolated"].mean(),
        }
    ).reset_index()


def _redshift_bins(objects: pd.DataFrame, *, n_bins: int = 8) -> pd.DataFrame:
    z = pd.to_numeric(objects["z"], errors="coerce")
    x = np.log10(1.0 + z)
    finite_x = np.isfinite(x)
    if not finite_x.any():
        return pd.DataFrame()
    edges = np.linspace(float(np.nanmin(x[finite_x])), float(np.nanmax(x[finite_x])), n_bins + 1)

    rows: list[dict] = []
    for idx in range(n_bins):
        lo, hi = edges[idx], edges[idx + 1]
        in_bin = (x >= lo) & (x < hi if idx < n_bins - 1 else x <= hi)
        if in_bin.sum() < 1:
            continue
        sub = objects.loc[in_bin]
        rows.append(
            {
                "bin": idx,
                "z_mid": float(10.0 ** (0.5 * (lo + hi)) - 1.0),
                "n": int(len(sub)),
                "median_aperture_ratio": float(
                    np.nanmedian(sub["median_aperture_ratio"].dropna())
                ) if sub["median_aperture_ratio"].notna().any() else np.nan,
                "median_deblended_ratio": float(
                    np.nanmedian(sub["median_deblended_ratio"].dropna())
                ) if sub["median_deblended_ratio"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_bins(bins: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = archive_existing(path)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(bins["z_mid"], bins["median_aperture_ratio"], marker="o", label="circular aperture")
    ax.plot(bins["z_mid"], bins["median_deblended_ratio"], marker="s", label="Voronoi deblended")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Flux / catalogue flux")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_deblending_diagnostics(
    catalog: str | Path,
    fits_paths: list[str | Path],
    *,
    out_dir: str | Path,
    plot_dir: str | Path,
    sample_name: str = "tng100_imageap_deblended",
    selected_col: str = "mock_spec_selected",
    threshold_sigma: float = 3.0,
    min_pixels: int = 8,
    edge_buffer_pix: int = 3,
) -> dict[str, pd.DataFrame]:
    from .io import read_catalog

    df = read_catalog(catalog)
    out_dir = Path(out_dir)
    plot_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    per_filter_parts: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for path in fits_paths:
        filter_name = filter_name_from_path(Path(path))
        x_col = f"x_{filter_name}_pix"
        y_col = f"y_{filter_name}_pix"
        cat_col = f"catalog_flux_{filter_name}_uJy"
        ap_col = f"flux_{filter_name}_uJy"
        if x_col not in df.columns or cat_col not in df.columns:
            continue

        per_obj, summary = _deblend_one_filter(
            df,
            path,
            all_positions_col_x=x_col,
            all_positions_col_y=y_col,
            selected_col=selected_col,
            cat_flux_col=cat_col,
            aperture_flux_col=ap_col,
            threshold_sigma=threshold_sigma,
            min_pixels=min_pixels,
            edge_buffer_pix=edge_buffer_pix,
        )
        per_filter_parts.append(per_obj)
        summaries.append(summary)

    if not per_filter_parts:
        raise RuntimeError("No valid filter data")

    per_filter = pd.concat(per_filter_parts, ignore_index=True)
    objects = _object_summary(per_filter)
    bins = _redshift_bins(objects)
    filter_summary = pd.DataFrame(summaries).sort_values("filter").reset_index(drop=True)

    fit_ap = fit_power_law(objects["z"].to_numpy(), objects["median_aperture_ratio"].to_numpy())
    fit_dbl = fit_power_law(objects["z"].to_numpy(), objects["median_deblended_ratio"].to_numpy())
    slope_table = pd.DataFrame(
        [
            {"quantity": "aperture", **fit_ap.as_dict()},
            {"quantity": "deblended", **fit_dbl.as_dict()},
        ]
    )

    write_table(per_filter, out_dir / "deblending_per_filter.csv")
    write_table(objects, out_dir / "deblending_object_summary.csv")
    write_table(bins, out_dir / "deblending_redshift_bins.csv")
    write_table(filter_summary, out_dir / "deblending_filter_summary.csv")
    write_table(slope_table, out_dir / "deblending_slope_table.csv")
    _plot_bins(bins, plot_dir / f"{sample_name}_deblending_redshift_bins.png")

    lines = [
        "# Voronoi Deblending Diagnostic",
        "",
        f"Objects: {len(objects):,}  |  Filters: {len(summaries)}",
        "",
        "| Quantity | alpha | N |",
        "|---|---:|---:|",
        f"| aperture | {fit_ap.alpha:.3f} ± {fit_ap.alpha_err:.3f} | {fit_ap.n:d} |",
        f"| deblended | {fit_dbl.alpha:.3f} ± {fit_dbl.alpha_err:.3f} | {fit_dbl.n:d} |",
        "",
        "Filter summary:",
        filter_summary[["filter", "median_aperture_ratio", "median_deblended_ratio",
                         "alpha_aperture", "alpha_deblended"]].to_string(index=False),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "per_filter": per_filter,
        "objects": objects,
        "bins": bins,
        "filter_summary": filter_summary,
        "slopes": slope_table,
    }


def write_deblended_tng_slopes(
    catalog: str | Path,
    per_filter_csv: str | Path,
    config: str | Path,
    *,
    out_dir: str | Path,
    plot_dir: str | Path,
    sample_name: str = "tng100_imageap_deblended_sb",
) -> pd.DataFrame:
    """Compute TNG SB power-law slopes using Voronoi-deblended fluxes.

    Reads *per_filter_csv* (output of ``write_deblending_diagnostics``), pivots
    deblended fluxes wide, patches the flux columns in *catalog*, and runs the
    standard SB pipeline.  Where a deblended flux is non-finite or non-positive
    the original aperture flux is kept as a fallback.

    Returns the slope table DataFrame.
    """
    from .io import read_catalog
    from .pipeline import run_catalog_pipeline
    from .filters import schema_from_toml

    df = read_catalog(catalog)
    per_filter = pd.read_csv(per_filter_csv)
    schema = schema_from_toml(config)
    out_dir = Path(out_dir)
    plot_dir = Path(plot_dir)

    for filt in per_filter["filter"].unique():
        flux_col = f"flux_{filt}_uJy"
        if flux_col not in df.columns:
            continue
        sub = per_filter[per_filter["filter"] == filt][["object_index", "deblended_flux_uJy"]].copy()
        sub = sub.set_index("object_index")["deblended_flux_uJy"]
        valid = sub[sub.notna() & (sub > 0.0)]
        df.loc[df.index.isin(valid.index), flux_col] = valid.reindex(df.index[df.index.isin(valid.index)]).values

    _, slope_table = run_catalog_pipeline(
        df,
        schema,
        result_dir=out_dir,
        plot_dir=plot_dir,
        sample_name=sample_name,
    )
    return slope_table

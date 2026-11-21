"""Diagnostics for catalogue-vs-image aperture TNG surface-brightness shifts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fitting import fit_power_law
from .io import read_catalog, write_table
from .plots import archive_existing


OBSERVABLE_COLUMNS = {
    "mean_sb": "sb_mean",
    "max_sb": "sb_max",
    "rest_uv_1500_2000": "sb_rest_uv_1500_2000",
}


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0) != 0.0
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _merge_catalogues(catalogue: pd.DataFrame, image: pd.DataFrame) -> pd.DataFrame:
    key = "object_key" if "object_key" in catalogue and "object_key" in image else "row_id"
    keep_catalogue = [
        col
        for col in (
            key,
            "row_id",
            "z",
            "mock_spec_selected",
            "p_spec",
            "n_valid_filters",
            "n_rest_uv_filters",
            "aperture_area_arcsec2_used",
            *OBSERVABLE_COLUMNS.values(),
        )
        if col in catalogue
    ]
    keep_image = [
        col
        for col in (
            key,
            "row_id",
            "z",
            "mock_spec_selected",
            "p_spec",
            "n_valid_filters",
            "n_rest_uv_filters",
            "aperture_area_arcsec2_used",
            "is_edge",
            *OBSERVABLE_COLUMNS.values(),
        )
        if col in image
    ]
    cat = catalogue[keep_catalogue].copy()
    img = image[keep_image].copy()
    merged = cat.merge(img, on=key, how="inner", suffixes=("_catalogue", "_image"))
    if "z_image" in merged:
        z = pd.to_numeric(merged["z_image"], errors="coerce")
        if "z_catalogue" in merged:
            z = z.fillna(pd.to_numeric(merged["z_catalogue"], errors="coerce"))
        merged["z"] = z
    elif "z_catalogue" in merged:
        merged["z"] = pd.to_numeric(merged["z_catalogue"], errors="coerce")
    elif "z" in merged:
        merged["z"] = pd.to_numeric(merged["z"], errors="coerce")
    else:
        raise KeyError("Merged catalogues do not contain a redshift column")
    return merged


def _selected_mask(df: pd.DataFrame, source: str) -> pd.Series:
    col = f"mock_spec_selected_{source}"
    if col not in df:
        raise KeyError(f"Missing selection column: {col}")
    return _as_bool(df[col])


def _fit_row(
    df: pd.DataFrame,
    *,
    observable: str,
    value_col: str,
    mask: pd.Series,
    component: str,
) -> dict[str, float | int | str]:
    fit = fit_power_law(
        df.loc[mask, "z"].to_numpy(),
        pd.to_numeric(df.loc[mask, value_col], errors="coerce").to_numpy(),
        observable=observable,
        sample="tng100_image_aperture_shift",
        selection=component,
    )
    row = fit.as_dict()
    row["component"] = component
    row["value_column"] = value_col
    return row


def slope_decomposition(merged: pd.DataFrame) -> pd.DataFrame:
    """Separate empirical-spec selection and image-aperture measurement effects."""

    cat_selected = _selected_mask(merged, "catalogue")
    image_selected = _selected_mask(merged, "image")
    rows: list[dict[str, float | int | str]] = []
    for observable, col in OBSERVABLE_COLUMNS.items():
        cat_col = f"{col}_catalogue"
        image_col = f"{col}_image"
        if cat_col not in merged or image_col not in merged:
            continue
        rows.append(
            _fit_row(
                merged,
                observable=observable,
                value_col=cat_col,
                mask=cat_selected,
                component="catalogue_sb_catalogue_spec_selection",
            )
        )
        rows.append(
            _fit_row(
                merged,
                observable=observable,
                value_col=cat_col,
                mask=image_selected,
                component="catalogue_sb_image_spec_selection",
            )
        )
        rows.append(
            _fit_row(
                merged,
                observable=observable,
                value_col=image_col,
                mask=image_selected,
                component="image_aperture_sb_image_spec_selection",
            )
        )

        ratio = pd.to_numeric(merged[image_col], errors="coerce") / pd.to_numeric(merged[cat_col], errors="coerce")
        tmp = merged.assign(_ratio=ratio)
        rows.append(
            _fit_row(
                tmp,
                observable=f"{observable}_image_to_catalogue_ratio",
                value_col="_ratio",
                mask=image_selected,
                component="image_to_catalogue_ratio_image_spec_selection",
            )
        )

    table = pd.DataFrame(rows)
    summary_rows: list[dict[str, float | int | str]] = []
    for observable in OBSERVABLE_COLUMNS:
        sub = table[table["observable"].isin({observable, f"{observable}_image_to_catalogue_ratio"})]
        values = dict(zip(sub["component"], sub["alpha"], strict=False))
        n_values = dict(zip(sub["component"], sub["n"], strict=False))
        a0 = values.get("catalogue_sb_catalogue_spec_selection", np.nan)
        a1 = values.get("catalogue_sb_image_spec_selection", np.nan)
        a2 = values.get("image_aperture_sb_image_spec_selection", np.nan)
        ar = values.get("image_to_catalogue_ratio_image_spec_selection", np.nan)
        summary_rows.append(
            {
                "observable": observable,
                "alpha_catalogue_original_selection": a0,
                "alpha_catalogue_image_selection": a1,
                "alpha_image_aperture_image_selection": a2,
                "delta_selection_only": a1 - a0,
                "delta_aperture_on_image_selection": a2 - a1,
                "delta_total": a2 - a0,
                "alpha_image_to_catalogue_ratio": ar,
                "n_catalogue_original_selection": n_values.get("catalogue_sb_catalogue_spec_selection", 0),
                "n_image_selection": n_values.get("image_aperture_sb_image_spec_selection", 0),
            }
        )
    return table, pd.DataFrame(summary_rows)


def filter_flux_ratio_slopes(image: pd.DataFrame) -> pd.DataFrame:
    selected = _as_bool(image["mock_spec_selected"]) if "mock_spec_selected" in image else pd.Series(True, index=image.index)
    rows: list[dict[str, float | int | str]] = []
    flux_cols = sorted(col for col in image if col.startswith("flux_") and col.endswith("_uJy"))
    for flux_col in flux_cols:
        filter_name = flux_col.removeprefix("flux_").removesuffix("_uJy")
        cat_col = f"catalog_flux_{filter_name}_uJy"
        if cat_col not in image:
            continue
        ratio = pd.to_numeric(image[flux_col], errors="coerce") / pd.to_numeric(image[cat_col], errors="coerce")
        fit = fit_power_law(
            pd.to_numeric(image.loc[selected, "z"], errors="coerce").to_numpy(),
            ratio.loc[selected].to_numpy(),
            observable="aperture_to_catalogue_flux_ratio",
            sample=filter_name,
            selection="image_spec_selection",
        )
        row = fit.as_dict()
        finite = np.isfinite(ratio) & (ratio > 0.0)
        row.update(
            {
                "filter": filter_name,
                "median_ratio_all": float(np.nanmedian(ratio[finite])) if finite.any() else np.nan,
                "median_ratio_selected": float(np.nanmedian(ratio[finite & selected])) if (finite & selected).any() else np.nan,
                "p16_ratio_selected": float(np.nanpercentile(ratio[finite & selected], 16.0))
                if (finite & selected).any()
                else np.nan,
                "p84_ratio_selected": float(np.nanpercentile(ratio[finite & selected], 84.0))
                if (finite & selected).any()
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def object_flux_ratio_table(image: pd.DataFrame) -> pd.DataFrame:
    """Return one row per object with median image/catalogue flux ratio and predictors."""

    flux_cols = sorted(col for col in image if col.startswith("flux_") and col.endswith("_uJy"))
    ratios = []
    for flux_col in flux_cols:
        filter_name = flux_col.removeprefix("flux_").removesuffix("_uJy")
        cat_col = f"catalog_flux_{filter_name}_uJy"
        if cat_col not in image:
            continue
        ratio = pd.to_numeric(image[flux_col], errors="coerce") / pd.to_numeric(image[cat_col], errors="coerce")
        ratios.append(ratio.where(np.isfinite(ratio) & (ratio > 0.0)).rename(filter_name))
    if not ratios:
        raise ValueError("No matching flux/catalog_flux columns found")
    ratio_df = pd.concat(ratios, axis=1)

    radius_cols = [col for col in image if col.startswith("aperture_radius_arcsec_")]
    radius_arcsec = (
        image[radius_cols].apply(pd.to_numeric, errors="coerce").median(axis=1, skipna=True)
        if radius_cols
        else pd.Series(np.nan, index=image.index)
    )
    out = pd.DataFrame(
        {
            "z": pd.to_numeric(image["z"], errors="coerce"),
            "selected": _as_bool(image["mock_spec_selected"])
            if "mock_spec_selected" in image
            else pd.Series(True, index=image.index),
            "median_flux_ratio": ratio_df.median(axis=1, skipna=True),
            "p16_flux_ratio": ratio_df.quantile(0.16, axis=1),
            "p84_flux_ratio": ratio_df.quantile(0.84, axis=1),
            "n_ratio_filters": ratio_df.notna().sum(axis=1),
            "aperture_radius_arcsec": radius_arcsec,
            "kpc_per_arcsec": pd.to_numeric(image.get("kpc_per_arcsec", np.nan), errors="coerce"),
            "mstar_msun_2rhalf": pd.to_numeric(image.get("mstar_msun_2rhalf", np.nan), errors="coerce"),
            "sfr_msun_per_yr_2rhalf": pd.to_numeric(image.get("sfr_msun_per_yr_2rhalf", np.nan), errors="coerce"),
            "app_gmag_rest": pd.to_numeric(image.get("app_gmag_rest", np.nan), errors="coerce"),
        }
    )
    out["aperture_radius_kpc"] = out["aperture_radius_arcsec"] * out["kpc_per_arcsec"]
    return out


def _transformed_predictors(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log1pz"] = np.log10(1.0 + pd.to_numeric(df["z"], errors="coerce"))
    for source, target in (
        ("aperture_radius_arcsec", "log_aperture_radius_arcsec"),
        ("aperture_radius_kpc", "log_aperture_radius_kpc"),
        ("mstar_msun_2rhalf", "log_mstar_msun_2rhalf"),
    ):
        values = pd.to_numeric(df[source], errors="coerce")
        out[target] = np.where(values > 0.0, np.log10(values), np.nan)
    sfr = pd.to_numeric(df["sfr_msun_per_yr_2rhalf"], errors="coerce")
    out["log_sfr_plus_1e-3"] = np.where(sfr >= 0.0, np.log10(sfr + 1.0e-3), np.nan)
    out["app_gmag_rest"] = pd.to_numeric(df["app_gmag_rest"], errors="coerce")
    out["n_ratio_filters"] = pd.to_numeric(df["n_ratio_filters"], errors="coerce")
    return out


def _ols(y: pd.Series, predictors: pd.DataFrame) -> dict[str, float | int]:
    frame = pd.concat([y.rename("_y"), predictors], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < len(predictors.columns) + 3:
        return {
            "n": int(len(frame)),
            "r2": np.nan,
            **{f"coef_{col}": np.nan for col in predictors.columns},
            **{f"err_{col}": np.nan for col in predictors.columns},
        }
    design = np.column_stack([np.ones(len(frame)), frame[predictors.columns].to_numpy(dtype=float)])
    yv = frame["_y"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    resid = yv - design @ beta
    dof = max(len(yv) - design.shape[1], 1)
    sigma2 = float(np.sum(resid * resid) / dof)
    cov = np.linalg.pinv(design.T @ design) * sigma2
    total = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1.0 - float(np.sum(resid * resid)) / total if total > 0.0 else np.nan
    row: dict[str, float | int] = {"n": int(len(frame)), "r2": r2, "intercept": float(beta[0])}
    for idx, col in enumerate(predictors.columns, start=1):
        row[f"coef_{col}"] = float(beta[idx])
        row[f"err_{col}"] = float(np.sqrt(cov[idx, idx]))
    return row


def aperture_physics_models(objects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit object-level ratio trends and redshift trends controlled for predictors."""

    use = (
        objects["selected"]
        & np.isfinite(objects["median_flux_ratio"])
        & (objects["median_flux_ratio"] > 0.0)
        & (objects["n_ratio_filters"] > 0)
    )
    selected = objects.loc[use].copy()
    predictors = _transformed_predictors(selected)
    y = np.log10(selected["median_flux_ratio"])

    univariate_rows: list[dict[str, float | int | str]] = []
    for predictor in predictors.columns:
        row = _ols(y, predictors[[predictor]])
        row.update({"model": f"log_ratio_vs_{predictor}", "predictors": predictor})
        univariate_rows.append(row)

    conditional_rows: list[dict[str, float | int | str]] = []
    controls = [
        [],
        ["log_aperture_radius_arcsec"],
        ["log_aperture_radius_kpc"],
        ["log_mstar_msun_2rhalf"],
        ["log_sfr_plus_1e-3"],
        ["app_gmag_rest"],
        ["log_aperture_radius_arcsec", "log_mstar_msun_2rhalf", "log_sfr_plus_1e-3", "app_gmag_rest"],
        ["log_aperture_radius_kpc", "log_mstar_msun_2rhalf", "log_sfr_plus_1e-3", "app_gmag_rest"],
    ]
    for control in controls:
        cols = ["log1pz", *control]
        pred = predictors[cols].copy()
        for col in control:
            finite = np.isfinite(pred[col])
            if finite.any():
                mean = float(np.nanmean(pred.loc[finite, col]))
                std = float(np.nanstd(pred.loc[finite, col]))
                if std > 0.0:
                    pred[col] = (pred[col] - mean) / std
        row = _ols(y, pred)
        row.update({"model": "redshift_plus_" + ("none" if not control else "_".join(control)), "controls": ",".join(control)})
        conditional_rows.append(row)

    return pd.DataFrame(univariate_rows), pd.DataFrame(conditional_rows)


def aperture_physics_bins(objects: pd.DataFrame, *, n_bins: int = 8) -> pd.DataFrame:
    use = (
        objects["selected"]
        & np.isfinite(objects["z"])
        & np.isfinite(objects["median_flux_ratio"])
        & (objects["median_flux_ratio"] > 0.0)
        & (objects["n_ratio_filters"] > 0)
    )
    df = objects.loc[use].copy()
    x = np.log10(1.0 + df["z"])
    edges = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for idx in range(n_bins):
        if idx == n_bins - 1:
            in_bin = (x >= edges[idx]) & (x <= edges[idx + 1])
        else:
            in_bin = (x >= edges[idx]) & (x < edges[idx + 1])
        if in_bin.sum() < 5:
            continue
        sub = df.loc[in_bin]
        rows.append(
            {
                "bin": idx,
                "z_mid": float(10.0 ** (0.5 * (edges[idx] + edges[idx + 1])) - 1.0),
                "log1pz_mid": float(0.5 * (edges[idx] + edges[idx + 1])),
                "n": int(len(sub)),
                "median_flux_ratio": float(np.nanmedian(sub["median_flux_ratio"])),
                "p16_flux_ratio": float(np.nanpercentile(sub["median_flux_ratio"], 16.0)),
                "p84_flux_ratio": float(np.nanpercentile(sub["median_flux_ratio"], 84.0)),
                "median_aperture_radius_arcsec": float(np.nanmedian(sub["aperture_radius_arcsec"])),
                "median_aperture_radius_kpc": float(np.nanmedian(sub["aperture_radius_kpc"])),
                "median_mstar_msun_2rhalf": float(np.nanmedian(sub["mstar_msun_2rhalf"])),
                "median_sfr_msun_per_yr_2rhalf": float(np.nanmedian(sub["sfr_msun_per_yr_2rhalf"])),
                "median_app_gmag_rest": float(np.nanmedian(sub["app_gmag_rest"])),
            }
        )
    return pd.DataFrame(rows)


def _plot_physics_tracks(bins: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = archive_existing(path)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), constrained_layout=True, sharex=True)
    ax = axes[0, 0]
    ax.plot(bins["z_mid"], bins["median_flux_ratio"], marker="o", color="#222222")
    ax.fill_between(bins["z_mid"], bins["p16_flux_ratio"], bins["p84_flux_ratio"], color="#222222", alpha=0.14)
    ax.set_yscale("log")
    ax.set_ylabel("median image/catalogue flux")

    axes[0, 1].plot(bins["z_mid"], bins["median_aperture_radius_arcsec"], marker="o", color="#2b5c9a")
    axes[0, 1].set_ylabel("median aperture radius [arcsec]")

    axes[1, 0].plot(bins["z_mid"], bins["median_aperture_radius_kpc"], marker="o", color="#aa5a2c")
    axes[1, 0].set_ylabel("median aperture radius [kpc]")

    axes[1, 1].plot(bins["z_mid"], bins["median_app_gmag_rest"], marker="o", color="#3b7f48")
    axes[1, 1].set_ylabel("median apparent gmag")
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel("Redshift")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_ratio_vs_aperture(objects: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = archive_existing(path)
    use = (
        objects["selected"]
        & np.isfinite(objects["median_flux_ratio"])
        & (objects["median_flux_ratio"] > 0.0)
        & np.isfinite(objects["aperture_radius_arcsec"])
        & (objects["aperture_radius_arcsec"] > 0.0)
    )
    df = objects.loc[use].copy()
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    sc = ax.scatter(
        df["aperture_radius_arcsec"],
        df["median_flux_ratio"],
        c=df["z"],
        s=8,
        alpha=0.22,
        edgecolors="none",
        cmap="viridis",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Aperture radius [arcsec]")
    ax.set_ylabel("median image/catalogue flux")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Redshift")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def redshift_binned_ratios(merged: pd.DataFrame, image: pd.DataFrame, *, n_bins: int = 8) -> pd.DataFrame:
    selected = _selected_mask(merged, "image")
    z = pd.to_numeric(merged["z"], errors="coerce")
    x = np.log10(1.0 + z)
    finite_x = np.isfinite(x)
    edges = np.linspace(float(np.nanmin(x[finite_x])), float(np.nanmax(x[finite_x])), n_bins + 1)

    rows: list[dict[str, float | int | str]] = []
    for idx in range(n_bins):
        if idx == n_bins - 1:
            in_bin = (x >= edges[idx]) & (x <= edges[idx + 1])
        else:
            in_bin = (x >= edges[idx]) & (x < edges[idx + 1])
        in_bin &= selected
        for observable, col in OBSERVABLE_COLUMNS.items():
            cat_col = f"{col}_catalogue"
            image_col = f"{col}_image"
            if cat_col not in merged or image_col not in merged:
                continue
            ratio = pd.to_numeric(merged[image_col], errors="coerce") / pd.to_numeric(merged[cat_col], errors="coerce")
            finite = in_bin & np.isfinite(ratio) & (ratio > 0.0)
            if not finite.any():
                continue
            rows.append(
                {
                    "bin": idx,
                    "quantity": f"{observable}_image_to_catalogue_ratio",
                    "z_mid": float(10.0 ** (0.5 * (edges[idx] + edges[idx + 1])) - 1.0),
                    "log1pz_mid": float(0.5 * (edges[idx] + edges[idx + 1])),
                    "n": int(finite.sum()),
                    "median": float(np.nanmedian(ratio[finite])),
                    "p16": float(np.nanpercentile(ratio[finite], 16.0)),
                    "p84": float(np.nanpercentile(ratio[finite], 84.0)),
                }
            )

    image_selected = _as_bool(image["mock_spec_selected"]) if "mock_spec_selected" in image else pd.Series(True, index=image.index)
    image_z = pd.to_numeric(image["z"], errors="coerce")
    image_x = np.log10(1.0 + image_z)
    stacked: list[pd.DataFrame] = []
    for flux_col in sorted(col for col in image if col.startswith("flux_") and col.endswith("_uJy")):
        filter_name = flux_col.removeprefix("flux_").removesuffix("_uJy")
        cat_col = f"catalog_flux_{filter_name}_uJy"
        if cat_col not in image:
            continue
        ratio = pd.to_numeric(image[flux_col], errors="coerce") / pd.to_numeric(image[cat_col], errors="coerce")
        stacked.append(pd.DataFrame({"z": image_z, "x": image_x, "ratio": ratio, "selected": image_selected}))
    if stacked:
        all_filters = pd.concat(stacked, ignore_index=True)
        for idx in range(n_bins):
            if idx == n_bins - 1:
                in_bin = (all_filters["x"] >= edges[idx]) & (all_filters["x"] <= edges[idx + 1])
            else:
                in_bin = (all_filters["x"] >= edges[idx]) & (all_filters["x"] < edges[idx + 1])
            finite = in_bin & all_filters["selected"] & np.isfinite(all_filters["ratio"]) & (all_filters["ratio"] > 0.0)
            if not finite.any():
                continue
            rows.append(
                {
                    "bin": idx,
                    "quantity": "per_filter_aperture_to_catalogue_flux_ratio",
                    "z_mid": float(10.0 ** (0.5 * (edges[idx] + edges[idx + 1])) - 1.0),
                    "log1pz_mid": float(0.5 * (edges[idx] + edges[idx + 1])),
                    "n": int(finite.sum()),
                    "median": float(np.nanmedian(all_filters.loc[finite, "ratio"])),
                    "p16": float(np.nanpercentile(all_filters.loc[finite, "ratio"], 16.0)),
                    "p84": float(np.nanpercentile(all_filters.loc[finite, "ratio"], 84.0)),
                }
            )
    return pd.DataFrame(rows)


def _plot_decomposition(summary: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = archive_existing(path)
    labels = summary["observable"].str.replace("_", " ").to_list()
    x = np.arange(len(labels), dtype=float)
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.bar(x - width, summary["alpha_catalogue_original_selection"], width=width, label="Catalogue, original spec")
    ax.bar(x, summary["alpha_catalogue_image_selection"], width=width, label="Catalogue, image spec")
    ax.bar(x + width, summary["alpha_image_aperture_image_selection"], width=width, label="Image aperture, image spec")
    ax.axhline(-3.0, color="#9d3d3d", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#3e7f4c", linestyle="--", linewidth=1.0)
    ax.set_ylabel("alpha in SB proportional to (1 + z)^alpha")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_ratio_tracks(binned: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = archive_existing(path)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    colors = {
        "mean_sb_image_to_catalogue_ratio": "#2b5c9a",
        "max_sb_image_to_catalogue_ratio": "#aa5a2c",
        "per_filter_aperture_to_catalogue_flux_ratio": "#333333",
    }
    labels = {
        "mean_sb_image_to_catalogue_ratio": "Mean SB ratio",
        "max_sb_image_to_catalogue_ratio": "Max SB ratio",
        "per_filter_aperture_to_catalogue_flux_ratio": "Per-filter flux ratio",
    }
    for quantity in colors:
        sub = binned[binned["quantity"] == quantity]
        if len(sub) == 0:
            continue
        ax.plot(sub["z_mid"], sub["median"], marker="o", color=colors[quantity], label=labels[quantity])
        ax.fill_between(sub["z_mid"], sub["p16"], sub["p84"], color=colors[quantity], alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Image aperture / catalogue")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_selection_comparison(merged: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = archive_existing(path)
    cat_selected = _selected_mask(merged, "catalogue")
    image_selected = _selected_mask(merged, "image")
    z = pd.to_numeric(merged["z"], errors="coerce")
    bins = np.linspace(0.0, 8.5, 18)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.hist(z[cat_selected], bins=bins, histtype="step", linewidth=1.6, color="#222222", label="Catalogue spec selection")
    ax.hist(z[image_selected], bins=bins, histtype="step", linewidth=1.6, color="#2b5c9a", label="Image-aperture spec selection")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Selected objects")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_image_aperture_diagnostics(
    catalogue_measured: str | Path,
    image_measured: str | Path,
    *,
    out_dir: str | Path,
    plot_dir: str | Path | None = None,
    sample_name: str = "tng100_image_aperture_shift",
) -> dict[str, pd.DataFrame]:
    """Write tables, plots, and a compact markdown summary."""

    out_dir = Path(out_dir)
    plot_dir = Path(plot_dir) if plot_dir is not None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    catalogue = read_catalog(catalogue_measured)
    image = read_catalog(image_measured)
    merged = _merge_catalogues(catalogue, image)
    detailed, summary = slope_decomposition(merged)
    flux = filter_flux_ratio_slopes(image)
    binned = redshift_binned_ratios(merged, image)
    objects = object_flux_ratio_table(image)
    physics_univariate, physics_conditional = aperture_physics_models(objects)
    physics_bins = aperture_physics_bins(objects)

    write_table(detailed, out_dir / "slope_decomposition_detailed.csv")
    write_table(summary, out_dir / "slope_decomposition_summary.csv")
    write_table(flux, out_dir / "filter_flux_ratio_slopes.csv")
    write_table(binned, out_dir / "redshift_binned_ratios.csv")
    write_table(objects, out_dir / "object_flux_ratio_predictors.csv")
    write_table(physics_univariate, out_dir / "aperture_physics_univariate_models.csv")
    write_table(physics_conditional, out_dir / "aperture_physics_conditional_redshift_models.csv")
    write_table(physics_bins, out_dir / "aperture_physics_redshift_bins.csv")

    _plot_decomposition(summary, plot_dir / f"{sample_name}_slope_decomposition.png")
    _plot_ratio_tracks(binned, plot_dir / f"{sample_name}_aperture_ratio_tracks.png")
    _plot_selection_comparison(merged, plot_dir / f"{sample_name}_spec_selection_redshift.png")
    _plot_physics_tracks(physics_bins, plot_dir / f"{sample_name}_aperture_physics_tracks.png")
    _plot_ratio_vs_aperture(objects, plot_dir / f"{sample_name}_ratio_vs_aperture_radius.png")

    cat_selected = _selected_mask(merged, "catalogue")
    image_selected = _selected_mask(merged, "image")
    overlap = cat_selected & image_selected
    lines = [
        "# TNG100 Image-Aperture Shift Diagnostics",
        "",
        f"Matched catalogue rows: {len(merged):,}.",
        f"Catalogue-flux empirical-spec selected rows in matched table: {int(cat_selected.sum()):,}.",
        f"Image-aperture empirical-spec selected rows in matched table: {int(image_selected.sum()):,}.",
        f"Selection overlap: {int(overlap.sum()):,} rows.",
        "",
        "| Observable | alpha cat/original | alpha cat/image selection | alpha image/image selection | selection delta | aperture delta | total delta | ratio alpha |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| {obs} | {a0:.3f} | {a1:.3f} | {a2:.3f} | {ds:+.3f} | {da:+.3f} | {dt:+.3f} | {ar:+.3f} |".format(
                obs=row.observable,
                a0=row.alpha_catalogue_original_selection,
                a1=row.alpha_catalogue_image_selection,
                a2=row.alpha_image_aperture_image_selection,
                ds=row.delta_selection_only,
                da=row.delta_aperture_on_image_selection,
                dt=row.delta_total,
                ar=row.alpha_image_to_catalogue_ratio,
            )
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "- `selection delta` isolates the changed empirical spectroscopic-selection sample while keeping catalogue fluxes.",
            "- `aperture delta` isolates image-aperture measurement at fixed image-aperture selected rows.",
            "- `ratio alpha` is the fitted redshift evolution of image-aperture SB divided by catalogue SB.",
            "",
            "Plots:",
            f"- `Plots/{sample_name}_slope_decomposition.png`",
            f"- `Plots/{sample_name}_aperture_ratio_tracks.png`",
            f"- `Plots/{sample_name}_spec_selection_redshift.png`",
            f"- `Plots/{sample_name}_aperture_physics_tracks.png`",
            f"- `Plots/{sample_name}_ratio_vs_aperture_radius.png`",
            "",
            "Object-level aperture-physics diagnostics:",
            "- `object_flux_ratio_predictors.csv` contains one row per image-aperture object with median image/catalogue flux ratio and physical predictors.",
            "- `aperture_physics_conditional_redshift_models.csv` tests whether the ratio-redshift slope remains after controlling for aperture and galaxy-property proxies.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "detailed": detailed,
        "summary": summary,
        "flux": flux,
        "binned": binned,
        "objects": objects,
        "physics_univariate": physics_univariate,
        "physics_conditional": physics_conditional,
        "physics_bins": physics_bins,
    }

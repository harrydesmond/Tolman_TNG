"""High-level pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .filters import CatalogSchema
from .fitting import fit_power_law, fit_results_table
from .io import write_table
from .measurements import add_all_observables, modal_typical_surface_brightness
from .plots import plot_modal_fit, plot_observable_fit, plot_percentile_tracks, plot_selection_ladder
from .selection import selection_ladder


OBSERVABLES = {
    "mean_sb": "sb_mean",
    "max_sb": "sb_max",
    "rest_uv_1500_2000": "sb_rest_uv_1500_2000",
}


def run_catalog_pipeline(
    raw: pd.DataFrame,
    schema: CatalogSchema,
    *,
    result_dir: str | Path,
    plot_dir: str | Path,
    sample_name: str,
    save_measured_catalog: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_dir = Path(result_dir)
    plot_dir = Path(plot_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    measured = add_all_observables(raw, schema)
    if save_measured_catalog:
        write_table(measured, result_dir / "measured_catalog.csv")

    results = []
    for stage in selection_ladder(measured, schema):
        for obs_name, column in OBSERVABLES.items():
            fit = fit_power_law(
                stage.df[schema.redshift_col].to_numpy(),
                stage.df[column].to_numpy(),
                observable=obs_name,
                sample=sample_name,
                selection=stage.name,
            )
            results.append(fit)

    table = fit_results_table(results)
    write_table(table, result_dir / "slope_table.csv")

    final_stage = selection_ladder(measured, schema)[-1]
    for obs_name, column in OBSERVABLES.items():
        fit = table[(table["observable"] == obs_name) & (table["selection"] == final_stage.name)]
        if len(fit) == 1:
            fit_obj = fit_power_law(
                final_stage.df[schema.redshift_col].to_numpy(),
                final_stage.df[column].to_numpy(),
                observable=obs_name,
                sample=sample_name,
                selection=final_stage.name,
            )
            plot_observable_fit(
                final_stage.df[schema.redshift_col].to_numpy(),
                final_stage.df[column].to_numpy(),
                fit_obj,
                plot_dir / f"{sample_name}_{obs_name}.png",
                ylabel=f"log10 {obs_name} [uJy / arcsec2]",
            )
            if obs_name in {"mean_sb", "max_sb"}:
                plot_percentile_tracks(
                    final_stage.df[schema.redshift_col].to_numpy(),
                    final_stage.df[column].to_numpy(),
                    plot_dir / f"{sample_name}_{obs_name}_percentiles.png",
                    ylabel=f"log10 {obs_name} [uJy / arcsec2]",
                )

    for obs_name, column in (("modal_mean_sb", "sb_mean"), ("modal_max_sb", "sb_max")):
        modal = modal_typical_surface_brightness(
            final_stage.df[schema.redshift_col].to_numpy(),
            final_stage.df[column].to_numpy(),
            n_z_bins=25,
            n_sb_bins=25,
            z_min=0.1,
            z_max=8.5,
        )
        if len(modal):
            write_table(modal, result_dir / f"{obs_name}.csv")
            fit_obj = fit_power_law(
                modal["z_center"].to_numpy(),
                modal["sb"].to_numpy(),
                y_sigma=modal["log_sb_err"].to_numpy(),
                observable=obs_name,
                sample=sample_name,
                selection=final_stage.name,
            )
            plot_modal_fit(
                modal,
                fit_obj,
                plot_dir / f"{sample_name}_{obs_name}.png",
                ylabel=f"log10 {obs_name} [uJy / arcsec2]",
            )
            table = pd.concat([table, pd.DataFrame([fit_obj.as_dict()])], ignore_index=True)
    write_table(table, result_dir / "slope_table.csv")

    plot_selection_ladder(table, plot_dir / f"{sample_name}_selection_ladder_mean_sb.png", observable="mean_sb")
    return measured, table

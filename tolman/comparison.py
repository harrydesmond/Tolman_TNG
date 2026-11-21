"""Comparison tables and plots for observed and mock Tolman slopes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_catalog, write_table
from .plots import archive_existing


DEFAULT_OBSERVABLE_LABELS = {
    "mean_sb": "Mean SB",
    "max_sb": "Maximum SB",
    "rest_uv_1500_2000": "Rest-frame 1500-2000 A",
    "modal_mean_sb": "Modal mean SB",
    "modal_max_sb": "Modal maximum SB",
}


def _final_selection(table: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in set(table["selection"]):
        return preferred
    for candidate in ("spectroscopic", "mock_spectroscopic"):
        if candidate in set(table["selection"]):
            return candidate
    if len(table) == 0:
        raise ValueError("Cannot infer selection from an empty slope table")
    return str(table["selection"].iloc[-1])


def compare_slope_tables(
    observed_table: str | Path,
    mock_table: str | Path,
    *,
    observed_selection: str | None = "spectroscopic",
    mock_selection: str | None = "mock_spectroscopic",
    observed_label: str = "ASTRODEEP",
    mock_label: str = "TNG mock",
) -> pd.DataFrame:
    """Return one row per common observable comparing alpha values."""

    obs = read_catalog(observed_table)
    mock = read_catalog(mock_table)
    obs_sel = _final_selection(obs, observed_selection)
    mock_sel = _final_selection(mock, mock_selection)
    obs = obs[obs["selection"] == obs_sel].copy()
    mock = mock[mock["selection"] == mock_sel].copy()

    rows: list[dict[str, float | int | str]] = []
    for observable in sorted(set(obs["observable"]) & set(mock["observable"])):
        o = obs[obs["observable"] == observable]
        m = mock[mock["observable"] == observable]
        if len(o) != 1 or len(m) != 1:
            continue
        o_row = o.iloc[0]
        m_row = m.iloc[0]
        delta = float(m_row["alpha"] - o_row["alpha"])
        obs_err = float(o_row["alpha_err"])
        mock_err = float(m_row["alpha_err"])
        combined = float(np.sqrt(obs_err * obs_err + mock_err * mock_err))
        rows.append(
            {
                "observable": observable,
                "observable_label": DEFAULT_OBSERVABLE_LABELS.get(observable, observable),
                "observed_dataset": observed_label,
                "mock_dataset": mock_label,
                "observed_selection": obs_sel,
                "mock_selection": mock_sel,
                "alpha_observed": float(o_row["alpha"]),
                "alpha_observed_err": obs_err,
                "n_observed": int(o_row["n"]),
                "alpha_mock": float(m_row["alpha"]),
                "alpha_mock_err": mock_err,
                "n_mock": int(m_row["n"]),
                "delta_alpha_mock_minus_observed": delta,
                "combined_alpha_err": combined,
                "delta_over_combined_err": delta / combined if combined > 0.0 else np.nan,
                "observed_x_min": float(o_row["x_min"]),
                "observed_x_max": float(o_row["x_max"]),
                "mock_x_min": float(m_row["x_min"]),
                "mock_x_max": float(m_row["x_max"]),
            }
        )
    return pd.DataFrame(rows)


def write_slope_comparison_report(table: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ASTRODEEP vs TNG Slope Comparison",
        "",
        "Comparison of fitted alpha in `SB proportional to (1+z)^alpha`.",
        "This is a slope-summary comparison, not yet a full source-extraction/noise/PSF matched result.",
        "",
        "| Observable | alpha obs | N obs | alpha mock | N mock | delta | delta/sigma |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            "| {label} | {ao:.3f} +/- {eo:.3f} | {no:d} | {am:.3f} +/- {em:.3f} | {nm:d} | {da:+.3f} | {ds:+.1f} |".format(
                label=row.observable_label,
                ao=row.alpha_observed,
                eo=row.alpha_observed_err,
                no=row.n_observed,
                am=row.alpha_mock,
                em=row.alpha_mock_err,
                nm=row.n_mock,
                da=row.delta_alpha_mock_minus_observed,
                ds=row.delta_over_combined_err,
            )
        )
    lines.extend(
        [
            "",
            "Known differences between the mock and observed measurements:",
            "- A catalogue-level mock table uses MAST total fluxes, not image-aperture fluxes.",
            "- An image-aperture mock table does not model source extraction, noise, PSF matching, segmentation or the ASTRODEEP aperture choice.",
            "- The MAST TNG filter set lacks the ASTRODEEP F775W, F105W and F410M bands.",
            "- The ASTRODEEP secure-spectroscopic sample has 7059 objects in the CDS release and 6860 in Tsymbal et al. (2026).",
            "- The rest-frame UV sample count and filter window are not matched to Tsymbal et al. (2026).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_slope_comparison(table: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = archive_existing(path)
    if len(table) == 0:
        raise ValueError("No rows to plot")

    labels = table["observable_label"].to_list()
    x = np.arange(len(labels), dtype=float)
    width = 0.28

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.errorbar(
        x - width / 2,
        table["alpha_observed"],
        yerr=table["alpha_observed_err"],
        fmt="o",
        color="#111111",
        capsize=2,
        label=str(table["observed_dataset"].iloc[0]),
    )
    ax.errorbar(
        x + width / 2,
        table["alpha_mock"],
        yerr=table["alpha_mock_err"],
        fmt="s",
        color="#3b6ea8",
        capsize=2,
        label=str(table["mock_dataset"].iloc[0]),
    )
    ax.axhline(-3.0, color="#a23b3b", linestyle="--", linewidth=1.0, label="FRW (1+z)^-3")
    ax.axhline(0.0, color="#3b7f48", linestyle="--", linewidth=1.0, label="Non-exp. (1+z)^0")
    ax.set_ylabel("alpha in SB proportional to (1 + z)^alpha")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_slope_comparison_outputs(
    observed_table: str | Path,
    mock_table: str | Path,
    *,
    out_csv: str | Path,
    out_plot: str | Path | None = None,
    out_report: str | Path | None = None,
    observed_selection: str | None = "spectroscopic",
    mock_selection: str | None = "mock_spectroscopic",
    observed_label: str = "ASTRODEEP",
    mock_label: str = "TNG mock",
) -> pd.DataFrame:
    table = compare_slope_tables(
        observed_table,
        mock_table,
        observed_selection=observed_selection,
        mock_selection=mock_selection,
        observed_label=observed_label,
        mock_label=mock_label,
    )
    write_table(table, out_csv)
    if out_plot:
        plot_slope_comparison(table, out_plot)
    if out_report:
        write_slope_comparison_report(table, out_report)
    return table

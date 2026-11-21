"""Plotting helpers for Tolman-test diagnostics."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .fitting import FitResult


def archive_existing(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return path
    old_dir = path.parent / "Old"
    old_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archived = old_dir / f"{path.stem}_{stamp}{path.suffix}"
    shutil.move(str(path), str(archived))
    return path


def plot_observable_fit(
    z: np.ndarray,
    sb: np.ndarray,
    fit: FitResult,
    path: str | Path,
    *,
    ylabel: str,
    comparison_slopes: tuple[float, ...] = (-3.0, 0.0),
) -> None:
    path = archive_existing(path)
    z = np.asarray(z, dtype=float)
    sb = np.asarray(sb, dtype=float)
    mask = np.isfinite(z) & np.isfinite(sb) & (z > -1.0) & (sb > 0.0)
    x = np.log10(1.0 + z[mask])
    y = np.log10(sb[mask])

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.scatter(x, y, s=7, alpha=0.18, edgecolors="none", color="#3b6ea8")
    if np.isfinite(fit.alpha):
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        yy = fit.beta + fit.alpha * xx
        ax.plot(xx, yy, color="#111111", linewidth=1.8, label=f"fit alpha={fit.alpha:.2f}")
        anchor_x = np.nanpercentile(x, 10.0)
        anchor_y = fit.beta + fit.alpha * anchor_x
        for slope in comparison_slopes:
            yy_ref = anchor_y + slope * (xx - anchor_x)
            ax.plot(xx, yy_ref, linewidth=1.2, linestyle="--", label=f"(1+z)^{slope:.0f}")
    ax.set_xlabel("log10(1 + z)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_modal_fit(
    modal,
    fit: FitResult,
    path: str | Path,
    *,
    ylabel: str,
) -> None:
    path = archive_existing(path)
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    if len(modal):
        ax.errorbar(
            modal["log1pz"],
            modal["log_sb"],
            yerr=modal["log_sb_err"],
            fmt="o",
            markersize=3.5,
            color="#3b6ea8",
            ecolor="#8aa6c8",
            elinewidth=0.8,
            capsize=2,
        )
    if np.isfinite(fit.alpha) and len(modal):
        xx = np.linspace(float(modal["log1pz"].min()), float(modal["log1pz"].max()), 200)
        ax.plot(xx, fit.beta + fit.alpha * xx, color="#111111", linewidth=1.8, label=f"fit alpha={fit.alpha:.2f}")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("log10(1 + z)")
    ax.set_ylabel(ylabel)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_selection_ladder(table, path: str | Path, *, observable: str) -> None:
    path = archive_existing(path)
    sub = table[table["observable"] == observable].copy()
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    if len(sub):
        x = np.arange(len(sub))
        ax.errorbar(
            x,
            sub["alpha"],
            yerr=sub["alpha_err"],
            fmt="o-",
            color="#111111",
            markersize=4,
            linewidth=1.4,
            capsize=2,
        )
        ax.axhline(-3.0, color="#a23b3b", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#3b7f48", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["selection"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("alpha in SB proportional to (1 + z)^alpha")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_percentile_tracks(
    z: np.ndarray,
    sb: np.ndarray,
    path: str | Path,
    *,
    ylabel: str,
    percentiles: tuple[int, ...] = (10, 25, 50, 75, 90),
    n_bins: int = 11,
) -> None:
    path = archive_existing(path)
    z = np.asarray(z, dtype=float)
    sb = np.asarray(sb, dtype=float)
    mask = np.isfinite(z) & np.isfinite(sb) & (z > -1.0) & (sb > 0.0)
    x = np.log10(1.0 + z[mask])
    y = np.log10(sb[mask])
    bins = np.linspace(np.min(x), np.max(x), n_bins + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.scatter(x, y, s=6, alpha=0.12, edgecolors="none", color="#3b6ea8")
    for pct in percentiles:
        xs: list[float] = []
        ys: list[float] = []
        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            in_bin = (x >= lo) & (x < hi)
            if np.sum(in_bin) < 20:
                continue
            xs.append(float(0.5 * (lo + hi)))
            ys.append(float(np.percentile(y[in_bin], pct)))
        if len(xs) >= 2:
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, label=f"p{pct}")
    ax.set_xlabel("log10(1 + z)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)

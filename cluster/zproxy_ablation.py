"""z_proxy ablation: retrain spec-classifier without the redshift proxy.

Cross-check on the spectroscopic classifier: verify that the empirical
mock-spectroscopic classifier does not encode the redshift distribution
directly. We drop the `z_proxy` feature (the
only feature carrying explicit redshift information), retrain on
ASTRODEEP, reapply to TNG100, and recompute:

  (a) Voronoi-deblended mean/max SB slopes (Tolman observable)
  (b) gamma_rho with intrinsic photrad_kpc volume (Li-DDR observable)

If both shift by less than the per-variant aperture systematic
(~+/-0.20 on gamma; ~0.05 on alpha), the main results do not depend on
the classifier seeing redshift directly.

Adapted from cluster/selection_ablation.py (which drops the SB
features) and cluster/tng_gamma_spec_selection_nulls.py
(volume-controlled gamma_rho).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

WORKDIR = os.environ.get("TOLMAN_WORKDIR", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, WORKDIR)

from tolman.li_ddr.li_reproduction import COSMO_LI, luminosity_at_band  # noqa: E402
from tolman.spec_selection import (  # noqa: E402
    add_selection_probabilities,
    add_spec_features,
    fit_model,
    save_model,
)

ROOT = Path(WORKDIR)
ASTRODEEP_CSV = ROOT / "results/astrodeep_cds/measured_catalog.csv"
TNG_IMAGEAP_CSV = ROOT / "data/processed_tng100_7_6_xyz_hst_jwst_imageap.csv"
TNG_IMAGEAP_SPECMODEL_CSV = ROOT / "results/tng100_7_6_xyz_hst_jwst_imageap_specmodel/measured_catalog.csv"
PER_FILTER_CSV = ROOT / "results/tng100_7_6_xyz_hst_jwst_imageap_deblended/deblending_per_filter.csv"
OUT_DIR = ROOT / "results/systematics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DEFAULT_FEATURES minus `z_proxy`:
ABLATED_FEATURES = [
    "mag_F200W",
    "mag_F356W",
    "color_F150W_F356W",
    "color_F200W_F444W",
    "log_sb_mean",
    "log_aperture_area",
    "n_valid_filters",
]

FILTER = "F200W"
ALPHA_SPEC = -0.3  # Li convention; equivalent to paper alpha_SED = 0.3.
TARGET_N = 2901


def fit_alpha(z, sb):
    z = np.asarray(z, dtype=float)
    sb = np.asarray(sb, dtype=float)
    m = np.isfinite(z) & np.isfinite(sb) & (z > -1) & (sb > 0)
    n = int(m.sum())
    if n < 3:
        return np.nan, np.nan, n
    x = np.log10(1.0 + z[m])
    y = np.log10(sb[m])
    A = np.c_[np.ones(n), x]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    s2 = float(np.sum(resid**2) / (n - 2))
    err = float((s2 * np.linalg.inv(A.T @ A)[1, 1]) ** 0.5)
    return float(coef[1]), err, n


def gamma_rho_volume_controlled(df: pd.DataFrame) -> tuple[float, float, int]:
    z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
    flux_uJy = pd.to_numeric(df[f"flux_{FILTER}_uJy"], errors="coerce").to_numpy(dtype=float)
    rad_arcsec = pd.to_numeric(df[f"aperture_radius_arcsec_{FILTER}"], errors="coerce").to_numpy(dtype=float)
    kpc_per_arcsec = pd.to_numeric(df["kpc_per_arcsec"], errors="coerce").to_numpy(dtype=float)
    photrad_kpc = rad_arcsec * kpc_per_arcsec
    flux_jy = flux_uJy * 1e-6
    L = luminosity_at_band(flux_jy, z, np.full_like(z, ALPHA_SPEC), cosmo=COSMO_LI)
    r_cm = photrad_kpc * 3.0857e21
    rho_L = L / ((4.0 / 3.0) * np.pi * r_cm**3)
    finite = np.isfinite(rho_L) & (rho_L > 0) & np.isfinite(z) & (z > 0)
    n = int(finite.sum())
    if n < 10:
        return float("nan"), float("nan"), n
    x = np.log10(1.0 + z[finite])
    y = np.log10(rho_L[finite])
    A = np.c_[np.ones(n), x]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    s2 = float(np.sum(resid**2) / (n - 2))
    err = float((s2 * np.linalg.inv(A.T @ A)[1, 1]) ** 0.5)
    return float(coef[1]), err, n


def bootstrap_gamma(df: pd.DataFrame, n_boot: int = 300, seed: int = 20260511) -> float:
    rng = np.random.default_rng(seed)
    arr = df.reset_index(drop=True)
    n = len(arr)
    if n < 10:
        return float("nan")
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        g, _, _ = gamma_rho_volume_controlled(arr.iloc[idx])
        if np.isfinite(g):
            samples.append(g)
    return float(np.std(samples, ddof=1)) if samples else float("nan")


def top_k_by_pspec(df: pd.DataFrame, model, features: list[str], k: int) -> np.ndarray:
    out = add_selection_probabilities(df, model, features)
    p = pd.to_numeric(out["p_spec"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()
    order = np.argsort(-p)
    sel = np.zeros(len(out), dtype=bool)
    sel[order[: max(0, min(k, len(out)))]] = True
    return sel


def main() -> None:
    # ── Train ablated model on ASTRODEEP ─────────────────────────────────────
    print("[load] ASTRODEEP measured catalog ...", flush=True)
    ast = pd.read_csv(ASTRODEEP_CSV, low_memory=False)
    print(f"  rows: {len(ast)}", flush=True)

    z = pd.to_numeric(ast["z"], errors="coerce")
    nv = pd.to_numeric(ast["n_valid_filters"], errors="coerce").fillna(0)
    flags = pd.to_numeric(ast["flags"], errors="coerce").fillna(0)
    ap = pd.to_numeric(ast["aperture_area_arcsec2_used"], errors="coerce").fillna(0)
    snr_cols = [c for c in ast.columns if c.startswith("snr_")]
    snr = ast[snr_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    mask = (
        np.isfinite(z) & (z > 0) & (z <= 14)
        & (nv > 0)
        & (np.nanmax(snr, axis=1) >= 5)
        & (flags < 80)
        & (ap > 0)
    )
    ast_train = ast[mask.to_numpy()].copy()
    print(
        f"  training set after quality cuts: {len(ast_train)} "
        f"(has_spec={int(ast_train['has_spec'].astype(bool).sum())})",
        flush=True,
    )

    print(f"[train] ablated features (no z_proxy): {ABLATED_FEATURES}", flush=True)
    model, used = fit_model(
        ast_train,
        target_col="has_spec",
        features=ABLATED_FEATURES,
        class_weight=None,
    )
    print(f"  used features: {used}", flush=True)

    feat_df = add_spec_features(ast_train)
    y_true = ast_train["has_spec"].astype(bool).astype(int).to_numpy()
    y_proba = model.predict_proba(feat_df[used])[:, 1]
    auc = roc_auc_score(y_true, y_proba)
    print(f"  in-sample ROC AUC (no z_proxy): {auc:.3f}   (canonical with z_proxy ~0.91)", flush=True)

    save_model(model, used, OUT_DIR / "astrodeep_spec_selection_nozproxy.joblib")
    print(f"  saved model -> {OUT_DIR}/astrodeep_spec_selection_nozproxy.joblib", flush=True)

    # ── Apply to TNG100 image-aperture+specmodel catalogue for gamma ──────────
    print(f"[gamma] reading {TNG_IMAGEAP_SPECMODEL_CSV.name} ...", flush=True)
    sm = pd.read_csv(TNG_IMAGEAP_SPECMODEL_CSV, low_memory=False)
    pool_mask = (
        sm["image_success_F200W"].astype(bool)
        & np.isfinite(sm[f"flux_{FILTER}_uJy"]) & (sm[f"flux_{FILTER}_uJy"] > 0)
        & (sm[f"aperture_radius_arcsec_{FILTER}"] > 0)
    )
    sm_pool = sm.loc[pool_mask].reset_index(drop=True)
    print(f"  pool (image-success, flux>0, ap>0): {len(sm_pool)}", flush=True)

    sel_ablated = top_k_by_pspec(sm_pool, model, used, TARGET_N)
    sm_ablated = sm_pool.iloc[sel_ablated].copy()
    g_abl, g_err_abl, n_abl = gamma_rho_volume_controlled(sm_ablated)
    g_boot_abl = bootstrap_gamma(sm_ablated, n_boot=300)
    print(
        f"  gamma_rho (ablated, vol-ctrl) = {g_abl:+.3f} +/- {g_err_abl:.3f} (OLS) | "
        f"{g_boot_abl:.3f} (bootstrap)  N={n_abl}",
        flush=True,
    )

    # Canonical reference: rerun on the catalogue's own mock_spec_selected flag.
    canon = sm.loc[sm["mock_spec_selected"].astype(bool) & pool_mask].copy()
    g_canon, g_err_canon, n_canon = gamma_rho_volume_controlled(canon)
    print(
        f"  gamma_rho (canonical, vol-ctrl) = {g_canon:+.3f} +/- {g_err_canon:.3f}  N={n_canon}",
        flush=True,
    )

    # Overlap between canonical and ablated selections
    if "subhalo_id" in sm.columns:
        key = "subhalo_id"
    elif "id" in sm.columns:
        key = "id"
    else:
        key = None
    if key is not None:
        canon_ids = set(canon[key].tolist())
        abl_ids = set(sm_ablated[key].tolist())
        overlap = len(canon_ids & abl_ids)
        print(
            f"  selection overlap (canonical ∩ ablated): {overlap} / "
            f"{len(canon_ids)} ({100 * overlap / max(1, len(canon_ids)):.1f}%)",
            flush=True,
        )
    else:
        overlap = -1

    # ── Voronoi-deblended SB slopes on the ablated selection ─────────────────
    print(f"[deblend] reading {PER_FILTER_CSV.name} ...", flush=True)
    pf = pd.read_csv(PER_FILTER_CSV, low_memory=False)
    print(f"  rows: {len(pf)}", flush=True)

    print(f"[deblend] reading {TNG_IMAGEAP_CSV.name} ...", flush=True)
    tng = pd.read_csv(TNG_IMAGEAP_CSV, low_memory=False)
    print(f"  rows: {len(tng)}", flush=True)

    # Apply ablated classifier to the same imageap pool used by the
    # original selection_ablation.py for like-for-like comparison.
    tng_with_p = add_selection_probabilities(tng, model, used)
    p = pd.to_numeric(tng_with_p["p_spec"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()
    order = np.argsort(-p)
    selected_mask = np.zeros(len(tng_with_p), dtype=bool)
    selected_mask[order[:TARGET_N]] = True
    selected_idx_new = np.flatnonzero(selected_mask)
    print(
        f"  imageap ablated selection: N={int(selected_mask.sum())} "
        f"(sum(p_spec)={p.sum():.1f})",
        flush=True,
    )

    new_selected_set = set(selected_idx_new.tolist())
    pf_new = pf[pf["object_index"].isin(new_selected_set)].copy()
    print(
        f"  per-filter rows for ablated selection: {len(pf_new)} "
        f"({pf_new['object_index'].nunique()} distinct objects)",
        flush=True,
    )

    flux_pivot = pf_new.pivot_table(
        index="object_index", columns="filter", values="deblended_flux_uJy", aggfunc="first"
    )
    ap_pivot = pf_new.pivot_table(
        index="object_index", columns="filter", values="aperture_flux_uJy", aggfunc="first"
    )
    flux_pivot = flux_pivot.where(flux_pivot.notna(), ap_pivot)

    ap_area = tng.set_index(tng.index)["aperture_area_arcsec2"]
    common = flux_pivot.index.intersection(ap_area.index)
    flux_pivot = flux_pivot.loc[common]
    ap_area_c = ap_area.loc[common]
    sb_per_filter = flux_pivot.div(ap_area_c, axis=0)
    sb_per_filter = sb_per_filter.where(sb_per_filter > 0)

    sb_mean = sb_per_filter.mean(axis=1, skipna=True).to_numpy()
    sb_max = sb_per_filter.max(axis=1, skipna=True).to_numpy()
    z_arr = tng.loc[common, "z"].to_numpy()

    a_mean, e_mean, n_mean = fit_alpha(z_arr, sb_mean)
    a_max, e_max, _ = fit_alpha(z_arr, sb_max)

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n[result] z_proxy ablation:", flush=True)
    print(f"  ROC AUC               = {auc:.3f}   (canonical ~0.91)")
    print(f"  Tolman N (deblended)  = {n_mean}")
    print(f"  mean SB alpha         = {a_mean:+.3f} +/- {e_mean:.3f}   (canonical -0.879 +/- 0.030)")
    print(f"  max  SB alpha         = {a_max:+.3f} +/- {e_max:.3f}   (canonical -0.866 +/- 0.029)")
    print(f"  Delta alpha (mean)    = {a_mean - (-0.879):+.3f}")
    print(f"  Delta alpha (max)     = {a_max - (-0.866):+.3f}")
    print(f"  gamma_rho (vol-ctrl)  = {g_abl:+.3f} +/- {g_err_abl:.3f} OLS | {g_boot_abl:.3f} boot")
    print(f"  gamma_rho (canonical) = {g_canon:+.3f} +/- {g_err_canon:.3f}")
    print(f"  Delta gamma           = {g_abl - g_canon:+.3f}")
    if overlap >= 0:
        print(f"  selection overlap     = {overlap} / {n_canon}")

    summary = pd.DataFrame([
        dict(
            sample="ablated_no_zproxy",
            features=";".join(used),
            auc_train=auc,
            n_deblended=n_mean,
            alpha_mean_sb=a_mean,
            err_mean_sb=e_mean,
            alpha_max_sb=a_max,
            err_max_sb=e_max,
            n_gamma=n_abl,
            gamma_rho=g_abl,
            gamma_rho_err_ols=g_err_abl,
            gamma_rho_err_boot=g_boot_abl,
            selection_overlap=overlap,
            alpha_spec_li=ALPHA_SPEC,
            alpha_sed_paper=-ALPHA_SPEC,
        ),
        dict(
            sample="canonical_reference",
            features=";".join([*ABLATED_FEATURES[:6], "z_proxy", "n_valid_filters"]),
            auc_train=float("nan"),
            n_deblended=2901,
            alpha_mean_sb=-0.879,
            err_mean_sb=0.030,
            alpha_max_sb=-0.866,
            err_max_sb=0.029,
            n_gamma=n_canon,
            gamma_rho=g_canon,
            gamma_rho_err_ols=g_err_canon,
            gamma_rho_err_boot=float("nan"),
            selection_overlap=int(n_canon),
            alpha_spec_li=ALPHA_SPEC,
            alpha_sed_paper=-ALPHA_SPEC,
        ),
    ])
    out_csv = OUT_DIR / "zproxy_ablation_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

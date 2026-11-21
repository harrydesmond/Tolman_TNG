"""Empirical spectroscopic-selection models."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .selection import flux_ujy_to_abmag


DEFAULT_FEATURES = (
    "mag_F200W",
    "mag_F356W",
    "color_F150W_F356W",
    "color_F200W_F444W",
    "log_sb_mean",
    "log_aperture_area",
    "z_proxy",
    "n_valid_filters",
)


def _flux_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col.startswith("flux_") and col.endswith("_uJy") and not col.startswith("fluxerr_")
    ]


def add_spec_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for filter_name in ("F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W"):
        col = f"flux_{filter_name}_uJy"
        if col in out:
            out[f"mag_{filter_name}"] = flux_ujy_to_abmag(out[col])
    for left, right in (("F150W", "F356W"), ("F200W", "F444W"), ("F115W", "F200W")):
        if f"mag_{left}" in out and f"mag_{right}" in out:
            out[f"color_{left}_{right}"] = out[f"mag_{left}"] - out[f"mag_{right}"]
    if "n_valid_filters" not in out:
        flux_cols = _flux_columns(out)
        if flux_cols:
            flux = out[flux_cols].apply(pd.to_numeric, errors="coerce")
            out["n_valid_filters"] = (flux > 0.0).sum(axis=1)
    if "sb_mean" not in out:
        flux_cols = _flux_columns(out)
        if flux_cols:
            flux = out[flux_cols].apply(pd.to_numeric, errors="coerce")
            if "aperture_area_arcsec2" in out:
                area = pd.to_numeric(out["aperture_area_arcsec2"], errors="coerce")
            elif "aperture_area_arcsec2_used" in out:
                area = pd.to_numeric(out["aperture_area_arcsec2_used"], errors="coerce")
            else:
                area = pd.Series(np.nan, index=out.index)
            positive_flux = flux.where(flux > 0.0)
            out["sb_mean"] = positive_flux.mean(axis=1, skipna=True) / area
    if "sb_mean" in out:
        sb = pd.to_numeric(out["sb_mean"], errors="coerce")
        out["log_sb_mean"] = np.where(sb > 0.0, np.log10(sb), np.nan)
    if "aperture_area_arcsec2" in out:
        area = pd.to_numeric(out["aperture_area_arcsec2"], errors="coerce")
    elif "aperture_area_arcsec2_used" in out:
        area = pd.to_numeric(out["aperture_area_arcsec2_used"], errors="coerce")
    else:
        area = pd.Series(np.nan, index=out.index)
    out["log_aperture_area"] = np.where(area > 0.0, np.log10(area), np.nan)
    if "z_phot" in out:
        out["z_proxy"] = pd.to_numeric(out["z_phot"], errors="coerce")
    elif "z" in out:
        out["z_proxy"] = pd.to_numeric(out["z"], errors="coerce")
    else:
        out["z_proxy"] = np.nan
    return out


def available_features(df: pd.DataFrame, requested: list[str] | None = None) -> list[str]:
    features = requested or list(DEFAULT_FEATURES)
    return [feature for feature in features if feature in df.columns]


def fit_model(
    df: pd.DataFrame,
    *,
    target_col: str = "has_spec",
    features: list[str] | None = None,
    class_weight: str | dict | None = None,
) -> tuple[Pipeline, list[str]]:
    feat_df = add_spec_features(df)
    used = available_features(feat_df, features)
    if not used:
        raise ValueError("No requested spectroscopic-selection features are available")
    target = pd.to_numeric(feat_df[target_col], errors="coerce").fillna(0).astype(bool)
    if target.nunique() != 2:
        raise ValueError(f"Target column {target_col!r} must contain both selected and unselected objects")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(max_iter=2000, class_weight=class_weight)),
        ]
    )
    model.fit(feat_df[used], target)
    return model, used


def holdout_roc_auc(
    df: pd.DataFrame,
    *,
    target_col: str = "has_spec",
    features: list[str] | None = None,
    class_weight: str | dict | None = None,
    test_size: float = 0.2,
    seed: int = 0,
) -> float:
    """Return ROC AUC on a held-out split of *df*.

    Trains the same logistic-regression pipeline as :func:`fit_model` on a
    random train fraction and evaluates ROC AUC on the complementary test
    fraction. The split is stratified on *target_col* so that both halves
    contain selected and unselected sources.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    feat_df = add_spec_features(df)
    used = available_features(feat_df, features)
    if not used:
        raise ValueError("No requested spectroscopic-selection features are available")
    target = pd.to_numeric(feat_df[target_col], errors="coerce").fillna(0).astype(int)
    if target.nunique() != 2:
        raise ValueError(f"Target column {target_col!r} must contain both selected and unselected objects")
    X_train, X_test, y_train, y_test = train_test_split(
        feat_df[used], target, test_size=test_size, random_state=seed, stratify=target,
    )
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(max_iter=2000, class_weight=class_weight)),
        ]
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, proba))


def save_model(model: Pipeline, features: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features}, path)


def load_model(path: str | Path) -> tuple[Pipeline, list[str]]:
    payload = joblib.load(path)
    return payload["model"], list(payload["features"])


def add_selection_probabilities(df: pd.DataFrame, model: Pipeline, features: list[str]) -> pd.DataFrame:
    out = add_spec_features(df)
    missing = [feature for feature in features if feature not in out]
    if missing:
        for feature in missing:
            out[feature] = np.nan
    out["p_spec"] = model.predict_proba(out[features])[:, 1]
    return out


def select_by_probability(
    df: pd.DataFrame,
    *,
    mode: str,
    target_count: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    out = df.copy()
    p = pd.to_numeric(out["p_spec"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if mode == "deterministic":
        if target_count is None:
            target_count = int(round(float(p.sum())))
        order = np.argsort(-p.to_numpy())
        selected = np.zeros(len(out), dtype=bool)
        selected[order[: max(0, min(int(target_count), len(out)))]] = True
    elif mode == "probabilistic":
        rng = np.random.default_rng(seed)
        selected = rng.random(len(out)) < p.to_numpy()
    else:
        raise ValueError("mode must be 'deterministic' or 'probabilistic'")
    out["mock_spec_selected"] = selected
    return out

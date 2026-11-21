"""Lightweight source detection for mock-image diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import ndimage


def robust_background(image: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(image[np.isfinite(image)], dtype=float)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    sigma = 1.4826 * mad if mad > 0.0 else float(np.std(finite))
    return med, sigma


def segment_sources(
    image: np.ndarray,
    *,
    threshold_sigma: float = 3.0,
    min_pixels: int = 8,
    edge_buffer_pix: int = 3,
) -> tuple[pd.DataFrame, np.ndarray, float, float]:
    """Detect connected threshold islands and return a 1-based label image."""

    bg, sigma = robust_background(image)
    mask = np.isfinite(image) & (image > bg + threshold_sigma * sigma)
    labels, n_labels = ndimage.label(mask)
    source_labels = np.zeros(labels.shape, dtype=np.int32)
    if n_labels == 0:
        return pd.DataFrame(), source_labels, bg, sigma

    counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
    weights_image = np.where(labels > 0, image - bg, 0.0)
    fluxes = np.bincount(labels.ravel(), weights=weights_image.ravel(), minlength=n_labels + 1)
    valid = np.flatnonzero((counts >= min_pixels) & (fluxes > 0.0))
    valid = valid[valid != 0]
    if len(valid) == 0:
        return pd.DataFrame(), source_labels, bg, sigma

    mapping = np.zeros(n_labels + 1, dtype=np.int32)
    mapping[valid] = np.arange(1, len(valid) + 1, dtype=np.int32)
    source_labels = mapping[labels]

    centers = ndimage.center_of_mass(np.maximum(weights_image, 0.0), labels, valid)
    objects = ndimage.find_objects(labels)
    rows: list[dict[str, float | int | bool]] = []
    ny, nx = image.shape
    for source_id, (label, center) in enumerate(zip(valid, centers, strict=True)):
        obj = objects[label - 1]
        if obj is None:
            continue
        y_slice, x_slice = obj
        y_center, x_center = center
        n_pix = int(counts[label])
        flux = float(fluxes[label])
        edge = bool(
            x_slice.start < edge_buffer_pix
            or y_slice.start < edge_buffer_pix
            or x_slice.stop > nx - edge_buffer_pix
            or y_slice.stop > ny - edge_buffer_pix
        )
        rows.append(
            {
                "source_id": source_id,
                "x": float(x_center + 0.5),
                "y": float(y_center + 0.5),
                "seg_area_pix": n_pix,
                "aperture_radius_pix": float(np.sqrt(n_pix / np.pi)),
                "detection_flux": flux,
                "is_edge": edge,
            }
        )
    return pd.DataFrame(rows), source_labels, bg, sigma


def detect_sources(
    image: np.ndarray,
    *,
    threshold_sigma: float = 3.0,
    min_pixels: int = 8,
    edge_buffer_pix: int = 3,
) -> pd.DataFrame:
    detections, _, _, _ = segment_sources(
        image,
        threshold_sigma=threshold_sigma,
        min_pixels=min_pixels,
        edge_buffer_pix=edge_buffer_pix,
    )
    return detections


def detect_sources_from_fits(
    image_path: str | Path,
    *,
    threshold_sigma: float = 3.0,
    min_pixels: int = 8,
    edge_buffer_pix: int = 3,
) -> pd.DataFrame:
    with fits.open(image_path, memmap=True) as hdul:
        image = np.asarray(hdul[0].data, dtype=float)
    return detect_sources(
        image,
        threshold_sigma=threshold_sigma,
        min_pixels=min_pixels,
        edge_buffer_pix=edge_buffer_pix,
    )

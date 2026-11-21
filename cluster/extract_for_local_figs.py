"""Cluster-side data extraction so the final figures can be made locally.

Produces, in ``$WORKDIR/data/``:
    tng100_mockspec_z.csv
    tng100_mockspec_z_sbmean_deblended.csv
    psf_noise_injection_per_galaxy.csv

``WORKDIR`` defaults to the repository root; set the environment
variable ``TOLMAN_WORKDIR`` to override when running on a different cluster
or workstation.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

WORKDIR = Path(os.environ.get("TOLMAN_WORKDIR", Path(__file__).resolve().parents[1]))
OUT_DIR = WORKDIR / "data"

TNG_COMMON = ["F435W","F606W","F814W","F125W","F140W","F160W",
              "F090W","F115W","F150W","F200W","F277W","F356W","F444W"]


def extract_redshift_data():
    spec = pd.read_csv(
        WORKDIR / "data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv",
        usecols=["z", "mock_spec_selected"],
        low_memory=False,
    )
    z = spec.loc[spec["mock_spec_selected"].astype(bool), "z"].dropna()
    z.to_csv(OUT_DIR / "tng100_mockspec_z.csv", index=False, header=["z"])
    print(f"saved tng100_mockspec_z.csv ({len(z)} rows, median {z.median():.3f})")


def extract_tng_sb_data():
    flux_cols = [f"flux_{f}_uJy" for f in TNG_COMMON]
    use_cols = ["z", "mock_spec_selected"] + flux_cols
    # Try to also pull aperture_area from any filter in the catalogue
    candidate_area_cols = [f"aperture_area_arcsec2_{f}" for f in TNG_COMMON]
    candidate_area_cols.append("aperture_area_arcsec2")
    # First read only header to see what's there
    hdr = pd.read_csv(WORKDIR / "data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv",
                      nrows=0).columns.tolist()
    area_cols = [c for c in candidate_area_cols if c in hdr]
    if not area_cols:
        raise RuntimeError("No aperture_area column found in TNG specmodel CSV")
    use_cols += area_cols
    use_cols = [c for c in use_cols if c in hdr]
    tng = pd.read_csv(WORKDIR / "data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv",
                      usecols=use_cols, low_memory=False)
    tng = tng.reset_index(drop=False).rename(columns={"index": "object_index"})

    # patch fluxes with deblended values per filter
    perfilt = pd.read_csv(WORKDIR / "results/tng100_7_6_xyz_hst_jwst_imageap_deblended/deblending_per_filter.csv")
    for filt in perfilt["filter"].unique():
        flux_col = f"flux_{filt}_uJy"
        if flux_col not in tng.columns:
            continue
        sub = perfilt[perfilt["filter"] == filt][["object_index", "deblended_flux_uJy"]]
        sub = sub.set_index("object_index")["deblended_flux_uJy"]
        valid = sub[sub.notna() & (sub > 0.0)]
        idx = tng.index[tng["object_index"].isin(valid.index)]
        # map back via object_index
        merged = tng.loc[idx, "object_index"].map(valid)
        tng.loc[idx, flux_col] = merged.values

    # compute per-galaxy mean SB across the TNG-common filters present
    sb_cols = []
    for f in TNG_COMMON:
        flux_col = f"flux_{f}_uJy"
        area_col = f"aperture_area_arcsec2_{f}" if f"aperture_area_arcsec2_{f}" in tng.columns else "aperture_area_arcsec2"
        if flux_col not in tng.columns or area_col not in tng.columns:
            continue
        flux = tng[flux_col].to_numpy(dtype=float)
        area = tng[area_col].to_numpy(dtype=float)
        sb = np.where((flux > 0) & np.isfinite(flux) & (area > 0) & np.isfinite(area),
                      flux / area, np.nan)
        sb_cols.append(sb)
    sb_mat = np.column_stack(sb_cols)
    with np.errstate(invalid="ignore"):
        sb_mean = np.nanmean(sb_mat, axis=1)

    z = tng["z"].to_numpy(dtype=float)
    sel = tng["mock_spec_selected"].astype(bool).to_numpy()
    mask = sel & np.isfinite(z) & np.isfinite(sb_mean) & (sb_mean > 0)

    out = pd.DataFrame({"z": z[mask], "sb_mean": sb_mean[mask]})
    out.to_csv(OUT_DIR / "tng100_mockspec_z_sbmean_deblended.csv", index=False)
    print(f"saved tng100_mockspec_z_sbmean_deblended.csv ({len(out)} rows)")


def extract_psf_noise_per_galaxy():
    from astropy.io import fits
    from scipy import ndimage

    PSF_SIGMA_PIX = 1.0
    SKY_SIGMA_NJY = 0.1
    RNG_SEED = 0
    FITS_F200W = WORKDIR / "data/raw/illustris/tng100-7-6-xyz_137sqarcmin/hlsp_illustris_jwst_nircam_tng100-7-6-xyz_f200w_v1_sim.fits"
    SPECMODEL_CSV = WORKDIR / "data/processed_tng100_7_6_xyz_hst_jwst_imageap_specmodel.csv"

    def astrodeep_radius_arcsec(z):
        # apopt is a diameter in Merlin+2024; the matching aperture radius is apopt/2.
        z = np.asarray(z, dtype=float)
        r = np.full_like(z, np.nan)
        r[(z >= 0) & (z < 2)] = 0.66
        r[(z >= 2) & (z < 3)] = 0.25
        r[(z >= 3) & (z < 4)] = 0.165
        r[z >= 4] = 0.14
        return r

    def aperture_flux(image, x, y, r_pix):
        out = np.full(len(x), np.nan)
        ny, nx = image.shape
        for i in range(len(x)):
            if not (np.isfinite(x[i]) and np.isfinite(y[i]) and np.isfinite(r_pix[i])):
                continue
            rp = float(r_pix[i])
            if rp <= 0:
                continue
            xc = float(x[i]); yc = float(y[i])
            x0 = max(0, int(np.floor(xc - rp)))
            x1 = min(nx, int(np.ceil(xc + rp)) + 1)
            y0 = max(0, int(np.floor(yc - rp)))
            y1 = min(ny, int(np.ceil(yc + rp)) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = image[y0:y1, x0:x1]
            py = np.arange(y0, y1) + 0.5
            px = np.arange(x0, x1) + 0.5
            dx = (px - xc); dy = (py - yc)
            d2 = dy[:, None] ** 2 + dx[None, :] ** 2
            mask = d2 <= rp ** 2
            out[i] = float(np.nansum(sub[mask]))
        return out

    print("[psf-noise] loading FITS ...")
    with fits.open(FITS_F200W, memmap=False) as hdul:
        raw = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header

    print("[psf-noise] loading specmodel ...")
    spec = pd.read_csv(SPECMODEL_CSV, low_memory=False)
    sel = spec["mock_spec_selected"].astype(bool).to_numpy()
    selected = spec.iloc[np.flatnonzero(sel)].copy()
    x = selected["x_F200W_pix"].to_numpy()
    y = selected["y_F200W_pix"].to_numpy()
    z = selected["z"].to_numpy()
    kpc_per_arcsec = selected["kpc_per_arcsec"].to_numpy()
    photrad_kpc = selected["aperture_radius_arcsec_F200W"].to_numpy() * kpc_per_arcsec

    try:
        from astropy.wcs import WCS
        import astropy.units as u
        wcs = WCS(hdr)
        pixscales = wcs.proj_plane_pixel_scales()
        arcsec_per_pix = float(pixscales[0].to(u.arcsec).value)
    except Exception:
        arcsec_per_pix = float(np.sqrt(137.0 * 3600.0 / float(raw.shape[0] * raw.shape[1])))
    if not (0.005 < arcsec_per_pix < 0.5):
        arcsec_per_pix = float(np.sqrt(137.0 * 3600.0 / float(raw.shape[0] * raw.shape[1])))

    r_fixed_pix = (photrad_kpc / kpc_per_arcsec) / arcsec_per_pix
    r_adaptive_arcsec = astrodeep_radius_arcsec(z)
    r_adaptive_pix = r_adaptive_arcsec / arcsec_per_pix

    print("[psf-noise] noise-free aperture flux ...")
    flux_fixed_clean = aperture_flux(raw, x, y, r_fixed_pix) / 1000.0
    flux_adapt_clean = aperture_flux(raw, x, y, r_adaptive_pix) / 1000.0

    print("[psf-noise] PSF + noise ...")
    img_psf = ndimage.gaussian_filter(raw, sigma=PSF_SIGMA_PIX).astype(np.float32)
    rng = np.random.default_rng(RNG_SEED)
    img_noisy = img_psf + rng.normal(0.0, SKY_SIGMA_NJY, size=raw.shape).astype(np.float32)
    del img_psf
    flux_fixed_noisy = aperture_flux(img_noisy, x, y, r_fixed_pix) / 1000.0
    flux_adapt_noisy = aperture_flux(img_noisy, x, y, r_adaptive_pix) / 1000.0

    ap_fixed_arcsec2 = np.pi * (r_fixed_pix * arcsec_per_pix) ** 2
    ap_adapt_arcsec2 = np.pi * r_adaptive_arcsec ** 2
    df = pd.DataFrame({
        "z": z,
        "sb_fixed_clean": flux_fixed_clean / ap_fixed_arcsec2,
        "sb_fixed_noisy": flux_fixed_noisy / ap_fixed_arcsec2,
        "sb_adapt_clean": flux_adapt_clean / ap_adapt_arcsec2,
        "sb_adapt_noisy": flux_adapt_noisy / ap_adapt_arcsec2,
    })
    df.to_csv(OUT_DIR / "psf_noise_injection_per_galaxy.csv", index=False)
    print(f"saved psf_noise_injection_per_galaxy.csv ({len(df)} rows)")


def main():
    extract_redshift_data()
    extract_tng_sb_data()
    extract_psf_noise_per_galaxy()
    print("DONE")


if __name__ == "__main__":
    main()

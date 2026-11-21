# TNG100 Image-Aperture Shift Diagnostics

Matched catalogue rows: 71,463.
Catalogue-flux empirical-spec selected rows in matched table: 3,704.
Image-aperture empirical-spec selected rows in matched table: 2,901.
Selection overlap: 2,803 rows.

| Observable | alpha cat/original | alpha cat/image selection | alpha image/image selection | selection delta | aperture delta | total delta | ratio alpha |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean_sb | -1.015 | -1.134 | -0.718 | -0.119 | +0.415 | +0.296 | +0.415 |
| max_sb | -1.003 | -1.112 | -0.703 | -0.109 | +0.410 | +0.301 | +0.410 |
| rest_uv_1500_2000 | 0.022 | -0.387 | 0.594 | -0.409 | +0.980 | +0.571 | +0.980 |

Interpretation guide:
- `selection delta` isolates the changed empirical spectroscopic-selection sample while keeping catalogue fluxes.
- `aperture delta` isolates image-aperture measurement at fixed image-aperture selected rows.
- `ratio alpha` is the fitted redshift evolution of image-aperture SB divided by catalogue SB.

Plots:
- `Plots/tng100_7_6_xyz_hst_jwst_imageap_shift_slope_decomposition.png`
- `Plots/tng100_7_6_xyz_hst_jwst_imageap_shift_aperture_ratio_tracks.png`
- `Plots/tng100_7_6_xyz_hst_jwst_imageap_shift_spec_selection_redshift.png`
- `Plots/tng100_7_6_xyz_hst_jwst_imageap_shift_aperture_physics_tracks.png`
- `Plots/tng100_7_6_xyz_hst_jwst_imageap_shift_ratio_vs_aperture_radius.png`

Object-level aperture-physics diagnostics:
- `object_flux_ratio_predictors.csv` contains one row per image-aperture object with median image/catalogue flux ratio and physical predictors.
- `aperture_physics_conditional_redshift_models.csv` tests whether the ratio-redshift slope remains after controlling for aperture and galaxy-property proxies.

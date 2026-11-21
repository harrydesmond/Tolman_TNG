# Data Staging

Raw catalogues and FITS images are not stored in this repository.

Expected inputs for a full reproduction:

- ASTRODEEP/CDS catalogue tables from VizieR `J/A+A/691/A240`, staged under
  `data/raw/astrodeep/cds/`.
- MAST ILLUSTRIS/TNG FITS products for the TNG100
  `tng100-7-6-xyz_137sqarcmin` field (~24 GB), staged locally or on a
  cluster filesystem.
- The trained ASTRODEEP spectroscopic-selection model is small and is included
  under `results/selection/astrodeep_spec_selection.joblib`.
- Compact ultracompact-radio-source catalogues used by the Li (2023)
  distance-duality reproduction are bundled under `data/li2023/` (JJ2.29 from
  Jackson & Jannetta 2006 and GKF5.0 from Gurvits, Kellermann & Frey 1999).

The repository includes compact result tables and plots for the paper's
results, so the published comparisons can be checked without downloading the
full raw data. It also includes five lightweight derived plotting caches in
`data/` (`astrodeep_spec_z.csv`, `astrodeep_tng_common_spec_z_sbmean.csv`,
`tng100_mockspec_z.csv`, `tng100_mockspec_z_sbmean_deblended.csv` and
`psf_noise_injection_per_galaxy.csv`) so that the selected final and diagnostic
figures can be regenerated without the large measured catalogues.

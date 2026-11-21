# Forward-modelling the Tolman and distance-duality tests with IllustrisTNG

Code and compact result tables for Desmond, Yasin, Stiskalek & von Hausegger
(2026), MNRAS, [doi:10.1093/mnras/stag2124](https://doi.org/10.1093/mnras/stag2124)
([arXiv:2606.27182](https://arxiv.org/abs/2606.27182)).

The code forward-models the JWST/ASTRODEEP Tolman surface-brightness (SB) test
and the ultracompact-radio distance-duality (DDR) test with IllustrisTNG mock
images. The two observables are:

```text
I_SB              = F_nu / A_ap
log10 (I_SB)      = beta_SB  + alpha_Tol log10(1 + z)

D_L / D_A         = D_L * theta_calib(z) / d_0
log10(D_L / D_A)  = c_DDR + alpha_DDR log10(1 + z)
```

Both are sensitive to intrinsic source evolution. The Tolman slope responds
to the projected luminosity density `Sigma_L = L / A_phys` and the Li (2023)
DDR estimator to the volumetric luminosity density `rho_L = L / V`. In the
matched fixed-aperture TNG sample the two evolve at the same rate
(`gamma_Sigma = 2.462`, `gamma_rho = 2.457`), so the paper uses a single
exponent `gamma`:

```text
alpha_Tol           = -(3 + alpha_SED) + gamma
alpha_DDR           = 2 - gamma/3
D_L / D_A_Li        ~ (1+z)^alpha_DDR
```

This equality is a property of the matched TNG sample. It is not assumed for
other populations or for the redshift-dependent aperture stress test.

The code (i) reproduces the ASTRODEEP-JWST Tolman slopes of Tsymbal et al.
(2026) and the Li (2023) DDR reduction of the JJ2.29 and GKF5.0
ultracompact-radio samples, (ii) measures the TNG100 SB slope for four
aperture definitions (catalogue fluxes, circular image apertures,
Voronoi-deblended image fluxes and ASTRODEEP-like adaptive apertures) and
(iii) measures `gamma` directly on the mock-spectroscopic TNG sample and
passes it through Li's procedure to give a TNG DDR track. The Tolman
prediction and the TNG DDR track use the same TNG forward model and the same
empirical mock-spectroscopic selection. The observed radio DDR slope gives a
separate estimate of `gamma` for compact radio sources.

## Main results

The result tables are in `results/` and the six paper figures in
`Plots/final/`. Bright-galaxy evolution in TNG flattens the Tolman slope by
an amount comparable to that measured with JWST and produces much of the
apparently non-expanding behaviour of the radio DDR. The radio data prefer slightly stronger luminosity-density
evolution for compact radio sources than for the bright galaxies modelled in
TNG.

All `gamma` values assume `alpha_SED = 0.3`. The code follows Li's convention
`F_nu propto nu^alpha_spec`, so `alpha_spec = -alpha_SED`, and `gamma` shifts
by `+Delta alpha_SED` if `alpha_SED` is changed. For the broad-band Tolman
observable used here the `alpha_SED = 0.3` reference slopes are `-3.3` (FRW)
and `-0.3` (non-expanding). The classical `-4` is the bolometric slope.

| Estimator | Value | Source |
|---|---:|---|
| `gamma` from TNG (half-range across four aperture variants) | `2.23 +/- 0.20` | `results/li_ddr/tng_gamma_aperture_table.csv` |
| `gamma` from radio DDR (JJ2.29 and GKF5.0 reduced separately, then combined; `L propto d^3`) | `2.50 +/- 0.26` | `results/li_ddr/ddr_radio_gamma_eff.csv` |
| `gamma` from the ASTRODEEP Tolman slope | `2.02 +/- 0.05` | OLS `0.018` and cut-to-cut `0.045` in quadrature |

The radio and TNG values differ by `Delta gamma = 0.27 +/- 0.33`, i.e.
`rho_L_radio / rho_L_gal ~ (1+z)^(0.27 +/- 0.33)`.

Tolman slopes:

| Measurement | mean SB `alpha_Tol` | max SB `alpha_Tol` | modal max `alpha_Tol` | Notes |
|---|---:|---:|---:|---|
| ASTRODEEP, TNG-common filters | `-1.279 +/- 0.05` | `-1.130 +/- 0.020` | `-1.101 +/- 0.095` | Mean-SB error: OLS `0.018` and cut-to-cut `0.045` in quadrature |
| TNG100 catalogue flux | `-1.015 +/- 0.019` | `-1.003 +/- 0.018` | `-0.841 +/- 0.112` | MAST `total_quant`, empirical ASTRODEEP spec selection |
| TNG100 circular image aperture | `-0.718 +/- 0.030` | `-0.703 +/- 0.029` | `-0.881 +/- 0.062` | Aperture fluxes from the FITS images |
| TNG100 Voronoi-deblended (fiducial) | `-0.879 +/- 0.030` | `-0.866 +/- 0.029` | `-1.009 +/- 0.052` | |
| TNG100 ASTRODEEP-like adaptive aperture | `+0.695 +/- 0.046` | `+0.717 +/- 0.047` | `+1.640 +/- 0.299` | Stress test of the aperture rule |

The TNG DDR track in `Plots/final/ddr_bridge.pdf` lies close to the radio
data because the mock-spectroscopic TNG sample has `gamma ~ 2-3`, the regime
that Li's procedure picks up in the radio sample.

## Script index

Outputs are under `results/li_ddr/` and scripts under `cluster/` unless
stated otherwise.

| Topic | Script | Output |
|---|---|---|
| ASTRODEEP normalisation | `tng100_image_aperture_*.sh` (catalogue pipeline) + `tolman.cli astrodeep-normalize` | `results/astrodeep_cds/` |
| ASTRODEEP Tolman slopes | `tolman.cli measure` | `results/astrodeep_cds_tng_common/slope_table.csv` |
| TNG SB slopes for the four aperture variants | `tng100_image_aperture_analyse.sh`, `tng100_deblending_diagnostics.sh`, `tng100_deblended_sb_slopes.sh`, `tng100_adaptive_aperture_sb.sh` | `results/tng100_7_6_xyz_hst_jwst_*_specmodel/slope_table.csv` |
| TNG `gamma` for the four aperture variants | `tng_gamma_aperture_table.{py,sh}` | `tng_gamma_aperture_table.csv` |
| Radio-data `gamma` | `ddr_radio_gamma_eff.{py,sh}` | `ddr_radio_gamma_eff.csv` |
| Dependence of `gamma` on the selection (full pool, brightest K, random and permuted-label selections) | `full_pool_gamma.{py,sh}`, `tng_gamma_spec_selection_nulls.{py,sh}` | `full_pool_gamma.csv`, `tng_gamma_spec_selection_nulls.csv` |
| Redshift-distribution reweighting | `reweight_extended.{py,sh}` | `results/systematics/reweight_extended_summary.csv` |
| Compactness selection bias | `zdist_compactness.{py,sh}` | `results/systematics/compactness_bias_summary.csv` |
| ASTRODEEP sample-cut variations | `scripts/astrodeep_slope_stability.py` | printed to screen |
| Gaussian-weighted deblending | `gaussian_deblend.{py,sh}` | `results/systematics/deblend_comparison_f200w.csv` |
| PSF and sky-noise injection | `psf_noise_injection.{py,sh}` | `results/systematics/psf_noise_injection_f200w.csv` |
| Cosmic variance (line-of-sight rotation and sub-blocks) | `cosmic_variance_yxz.sh`, `cosmic_variance_yxz_postdownload.sh`, `cosmic_variance_yxz_imageap.sh`, `cosmic_variance_subblocks.{py,sh}` | `results/li_ddr/cosmic_variance/*.csv` |
| Classifier feature ablations | `selection_ablation.{py,sh}` (drops `log_sb_mean` and `log_aperture_area`), `zproxy_ablation.{py,sh}` (drops `z_proxy`) | `results/systematics/{selection_ablation,zproxy_ablation}_summary.csv` |
| Li (2023) DDR reproduction | `scripts/li_ddr_run.py` + `tolman.li_ddr` | `li_radio_table1.csv`, `radio_per_source_ddr.csv`, `tng_per_source_ddr.csv` |
| SB normalisation of the TNG variants relative to ASTRODEEP | `scripts/li_ddr_normalisation_check.py` | `sb_normalisation_check.csv` |
| Fig. 1 (`gamma` schematic) | `scripts/make_gamma_schematic.py` | `Plots/final/gamma_schematic.pdf` |
| Figs 2, 3 and 5 (redshift distributions, mean-SB fits, slope comparison) | `scripts/make_final_figures.py` | `Plots/final/zdist.pdf`, `mean_sb_fits.pdf`, `aperture_ladder_bars.pdf` |
| Fig. 4 (DDR comparison) | `scripts/fig_ddr_normalisation_compare.py` | `Plots/final/ddr_bridge.pdf` |
| Fig. 6 (`rho_L` evolution) | `scripts/fig_ddr_alt_views.py` | `Plots/final/rho_L_evolution.pdf` |

The two DDR figure scripts write only the paper figure by default; pass
`--write-diagnostics` to also write exploratory PNG versions under `Plots/`.

## Contents

- `tolman/`: analysis package. The Tolman SB pipeline is at the top level and
  the DDR pipeline in `tolman.li_ddr` (Li 2023 reproduction on the radio
  samples and the matched TNG analogue).
- `configs/`: TOML schemas for ASTRODEEP and TNG-style catalogues.
- `cluster/`: batch scripts for the TNG100 FITS-image pipelines, the `gamma`
  measurements and the systematics tests (see the resource table below).
- `scripts/`: analysis drivers and figure scripts that run on a workstation.
  `reproduce_main_results.sh` lists the full command sequence.
- `tests/`: unit tests, including `test_li_ddr.py`, which reproduces Li (2023)
  Table 1 row by row.
- `results/`: compact result tables behind the numbers in the paper.
- `Plots/`: `Plots/final/*.pdf` are the six paper figures,
  `Plots/diagnostics/*.pdf` supporting diagnostics and `Plots/li_ddr/*.png`
  DDR-reduction diagnostics.

Raw CDS tables, MAST/TNG FITS images and large measured catalogues are not
included; see `data/README.md`.

Paper figures:

- `gamma_schematic.pdf` (Fig. 1): maps `alpha_Tol` and `alpha_DDR` onto
  `gamma`, with FRW and non-expanding references and bands at
  `gamma = 2.23 +/- 0.20` (TNG), `2.02 +/- 0.05` (Tolman data) and
  `2.50 +/- 0.26` (DDR data).
- `zdist.pdf` (Fig. 2): ASTRODEEP and TNG redshift distributions.
- `mean_sb_fits.pdf` (Fig. 3): mean-SB power-law fits for ASTRODEEP and TNG
  with the `alpha_SED = 0.3` reference slopes `-3.3` and `-0.3`.
- `ddr_bridge.pdf` (Fig. 4): binned DDR ratio with non-expanding `d_0`, FRW
  `d_0` and self-anchored normalisations.
- `aperture_ladder_bars.pdf` (Fig. 5): Tolman slopes for the ASTRODEEP data
  and the TNG aperture variants.
- `rho_L_evolution.pdf` (Fig. 6): `rho_L(z)/rho_L(0) = (1+z)^gamma` for the
  four TNG aperture variants and the radio data.

## Install

Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the tests:

```bash
MPLCONFIGDIR=/tmp/matplotlib-tolman pytest -q
```

Synthetic smoke test:

```bash
MPLCONFIGDIR=/tmp/matplotlib-tolman python3 -m tolman.cli demo \
  --result-dir results/demo \
  --plot-dir Plots \
  --sample-name demo \
  --save-measured-catalog
```

## Reproducing from raw data

### 1. ASTRODEEP/CDS catalogues

Download the CDS/VizieR tables:

```bash
mkdir -p data/raw/astrodeep/cds

tables=(
  a2744o a2744p ceerso ceersp jadesgno jadesgnp
  jadesgso jadesgsp ngdeepo ngdeepp primerco primercp
  primeruo primerup
)
base_url="https://vizier.cds.unistra.fr/viz-bin/asu-tsv"

for table in "${tables[@]}"; do
  query="?-source=J/A%2BA/691/A240/${table}&-out.all&-out.max=200000"
  curl -L -s --fail --connect-timeout 20 --max-time 300 \
    "${base_url}${query}" \
    -o "data/raw/astrodeep/cds/${table}.tsv"
done
```

Normalise each field with `astrodeep-normalize`. The catalogue `apopt`
column is the per-source aperture *diameter* (Merlin et al. 2024), so pass
`--aperture-kind diameter` (the default):

```bash
python3 -m tolman.cli astrodeep-normalize \
  --phot-catalog data/raw/astrodeep/cds/a2744o.tsv \
  --redshift-catalog data/raw/astrodeep/cds/a2744p.tsv \
  --field-name a2744 \
  --aperture-kind diameter \
  --out data/astrodeep_a2744_cds.csv \
  --config-out configs/astrodeep_cds.toml
```

Repeat for the other fields, concatenate into `data/astrodeep_cds_all.csv`
and run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-tolman python3 -m tolman.cli measure \
  --catalog data/astrodeep_cds_all.csv \
  --config configs/astrodeep_cds_tng_common.toml \
  --result-dir results/astrodeep_cds_tng_common \
  --plot-dir Plots \
  --sample-name astrodeep_cds_tng_common \
  --save-measured-catalog
```

### 2. TNG100 MAST/Illustris products

The `tng100-7-6-xyz_137sqarcmin` FITS set is about 24 GB. Stage it under

```text
data/raw/illustris/tng100-7-6-xyz_137sqarcmin/
```

or set `TNG100_FITS_GLOB` to its location. `scripts/reproduce_main_results.sh`
lists the full command sequence. The image-level TNG100 runs use the scripts
in `cluster/`. Each shell script reads a `WORKDIR` environment variable
(default: the repository root) and the Python scripts read `TOLMAN_WORKDIR`,
so both can point to wherever the FITS images are staged.

## Cluster resource requirements

The shell scripts in `cluster/` are written for a Slurm batch queue. They
expect Python 3.11+ on `PATH` and read `WORKDIR` and `MPLCONFIGDIR`. Typical
resources:

| Script(s) | Cores | Memory | Wall time | Notes |
|---|---:|---:|---:|---|
| `tng100_xyz_download_analyse.sh`, `cosmic_variance_yxz.sh` | 4 | ~32 GB total (8 GB/core) | ~12–24 h | Includes the ~24 GB FITS download and the image-aperture pipeline |
| `tng100_image_aperture_analyse.sh`, `cosmic_variance_yxz_imageap.sh`, `cosmic_variance_yxz_postdownload.sh` | 4 | ~32 GB | ~6–12 h | Image-aperture photometry on the FITS pixels |
| `tng100_deblending_diagnostics.sh`, `tng100_deblended_sb_slopes.sh`, `tng100_adaptive_aperture_sb.sh`, `tng100_aperture_sweep.sh`, `tng100_annulus_contamination.sh`, `tng100_image_aperture_diagnostics.sh` | 4 | ~32 GB | ~2–6 h each | Pixel-level diagnostics |
| `cosmic_variance_subblocks.{py,sh}`, `psf_noise_injection.{py,sh}`, `gaussian_deblend.{py,sh}` | 2–4 | ~8–16 GB | ~1–3 h | Measured catalogues and small FITS cut-outs |
| `tng100_segmentation_diagnostics.sh` | 1 | ~4 GB | ~1 h | Single worker by default |
| `reweight_extended.{py,sh}`, `selection_ablation.{py,sh}`, `zdist_compactness.{py,sh}`, `zproxy_ablation.{py,sh}`, `tng_gamma_aperture_table.{py,sh}`, `tng_gamma_spec_selection_nulls.{py,sh}`, `ddr_radio_gamma_eff.{py,sh}`, `full_pool_gamma.{py,sh}` | 1 | ~2–4 GB | <2 min each | Catalogue-level; also run on a workstation |

The image-aperture CLI runs its own `ProcessPoolExecutor`, so submit each
shell script as a single Python process with the requested number of cores.
If the scheduler starts one copy of the script per core, the worker pools
multiply and the job runs out of memory at the pixel stage. The scripts in
the last row of the table are single-process.

## Output tables

- `results/astrodeep_cds_tng_common/slope_table.csv`: ASTRODEEP Tolman slopes
  on the TNG-common filter set (reproduction of Tsymbal et al. 2026).
- `results/tng100_7_6_xyz_hst_jwst_specmodel/slope_table.csv`,
  `results/tng100_7_6_xyz_hst_jwst_imageap_specmodel/slope_table.csv`,
  `results/tng100_7_6_xyz_hst_jwst_imageap_deblended_sb/slope_table.csv`,
  `results/tng100_7_6_xyz_hst_jwst_imageap_astrodeep_aperture/slope_table.csv`:
  TNG slopes for the catalogue, image-aperture, Voronoi-deblended and
  adaptive-aperture variants.
- `results/tng100_7_6_xyz_hst_jwst_imageap_shift/slope_decomposition_summary.csv`:
  decomposition of the slope into flux and aperture-area evolution.
- `results/tng100_7_6_xyz_hst_jwst_imageap_segmentation/segmentation_ratio_fits.csv`,
  `results/tng100_7_6_xyz_hst_jwst_imageap_sweep/aperture_sweep_slope_table.csv`:
  aperture and segmentation diagnostics used in the systematics discussion.
- `results/li_ddr/li_radio_table1.csv`: reproduction of Li (2023) Table 1.
- `results/li_ddr/tng_gamma_aperture_table.csv`: TNG `gamma` for each
  aperture variant (Table 4 and Fig. 6).
- `results/li_ddr/ddr_radio_gamma_eff.csv`: radio-data `gamma` for the
  `L propto d^2` and `L propto d^3` mappings, for the Li-style separate
  reductions and for a joint refit of the raw catalogues.
- `results/li_ddr/tng_gamma_spec_selection_nulls.csv` (and `_summary.csv`):
  `gamma` for randomised and permuted-label selections.
- `results/li_ddr/full_pool_gamma.csv`: `gamma` for the full
  image-aperture pool, its z<4 subset, the brightest K=2901 galaxies and the
  mock-spectroscopic sample.
- `results/li_ddr/tng_rho_L_per_galaxy_deblended.csv`: per-galaxy
  `rho_L = L/V` for the Voronoi-deblended sample (binned medians in Fig. 6).
- `results/systematics/`: outputs of the systematics tests in the script
  index.

## Contact

Harry Desmond (harry.desmond@port.ac.uk).

## Citation

If you use this code, please cite the paper:

```bibtex
@ARTICLE{2026arXiv260627182D,
       author = {{Desmond}, Harry and {Yasin}, Tariq and {Stiskalek}, Richard
                 and {von Hausegger}, Sebastian},
        title = "{Forward-modelling the Tolman and distance-duality tests
                 with IllustrisTNG}",
      journal = {arXiv e-prints},
     keywords = {Cosmology and Nongalactic Astrophysics, Astrophysics of Galaxies},
         year = 2026,
        month = jun,
          eid = {arXiv:2606.27182},
        pages = {arXiv:2606.27182},
archivePrefix = {arXiv},
       eprint = {2606.27182},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260627182D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

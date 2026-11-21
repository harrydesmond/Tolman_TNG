"""Command line interface for the Tolman hydro-simulation test."""

from __future__ import annotations

import argparse
from pathlib import Path

from .filters import ASTRODEEP_JWST_FILTERS, CatalogSchema, FilterSpec, SelectionConfig, schema_from_toml


def _demo_schema() -> CatalogSchema:
    filters = tuple(
        FilterSpec(
            name=name,
            lambda_eff_angstrom=lam,
            flux_col=f"flux_{name}_uJy",
            flux_err_col=f"fluxerr_{name}_uJy",
        )
        for name, lam in ASTRODEEP_JWST_FILTERS.items()
    )
    return CatalogSchema(
        redshift_col="z",
        filters=filters,
        area_col="aperture_area_arcsec2",
        object_id_col="id",
        field_col="field",
        flags_col="flags",
        bad_flag_min=80.0,
        blend_col="is_blended",
        edge_col="is_edge",
        point_source_col="is_point_source",
        selection=SelectionConfig(
            min_snr=5.0,
            spec_mag_filter="F200W",
            spec_mag_limit=29.0,
            z_min=0.0,
            z_max=14.0,
        ),
    )


def run_demo(args: argparse.Namespace) -> None:
    from .pipeline import run_catalog_pipeline
    from .synthetic import generate_synthetic_catalog

    raw = generate_synthetic_catalog(n=args.n, seed=args.seed)
    run_catalog_pipeline(
        raw,
        _demo_schema(),
        result_dir=args.result_dir,
        plot_dir=args.plot_dir,
        sample_name=args.sample_name,
        save_measured_catalog=args.save_measured_catalog,
    )


def run_measure(args: argparse.Namespace) -> None:
    from .io import read_catalog
    from .pipeline import run_catalog_pipeline

    schema = schema_from_toml(args.config)
    raw = read_catalog(args.catalog)
    run_catalog_pipeline(
        raw,
        schema,
        result_dir=args.result_dir,
        plot_dir=args.plot_dir,
        sample_name=args.sample_name,
        save_measured_catalog=args.save_measured_catalog,
    )


def run_aperture_photometry(args: argparse.Namespace) -> None:
    from .image_photometry import add_circular_aperture_fluxes
    from .io import read_catalog

    raw = read_catalog(args.catalog)
    image_paths = {}
    for item in args.image:
        name, sep, path = item.partition("=")
        if not sep:
            raise SystemExit("--image entries must have the form FILTER=/path/to/image.fits")
        image_paths[name] = Path(path)
    measured = add_circular_aperture_fluxes(
        raw,
        image_paths=image_paths,
        x_col=args.x_col,
        y_col=args.y_col,
        radius_pix_col=args.radius_pix_col,
        pixel_scale_arcsec=args.pixel_scale_arcsec,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    measured.to_csv(args.out, index=False)


def run_mast_normalize(args: argparse.Namespace) -> None:
    from .mast_illustris import filter_name_from_path, normalize_tng_filter_fits, write_schema_for_normalized_catalog

    paths = list(args.fits or [])
    for pattern in args.fits_glob or []:
        paths.extend(sorted(Path().glob(pattern)))
    if not paths:
        raise SystemExit("Provide at least one --fits or --fits-glob input")
    filters = [filter_name_from_path(path) for path in paths]
    normalized = normalize_tng_filter_fits(paths, sample_name=args.sample_name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.out, index=False)
    if args.config_out:
        write_schema_for_normalized_catalog(filters, args.config_out)


def run_astrodeep_normalize(args: argparse.Namespace) -> None:
    from .astrodeep import normalize_astrodeep_catalogs, write_astrodeep_schema

    normalized = normalize_astrodeep_catalogs(
        args.phot_catalog,
        args.redshift_catalog,
        field_name=args.field_name,
        id_col=args.id_col,
        redshift_id_col=args.redshift_id_col,
        zspec_col=args.zspec_col,
        zphot_col=args.zphot_col,
        aperture_col=args.aperture_col,
        aperture_kind=args.aperture_kind,
        flag_col=args.flag_col,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.out, index=False)
    if args.config_out:
        write_astrodeep_schema(args.config_out)


def run_fit_spec_selection(args: argparse.Namespace) -> None:
    from .io import read_catalog
    from .spec_selection import fit_model, save_model

    df = read_catalog(args.catalog)
    features = args.feature or None
    class_weight = "balanced" if args.balanced_class_weight else None
    model, used = fit_model(df, target_col=args.target_col, features=features, class_weight=class_weight)
    save_model(model, used, args.out)
    print("features=" + ",".join(used))


def run_apply_spec_selection(args: argparse.Namespace) -> None:
    from .io import read_catalog
    from .spec_selection import add_selection_probabilities, load_model, select_by_probability

    df = read_catalog(args.catalog)
    model, features = load_model(args.model)
    out = add_selection_probabilities(df, model, features)
    out = select_by_probability(out, mode=args.mode, target_count=args.target_count, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)


def run_detect_sources(args: argparse.Namespace) -> None:
    from .image_detection import detect_sources_from_fits

    detections = detect_sources_from_fits(
        args.image,
        threshold_sigma=args.threshold_sigma,
        min_pixels=args.min_pixels,
        edge_buffer_pix=args.edge_buffer_pix,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    detections.to_csv(args.out, index=False)


def run_tng_image_aperture(args: argparse.Namespace) -> None:
    from .mast_illustris import filter_name_from_path
    from .tng_image_aperture import normalize_tng_image_aperture_fits, write_tng_image_aperture_schema

    paths = list(args.fits or [])
    for pattern in args.fits_glob or []:
        paths.extend(sorted(Path().glob(pattern)))
    if not paths:
        raise SystemExit("Provide at least one --fits or --fits-glob input")
    filters = [filter_name_from_path(path) for path in paths]
    normalized, diagnostics = normalize_tng_image_aperture_fits(
        paths,
        sample_name=args.sample_name,
        max_rows=args.max_rows,
        pixel_scale_arcsec=args.pixel_scale_arcsec,
        coordinate_mode=args.coordinate_mode,
        n_workers=args.n_workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.out, index=False)
    if args.config_out:
        write_tng_image_aperture_schema(filters, args.config_out)
    if args.diagnostics_out:
        args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(args.diagnostics_out, index=False)
    print(diagnostics.to_string(index=False))


def run_compare_slopes(args: argparse.Namespace) -> None:
    from .comparison import write_slope_comparison_outputs

    table = write_slope_comparison_outputs(
        args.observed,
        args.mock,
        out_csv=args.out,
        out_plot=args.plot,
        out_report=args.report,
        observed_selection=args.observed_selection,
        mock_selection=args.mock_selection,
        observed_label=args.observed_label,
        mock_label=args.mock_label,
    )
    print(table.to_string(index=False))


def run_astrodeep_audit(args: argparse.Namespace) -> None:
    from .astrodeep_audit import write_astrodeep_sample_audit
    from .io import read_catalog

    df = read_catalog(args.catalog)
    counts, fields = write_astrodeep_sample_audit(df, args.out_dir)
    print(counts.to_string(index=False))
    if len(fields):
        print(fields.to_string(index=False))


def run_image_aperture_diagnostics(args: argparse.Namespace) -> None:
    from .imageap_diagnostics import write_image_aperture_diagnostics

    tables = write_image_aperture_diagnostics(
        args.catalogue_measured,
        args.image_measured,
        out_dir=args.out_dir,
        plot_dir=args.plot_dir,
        sample_name=args.sample_name,
    )
    print(tables["summary"].to_string(index=False))


def run_annulus_contamination(args: argparse.Namespace) -> None:
    from .aperture_contamination import write_annulus_contamination_diagnostics

    paths = list(args.fits or [])
    for pattern in args.fits_glob or []:
        paths.extend(sorted(Path().glob(pattern)))
    if not paths:
        raise SystemExit("Provide at least one --fits or --fits-glob input")
    tables = write_annulus_contamination_diagnostics(
        args.catalog,
        paths,
        out_dir=args.out_dir,
        plot_dir=args.plot_dir,
        sample_name=args.sample_name,
        inner_factor=args.inner_factor,
        outer_factor=args.outer_factor,
        n_workers=args.n_workers,
    )
    print(tables["fits"].to_string(index=False))


def run_deblending_diagnostics(args: argparse.Namespace) -> None:
    from .deblending_diagnostics import write_deblending_diagnostics

    paths = list(args.fits or [])
    for pattern in args.fits_glob or []:
        paths.extend(sorted(Path().glob(pattern)))
    if not paths:
        raise SystemExit("Provide at least one --fits or --fits-glob input")
    tables = write_deblending_diagnostics(
        args.catalog,
        paths,
        out_dir=args.out_dir,
        plot_dir=args.plot_dir,
        sample_name=args.sample_name,
        selected_col=args.selected_col,
        threshold_sigma=args.threshold_sigma,
        min_pixels=args.min_pixels,
        edge_buffer_pix=args.edge_buffer_pix,
    )
    print(tables["slopes"].to_string(index=False))


def run_deblended_tng_slopes(args: argparse.Namespace) -> None:
    from .deblending_diagnostics import write_deblended_tng_slopes

    table = write_deblended_tng_slopes(
        args.catalog,
        args.per_filter_csv,
        args.config,
        out_dir=args.out_dir,
        plot_dir=args.plot_dir,
        sample_name=args.sample_name,
    )
    print(table.to_string(index=False))


def run_adaptive_aperture_sb(args: argparse.Namespace) -> None:
    from .adaptive_aperture import write_adaptive_aperture_sb

    paths = list(args.fits or [])
    for pattern in args.fits_glob or []:
        paths.extend(sorted(Path().glob(pattern)))
    if not paths:
        raise SystemExit("Provide at least one --fits or --fits-glob input")
    table = write_adaptive_aperture_sb(
        args.catalog,
        paths,
        args.config,
        out_dir=args.out_dir,
        plot_dir=args.plot_dir,
        sample_name=args.sample_name,
        selected_col=args.selected_col,
    )
    print(table.to_string(index=False))


def run_aperture_sweep(args: argparse.Namespace) -> None:
    from .aperture_sweep import write_aperture_sweep

    paths = list(args.fits or [])
    for pattern in args.fits_glob or []:
        paths.extend(sorted(Path().glob(pattern)))
    if not paths:
        raise SystemExit("Provide at least one --fits or --fits-glob input")
    tables = write_aperture_sweep(
        args.catalog,
        paths,
        out_dir=args.out_dir,
        plot_dir=args.plot_dir,
        sample_name=args.sample_name,
        selected_col=args.selected_col,
        scale_factors=tuple(args.scale_factors),
        n_workers=args.n_workers,
    )
    print(tables["slopes"].to_string(index=False))


def run_segmentation_diagnostics(args: argparse.Namespace) -> None:
    from .segmentation_diagnostics import write_segmentation_diagnostics

    paths = list(args.fits or [])
    for pattern in args.fits_glob or []:
        paths.extend(sorted(Path().glob(pattern)))
    if not paths:
        raise SystemExit("Provide at least one --fits or --fits-glob input")
    tables = write_segmentation_diagnostics(
        args.catalog,
        paths,
        out_dir=args.out_dir,
        plot_dir=args.plot_dir,
        sample_name=args.sample_name,
        selected_col=args.selected_col,
        threshold_sigma=args.threshold_sigma,
        min_pixels=args.min_pixels,
        edge_buffer_pix=args.edge_buffer_pix,
        n_workers=args.n_workers,
    )
    print(tables["fits"].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run the full pipeline on a synthetic catalogue.")
    demo.add_argument("--n", type=int, default=6000)
    demo.add_argument("--seed", type=int, default=1)
    demo.add_argument("--result-dir", type=Path, default=Path("results/demo"))
    demo.add_argument("--plot-dir", type=Path, default=Path("Plots"))
    demo.add_argument("--sample-name", default="demo")
    demo.add_argument("--save-measured-catalog", action="store_true")
    demo.set_defaults(func=run_demo)

    measure = sub.add_parser("measure", help="Run the pipeline on a CSV/TSV/FITS/parquet catalogue.")
    measure.add_argument("--catalog", type=Path, required=True)
    measure.add_argument("--config", type=Path, required=True)
    measure.add_argument("--result-dir", type=Path, default=Path("results/measure"))
    measure.add_argument("--plot-dir", type=Path, default=Path("Plots"))
    measure.add_argument("--sample-name", default="mock")
    measure.add_argument("--save-measured-catalog", action="store_true")
    measure.set_defaults(func=run_measure)

    phot = sub.add_parser(
        "aperture-photometry",
        help="Measure circular-aperture fluxes from FITS images at known catalogue positions.",
    )
    phot.add_argument("--catalog", type=Path, required=True)
    phot.add_argument("--image", action="append", required=True, help="FILTER=/path/to/image.fits")
    phot.add_argument("--x-col", required=True)
    phot.add_argument("--y-col", required=True)
    phot.add_argument("--radius-pix-col", required=True)
    phot.add_argument("--pixel-scale-arcsec", type=float, required=True)
    phot.add_argument("--out", type=Path, required=True)
    phot.set_defaults(func=run_aperture_photometry)

    mast = sub.add_parser(
        "mast-normalize",
        help="Normalize MAST ILLUSTRIS/TNG per-filter FITS OutputData tables.",
    )
    mast.add_argument("--fits", type=Path, action="append")
    mast.add_argument("--fits-glob", action="append", help="Glob pattern for FITS files, quoted in the shell.")
    mast.add_argument("--out", type=Path, required=True)
    mast.add_argument("--config-out", type=Path)
    mast.add_argument("--sample-name", default="tng")
    mast.set_defaults(func=run_mast_normalize)

    astro = sub.add_parser(
        "astrodeep-normalize",
        help="Normalize ASTRODEEP-JWST optap photometry and redshift catalogues.",
    )
    astro.add_argument("--phot-catalog", type=Path, required=True)
    astro.add_argument("--redshift-catalog", type=Path)
    astro.add_argument("--field-name", required=True)
    astro.add_argument("--out", type=Path, required=True)
    astro.add_argument("--config-out", type=Path)
    astro.add_argument("--id-col")
    astro.add_argument("--redshift-id-col")
    astro.add_argument("--zspec-col")
    astro.add_argument("--zphot-col")
    astro.add_argument("--aperture-col")
    astro.add_argument("--aperture-kind", choices=["diameter", "radius"], default="diameter")
    astro.add_argument("--flag-col")
    astro.set_defaults(func=run_astrodeep_normalize)

    fit_spec = sub.add_parser("fit-spec-selection", help="Fit empirical spectroscopic-selection model.")
    fit_spec.add_argument("--catalog", type=Path, required=True)
    fit_spec.add_argument("--out", type=Path, required=True)
    fit_spec.add_argument("--target-col", default="has_spec")
    fit_spec.add_argument("--feature", action="append")
    fit_spec.add_argument("--balanced-class-weight", action="store_true")
    fit_spec.set_defaults(func=run_fit_spec_selection)

    apply_spec = sub.add_parser("apply-spec-selection", help="Apply a spectroscopic-selection model to a mock catalogue.")
    apply_spec.add_argument("--catalog", type=Path, required=True)
    apply_spec.add_argument("--model", type=Path, required=True)
    apply_spec.add_argument("--out", type=Path, required=True)
    apply_spec.add_argument("--mode", choices=["deterministic", "probabilistic"], default="deterministic")
    apply_spec.add_argument("--target-count", type=int)
    apply_spec.add_argument("--seed", type=int, default=0)
    apply_spec.set_defaults(func=run_apply_spec_selection)

    detect = sub.add_parser("detect-sources", help="Detect sources in a FITS image using a simple threshold segmenter.")
    detect.add_argument("--image", type=Path, required=True)
    detect.add_argument("--out", type=Path, required=True)
    detect.add_argument("--threshold-sigma", type=float, default=3.0)
    detect.add_argument("--min-pixels", type=int, default=8)
    detect.add_argument("--edge-buffer-pix", type=int, default=3)
    detect.set_defaults(func=run_detect_sources)

    image_ap = sub.add_parser(
        "tng-image-aperture",
        help="Measure TNG FITS images with catalogue-position circular apertures.",
    )
    image_ap.add_argument("--fits", type=Path, action="append")
    image_ap.add_argument("--fits-glob", action="append", help="Glob pattern for FITS files, quoted in the shell.")
    image_ap.add_argument("--out", type=Path, required=True)
    image_ap.add_argument("--config-out", type=Path)
    image_ap.add_argument("--diagnostics-out", type=Path)
    image_ap.add_argument("--sample-name", default="tng_image_aperture")
    image_ap.add_argument("--max-rows", type=int)
    image_ap.add_argument("--pixel-scale-arcsec", type=float)
    image_ap.add_argument("--coordinate-mode", choices=["linear", "wcs"], default="linear")
    image_ap.add_argument("--n-workers", type=int, default=1)
    image_ap.set_defaults(func=run_tng_image_aperture)

    compare = sub.add_parser("compare-slopes", help="Compare observed and mock slope tables.")
    compare.add_argument("--observed", type=Path, required=True)
    compare.add_argument("--mock", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--plot", type=Path)
    compare.add_argument("--report", type=Path)
    compare.add_argument("--observed-selection", default="spectroscopic")
    compare.add_argument("--mock-selection", default="mock_spectroscopic")
    compare.add_argument("--observed-label", default="ASTRODEEP")
    compare.add_argument("--mock-label", default="TNG mock")
    compare.set_defaults(func=run_compare_slopes)

    audit = sub.add_parser("astrodeep-audit", help="Audit candidate ASTRODEEP paper sample cuts.")
    audit.add_argument("--catalog", type=Path, required=True)
    audit.add_argument("--out-dir", type=Path, required=True)
    audit.set_defaults(func=run_astrodeep_audit)

    image_diag = sub.add_parser(
        "image-aperture-diagnostics",
        help="Decompose TNG catalogue-vs-image aperture slope shifts.",
    )
    image_diag.add_argument("--catalogue-measured", type=Path, required=True)
    image_diag.add_argument("--image-measured", type=Path, required=True)
    image_diag.add_argument("--out-dir", type=Path, required=True)
    image_diag.add_argument("--plot-dir", type=Path, default=Path("Plots"))
    image_diag.add_argument("--sample-name", default="tng100_image_aperture_shift")
    image_diag.set_defaults(func=run_image_aperture_diagnostics)

    annulus = sub.add_parser(
        "annulus-contamination",
        help="Estimate local-annulus contamination in TNG image apertures.",
    )
    annulus.add_argument("--catalog", type=Path, required=True)
    annulus.add_argument("--fits", type=Path, action="append")
    annulus.add_argument("--fits-glob", action="append", help="Glob pattern for FITS files, quoted in the shell.")
    annulus.add_argument("--out-dir", type=Path, required=True)
    annulus.add_argument("--plot-dir", type=Path, default=Path("Plots"))
    annulus.add_argument("--sample-name", default="tng100_imageap_annulus")
    annulus.add_argument("--inner-factor", type=float, default=2.0)
    annulus.add_argument("--outer-factor", type=float, default=3.0)
    annulus.add_argument("--n-workers", type=int, default=1)
    annulus.set_defaults(func=run_annulus_contamination)

    deblend = sub.add_parser(
        "deblending-diagnostics",
        help="Voronoi-deblend TNG image segments and compare deblended/aperture/catalogue flux ratios.",
    )
    deblend.add_argument("--catalog", type=Path, required=True)
    deblend.add_argument("--fits", type=Path, action="append")
    deblend.add_argument("--fits-glob", action="append", help="Glob pattern for FITS files, quoted in the shell.")
    deblend.add_argument("--out-dir", type=Path, required=True)
    deblend.add_argument("--plot-dir", type=Path, default=Path("Plots"))
    deblend.add_argument("--sample-name", default="tng100_imageap_deblended")
    deblend.add_argument("--selected-col", default="mock_spec_selected")
    deblend.add_argument("--threshold-sigma", type=float, default=3.0)
    deblend.add_argument("--min-pixels", type=int, default=8)
    deblend.add_argument("--edge-buffer-pix", type=int, default=3)
    deblend.set_defaults(func=run_deblending_diagnostics)

    dblsb = sub.add_parser(
        "deblended-tng-slopes",
        help="Compute TNG SB power-law slopes using Voronoi-deblended fluxes from a prior deblending-diagnostics run.",
    )
    dblsb.add_argument("--catalog", type=Path, required=True, help="Specmodel CSV (image-aperture run).")
    dblsb.add_argument("--per-filter-csv", type=Path, required=True, help="deblending_per_filter.csv from deblending-diagnostics run.")
    dblsb.add_argument("--config", type=Path, required=True, help="TOML schema for this catalogue.")
    dblsb.add_argument("--out-dir", type=Path, required=True)
    dblsb.add_argument("--plot-dir", type=Path, default=Path("Plots"))
    dblsb.add_argument("--sample-name", default="tng100_imageap_deblended_sb")
    dblsb.set_defaults(func=run_deblended_tng_slopes)

    adap = sub.add_parser(
        "adaptive-aperture-sb",
        help="Compute TNG SB slopes using ASTRODEEP-equivalent z-dependent aperture radii.",
    )
    adap.add_argument("--catalog", type=Path, required=True, help="Specmodel CSV (image-aperture run, with x/y pixel positions).")
    adap.add_argument("--fits", type=Path, action="append")
    adap.add_argument("--fits-glob", action="append", help="Glob pattern for FITS files.")
    adap.add_argument("--config", type=Path, required=True, help="TOML schema for this catalogue.")
    adap.add_argument("--out-dir", type=Path, required=True)
    adap.add_argument("--plot-dir", type=Path, default=Path("Plots"))
    adap.add_argument("--sample-name", default="tng100_imageap_astrodeep_aperture")
    adap.add_argument("--selected-col", default="mock_spec_selected")
    adap.set_defaults(func=run_adaptive_aperture_sb)

    sweep = sub.add_parser(
        "aperture-sweep",
        help="Measure circular apertures at multiple scale factors to map flux recovery vs redshift.",
    )
    sweep.add_argument("--catalog", type=Path, required=True)
    sweep.add_argument("--fits", type=Path, action="append")
    sweep.add_argument("--fits-glob", action="append", help="Glob pattern for FITS files, quoted in the shell.")
    sweep.add_argument("--out-dir", type=Path, required=True)
    sweep.add_argument("--plot-dir", type=Path, default=Path("Plots"))
    sweep.add_argument("--sample-name", default="tng100_imageap_sweep")
    sweep.add_argument("--selected-col", default="mock_spec_selected")
    sweep.add_argument(
        "--scale-factors",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    )
    sweep.add_argument("--n-workers", type=int, default=1)
    sweep.set_defaults(func=run_aperture_sweep)

    segment = sub.add_parser(
        "segmentation-diagnostics",
        help="Assign catalogue objects to threshold image segments and compare segment/aperture flux ratios.",
    )
    segment.add_argument("--catalog", type=Path, required=True)
    segment.add_argument("--fits", type=Path, action="append")
    segment.add_argument("--fits-glob", action="append", help="Glob pattern for FITS files, quoted in the shell.")
    segment.add_argument("--out-dir", type=Path, required=True)
    segment.add_argument("--plot-dir", type=Path, default=Path("Plots"))
    segment.add_argument("--sample-name", default="tng100_imageap_segmentation")
    segment.add_argument("--selected-col", default="mock_spec_selected")
    segment.add_argument("--threshold-sigma", type=float, default=3.0)
    segment.add_argument("--min-pixels", type=int, default=8)
    segment.add_argument("--edge-buffer-pix", type=int, default=3)
    segment.add_argument("--n-workers", type=int, default=1)
    segment.set_defaults(func=run_segmentation_diagnostics)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

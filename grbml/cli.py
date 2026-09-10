"""Command line interface: ``grbml <command>``.

    grbml collect   copy per-burst CSVs out of a reduction tree
    grbml inspect   report what light curves are available
    grbml run       the full embed-and-cluster analysis
    grbml simulate  the noise experiment
    grbml denoise   plot one burst before and after denoising
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from grbml.config import (
    ClusterConfig,
    DenoiseConfig,
    FeatureConfig,
    PipelineConfig,
    ReduceConfig,
)


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    """Turn parsed arguments into a configuration, honouring --config."""
    if getattr(args, "config", None):
        config = PipelineConfig.from_json(args.config)
    else:
        config = PipelineConfig()

    if args.data_dir:
        config.data_dirs = list(args.data_dir)
    if args.output:
        config.output_dir = args.output
    if args.bands:
        config.features = FeatureConfig(**{**vars(config.features), "bands": tuple(args.bands)})
    if args.catalog:
        config.catalog = args.catalog
    if args.annotations:
        config.annotations = args.annotations
    if args.no_denoise:
        config.denoise = DenoiseConfig(**{**vars(config.denoise), "enabled": False})
    if args.window:
        config.features = FeatureConfig(**{**vars(config.features), "window": args.window})
    if args.resample_dt is not None:
        config.features = FeatureConfig(
            **{**vars(config.features), "resample_dt": args.resample_dt}
        )

    reduce_kwargs = dict(vars(config.reduce))
    if args.pca:
        reduce_kwargs["use_pca"] = True
    if args.pca_components is not None:
        reduce_kwargs["pca_components"] = args.pca_components
    if args.n_neighbors is not None:
        reduce_kwargs["n_neighbors"] = args.n_neighbors
    if args.min_dist is not None:
        reduce_kwargs["min_dist"] = args.min_dist
    if args.seed is not None:
        reduce_kwargs["random_state"] = args.seed
    config.reduce = ReduceConfig(**reduce_kwargs)

    cluster_kwargs = dict(vars(config.cluster))
    if args.algorithm:
        cluster_kwargs["algorithm"] = args.algorithm
    if args.min_cluster_size is not None:
        cluster_kwargs["min_cluster_size"] = args.min_cluster_size
    if args.n_clusters is not None:
        cluster_kwargs["n_clusters"] = args.n_clusters
    config.cluster = ClusterConfig(**cluster_kwargs)

    if args.no_plots:
        config.make_plots = False
    return config


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="JSON configuration file to start from")
    parser.add_argument("--data-dir", action="append",
                        help="directory of light-curve CSVs (repeatable)")
    parser.add_argument("-o", "--output", help="output directory")
    parser.add_argument("--bands", nargs="+",
                        help="energy bands to use, e.g. 8-50 50-300 300-900")
    parser.add_argument("--catalog", help="burst catalogue CSV with T90/fluence")
    parser.add_argument("--annotations", help="annotated-GRB CSV")
    parser.add_argument("--window", choices=["t90", "full"],
                        help="light-curve interval to keep")
    parser.add_argument("--resample-dt", type=float,
                        help="rebin every light curve to this bin width in "
                             "seconds (e.g. 0.016 or 0.064) so the sample "
                             "shares one frequency grid")
    parser.add_argument("--no-denoise", action="store_true",
                        help="skip wavelet denoising")
    parser.add_argument("--pca", action="store_true",
                        help="initialise UMAP with PCA")
    parser.add_argument("--pca-components", type=float,
                        help="component count, or a fraction of variance to keep")
    parser.add_argument("--n-neighbors", type=int, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, help="UMAP min_dist")
    parser.add_argument("--algorithm",
                        choices=["hdbscan", "dbscan", "kmeans", "gmm"],
                        help="clustering algorithm")
    parser.add_argument("--min-cluster-size", type=int,
                        help="HDBSCAN/DBSCAN minimum cluster size")
    parser.add_argument("--n-clusters", type=int,
                        help="number of clusters for kmeans/gmm")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--no-plots", action="store_true", help="skip figures")


def command_inspect(args: argparse.Namespace) -> int:
    from grbml.dataio import available_bands, discover_light_curves

    roots = args.data_dir or ["."]
    index = discover_light_curves(roots)
    bands = available_bands(roots)
    per_band: dict = {}
    for curves in index.values():
        for band in curves:
            per_band[band] = per_band.get(band, 0) + 1

    print(f"directories: {', '.join(str(root) for root in roots)}")
    print(f"bursts:      {len(index)}")
    print(f"bands:       {', '.join(bands) if bands else '(none)'}")
    for band in bands:
        print(f"  {band:>10} keV : {per_band[band]} bursts")
    complete = sum(1 for curves in index.values() if len(curves) == len(bands))
    print(f"bursts with every band: {complete}")
    if args.list:
        for trigger in sorted(index):
            print(f"  {trigger}: {', '.join(sorted(index[trigger]))}")
    return 0


def command_collect(args: argparse.Namespace) -> int:
    """Copy per-burst CSVs into one flat directory, prefixed by burst.

    The Python counterpart of ``Batch_csv_collector_shell_script``: walk a
    reduction tree, find the CSVs in each burst's band subdirectory, and copy
    them out as ``<burst>__<file>.csv``.
    """
    source = Path(args.input)
    destination = Path(args.output)
    if not source.is_dir():
        print(f"error: input directory {source} does not exist", file=sys.stderr)
        return 1
    destination.mkdir(parents=True, exist_ok=True)

    copied = skipped = 0
    for burst_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        band_dir = burst_dir / args.subdir if args.subdir else burst_dir
        if not band_dir.is_dir():
            continue
        for csv in sorted(band_dir.glob("*.csv")):
            target = destination / f"{burst_dir.name}__{csv.name}"
            if target.exists() and not args.overwrite:
                skipped += 1
                continue
            if args.dry_run:
                print(f"would copy {csv} -> {target}")
            else:
                shutil.copy2(csv, target)
            copied += 1
    print(f"copied {copied} file(s), skipped {skipped} existing, into {destination}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    from grbml.pipeline import run

    config = _build_config(args)
    result = run(config)
    print(f"bursts:    {result.features.n_bursts}")
    print(f"features:  {result.features.n_features}")
    print(f"clusters:  {result.clusters.n_clusters} "
          f"(+{result.clusters.n_noise} outliers)")
    if "t90_source" in result.table.columns:
        counts = result.table["t90_source"].value_counts().to_dict()
        print(f"T90 source: {counts}")
        missing = int(counts.get("none", 0))
        if missing:
            print(
                f"note:      {missing} burst(s) have no usable T90 and fell "
                "back to the full extracted interval. Check cluster_summary.csv: "
                "a cluster made only of those bursts is grouped by the analysis "
                "window, not by the burst."
            )
    print(f"sizes:     {result.clusters.sizes()}")
    print(f"output:    {config.output_dir}")
    return 0


def command_simulate(args: argparse.Namespace) -> int:
    from grbml.pipeline import run_simulation

    config = _build_config(args)
    outcome = run_simulation(
        n_per_class=args.n_per_class,
        n_bins=args.n_bins,
        dt=args.dt,
        output_dir=args.output or "results/simulation",
        config=config,
    )
    print(f"classes simulated: {len(outcome['class_names'])} "
          f"({', '.join(outcome['class_names'])})")
    print(f"clusters found:    {outcome['clusters'].n_clusters}")
    print(f"adjusted Rand:     {outcome['adjusted_rand_score']:.3f}")
    print(outcome["confusion"].to_string())
    return 0


def command_denoise(args: argparse.Namespace) -> int:
    from grbml import plots
    from grbml.dataio import read_light_curve
    from grbml.denoise import select_denoising

    curve = read_light_curve(args.file)
    choice = select_denoising(curve.net_rate, DenoiseConfig())
    print(f"{curve.trigger} [{curve.band} keV]: wavelet {choice.wavelet}, "
          f"level {choice.level}, BIC {choice.bic:.1f}")
    output = args.output or f"{curve.trigger}_{curve.band}_denoised.png"
    plots.plot_denoising(
        curve.time,
        curve.net_rate,
        choice.signal,
        title=(f"GRB {curve.trigger} ({curve.band} keV) - "
               f"{choice.wavelet} L{choice.level}"),
        path=output,
    )
    print(f"wrote {output}")
    return 0


def command_config(args: argparse.Namespace) -> int:
    config = _build_config(args)
    text = json.dumps(config.to_dict(), indent=2, default=list)
    if args.output:
        Path(args.output).write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grbml",
        description="Clustering and classification of GRB prompt-emission light curves",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="report available light curves")
    inspect.add_argument("--data-dir", action="append")
    inspect.add_argument("--list", action="store_true", help="list every burst")
    inspect.set_defaults(func=command_inspect)

    collect = subparsers.add_parser(
        "collect", help="copy per-burst CSVs into one directory"
    )
    collect.add_argument("input", help="root of the reduction output tree")
    collect.add_argument("output", help="destination directory")
    collect.add_argument("--subdir", default="300-900",
                         help="per-burst subdirectory to read (default: 300-900)")
    collect.add_argument("--overwrite", action="store_true")
    collect.add_argument("--dry-run", action="store_true")
    collect.set_defaults(func=command_collect)

    run_parser = subparsers.add_parser("run", help="run the full analysis")
    _add_analysis_arguments(run_parser)
    run_parser.set_defaults(func=command_run)

    simulate = subparsers.add_parser(
        "simulate", help="cluster simulated noise of known colour"
    )
    _add_analysis_arguments(simulate)
    simulate.add_argument("--n-per-class", type=int, default=200)
    simulate.add_argument("--n-bins", type=int, default=512)
    simulate.add_argument("--dt", type=float, default=0.064)
    simulate.set_defaults(func=command_simulate)

    denoise = subparsers.add_parser(
        "denoise", help="denoise one light curve and plot the comparison"
    )
    denoise.add_argument("file", help="light-curve CSV")
    denoise.add_argument("-o", "--output", help="output image path")
    denoise.set_defaults(func=command_denoise)

    config_parser = subparsers.add_parser(
        "config", help="print a configuration file for the given options"
    )
    _add_analysis_arguments(config_parser)
    config_parser.set_defaults(func=command_config)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

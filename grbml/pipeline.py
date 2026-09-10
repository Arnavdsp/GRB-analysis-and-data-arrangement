"""End-to-end analysis: light curves in, embedding and clusters out.

    load -> denoise -> window/pad/normalise -> |FFT|
         -> (PCA) -> UMAP -> HDBSCAN
         -> duration and power-index maps

One call to :func:`run` reproduces one row of the thesis's comparison grid, and
writes everything needed to interpret it: the embedding, the cluster labels,
the per-burst table, the figures and the exact configuration used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from grbml import catalog as catalog_module
from grbml.cluster import ClusterResult, cluster, silhouette
from grbml.config import PipelineConfig
from grbml.dataio import LightCurve, load_dataset
from grbml.embed import EmbeddingResult, embed
from grbml.features import FeatureMatrix, build_feature_matrix
from grbml.powerspec import power_law_index


@dataclass
class PipelineResult:
    """Everything one run produced."""

    config: PipelineConfig
    features: FeatureMatrix
    embedding: EmbeddingResult
    clusters: ClusterResult
    table: pd.DataFrame
    cluster_summary: pd.DataFrame
    silhouette: float = float("nan")
    figures: Dict[str, str] = field(default_factory=dict)

    @property
    def triggers(self) -> Sequence[str]:
        return self.features.triggers

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PipelineResult(n_bursts={self.features.n_bursts}, "
            f"n_clusters={self.clusters.n_clusters}, "
            f"n_outliers={self.clusters.n_noise})"
        )


def compute_power_indices(
    dataset: Mapping[str, Mapping[str, LightCurve]],
    triggers: Sequence[str],
    band: str,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> pd.DataFrame:
    """Power-law index per burst, fitted to the raw light curve's spectrum.

    The raw curve is used on purpose: denoising is a filter that suppresses
    exactly the high-frequency power the index measures, so an index fitted to
    a denoised curve describes the filter as much as the burst.
    """
    rows = []
    for trigger in triggers:
        curves = dataset.get(trigger, {})
        curve = curves.get(band)
        if curve is None:
            rows.append({"trigger": trigger, "power_index": np.nan,
                         "power_index_error": np.nan})
            continue
        fit = power_law_index(curve.net_rate, curve.dt, fmin=fmin, fmax=fmax)
        rows.append(
            {
                "trigger": trigger,
                "power_index": fit.index,
                "power_index_error": fit.index_error,
            }
        )
    return pd.DataFrame(rows).set_index("trigger")


def _write_figures(
    result_dir: Path,
    embedding: np.ndarray,
    clusters: ClusterResult,
    table: pd.DataFrame,
    triggers: Sequence[str],
    annotations: Optional[Dict[str, Dict[str, str]]],
    explained_variance: Optional[np.ndarray],
) -> Dict[str, str]:
    from grbml import plots

    figure_dir = result_dir / "figures"
    written: Dict[str, str] = {}

    figure = plots.plot_embedding(embedding, path=figure_dir / "embedding.png")
    if annotations:
        plots.annotate_triggers(figure.axes[0], embedding, triggers, annotations)
        figure.savefig(figure_dir / "embedding_annotated.png", dpi=150,
                       bbox_inches="tight")
        written["embedding_annotated"] = str(figure_dir / "embedding_annotated.png")
    written["embedding"] = str(figure_dir / "embedding.png")

    plots.plot_clusters(embedding, clusters.labels, path=figure_dir / "clusters.png")
    written["clusters"] = str(figure_dir / "clusters.png")

    if "t90" in table.columns and np.any(np.isfinite(table["t90"].to_numpy(float))):
        plots.plot_value_map(
            embedding, table["t90"].to_numpy(float), label="T90 (s)", log=True,
            path=figure_dir / "duration_map.png",
        )
        written["duration_map"] = str(figure_dir / "duration_map.png")

    if "power_index" in table.columns:
        values = table["power_index"].to_numpy(float)
        if np.any(np.isfinite(values)):
            plots.plot_value_map(
                embedding, values, label="power-law index", cmap="plasma",
                title="UMAP embedding coloured by power-law index",
                path=figure_dir / "power_index_map.png",
            )
            written["power_index_map"] = str(figure_dir / "power_index_map.png")

    if explained_variance is not None:
        plots.plot_scree(explained_variance, path=figure_dir / "scree.png")
        written["scree"] = str(figure_dir / "scree.png")

    return written


def run(
    config: Optional[PipelineConfig] = None,
    dataset: Optional[Mapping[str, Mapping[str, LightCurve]]] = None,
    save: bool = True,
) -> PipelineResult:
    """Run the full analysis.

    ``dataset`` short-circuits loading, which is what the tests and notebooks
    use; otherwise the light curves come from ``config.data_dirs``.
    """
    config = config or PipelineConfig()
    bands = list(config.features.bands)

    if dataset is None:
        dataset = load_dataset(config.data_dirs, bands=bands)
    if not dataset:
        raise ValueError(
            f"no bursts found under {list(config.data_dirs)} for bands {bands}"
        )

    burst_catalog = (
        catalog_module.load_catalog(config.catalog) if config.catalog else None
    )

    features = build_feature_matrix(
        dataset,
        config=config.features,
        denoise_config=config.denoise,
        catalog=burst_catalog,
    )
    embedding_result = embed(features.X, config.reduce)
    cluster_result = cluster(embedding_result.embedding, config.cluster)

    reference_band = "50-300" if "50-300" in bands else bands[0]
    power_indices = compute_power_indices(
        dataset,
        features.triggers,
        reference_band,
        fmin=config.powerlaw_fmin,
        fmax=config.powerlaw_fmax,
    )

    table = features.metadata.join(power_indices, how="left")
    table = catalog_module.merge_metadata(table, burst_catalog)
    table["cluster"] = cluster_result.labels
    table["umap_1"] = embedding_result.embedding[:, 0]
    table["umap_2"] = embedding_result.embedding[:, 1]
    if "t90" in table.columns:
        table["duration_class"] = table["t90"].map(catalog_module.duration_class)

    annotations: Optional[Dict[str, Dict[str, str]]] = None
    if config.annotations:
        annotation_table = catalog_module.load_annotations(
            config.annotations, triggers=features.triggers
        )
        annotations = catalog_module.annotation_map(annotation_table)
        table["annotation"] = [
            annotations.get(trigger, {}).get("class", "")
            for trigger in features.triggers
        ]

    summary = catalog_module.summarise_clusters(table, cluster_result.labels)
    score = silhouette(embedding_result.embedding, cluster_result.labels)

    figures: Dict[str, str] = {}
    if save:
        result_dir = Path(config.output_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(result_dir / "bursts.csv")
        summary.to_csv(result_dir / "cluster_summary.csv", index=False)
        config.to_json(result_dir / "config.json")
        (result_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "n_bursts": features.n_bursts,
                    "n_features": features.n_features,
                    "bands": bands,
                    "pad_length": features.pad_length,
                    "n_clusters": cluster_result.n_clusters,
                    "n_outliers": cluster_result.n_noise,
                    "cluster_sizes": cluster_result.sizes(),
                    "silhouette": None if np.isnan(score) else score,
                    "pca_components": embedding_result.n_pca_components,
                    "n_missing_t90": int(table["t90"].isna().sum())
                    if "t90" in table.columns
                    else None,
                },
                indent=2,
            )
        )
        if config.make_plots:
            figures = _write_figures(
                result_dir,
                embedding_result.embedding,
                cluster_result,
                table,
                features.triggers,
                annotations,
                embedding_result.explained_variance_ratio,
            )

    return PipelineResult(
        config=config,
        features=features,
        embedding=embedding_result,
        clusters=cluster_result,
        table=table,
        cluster_summary=summary,
        silhouette=score,
        figures=figures,
    )


def run_simulation(
    n_per_class: int = 200,
    n_bins: int = 512,
    dt: float = 0.064,
    combinations: Optional[Mapping[str, Sequence[float]]] = None,
    output_dir: str | Path = "results/simulation",
    config: Optional[PipelineConfig] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """The noise experiment: can noise colour alone produce clusters?

    Simulates labelled noise-dominated light curves, pushes them through the
    same feature/embedding/cluster path as real data, and reports how the
    recovered clusters line up with the known input colours.  In the thesis the
    colours separate cleanly, and HDBSCAN finds five clusters where seven
    classes went in - two pairs of mixtures are indistinguishable.
    """
    from sklearn.metrics import adjusted_rand_score

    from grbml.simulate import simulate_noise_experiment

    config = config or PipelineConfig()
    simulation = simulate_noise_experiment(
        n_per_class=n_per_class,
        n_bins=n_bins,
        dt=dt,
        combinations=dict(combinations) if combinations else None,
    )

    # The simulated curves are already a uniform length, so they enter the
    # feature step directly - only normalisation and |FFT| still apply.
    from grbml.features import fourier_amplitudes, normalize

    rows = [
        fourier_amplitudes(
            normalize(curve, config.features.normalize),
            drop_dc=config.features.drop_dc,
        )
        for curve in simulation.curves
    ]
    X = np.nan_to_num(np.vstack(rows), nan=0.0, posinf=0.0, neginf=0.0)

    embedding_result = embed(X, config.reduce)
    cluster_result = cluster(embedding_result.embedding, config.cluster)
    agreement = float(adjusted_rand_score(simulation.labels, cluster_result.labels))

    table = pd.DataFrame(
        {
            "true_class": simulation.label_names(),
            "cluster": cluster_result.labels,
            "umap_1": embedding_result.embedding[:, 0],
            "umap_2": embedding_result.embedding[:, 1],
        }
    )
    confusion = pd.crosstab(table["true_class"], table["cluster"])

    if save:
        result_dir = Path(output_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(result_dir / "simulated_bursts.csv", index=False)
        confusion.to_csv(result_dir / "confusion.csv")
        if config.make_plots:
            from grbml import plots

            figure_dir = result_dir / "figures"
            plots.plot_clusters(
                embedding_result.embedding,
                cluster_result.labels,
                title="Simulated noise: HDBSCAN clusters",
                path=figure_dir / "simulation_clusters.png",
            )
            codes = pd.Categorical(
                table["true_class"], categories=list(simulation.class_names)
            ).codes
            plots.plot_value_map(
                embedding_result.embedding,
                codes.astype(float),
                label="noise class index",
                title="Simulated noise: embedding coloured by true class",
                cmap="tab10",
                path=figure_dir / "simulation_true_class.png",
            )

    return {
        "embedding": embedding_result,
        "clusters": cluster_result,
        "table": table,
        "confusion": confusion,
        "adjusted_rand_score": agreement,
        "class_names": list(simulation.class_names),
    }

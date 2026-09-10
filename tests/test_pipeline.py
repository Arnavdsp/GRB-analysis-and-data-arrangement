import numpy as np
import pytest

from grbml.config import (
    ClusterConfig,
    DenoiseConfig,
    FeatureConfig,
    PipelineConfig,
    ReduceConfig,
)
from grbml.pipeline import compute_power_indices, run, run_simulation

umap = pytest.importorskip("umap", reason="umap-learn is needed for the embedding")


def small_config(tmp_path, **overrides) -> PipelineConfig:
    config = PipelineConfig(
        output_dir=str(tmp_path / "results"),
        features=FeatureConfig(bands=("50-300",)),
        denoise=DenoiseConfig(wavelets=("db1",), levels=(3,)),
        reduce=ReduceConfig(n_neighbors=5, random_state=0),
        cluster=ClusterConfig(min_cluster_size=5),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_run_end_to_end(dataset, tmp_path):
    result = run(small_config(tmp_path), dataset=dataset)

    assert result.features.n_bursts == len(dataset)
    assert result.embedding.embedding.shape == (len(dataset), 2)
    assert len(result.clusters.labels) == len(dataset)
    assert set(result.table.index) == set(dataset)
    for column in ("cluster", "umap_1", "umap_2", "power_index", "t90", "t90_source"):
        assert column in result.table.columns


def test_run_writes_its_outputs(dataset, tmp_path):
    config = small_config(tmp_path)
    run(config, dataset=dataset)

    output = tmp_path / "results"
    for name in ("bursts.csv", "cluster_summary.csv", "config.json", "run_summary.json"):
        assert (output / name).exists()
    assert (output / "figures" / "embedding.png").exists()
    assert (output / "figures" / "clusters.png").exists()


def test_run_with_pca(dataset, tmp_path):
    config = small_config(tmp_path)
    config.reduce = ReduceConfig(
        use_pca=True, pca_components=0.9, n_neighbors=5, random_state=0
    )
    result = run(config, dataset=dataset, save=False)
    assert result.embedding.pca is not None
    assert result.embedding.n_pca_components >= 1


def test_run_without_plots_skips_figures(dataset, tmp_path):
    config = small_config(tmp_path)
    config.make_plots = False
    result = run(config, dataset=dataset)
    assert result.figures == {}
    assert not (tmp_path / "results" / "figures").exists()


def test_run_rejects_an_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="no bursts"):
        run(small_config(tmp_path), dataset={})


def test_compute_power_indices(dataset):
    triggers = sorted(dataset)
    frame = compute_power_indices(dataset, triggers, "50-300")
    assert list(frame.index) == triggers
    assert np.isfinite(frame["power_index"]).any()


def test_compute_power_indices_tolerates_a_missing_band(dataset):
    triggers = sorted(dataset)
    frame = compute_power_indices(dataset, triggers, "300-900")
    assert frame["power_index"].isna().all()


@pytest.mark.slow
def test_noise_simulation_separates_the_pure_colours(tmp_path):
    """The thesis's central control (Figure 4.39).

    Light curves that are *pure noise* of a known colour, with no astrophysics
    in them at all, must still land in clean separate clusters - which is why a
    cluster found in real data cannot be read as a burst population until the
    background's influence is ruled out.
    """
    config = small_config(tmp_path)
    config.reduce = ReduceConfig(n_neighbors=15, min_dist=0.001, random_state=0)
    config.cluster = ClusterConfig(min_cluster_size=20)
    outcome = run_simulation(
        n_per_class=60,
        n_bins=512,
        combinations={"WN": (0.0,), "PN": (1.0,), "RN": (2.0,)},
        output_dir=tmp_path / "simulation",
        config=config,
        save=False,
    )
    assert outcome["clusters"].n_clusters == 3
    assert outcome["adjusted_rand_score"] > 0.9

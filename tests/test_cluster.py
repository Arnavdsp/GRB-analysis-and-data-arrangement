import numpy as np
import pytest

from grbml.cluster import cluster, silhouette
from grbml.config import ClusterConfig


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    return np.vstack(
        [
            rng.normal([0, 0], 0.3, (60, 2)),
            rng.normal([8, 0], 0.3, (60, 2)),
            rng.normal([0, 8], 0.3, (60, 2)),
        ]
    )


@pytest.mark.parametrize("algorithm", ["hdbscan", "dbscan", "kmeans", "gmm"])
def test_every_algorithm_finds_three_separated_blobs(blobs, algorithm):
    result = cluster(
        blobs,
        ClusterConfig(algorithm=algorithm, min_cluster_size=15, n_clusters=3, eps=1.0),
    )
    assert result.n_clusters == 3
    assert sum(result.sizes().values()) == blobs.shape[0]


def test_hdbscan_labels_outliers_as_noise(blobs):
    scattered = np.vstack([blobs, np.array([[50.0, 50.0], [-40.0, 30.0]])])
    result = cluster(scattered, ClusterConfig(min_cluster_size=15))
    assert result.n_noise >= 2


def test_unknown_algorithm_is_rejected(blobs):
    with pytest.raises(ValueError, match="unknown clustering algorithm"):
        cluster(blobs, ClusterConfig(algorithm="magic"))


def test_embedding_must_be_two_dimensional():
    with pytest.raises(ValueError, match="two-dimensional"):
        cluster(np.arange(10.0), ClusterConfig())


def test_silhouette(blobs):
    result = cluster(blobs, ClusterConfig(min_cluster_size=15))
    assert silhouette(blobs, result.labels) > 0.8


def test_silhouette_is_undefined_for_one_cluster(blobs):
    assert np.isnan(silhouette(blobs, np.zeros(blobs.shape[0], dtype=int)))

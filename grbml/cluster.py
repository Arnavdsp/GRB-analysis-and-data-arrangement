"""Clustering of the two-dimensional embedding (thesis sec. 3.3).

HDBSCAN is the default: it is hierarchical and density based, so it does not
need the number of clusters up front and it labels population outliers as noise
(-1) instead of forcing them into a group.  K-means, GMM and DBSCAN are here so
the "does the choice of algorithm change the answer?" comparison can be run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from grbml.config import ClusterConfig


@dataclass
class ClusterResult:
    """Cluster labels and a summary of what the algorithm found."""

    labels: np.ndarray
    algorithm: str
    model: Any = None
    probabilities: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        """Number of real clusters, excluding the noise label."""
        return int(len({label for label in self.labels if label >= 0}))

    @property
    def n_noise(self) -> int:
        return int(np.count_nonzero(self.labels < 0))

    def sizes(self) -> Dict[int, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique, counts)}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ClusterResult(algorithm={self.algorithm!r}, "
            f"n_clusters={self.n_clusters}, n_noise={self.n_noise})"
        )


def _hdbscan_model(config: ClusterConfig, n_samples: int):
    """HDBSCAN from scikit-learn, falling back to the standalone package."""
    min_cluster_size = int(max(2, min(config.min_cluster_size, n_samples)))
    kwargs: Dict[str, Any] = {"min_cluster_size": min_cluster_size}
    if config.min_samples is not None:
        kwargs["min_samples"] = int(config.min_samples)
    try:
        import inspect

        from sklearn.cluster import HDBSCAN  # scikit-learn >= 1.3

        # Newer scikit-learn warns that `copy` will change default; set it
        # explicitly where the parameter exists so runs stay quiet.
        if "copy" in inspect.signature(HDBSCAN).parameters:
            kwargs["copy"] = True
        return HDBSCAN(**kwargs)
    except ImportError:  # pragma: no cover - depends on environment
        try:
            from hdbscan import HDBSCAN as StandaloneHDBSCAN
        except ImportError as exc:
            raise ImportError(
                "HDBSCAN needs scikit-learn >= 1.3 or the standalone `hdbscan` "
                "package. Install one with `pip install -U scikit-learn` or "
                "`pip install hdbscan`."
            ) from exc
        return StandaloneHDBSCAN(**kwargs)


def cluster(embedding: np.ndarray, config: Optional[ClusterConfig] = None) -> ClusterResult:
    """Cluster an embedding with the configured algorithm."""
    config = config or ClusterConfig()
    points = np.asarray(embedding, dtype=float)
    if points.ndim != 2:
        raise ValueError("embedding must be two-dimensional")
    n_samples = points.shape[0]
    algorithm = config.algorithm.lower()

    probabilities: Optional[np.ndarray] = None

    if algorithm == "hdbscan":
        model = _hdbscan_model(config, n_samples)
        labels = model.fit_predict(points)
        probabilities = getattr(model, "probabilities_", None)
    elif algorithm == "dbscan":
        from sklearn.cluster import DBSCAN

        min_samples = config.min_samples or config.min_cluster_size
        model = DBSCAN(eps=config.eps, min_samples=int(min_samples))
        labels = model.fit_predict(points)
    elif algorithm == "kmeans":
        from sklearn.cluster import KMeans

        n_clusters = int(max(1, min(config.n_clusters, n_samples)))
        model = KMeans(
            n_clusters=n_clusters, random_state=config.random_state, n_init=10
        )
        labels = model.fit_predict(points)
    elif algorithm == "gmm":
        from sklearn.mixture import GaussianMixture

        n_components = int(max(1, min(config.n_clusters, n_samples)))
        model = GaussianMixture(
            n_components=n_components, random_state=config.random_state
        )
        labels = model.fit_predict(points)
        probabilities = model.predict_proba(points).max(axis=1)
    else:
        raise ValueError(f"unknown clustering algorithm {config.algorithm!r}")

    return ClusterResult(
        labels=np.asarray(labels, dtype=int),
        algorithm=algorithm,
        model=model,
        probabilities=probabilities,
        meta={"n_samples": n_samples},
    )


def silhouette(embedding: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score over the clustered points, ignoring noise.

    Returns NaN when fewer than two clusters survive, where the score is not
    defined.
    """
    from sklearn.metrics import silhouette_score

    points = np.asarray(embedding, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mask = labels >= 0
    if int(len(set(labels[mask]))) < 2 or int(np.count_nonzero(mask)) < 3:
        return float("nan")
    return float(silhouette_score(points[mask], labels[mask]))

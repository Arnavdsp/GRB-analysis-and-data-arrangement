"""Dimensionality reduction: PCA initialisation and UMAP (thesis sec. 3.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from grbml.config import ReduceConfig

UMAP_HINT = (
    "umap-learn is required for the UMAP embedding. Install it with "
    "`pip install umap-learn` (or `pip install .[umap]`)."
)


@dataclass
class EmbeddingResult:
    """The two-dimensional embedding plus the fitted transformers."""

    embedding: np.ndarray
    reducer: Any = None
    pca: Any = None
    scaler: Any = None
    explained_variance_ratio: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def n_pca_components(self) -> Optional[int]:
        if self.explained_variance_ratio is None:
            return None
        return int(self.explained_variance_ratio.size)


def standardize(X: np.ndarray, scaler: Optional[Any] = None):
    """Zero-mean, unit-variance columns; returns the array and the scaler."""
    from sklearn.preprocessing import StandardScaler

    if scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(X), scaler
    return scaler.transform(X), scaler


def apply_pca(X: np.ndarray, components: Any = 0.95, random_state: Optional[int] = 42):
    """Project onto principal components.

    ``components`` is either an integer number of components or a fraction of
    variance to retain - the scree-plot criterion the thesis uses to pick how
    many components go into UMAP.
    """
    from sklearn.decomposition import PCA

    n_samples, n_features = X.shape
    if isinstance(components, float) and 0 < components < 1:
        n_components: Any = components
    else:
        n_components = int(min(int(components), n_samples, n_features))
        if n_components < 1:
            raise ValueError("PCA needs at least one component")
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X), pca


def scree(pca: Any) -> np.ndarray:
    """Cumulative explained-variance ratio, for the scree plot."""
    return np.cumsum(pca.explained_variance_ratio_)


def umap_embed(X: np.ndarray, config: Optional[ReduceConfig] = None):
    """Run UMAP on a feature matrix."""
    config = config or ReduceConfig()
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(UMAP_HINT) from exc

    n_samples = X.shape[0]
    # UMAP needs at least two neighbours and cannot use more than it has.
    n_neighbors = int(max(2, min(config.n_neighbors, n_samples - 1)))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=config.min_dist,
        n_components=config.n_components,
        metric=config.metric,
        random_state=config.random_state,
    )
    return reducer.fit_transform(X), reducer


def embed(X: np.ndarray, config: Optional[ReduceConfig] = None) -> EmbeddingResult:
    """Full reduction: optional scaling, optional PCA, then UMAP.

    The thesis runs every analysis twice, with and without the PCA step, and
    finds that the choice changes the number of clusters - so it is a switch,
    not a detail.
    """
    config = config or ReduceConfig()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("feature matrix must be two-dimensional")
    if X.shape[0] < 3:
        raise ValueError("need at least three bursts to embed")

    scaler = None
    working = X
    if config.standardize:
        working, scaler = standardize(working)

    pca = None
    explained = None
    if config.use_pca:
        working, pca = apply_pca(
            working, config.pca_components, random_state=config.random_state
        )
        explained = pca.explained_variance_ratio_

    embedding, reducer = umap_embed(working, config)
    return EmbeddingResult(
        embedding=np.asarray(embedding),
        reducer=reducer,
        pca=pca,
        scaler=scaler,
        explained_variance_ratio=explained,
        meta={
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "used_pca": bool(config.use_pca),
        },
    )

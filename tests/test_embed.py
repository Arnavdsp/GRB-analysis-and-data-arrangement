import numpy as np
import pytest

from grbml.config import ReduceConfig
from grbml.embed import apply_pca, embed, scree, standardize

pytest.importorskip("umap", reason="umap-learn is needed for the embedding")


@pytest.fixture
def features():
    rng = np.random.default_rng(0)
    # Two families with different spectral shapes.
    first = rng.normal(0.0, 1.0, (30, 40))
    second = rng.normal(3.0, 1.0, (30, 40))
    return np.vstack([first, second])


def test_standardize(features):
    scaled, scaler = standardize(features)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-9)
    again, _ = standardize(features, scaler)
    assert np.allclose(scaled, again)


def test_apply_pca_with_a_variance_fraction(features):
    reduced, pca = apply_pca(features, components=0.9)
    assert reduced.shape[0] == features.shape[0]
    assert float(np.sum(pca.explained_variance_ratio_)) >= 0.9
    assert scree(pca)[-1] == pytest.approx(float(np.sum(pca.explained_variance_ratio_)))


def test_apply_pca_with_a_component_count(features):
    reduced, _ = apply_pca(features, components=5)
    assert reduced.shape == (features.shape[0], 5)


def test_apply_pca_caps_components_at_the_sample_size():
    small = np.random.default_rng(0).normal(size=(4, 50))
    reduced, _ = apply_pca(small, components=20)
    assert reduced.shape[1] <= 4


def test_embed_returns_two_dimensions(features):
    result = embed(features, ReduceConfig(n_neighbors=5, random_state=0))
    assert result.embedding.shape == (features.shape[0], 2)
    assert result.pca is None


def test_embed_with_pca_records_the_variance(features):
    result = embed(
        features, ReduceConfig(use_pca=True, pca_components=0.95, n_neighbors=5,
                               random_state=0)
    )
    assert result.explained_variance_ratio is not None
    assert result.n_pca_components >= 1


def test_embed_rejects_bad_shapes():
    with pytest.raises(ValueError, match="two-dimensional"):
        embed(np.arange(10.0))
    with pytest.raises(ValueError, match="at least three"):
        embed(np.zeros((2, 5)))

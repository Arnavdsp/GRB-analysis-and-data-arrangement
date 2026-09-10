import json

import pytest

from grbml.config import ClusterConfig, FeatureConfig, PipelineConfig, ReduceConfig


def test_round_trip_through_json(tmp_path):
    config = PipelineConfig(
        data_dirs=["bg_sub_lightcurve1"],
        features=FeatureConfig(bands=("8-50", "50-300"), resample_dt=0.064),
        reduce=ReduceConfig(use_pca=True, n_neighbors=15),
        cluster=ClusterConfig(algorithm="gmm", n_clusters=5),
    )
    path = config.to_json(tmp_path / "config.json")
    restored = PipelineConfig.from_json(path)

    assert list(restored.features.bands) == ["8-50", "50-300"]
    assert restored.features.resample_dt == 0.064
    assert restored.reduce.use_pca is True
    assert restored.cluster.algorithm == "gmm"


def test_unknown_keys_are_rejected():
    """A typo in a config file must fail loudly, not be silently ignored."""
    with pytest.raises(ValueError, match="unknown configuration keys"):
        PipelineConfig.from_dict({"output_dir": "x", "outputdir": "y"})


def test_unknown_nested_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown reduce keys"):
        PipelineConfig.from_dict({"reduce": {"n_neighbours": 5}})


def test_defaults_follow_the_thesis():
    config = PipelineConfig()
    assert config.reduce.n_neighbors == 5
    assert config.reduce.min_dist == 0.001
    assert config.features.window == "t90"
    assert config.features.normalize == "fluence"
    assert config.features.representation == "fft"
    assert config.cluster.algorithm == "hdbscan"
    assert config.denoise.criterion == "bic"


def test_to_dict_is_json_serialisable():
    json.dumps(PipelineConfig().to_dict(), default=list)

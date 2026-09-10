import numpy as np
import pandas as pd
import pytest

from grbml.dataio import (
    available_bands,
    discover_light_curves,
    load_dataset,
    parse_filename,
    read_light_curve,
)


@pytest.mark.parametrize(
    "name, trigger, band",
    [
        ("bn230806168__background_subtracted_light_curve_300-900.csv", "230806168", "300-900"),
        # The trigger number is repeated in some files; the band is still the
        # trailing token.
        ("bn231109274__background_subtracted_light_curve_231109274_300-900.csv", "231109274", "300-900"),
        ("bn150630958__background_subtracted_light_curve_8-50.csv", "150630958", "8-50"),
        ("230812790_50-300.csv", "230812790", "50-300"),
    ],
)
def test_parse_filename(name, trigger, band):
    info = parse_filename(name)
    assert info is not None
    assert (info.trigger, info.band) == (trigger, band)


def test_parse_filename_rejects_unrelated_files():
    assert parse_filename("notes.csv") is None
    assert parse_filename("cluster_summary.csv") is None


def test_discover_collapses_duplicate_products(tmp_path):
    """Both spellings of one product must index as a single file."""
    for name in (
        "bn231109274__background_subtracted_light_curve_300-900.csv",
        "bn231109274__background_subtracted_light_curve_231109274_300-900.csv",
    ):
        (tmp_path / name).write_text("Centroid,Background_Subtracted_Rate\n0,1\n1,2\n")

    index = discover_light_curves([tmp_path])
    assert list(index) == ["231109274"]
    assert list(index["231109274"]) == ["300-900"]
    # The canonical (shorter) name wins, deterministically.
    assert index["231109274"]["300-900"].name.endswith("light_curve_300-900.csv")


def test_read_light_curve(csv_tree):
    path = next(csv_tree.glob("*.csv"))
    curve = read_light_curve(path)
    assert curve.band == "50-300"
    assert curve.n_bins == 512
    assert np.isclose(curve.dt, 0.064)
    assert curve.background is not None


def test_read_light_curve_derives_net_rate(tmp_path):
    """A file with only total and background rates is still usable."""
    path = tmp_path / "bn230812790__lc_50-300.csv"
    pd.DataFrame(
        {"Centroid": [0.0, 1.0, 2.0], "Rate": [10.0, 20.0, 30.0],
         "Background_Rate": [5.0, 5.0, 5.0]}
    ).to_csv(path, index=False)
    curve = read_light_curve(path)
    assert np.allclose(curve.net_rate, [5.0, 15.0, 25.0])


def test_read_light_curve_rejects_missing_columns(tmp_path):
    path = tmp_path / "bn230812790__lc_50-300.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="time column"):
        read_light_curve(path)


def test_load_dataset_and_bands(csv_tree):
    dataset = load_dataset([csv_tree], bands=["50-300"])
    assert len(dataset) == 6
    assert available_bands([csv_tree]) == ["50-300"]


def test_load_dataset_drops_incomplete_bursts(csv_tree):
    """A burst missing a requested band has no well-defined feature vector."""
    dataset = load_dataset([csv_tree], bands=["50-300", "8-50"], require_all_bands=True)
    assert dataset == {}


def test_slice_time(burst):
    sliced = burst.slice_time(0.0, 5.0)
    assert sliced.time.min() >= 0.0
    assert sliced.time.max() < 5.0
    assert sliced.trigger == burst.trigger

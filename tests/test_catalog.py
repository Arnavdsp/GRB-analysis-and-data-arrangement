import numpy as np
import pandas as pd
import pytest

from grbml.catalog import (
    annotation_map,
    duration_class,
    load_annotations,
    load_catalog,
    merge_metadata,
    normalize_trigger,
    resolve_trigger,
    summarise_clusters,
)


@pytest.mark.parametrize(
    "value",
    ["bn230812790", "230812790", "GRB230812790", " bn230812790 ", "BN230812790"],
)
def test_normalize_trigger_accepts_every_spelling(value):
    assert normalize_trigger(value) == "230812790"


def test_load_catalog_renames_and_indexes(tmp_path):
    path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "trigger_name": ["bn230812790", "bn230814788"],
            "t90": [1.5, 30.0],
            "fluence": [1e-6, 5e-5],
        }
    ).to_csv(path, index=False)

    catalog = load_catalog(path)
    assert list(catalog.index) == ["230812790", "230814788"]
    assert catalog.loc["230812790", "t90"] == 1.5


def test_load_catalog_needs_a_trigger_column(tmp_path):
    path = tmp_path / "catalog.csv"
    pd.DataFrame({"t90": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="trigger column"):
        load_catalog(path)


def test_resolve_trigger_uses_same_day_ordering():
    triggers = ["230812395", "230812790", "230814788"]
    assert resolve_trigger("GRB 230812A", triggers) == "230812395"
    assert resolve_trigger("GRB 230812B", triggers) == "230812790"
    assert resolve_trigger("GRB 230814A", triggers) == "230814788"
    # A date not in the sample cannot be resolved.
    assert resolve_trigger("GRB 991231A", triggers) is None
    # Neither can a letter beyond the bursts available that day.
    assert resolve_trigger("GRB 230814Z", triggers) is None


def test_load_annotations_prefers_an_explicit_trigger(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text(
        "# comment line\n"
        "grb,class,trigger\n"
        "GRB 230812A,kilonova,bn230899999\n"
    )
    frame = load_annotations(path, triggers=["230812395"])
    assert frame.loc[0, "trigger"] == "230899999"


def test_load_annotations_warns_about_unmatched_names(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text("grb,class\nGRB 991231A,kilonova\n")
    with pytest.warns(RuntimeWarning, match="could not be matched"):
        frame = load_annotations(path, triggers=["230812395"])
    assert frame["trigger"].isna().all()


def test_load_annotations_needs_a_grb_column(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text("name,class\nGRB 230812A,kilonova\n")
    with pytest.raises(ValueError, match="'grb' column"):
        load_annotations(path)


def test_shipped_annotation_file_loads():
    frame = load_annotations("data/annotated_grbs.csv", resolve=False)
    assert set(frame["class"]) == {"kilonova", "supernova"}
    assert len(frame) >= 10


def test_annotation_map_skips_unresolved_rows():
    frame = pd.DataFrame(
        {"grb": ["GRB 230812A", "GRB 991231A"], "class": ["kilonova", "supernova"],
         "trigger": ["230812395", None]}
    )
    mapping = annotation_map(frame)
    assert list(mapping) == ["230812395"]
    assert mapping["230812395"]["class"] == "kilonova"


def test_duration_class_boundary():
    assert duration_class(1.9) == "short"
    assert duration_class(2.1) == "long"
    assert duration_class(np.nan) == "unknown"


def test_merge_metadata_keeps_measured_values_separate():
    metadata = pd.DataFrame({"t90": [1.0]}, index=["230812790"])
    catalog = pd.DataFrame({"t90": [2.0]}, index=["230812790"])
    merged = merge_metadata(metadata, catalog, columns=("t90",))
    assert merged.loc["230812790", "t90"] == 1.0
    assert merged.loc["230812790", "catalog_t90"] == 2.0


def test_summarise_clusters_counts_duration_provenance():
    metadata = pd.DataFrame(
        {
            "t90": [1.0, 30.0, np.nan, np.nan],
            "t90_source": ["measured", "measured", "none", "none"],
        },
        index=["a", "b", "c", "d"],
    )
    summary = summarise_clusters(metadata, [0, 0, 1, 1])
    row = summary.set_index("cluster")
    assert row.loc[0, "n_short"] == 1 and row.loc[0, "n_long"] == 1
    # A cluster made only of bursts with no usable duration is visible as such.
    assert row.loc[1, "n_t90_none"] == 2
    assert np.isnan(row.loc[1, "median_t90"])

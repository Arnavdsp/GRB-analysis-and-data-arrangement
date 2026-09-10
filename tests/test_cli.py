import json

import pytest

from grbml.cli import build_parser, main


def test_inspect_reports_the_sample(csv_tree, capsys):
    assert main(["inspect", "--data-dir", str(csv_tree)]) == 0
    output = capsys.readouterr().out
    assert "bursts:      6" in output
    assert "50-300" in output


def test_inspect_can_list_bursts(csv_tree, capsys):
    main(["inspect", "--data-dir", str(csv_tree), "--list"])
    assert "230810000" in capsys.readouterr().out


def test_collect_copies_and_prefixes(tmp_path, capsys):
    source = tmp_path / "reduction"
    for burst in ("bn230812790", "bn230814788"):
        band_dir = source / burst / "300-900"
        band_dir.mkdir(parents=True)
        (band_dir / "background_subtracted_light_curve_300-900.csv").write_text(
            "Centroid,Background_Subtracted_Rate\n0,1\n1,2\n"
        )
    destination = tmp_path / "collected"

    assert main(["collect", str(source), str(destination)]) == 0

    names = sorted(path.name for path in destination.glob("*.csv"))
    assert names == [
        "bn230812790__background_subtracted_light_curve_300-900.csv",
        "bn230814788__background_subtracted_light_curve_300-900.csv",
    ]
    assert "copied 2 file(s)" in capsys.readouterr().out


def test_collect_skips_existing_without_overwrite(tmp_path, capsys):
    source = tmp_path / "reduction"
    band_dir = source / "bn230812790" / "300-900"
    band_dir.mkdir(parents=True)
    (band_dir / "lc_300-900.csv").write_text("Centroid,Background_Subtracted_Rate\n0,1\n")
    destination = tmp_path / "collected"

    main(["collect", str(source), str(destination)])
    main(["collect", str(source), str(destination)])

    assert "skipped 1 existing" in capsys.readouterr().out


def test_collect_reports_a_missing_input(tmp_path, capsys):
    assert main(["collect", str(tmp_path / "nope"), str(tmp_path / "out")]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_config_command_reflects_the_flags(tmp_path, capsys):
    main(
        [
            "config",
            "--bands", "8-50", "50-300",
            "--pca",
            "--n-neighbors", "15",
            "--resample-dt", "0.064",
            "--algorithm", "kmeans",
            "--n-clusters", "4",
        ]
    )
    config = json.loads(capsys.readouterr().out)
    assert config["features"]["bands"] == ["8-50", "50-300"]
    assert config["features"]["resample_dt"] == 0.064
    assert config["reduce"]["use_pca"] is True
    assert config["reduce"]["n_neighbors"] == 15
    assert config["cluster"]["algorithm"] == "kmeans"
    assert config["cluster"]["n_clusters"] == 4


def test_config_round_trips_through_a_file(tmp_path, capsys):
    path = tmp_path / "config.json"
    main(["config", "--pca", "--min-dist", "0.05", "-o", str(path)])
    main(["config", "--config", str(path)])
    config = json.loads(capsys.readouterr().out.split("wrote")[-1].split("\n", 1)[-1])
    assert config["reduce"]["use_pca"] is True
    assert config["reduce"]["min_dist"] == 0.05


def test_denoise_command_writes_a_figure(csv_tree, tmp_path, capsys):
    source = next(csv_tree.glob("*.csv"))
    output = tmp_path / "denoised.png"
    assert main(["denoise", str(source), "-o", str(output)]) == 0
    assert output.exists()
    assert "wavelet" in capsys.readouterr().out


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])

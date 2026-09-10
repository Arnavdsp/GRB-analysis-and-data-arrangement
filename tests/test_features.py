import numpy as np
import pytest

from grbml.config import DenoiseConfig, FeatureConfig
from grbml.features import (
    build_feature_matrix,
    estimate_duration,
    fluence,
    fourier_amplitudes,
    normalize,
    pad_or_truncate,
    rebin,
)
from tests.conftest import make_curve


def test_pad_or_truncate():
    assert pad_or_truncate(np.arange(3), 5).tolist() == [0, 1, 2, 0, 0]
    assert pad_or_truncate(np.arange(5), 3).tolist() == [0, 1, 2]
    assert pad_or_truncate(np.arange(4), 4).tolist() == [0, 1, 2, 3]


def test_normalize_methods():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(normalize(values, "fluence").sum(), 1.0)
    assert np.isclose(normalize(values, "peak").max(), 1.0)
    assert np.isclose(np.linalg.norm(normalize(values, "l2")), 1.0)
    assert np.allclose(normalize(values, "none"), values)
    with pytest.raises(ValueError):
        normalize(values, "nonsense")


def test_normalize_leaves_a_dead_curve_alone():
    """A curve with no net signal must not turn into NaNs."""
    values = np.zeros(8)
    assert np.allclose(normalize(values, "fluence"), values)


def test_fourier_amplitudes_drop_dc():
    values = np.ones(64) * 5.0
    with_dc = fourier_amplitudes(values, drop_dc=False)
    without_dc = fourier_amplitudes(values, drop_dc=True)
    assert with_dc.size == without_dc.size + 1
    assert with_dc[0] > 0  # the constant lives entirely in the DC term
    assert np.allclose(without_dc, 0.0, atol=1e-9)


def test_rebin_preserves_counts():
    time = np.arange(0.0, 10.0, 0.5)
    rate = np.full(time.size, 10.0)
    coarse_time, coarse_rate = rebin(time, rate, 1.0)
    assert coarse_time.size == 10
    assert np.isclose(rate.sum() * 0.5, coarse_rate.sum() * 1.0)
    assert np.allclose(coarse_rate, 10.0)


def test_rebin_refuses_to_invent_resolution():
    time = np.arange(0.0, 10.0, 0.5)
    with pytest.raises(ValueError, match="finer"):
        rebin(time, np.ones(time.size), 0.25)


def test_rebin_is_a_no_op_at_the_same_width():
    time = np.arange(0.0, 10.0, 0.5)
    rate = np.arange(time.size, dtype=float)
    new_time, new_rate = rebin(time, rate, 0.5)
    assert np.allclose(new_time, time) and np.allclose(new_rate, rate)


def test_fluence_is_the_counts_integral():
    time = np.arange(0.0, 10.0, 0.5)
    assert fluence(time, np.full(time.size, 4.0)) == pytest.approx(4.0 * 0.5 * time.size)


@pytest.mark.parametrize(
    "width, expected",
    # 90% of a Gaussian lies within +-1.645 sigma, so T90 ~ 3.29 * width.
    [(0.5, 3.29 * 0.5), (3.0, 3.29 * 3.0), (12.0, 3.29 * 12.0)],
)
def test_estimate_duration_recovers_a_gaussian_burst(width, expected):
    time = np.arange(-100.0, 400.0, 0.5)
    rng = np.random.default_rng(0)
    signal = 300.0 * np.exp(-((time - 5.0) ** 2) / (2.0 * width ** 2))
    result = estimate_duration(time, signal + rng.normal(0.0, 6.0, time.size))
    assert result.is_valid
    assert result.t90 == pytest.approx(expected, rel=0.35)


def test_estimate_duration_rejects_pure_noise():
    """Integrating noise across the whole window would invent a long burst."""
    time = np.arange(-100.0, 400.0, 0.5)
    rng = np.random.default_rng(7)
    result = estimate_duration(time, rng.normal(0.0, 8.0, time.size))
    assert not result.is_valid
    assert result.source == "none"


def test_estimate_duration_ignores_an_isolated_spike_at_the_edge():
    """A one-bin artefact must not stretch the burst across the file."""
    time = np.arange(-100.0, 400.0, 0.5)
    rng = np.random.default_rng(8)
    values = rng.normal(0.0, 5.0, time.size)
    values += 300.0 * np.exp(-((time - 3.0) ** 2) / (2.0 * 2.0 ** 2))
    values[0] = 5000.0  # reconstruction artefact at the left edge

    result = estimate_duration(time, values)

    assert result.is_valid
    assert result.t05 > -20.0
    assert result.t90 < 60.0


def test_estimate_duration_bridges_a_quiescent_gap():
    """T90 spans the quiet interval between two pulses of one burst."""
    time = np.arange(-50.0, 200.0, 0.5)
    rng = np.random.default_rng(9)
    two_pulses = (
        300.0 * np.exp(-((time - 2.0) ** 2) / (2.0 * 1.0 ** 2))
        + 250.0 * np.exp(-((time - 12.0) ** 2) / (2.0 * 1.0 ** 2))
    )
    result = estimate_duration(time, two_pulses + rng.normal(0.0, 5.0, time.size))
    assert result.is_valid
    assert 8.0 < result.t90 < 25.0


def test_estimate_duration_gap_tolerance_is_binning_independent():
    """The same burst must give the same T90 at two different bin widths."""
    rng = np.random.default_rng(10)
    durations = []
    for dt in (0.5, 0.1):
        time = np.arange(-50.0, 200.0, dt)
        pulses = (
            300.0 * np.exp(-((time - 2.0) ** 2) / (2.0 * 1.0 ** 2))
            + 250.0 * np.exp(-((time - 12.0) ** 2) / (2.0 * 1.0 ** 2))
        )
        noisy = pulses + rng.normal(0.0, 5.0, time.size)
        durations.append(estimate_duration(time, noisy).t90)
    assert durations[0] == pytest.approx(durations[1], rel=0.25)


def test_estimate_duration_does_not_bridge_an_unrelated_far_spike():
    """A significant excursion 100 s away is a separate event, not this burst."""
    time = np.arange(-50.0, 300.0, 0.5)
    rng = np.random.default_rng(11)
    values = rng.normal(0.0, 5.0, time.size)
    values += 300.0 * np.exp(-((time - 2.0) ** 2) / (2.0 * 1.0 ** 2))
    values += 200.0 * np.exp(-((time - 150.0) ** 2) / (2.0 * 1.0 ** 2))
    result = estimate_duration(time, values)
    assert result.is_valid
    assert result.t90 < 30.0


def test_build_feature_matrix_shapes(dataset):
    matrix = build_feature_matrix(dataset, FeatureConfig(bands=("50-300",)))
    assert matrix.n_bursts == len(dataset)
    assert matrix.X.shape[1] == matrix.n_features
    assert np.isfinite(matrix.X).all()
    assert "t90" in matrix.metadata.columns
    assert "t90_source" in matrix.metadata.columns


def test_build_feature_matrix_raw_representation(dataset):
    config = FeatureConfig(bands=("50-300",), representation="raw")
    matrix = build_feature_matrix(dataset, config)
    assert matrix.X.shape[1] == matrix.pad_length


def test_build_feature_matrix_concatenates_bands():
    dataset = {
        trigger: {
            "8-50": make_curve(trigger, band="8-50", seed=index),
            "50-300": make_curve(trigger, band="50-300", seed=index + 100),
        }
        for index, trigger in enumerate(["230812790", "230812791", "230812792"])
    }
    one_band = build_feature_matrix(dataset, FeatureConfig(bands=("50-300",)))
    two_bands = build_feature_matrix(dataset, FeatureConfig(bands=("8-50", "50-300")))
    assert two_bands.n_features > one_band.n_features


def test_build_feature_matrix_reports_the_band_mismatch(dataset):
    with pytest.raises(ValueError, match="available bands"):
        build_feature_matrix(dataset, FeatureConfig(bands=("300-900",)))


def test_mixed_binning_warns(recwarn):
    dataset = {
        "230812790": {"50-300": make_curve("230812790", dt=0.064, seed=1)},
        "230812791": {"50-300": make_curve("230812791", dt=0.128, seed=2)},
    }
    with pytest.warns(RuntimeWarning, match="different bin widths"):
        build_feature_matrix(dataset, FeatureConfig(bands=("50-300",)))


def test_resample_dt_puts_the_sample_on_one_grid():
    dataset = {
        "230812790": {"50-300": make_curve("230812790", dt=0.064, seed=1)},
        "230812791": {"50-300": make_curve("230812791", dt=0.128, seed=2)},
    }
    config = FeatureConfig(bands=("50-300",), resample_dt=0.256)
    matrix = build_feature_matrix(dataset, config, DenoiseConfig(enabled=False))
    assert set(np.round(matrix.metadata["dt_50-300"], 6)) == {0.256}

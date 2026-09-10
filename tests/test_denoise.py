import numpy as np
import pytest

from grbml.config import DenoiseConfig
from grbml.denoise import (
    denoise,
    denoise_auto,
    estimate_sigma,
    information_criteria,
    max_useful_level,
    select_denoising,
    universal_threshold,
)


def test_estimate_sigma_matches_known_noise():
    rng = np.random.default_rng(0)
    sigma = estimate_sigma(rng.normal(0.0, 3.0, 20000))
    assert sigma == pytest.approx(3.0, rel=0.05)


def test_universal_threshold_grows_with_length():
    assert universal_threshold(1.0, 1024) > universal_threshold(1.0, 64)
    assert universal_threshold(0.0, 1024) == 0.0
    assert universal_threshold(1.0, 1) == 0.0


def test_denoise_reduces_noise_and_keeps_the_pulse():
    rng = np.random.default_rng(1)
    time = np.arange(1024) * 0.064
    pulse = 100.0 * np.exp(-((time - 20.0) ** 2) / (2.0 * 1.5 ** 2))
    noisy = pulse + rng.normal(0.0, 5.0, time.size)

    result = denoise(noisy, wavelet="db6", level=5)

    assert result.signal.size == noisy.size
    # Closer to the truth than the noisy input.
    assert np.std(result.signal - pulse) < np.std(noisy - pulse)
    # The pulse survives rather than being thresholded away.
    assert result.signal.max() > 0.5 * pulse.max()


def test_denoise_handles_odd_lengths():
    rng = np.random.default_rng(2)
    signal = rng.normal(0.0, 1.0, 501)
    assert denoise(signal, wavelet="db4", level=3).signal.size == 501


def test_denoise_on_signal_too_short_to_decompose():
    """Very short bursts must not abort a run."""
    result = denoise([1.0, 2.0, 3.0], wavelet="db6", level=5)
    assert result.level == 0
    assert np.allclose(result.signal, [1.0, 2.0, 3.0])


def test_information_criteria_penalise_extra_coefficients():
    original = np.arange(100, dtype=float)
    denoised = original.copy()
    lean_aic, lean_bic = information_criteria(original, denoised, 10, 1.0)
    rich_aic, rich_bic = information_criteria(original, denoised, 50, 1.0)
    assert rich_aic > lean_aic
    assert rich_bic > lean_bic


def test_information_criteria_undefined_without_noise():
    values = np.ones(10)
    assert information_criteria(values, values, 5, 0.0) == (np.inf, np.inf)


def test_select_denoising_searches_the_grid():
    rng = np.random.default_rng(3)
    time = np.arange(1024) * 0.064
    signal = 80.0 * np.exp(-((time - 30.0) ** 2) / (2.0 * 3.0 ** 2))
    noisy = signal + rng.normal(0.0, 4.0, time.size)

    config = DenoiseConfig(wavelets=("db1", "db6", "sym4"), levels=(1, 3, 5))
    best = select_denoising(noisy, config)

    assert best.wavelet in config.wavelets
    assert best.level in config.levels
    assert np.isfinite(best.bic)
    # The winner really is the lowest-BIC combination in the grid.
    scores = [
        denoise(noisy, wavelet=w, level=l).bic
        for w in config.wavelets
        for l in config.levels
    ]
    assert best.bic == pytest.approx(min(scores))


def test_denoise_auto_respects_the_disabled_flag():
    rng = np.random.default_rng(4)
    signal = rng.normal(0.0, 1.0, 256)
    untouched = denoise_auto(signal, DenoiseConfig(enabled=False))
    assert np.allclose(untouched, signal)


def test_max_useful_level():
    assert max_useful_level(1024, "db1") > max_useful_level(16, "db1")
    assert max_useful_level(4, "db6") == 0

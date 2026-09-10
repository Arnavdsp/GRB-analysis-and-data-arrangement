import numpy as np
import pytest

from grbml.powerspec import fit_power_law, power_law_index, power_spectrum


@pytest.mark.parametrize("beta", [0.0, 1.0, 2.0])
def test_power_law_index_recovers_the_input_colour(beta):
    """White, pink and red noise must come back as indices 0, -1 and -2."""
    from grbml.simulate import timmer_koenig

    rng = np.random.default_rng(5)
    indices = [
        power_law_index(timmer_koenig(4096, 0.064, beta, rng=rng), 0.064).index
        for _ in range(6)
    ]
    assert float(np.mean(indices)) == pytest.approx(-beta, abs=0.15)


def test_power_spectrum_shape_and_frequencies():
    frequency, power = power_spectrum(np.ones(1024), dt=0.064)
    assert frequency.size == power.size == 512
    assert frequency[0] > 0  # the DC term is dropped
    assert frequency[-1] == pytest.approx(1.0 / (2 * 0.064))


def test_power_spectrum_rejects_bad_input():
    with pytest.raises(ValueError):
        power_spectrum([1.0, 2.0], dt=0.1)
    with pytest.raises(ValueError):
        power_spectrum(np.ones(64), dt=0.0)


def test_fit_power_law_on_an_exact_power_law():
    frequency = np.logspace(-2, 1, 200)
    power = 3.0 * frequency ** -1.7
    fit = fit_power_law(frequency, power)
    assert fit.is_valid
    assert fit.index == pytest.approx(-1.7, abs=1e-6)
    assert 10 ** fit.log_amplitude == pytest.approx(3.0, rel=1e-6)
    assert np.allclose(fit.evaluate(frequency), power)


def test_fit_power_law_honours_the_frequency_range():
    frequency = np.logspace(-3, 1, 500)
    # Two regimes: steep at low frequency, flat above 0.1 Hz.
    power = np.where(frequency < 0.1, frequency ** -2.0, 0.1 ** -2.0 * np.ones_like(frequency))
    steep = fit_power_law(frequency, power, fmax=0.09)
    flat = fit_power_law(frequency, power, fmin=0.11)
    assert steep.index == pytest.approx(-2.0, abs=0.05)
    assert flat.index == pytest.approx(0.0, abs=0.05)


def test_fit_power_law_needs_enough_points():
    fit = fit_power_law(np.array([1.0, 2.0]), np.array([1.0, 0.5]))
    assert not fit.is_valid
    assert fit.n_points == 0


def test_power_law_index_survives_a_useless_curve():
    assert not power_law_index([1.0, 2.0], 0.1).is_valid

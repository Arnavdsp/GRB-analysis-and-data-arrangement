import numpy as np
import pytest

from grbml.powerspec import power_law_index
from grbml.simulate import (
    NOISE_COMBINATIONS,
    emmanoulopoulos,
    fred_pulse,
    noise_mixture,
    simulate_noise_experiment,
    timmer_koenig,
)


@pytest.mark.parametrize("beta", [0.0, 1.0, 2.0])
def test_timmer_koenig_produces_the_requested_spectrum(beta):
    rng = np.random.default_rng(11)
    recovered = np.mean(
        [power_law_index(timmer_koenig(4096, 0.064, beta, rng=rng), 0.064).index
         for _ in range(6)]
    )
    assert recovered == pytest.approx(-beta, abs=0.15)


def test_timmer_koenig_is_reproducible():
    first = timmer_koenig(512, 0.064, 1.0, rng=7)
    second = timmer_koenig(512, 0.064, 1.0, rng=7)
    assert np.allclose(first, second)


def test_timmer_koenig_scales_to_the_requested_moments():
    series = timmer_koenig(4096, 0.064, 1.0, rng=3, mean=10.0, std=2.0)
    assert np.mean(series) == pytest.approx(10.0, abs=0.2)
    assert np.std(series) == pytest.approx(2.0, rel=0.05)


def test_timmer_koenig_rejects_bad_input():
    with pytest.raises(ValueError):
        timmer_koenig(2, 0.064, 1.0)
    with pytest.raises(ValueError):
        timmer_koenig(512, 0.0, 1.0)


def test_emmanoulopoulos_reproduces_a_non_gaussian_pdf():
    """The point of E13 over TK95: the value distribution is preserved."""
    rng = np.random.default_rng(13)
    target = rng.lognormal(0.0, 0.7, 4096)
    simulated = emmanoulopoulos(4096, 0.064, 2.0, target_values=target, rng=rng)

    def skew(values):
        return float(np.mean((values - values.mean()) ** 3) / values.std() ** 3)

    assert skew(simulated) == pytest.approx(skew(target), abs=0.8)
    assert simulated.min() > 0  # a lognormal sample stays positive
    assert power_law_index(simulated, 0.064).index < -1.0  # still red


def test_emmanoulopoulos_accepts_a_sampler():
    drawn = emmanoulopoulos(
        512, 0.064, 1.0,
        pdf_sampler=lambda n, rng: rng.exponential(2.0, n),
        rng=17,
    )
    assert drawn.size == 512
    assert drawn.min() >= 0


def test_noise_mixture_lies_between_its_components():
    rng = np.random.default_rng(19)
    mixed = np.mean(
        [power_law_index(noise_mixture(4096, 0.064, (0.0, 2.0), rng=rng), 0.064).index
         for _ in range(6)]
    )
    assert -2.0 < mixed < 0.0


def test_noise_mixture_validates_weights():
    with pytest.raises(ValueError):
        noise_mixture(512, 0.064, (0.0, 1.0), weights=(1.0,))
    with pytest.raises(ValueError):
        noise_mixture(512, 0.064, ())


def test_fred_pulse_shape():
    time = np.arange(0.0, 100.0, 0.064)
    pulse = fred_pulse(time, start=10.0, rise=1.0, decay=8.0, amplitude=50.0)
    assert pulse.max() == pytest.approx(50.0)
    assert np.all(pulse[time < 10.0] == 0.0)  # nothing before the start
    peak_time = time[int(np.argmax(pulse))]
    assert peak_time > 10.0  # rises after the start
    # and decays afterwards
    assert pulse[-1] < 0.01 * pulse.max()


def test_simulate_noise_experiment_labels():
    simulation = simulate_noise_experiment(n_per_class=4, n_bins=256)
    assert simulation.curves.shape == (4 * len(NOISE_COMBINATIONS), 256)
    assert set(simulation.label_names()) == set(NOISE_COMBINATIONS)
    assert simulation.n_curves == 4 * len(NOISE_COMBINATIONS)


def test_simulate_noise_experiment_with_pulses_adds_signal():
    quiet = simulate_noise_experiment(n_per_class=3, n_bins=256, seed=1)
    loud = simulate_noise_experiment(n_per_class=3, n_bins=256, pulse=True, seed=1)
    assert loud.curves.max() > quiet.curves.max()

"""Simulated light curves for the noise experiment (thesis sec. 4.5).

The thesis tests whether background noise alone can structure the embedding, by
simulating light curves that are pure noise of a known colour, running them
through the same pipeline, and checking whether the colours separate.  They do,
which is the reason the clusters found in real data cannot be read as
astrophysics without first removing the background's influence.

Two generators are provided:

* :func:`timmer_koenig` - the standard method for a light curve with a given
  power-law power spectrum and Gaussian amplitudes.
* :func:`emmanoulopoulos` - the extension that also reproduces a chosen flux
  distribution, which real GRB light curves need since they are not Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

#: Power-law index (P ~ f^-beta) of the standard noise colours.
WHITE, PINK, RED = 0.0, 1.0, 2.0

#: The seven combinations the thesis simulates.
NOISE_COMBINATIONS: Dict[str, Tuple[float, ...]] = {
    "WN": (WHITE,),
    "PN": (PINK,),
    "RN": (RED,),
    "RN+WN": (RED, WHITE),
    "PN+WN": (PINK, WHITE),
    "PN+RN": (PINK, RED),
    "PN+RN+WN": (PINK, RED, WHITE),
}


def _as_generator(rng: Optional[np.random.Generator | int]) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def timmer_koenig(
    n_bins: int,
    dt: float,
    beta: float,
    rng: Optional[np.random.Generator | int] = None,
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """Gaussian light curve with power spectrum ``P(f) ~ f^-beta``.

    Timmer & Koenig (1995): give each Fourier amplitude the target power and a
    random phase, then transform back.  ``beta`` of 0, 1 and 2 gives white,
    pink and red noise.
    """
    if n_bins < 4:
        raise ValueError("need at least four bins")
    if dt <= 0:
        raise ValueError("dt must be positive")
    generator = _as_generator(rng)

    frequency = np.fft.rfftfreq(n_bins, d=dt)
    spectrum = np.zeros(frequency.size, dtype=complex)

    positive = frequency[1:]
    amplitude = positive ** (-beta / 2.0)
    real = generator.normal(0.0, 1.0, positive.size) * amplitude
    imaginary = generator.normal(0.0, 1.0, positive.size) * amplitude
    spectrum[1:] = (real + 1j * imaginary) / np.sqrt(2.0)

    if n_bins % 2 == 0:
        # The Nyquist component of a real series has no imaginary part.
        spectrum[-1] = spectrum[-1].real

    series = np.fft.irfft(spectrum, n=n_bins)
    spread = float(np.std(series))
    if spread > 0:
        series = (series - np.mean(series)) / spread
    return series * std + mean


def emmanoulopoulos(
    n_bins: int,
    dt: float,
    beta: float,
    target_values: Optional[Sequence[float]] = None,
    pdf_sampler: Optional[Callable[[int, np.random.Generator], np.ndarray]] = None,
    rng: Optional[np.random.Generator | int] = None,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Light curve matching both a power-law power spectrum and a flux PDF.

    Emmanoulopoulos et al. (2013) improve on Timmer & Koenig, whose output is
    always Gaussian.  The Fourier amplitudes of a Timmer-Koenig curve are held
    fixed while the values are repeatedly rank-matched to a sample drawn from
    the target distribution, so the final series has the requested spectrum and
    the requested value distribution at once.

    Supply the distribution either as ``target_values`` (values to resample,
    e.g. a real light curve) or as ``pdf_sampler``; the default is a lognormal,
    which is the usual description of GRB and blazar flux distributions.
    """
    generator = _as_generator(rng)

    if target_values is not None:
        pool = np.asarray(target_values, dtype=float)
        sample = generator.choice(pool, size=n_bins, replace=True)
    elif pdf_sampler is not None:
        sample = np.asarray(pdf_sampler(n_bins, generator), dtype=float)
    else:
        sample = generator.lognormal(mean=0.0, sigma=0.5, size=n_bins)

    reference = timmer_koenig(n_bins, dt, beta, rng=generator)
    target_amplitude = np.abs(np.fft.rfft(reference))

    sorted_sample = np.sort(sample)
    current = generator.permutation(sample)

    for _ in range(max_iterations):
        spectrum = np.fft.rfft(current)
        phases = np.angle(spectrum)
        adjusted = np.fft.irfft(
            target_amplitude * np.exp(1j * phases), n=n_bins
        )
        # Rank-match: give the spectrum-corrected series the target values,
        # keeping its ordering.  This is the step that fixes the PDF.
        ranks = np.argsort(np.argsort(adjusted))
        updated = sorted_sample[ranks]
        if np.allclose(updated, current, rtol=0.0, atol=tolerance):
            current = updated
            break
        current = updated

    return current


def fred_pulse(
    time: np.ndarray,
    start: float = 0.0,
    rise: float = 1.0,
    decay: float = 5.0,
    amplitude: float = 100.0,
) -> np.ndarray:
    """Fast-rise exponential-decay pulse, the usual GRB pulse shape."""
    time = np.asarray(time, dtype=float)
    shape = np.zeros_like(time)
    after = time >= start
    elapsed = time[after] - start
    shape[after] = amplitude * np.exp(-elapsed / decay) * (1.0 - np.exp(-elapsed / rise))
    peak = float(np.max(shape)) if shape.size else 0.0
    if peak > 0:
        shape *= amplitude / peak
    return shape


def noise_mixture(
    n_bins: int,
    dt: float,
    betas: Sequence[float],
    rng: Optional[np.random.Generator | int] = None,
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Sum of independent noise components of different colours.

    Each component is normalised to unit variance first, so ``weights`` set the
    mixing directly; the default is an equal mix.
    """
    generator = _as_generator(rng)
    betas = list(betas)
    if not betas:
        raise ValueError("need at least one noise component")
    if weights is None:
        weights = [1.0 / len(betas)] * len(betas)
    if len(weights) != len(betas):
        raise ValueError("weights and betas must have the same length")

    total = np.zeros(n_bins, dtype=float)
    for beta, weight in zip(betas, weights):
        total += weight * timmer_koenig(n_bins, dt, beta, rng=generator)
    return total


@dataclass
class SimulationSet:
    """Simulated curves with the labels that generated them."""

    curves: np.ndarray  # (n_curves, n_bins)
    labels: np.ndarray  # integer class per curve
    class_names: Sequence[str]
    dt: float

    @property
    def n_curves(self) -> int:
        return int(self.curves.shape[0])

    def label_names(self) -> np.ndarray:
        names = np.asarray(self.class_names)
        return names[self.labels]


def simulate_noise_experiment(
    n_per_class: int = 500,
    n_bins: int = 512,
    dt: float = 0.064,
    combinations: Optional[Dict[str, Sequence[float]]] = None,
    pulse: bool = False,
    seed: Optional[int] = 42,
) -> SimulationSet:
    """Build the labelled set of noise-dominated light curves.

    With ``combinations`` left at the default this reproduces the seven-class
    experiment: three pure colours and their four mixtures.  Setting ``pulse``
    adds a FRED pulse to every curve, giving the GRB-like curves of Figure 4.38
    rather than pure noise.
    """
    combinations = combinations or NOISE_COMBINATIONS
    generator = _as_generator(seed)
    names = list(combinations)

    time = np.arange(n_bins) * dt
    curves = []
    labels = []
    for class_index, name in enumerate(names):
        betas = combinations[name]
        for _ in range(n_per_class):
            curve = noise_mixture(n_bins, dt, betas, rng=generator)
            if pulse:
                start = float(generator.uniform(0.0, 0.2 * n_bins * dt))
                decay = float(generator.uniform(1.0, 0.1 * n_bins * dt))
                curve = curve + fred_pulse(
                    time, start=start, rise=max(decay / 5.0, dt), decay=decay,
                    amplitude=float(generator.uniform(3.0, 30.0)),
                )
            curves.append(curve)
            labels.append(class_index)

    return SimulationSet(
        curves=np.vstack(curves),
        labels=np.asarray(labels, dtype=int),
        class_names=names,
        dt=dt,
    )

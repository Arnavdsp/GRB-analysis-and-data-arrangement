"""Power spectra and power-law indices (thesis sec. 4.1.1, appendix A).

The colour of the noise in a light curve turns out to be one of the two things
that structure the UMAP embedding, so every burst gets a power-law index fitted
to its power spectrum.  An index near 0 is white noise (uncorrelated), near -1
pink, near -2 red (highly correlated); the thesis finds indices between about
-4 and 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


def power_spectrum(
    values: Sequence[float],
    dt: float,
    normalization: str = "none",
    detrend: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Periodogram of an evenly binned light curve.

    Returns positive frequencies in Hz and their power, with the zero-frequency
    term dropped (it only carries the mean level).
    """
    series = np.asarray(values, dtype=float)
    if series.size < 4:
        raise ValueError("need at least four samples for a power spectrum")
    if dt <= 0:
        raise ValueError("dt must be positive")

    if detrend:
        series = series - np.mean(series)

    spectrum = np.fft.rfft(series)
    power = np.abs(spectrum) ** 2
    frequency = np.fft.rfftfreq(series.size, d=dt)

    # Drop the DC term; it is not part of the power-law behaviour.
    frequency, power = frequency[1:], power[1:]

    if normalization == "leahy":
        total = float(np.sum(np.clip(series, 0.0, None)))
        if total > 0:
            power = 2.0 * power / total
    elif normalization == "fractional":
        mean = float(np.mean(np.asarray(values, dtype=float)))
        if mean != 0:
            power = power / (mean ** 2 * series.size)
    elif normalization != "none":
        raise ValueError(f"unknown normalization {normalization!r}")

    return frequency, power


@dataclass
class PowerLawFit:
    """Straight-line fit of ``log10(power)`` against ``log10(frequency)``."""

    index: float
    log_amplitude: float
    index_error: float
    n_points: int

    @property
    def is_valid(self) -> bool:
        return np.isfinite(self.index)

    def evaluate(self, frequency: np.ndarray) -> np.ndarray:
        """Model power at the given frequencies."""
        frequency = np.asarray(frequency, dtype=float)
        return 10.0 ** (self.log_amplitude + self.index * np.log10(frequency))


def fit_power_law(
    frequency: np.ndarray,
    power: np.ndarray,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    min_points: int = 5,
) -> PowerLawFit:
    """Fit ``P(f) = A f^index`` by least squares in log-log space.

    Fitting in log space weights the decades evenly, which matters because a
    periodogram carries far more points per decade at high frequency; a linear
    fit would be decided almost entirely by the highest decade.
    """
    frequency = np.asarray(frequency, dtype=float)
    power = np.asarray(power, dtype=float)
    invalid = PowerLawFit(np.nan, np.nan, np.nan, 0)

    mask = np.isfinite(frequency) & np.isfinite(power) & (frequency > 0) & (power > 0)
    if fmin is not None:
        mask &= frequency >= fmin
    if fmax is not None:
        mask &= frequency <= fmax
    if int(np.count_nonzero(mask)) < min_points:
        return invalid

    log_f = np.log10(frequency[mask])
    log_p = np.log10(power[mask])
    n_points = log_f.size

    slope, intercept = np.polyfit(log_f, log_p, 1)
    residual = log_p - (slope * log_f + intercept)
    degrees_of_freedom = n_points - 2
    if degrees_of_freedom > 0:
        variance = float(np.sum(residual ** 2) / degrees_of_freedom)
        spread = float(np.sum((log_f - np.mean(log_f)) ** 2))
        error = float(np.sqrt(variance / spread)) if spread > 0 else np.nan
    else:
        error = np.nan

    return PowerLawFit(
        index=float(slope),
        log_amplitude=float(intercept),
        index_error=error,
        n_points=int(n_points),
    )


def power_law_index(
    values: Sequence[float],
    dt: float,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    normalization: str = "none",
) -> PowerLawFit:
    """Power-law index of a light curve, in one call."""
    try:
        frequency, power = power_spectrum(values, dt, normalization=normalization)
    except ValueError:
        return PowerLawFit(np.nan, np.nan, np.nan, 0)
    return fit_power_law(frequency, power, fmin=fmin, fmax=fmax)

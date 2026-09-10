"""Wavelet denoising of light curves (thesis sec. 3.1.2).

The transient feature of a burst lives in the low-frequency (approximation)
coefficients, while the background fluctuations dominate the high-frequency
(detail) coefficients.  Soft-thresholding the detail coefficients at the
universal threshold

    threshold = sigma * sqrt(2 ln N),   sigma = MAD / 0.6745

and reconstructing keeps the burst and drops the fluctuations.  Because there is
a whole zoo of wavelets and decomposition levels to choose from, the grid is
scored with AIC/BIC and the lowest-BIC combination is selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pywt

from grbml.config import DenoiseConfig


def estimate_sigma(coefficients: np.ndarray) -> float:
    """Noise standard deviation from the median absolute deviation.

    ``sigma = MAD / 0.6745`` is the standard robust estimator; the finest
    detail coefficients are almost pure noise, so they are the usual input.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.size == 0:
        return 0.0
    mad = np.median(np.abs(coefficients - np.median(coefficients)))
    return float(mad / 0.6745)


def universal_threshold(sigma: float, n_samples: int) -> float:
    """Donoho-Johnstone universal threshold ``sigma * sqrt(2 ln N)``."""
    if n_samples < 2 or sigma <= 0:
        return 0.0
    return float(sigma * np.sqrt(2.0 * np.log(n_samples)))


def max_useful_level(n_samples: int, wavelet: str) -> int:
    """Largest decomposition level that is meaningful for this signal length."""
    filter_length = pywt.Wavelet(wavelet).dec_len
    if n_samples < filter_length:
        return 0
    return int(pywt.dwt_max_level(n_samples, filter_length))


@dataclass
class DenoiseResult:
    """Denoised signal plus everything needed to justify the choice."""

    signal: np.ndarray
    wavelet: str
    level: int
    threshold: float
    sigma: float
    n_nonzero: int
    aic: float
    bic: float

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"DenoiseResult(wavelet={self.wavelet!r}, level={self.level}, "
            f"bic={self.bic:.1f})"
        )


def denoise(
    signal: Sequence[float],
    wavelet: str = "db6",
    level: int = 5,
    mode: str = "soft",
    signal_extension: str = "symmetric",
) -> DenoiseResult:
    """Denoise ``signal`` with a fixed wavelet and decomposition level."""
    values = np.asarray(signal, dtype=float)
    n_samples = values.size
    if n_samples < 2:
        raise ValueError("signal must have at least two samples")

    level = int(min(level, max_useful_level(n_samples, wavelet)))
    if level < 1:
        # Too short to decompose: return the input unchanged rather than fail,
        # so a handful of very short bursts do not abort a whole run.
        return DenoiseResult(
            signal=values.copy(),
            wavelet=wavelet,
            level=0,
            threshold=0.0,
            sigma=0.0,
            n_nonzero=n_samples,
            aic=np.inf,
            bic=np.inf,
        )

    coefficients = pywt.wavedec(values, wavelet, mode=signal_extension, level=level)
    sigma = estimate_sigma(coefficients[-1])
    threshold = universal_threshold(sigma, n_samples)
    denoised_coefficients = [coefficients[0]] + [
        pywt.threshold(detail, value=threshold, mode=mode)
        for detail in coefficients[1:]
    ]
    reconstructed = pywt.waverec(denoised_coefficients, wavelet, mode=signal_extension)
    # waverec pads odd-length signals up to the next even length.
    reconstructed = reconstructed[:n_samples]

    n_nonzero = int(sum(int(np.count_nonzero(c)) for c in denoised_coefficients))
    aic, bic = information_criteria(values, reconstructed, n_nonzero, sigma)
    return DenoiseResult(
        signal=reconstructed,
        wavelet=wavelet,
        level=level,
        threshold=threshold,
        sigma=sigma,
        n_nonzero=n_nonzero,
        aic=aic,
        bic=bic,
    )


def information_criteria(
    original: np.ndarray,
    denoised: np.ndarray,
    n_nonzero: int,
    sigma: float,
) -> Tuple[float, float]:
    """AIC and BIC for a denoised reconstruction.

        AIC = 2k + (1/sigma^2) * sum (x_i - x_hat_i)^2
        BIC = k ln(n) + (1/sigma^2) * sum (x_i - x_hat_i)^2

    where ``k`` is the number of retained (non-zero) wavelet coefficients.  Both
    reward a close fit and penalise a model that keeps many coefficients.
    """
    original = np.asarray(original, dtype=float)
    denoised = np.asarray(denoised, dtype=float)
    n_samples = original.size
    residual = float(np.sum((original - denoised) ** 2))
    if sigma <= 0:
        # A perfectly flat detail band leaves the criteria undefined; treating
        # them as infinite makes such a fit lose the model comparison.
        return np.inf, np.inf
    scaled = residual / (sigma ** 2)
    aic = 2.0 * n_nonzero + scaled
    bic = n_nonzero * np.log(n_samples) + scaled
    return float(aic), float(bic)


def select_denoising(
    signal: Sequence[float],
    config: Optional[DenoiseConfig] = None,
) -> DenoiseResult:
    """Search the wavelet/level grid and return the best-scoring denoising."""
    config = config or DenoiseConfig()
    criterion = config.criterion.lower()
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'")

    values = np.asarray(signal, dtype=float)
    best: Optional[DenoiseResult] = None
    for wavelet in config.wavelets:
        try:
            usable = max_useful_level(values.size, wavelet)
        except ValueError:  # unknown wavelet name
            continue
        for level in config.levels:
            if level > usable:
                continue
            candidate = denoise(
                values,
                wavelet=wavelet,
                level=level,
                mode=config.mode,
                signal_extension=config.signal_extension,
            )
            score = candidate.aic if criterion == "aic" else candidate.bic
            best_score = np.inf if best is None else (
                best.aic if criterion == "aic" else best.bic
            )
            if score < best_score:
                best = candidate

    if best is None:
        # Nothing in the grid fits this signal length; hand back the input.
        return DenoiseResult(
            signal=values.copy(),
            wavelet="none",
            level=0,
            threshold=0.0,
            sigma=estimate_sigma(values),
            n_nonzero=values.size,
            aic=np.inf,
            bic=np.inf,
        )
    return best


def denoise_auto(
    signal: Sequence[float],
    config: Optional[DenoiseConfig] = None,
) -> np.ndarray:
    """Convenience wrapper returning only the denoised array."""
    config = config or DenoiseConfig()
    if not config.enabled:
        return np.asarray(signal, dtype=float).copy()
    return select_denoising(signal, config).signal

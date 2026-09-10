"""Standardisation, normalisation and feature extraction (thesis sec. 3.1.3).

For each burst the pipeline

1. denoises the background-subtracted light curve,
2. cuts it to the interval 0 s .. T90 s,
3. pads every curve with zeros to a common length,
4. divides by the fluence of that energy band, so that the known
   long-bright / short-faint result cannot drive the clustering,
5. concatenates the bands of one burst into a single time series, and
6. keeps the Fourier amplitudes, discarding the phases so that an unknown
   trigger-time offset does not matter.

Stacking those vectors gives the M x N feature matrix (M bursts, N features)
that the dimensionality reduction consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from grbml.config import DenoiseConfig, FeatureConfig
from grbml.dataio import LightCurve
from grbml.denoise import denoise_auto, select_denoising


@dataclass
class DurationResult:
    """Burst duration measured from the cumulative net counts."""

    t90: float
    t05: float
    t95: float
    total_counts: float
    #: Where the duration came from: "catalog", "measured" or "none".  The
    #: distinction matters when reading an embedding, because bursts with no
    #: usable duration fall back to a different window and can cluster on that
    #: difference alone.
    source: str = "measured"

    @property
    def is_valid(self) -> bool:
        return np.isfinite(self.t90) and self.t90 > 0


def _background_sigma(
    time: np.ndarray,
    net_rate: np.ndarray,
    background_end: float,
) -> float:
    """Noise level of the background-subtracted curve.

    Bins before the trigger contain background only, so their scatter measures
    the residual noise left by the background fit.  When the curve does not
    reach back before the trigger, fall back to the robust scatter of the whole
    curve.
    """
    pre_trigger = net_rate[time < background_end]
    sample = pre_trigger if pre_trigger.size >= 10 else net_rate
    if sample.size == 0:
        return 0.0
    mad = np.median(np.abs(sample - np.median(sample)))
    sigma = float(mad / 0.6745)
    if sigma > 0:
        return sigma
    return float(np.std(sample))


def _burst_region(
    net_rate: np.ndarray,
    sigma: float,
    threshold_sigma: float,
    detection_sigma: float,
    max_gap: int,
    min_bins: int,
) -> Optional[Tuple[int, int]]:
    """Bin range of the emission: the most significant contiguous block.

    Every run of bins above ``threshold_sigma`` is a candidate; runs separated
    by at most ``max_gap`` quiet bins are merged, so the dips between the
    pulses of a burst do not split it.  The surviving block with the largest
    integrated excess wins.

    The obvious alternatives both fail on this data.  Taking the outermost bins
    above threshold lets a single noise spike near either end of the extracted
    interval stretch the region across the whole file, which inflates T90 for
    every faint burst.  Growing outward from the global maximum instead follows
    whichever spike happens to be tallest - often a one-bin wavelet
    reconstruction artefact at the edge of the curve rather than the burst.
    Scoring whole blocks makes an isolated spike lose to real emission, since
    a burst spreads its excess over many bins.
    """
    if net_rate.size == 0 or sigma <= 0:
        return None

    above = net_rate >= threshold_sigma * sigma
    if not np.any(above):
        return None

    # Contiguous runs of significant bins.
    edges = np.flatnonzero(np.diff(above.astype(np.int8)))
    starts = np.concatenate(([0], edges + 1))
    stops = np.concatenate((edges + 1, [above.size]))
    runs = [(int(a), int(b) - 1) for a, b in zip(starts, stops) if above[a]]
    if not runs:
        return None

    # Merge runs separated by at most max_gap quiet bins.
    merged = [runs[0]]
    for run_start, run_stop in runs[1:]:
        previous_start, previous_stop = merged[-1]
        if run_start - previous_stop - 1 <= max_gap:
            merged[-1] = (previous_start, run_stop)
        else:
            merged.append((run_start, run_stop))

    best: Optional[Tuple[int, int]] = None
    best_score = -np.inf
    for block_start, block_stop in merged:
        block = net_rate[block_start : block_stop + 1]
        if block.size < min_bins:
            continue
        if float(np.max(block)) < detection_sigma * sigma:
            continue
        score = float(np.sum(np.clip(block, 0.0, None))) / sigma
        if score > best_score:
            best_score = score
            best = (block_start, block_stop)
    return best


def estimate_duration(
    time: np.ndarray,
    net_rate: np.ndarray,
    low: float = 0.05,
    high: float = 0.95,
    detection_curve: Optional[np.ndarray] = None,
    sigma: Optional[float] = None,
    threshold_sigma: float = 3.0,
    detection_sigma: float = 5.0,
    max_gap_seconds: float = 10.0,
    min_bins: int = 2,
    background_end: float = 0.0,
) -> DurationResult:
    """Measure T90 as the interval carrying the central 90% of net counts.

Pass the denoised curve as ``net_rate``: it is integrated to
    find the 5% and 95% points.  Pass the *raw* background-subtracted curve as
    ``detection_curve``, with ``sigma`` measured on it, to decide where the
    emission is.  Splitting the two matters.  Denoising is what removes the
    scatter, so a denoised curve's own pre-trigger spread understates the
    measurement noise and would put the significance threshold far below the
    real background; but soft thresholding also pulls the peak down by roughly
    the threshold value, so testing the denoised peak against the raw noise
    rejects real bursts.  Detecting on raw counts and integrating the denoised
    shape avoids both.  Both arguments default to ``net_rate``.

    It is a fallback for bursts with no catalogue
    entry: a published T90 is preferable, because the catalogue value comes
    from a fit that uses the full detector response and a hand-checked source
    interval, while this only sees one background-subtracted band.  Bursts too
    faint to reach ``detection_sigma`` above the background return a NaN
    duration, which the pipeline reports rather than guessing at.

    Integrating the whole extracted interval would be wrong: after clipping
    negative background residuals to zero, pure noise accumulates counts at a
    steady rate and the 5%-95% points drift towards the ends of the window.
    The emission region is therefore isolated first (see :func:`_burst_region`)
    and only that region is integrated.
    """
    time = np.asarray(time, dtype=float)
    net_rate = np.asarray(net_rate, dtype=float)
    invalid = DurationResult(np.nan, np.nan, np.nan, 0.0, source="none")
    if time.size < 2:
        return invalid

    detection = (
        net_rate if detection_curve is None
        else np.asarray(detection_curve, dtype=float)
    )
    if detection.shape != net_rate.shape:
        raise ValueError("detection_curve must have the same length as net_rate")

    dt = float(np.median(np.diff(time)))
    if sigma is None:
        sigma = _background_sigma(time, detection, background_end)
    if not np.isfinite(sigma) or sigma <= 0:
        return invalid

    # The bridging tolerance is a duration, not a bin count: a burst's
    # quiescent gaps last as long as they last regardless of how finely the
    # light curve happens to be binned, and T90 spans them by definition.
    max_gap = int(round(max_gap_seconds / dt)) if dt > 0 else 0
    region = _burst_region(
        detection, sigma, threshold_sigma, detection_sigma, max_gap, min_bins
    )
    if region is None:
        return invalid
    left, right = region

    span = slice(left, right + 1)
    span_time = time[span]
    counts = np.clip(net_rate[span], 0.0, None) * dt
    total = float(np.sum(counts))
    if not np.isfinite(total) or total <= 0:
        return invalid

    if span_time.size == 1:
        # A single significant bin: the burst is shorter than one time bin.
        start = float(span_time[0])
        return DurationResult(t90=dt, t05=start, t95=start + dt, total_counts=total)

    fraction = np.cumsum(counts) / total
    t_low = float(np.interp(low, fraction, span_time))
    t_high = float(np.interp(high, fraction, span_time))
    return DurationResult(
        t90=max(t_high - t_low, dt), t05=t_low, t95=t_high, total_counts=total
    )


def fluence(time: np.ndarray, net_rate: np.ndarray) -> float:
    """Integrated net counts of a light curve (counts, not erg/cm^2).

    The normalisation only needs a per-band brightness scale, and the counts
    integral of the background-subtracted curve is exactly that.
    """
    time = np.asarray(time, dtype=float)
    net_rate = np.asarray(net_rate, dtype=float)
    if time.size < 2:
        return 0.0
    dt = float(np.median(np.diff(time)))
    return float(np.sum(np.clip(net_rate, 0.0, None)) * dt)


def rebin(
    time: np.ndarray,
    rate: np.ndarray,
    target_dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rebin a light curve onto a coarser, evenly spaced grid.

    Counts are summed inside each new bin and divided by the new bin width, so
    the result is still a rate and the total counts are preserved.  Rebinning
    to a *finer* grid is refused: the information is not there, and
    interpolating it would invent structure at exactly the frequencies the
    power-law index is measured from.
    """
    time = np.asarray(time, dtype=float)
    rate = np.asarray(rate, dtype=float)
    if time.size < 2:
        return time.copy(), rate.copy()

    source_dt = float(np.median(np.diff(time)))
    if target_dt <= 0:
        raise ValueError("target_dt must be positive")
    if target_dt < source_dt - 1e-12:
        raise ValueError(
            f"cannot rebin from {source_dt:g}s to a finer {target_dt:g}s grid"
        )
    if abs(target_dt - source_dt) < 1e-12:
        return time.copy(), rate.copy()

    start = float(time[0] - source_dt / 2.0)
    stop = float(time[-1] + source_dt / 2.0)
    n_bins = int(np.floor((stop - start) / target_dt))
    if n_bins < 1:
        return time.copy(), rate.copy()

    edges = start + np.arange(n_bins + 1) * target_dt
    counts = rate * source_dt
    binned, _ = np.histogram(time, bins=edges, weights=counts)
    centres = edges[:-1] + target_dt / 2.0
    return centres, binned / target_dt


def pad_or_truncate(values: np.ndarray, length: int) -> np.ndarray:
    """Zero-pad on the right, or truncate, to exactly ``length`` samples."""
    values = np.asarray(values, dtype=float)
    if values.size == length:
        return values.copy()
    if values.size > length:
        return values[:length].copy()
    padded = np.zeros(length, dtype=float)
    padded[: values.size] = values
    return padded


def normalize(values: np.ndarray, method: str, scale: Optional[float] = None) -> np.ndarray:
    """Scale a light curve so that brightness cannot dominate the embedding."""
    values = np.asarray(values, dtype=float)
    if method == "none":
        return values.copy()
    if method == "fluence":
        divisor = scale if scale is not None else np.sum(np.clip(values, 0.0, None))
    elif method == "peak":
        divisor = np.max(np.abs(values)) if values.size else 0.0
    elif method == "l2":
        divisor = float(np.linalg.norm(values))
    else:
        raise ValueError(f"unknown normalisation {method!r}")
    if not np.isfinite(divisor) or divisor == 0:
        # A curve with no net signal stays as it is rather than becoming NaN.
        return values.copy()
    return values / divisor


def fourier_amplitudes(values: np.ndarray, drop_dc: bool = True) -> np.ndarray:
    """Amplitude spectrum ``|FFT|`` of a real series, phases discarded."""
    spectrum = np.abs(np.fft.rfft(np.asarray(values, dtype=float)))
    return spectrum[1:] if drop_dc else spectrum


@dataclass
class FeatureMatrix:
    """Feature vectors plus the per-burst quantities used to interpret them."""

    X: np.ndarray
    triggers: Sequence[str]
    bands: Sequence[str]
    pad_length: int
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    curves: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)

    @property
    def n_bursts(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FeatureMatrix(n_bursts={self.n_bursts}, n_features={self.n_features}, "
            f"bands={list(self.bands)}, pad_length={self.pad_length})"
        )


def _reference_band(bands: Sequence[str]) -> str:
    """Band used to measure the duration: the 50-300 keV band when present."""
    return "50-300" if "50-300" in bands else bands[0]


def prepare_curves(
    dataset: Mapping[str, Mapping[str, LightCurve]],
    config: Optional[FeatureConfig] = None,
    denoise_config: Optional[DenoiseConfig] = None,
    catalog: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], pd.DataFrame]:
    """Denoise and window every light curve, and record per-burst quantities.

    Returns the windowed (not yet padded) curves and a metadata table with the
    duration, fluence and denoising choice for each burst.
    """
    config = config or FeatureConfig()
    denoise_config = denoise_config or DenoiseConfig()
    bands = list(config.bands)
    reference = _reference_band(bands)

    prepared: Dict[str, Dict[str, np.ndarray]] = {}
    records = []
    observed_dt: set = set()

    for trigger in sorted(dataset):
        curves = dataset[trigger]
        if any(band not in curves for band in bands):
            continue
        if config.resample_dt is not None:
            curves = {
                band: _rebinned_curve(curve, config.resample_dt)
                for band, curve in curves.items()
            }
        observed_dt.update(round(curves[band].dt, 6) for band in bands)

        denoised: Dict[str, np.ndarray] = {}
        record: Dict[str, object] = {"trigger": trigger}
        for band in bands:
            curve = curves[band]
            if denoise_config.enabled:
                # Keep the chosen wavelet and level in the metadata so a run
                # can be audited after the fact.
                choice = select_denoising(curve.net_rate, denoise_config)
                denoised[band] = choice.signal
                record[f"wavelet_{band}"] = choice.wavelet
                record[f"wavelet_level_{band}"] = choice.level
                record[f"wavelet_bic_{band}"] = choice.bic
            else:
                denoised[band] = np.asarray(curve.net_rate, dtype=float).copy()
            record[f"dt_{band}"] = curve.dt
            record[f"n_bins_{band}"] = curve.n_bins

        reference_curve = curves[reference]
        duration = _duration_for(
            trigger, reference_curve, denoised[reference], catalog
        )
        record["t90"] = duration.t90
        record["t05"] = duration.t05
        record["t95"] = duration.t95
        record["t90_source"] = duration.source

        windowed: Dict[str, np.ndarray] = {}
        for band in bands:
            curve = curves[band]
            values = denoised[band]
            record["window"] = (
                "t90" if config.window == "t90" and duration.is_valid else "full"
            )
            if config.window == "t90" and duration.is_valid:
                start = duration.t05 + config.window_start
                stop = start + duration.t90
                mask = (curve.time >= start) & (curve.time < stop)
                if not np.any(mask):
                    mask = np.ones(curve.time.size, dtype=bool)
            elif config.window == "full":
                mask = np.ones(curve.time.size, dtype=bool)
            else:
                # No usable duration: fall back to the whole extracted interval
                # rather than dropping the burst silently.
                mask = np.ones(curve.time.size, dtype=bool)
            segment = values[mask]
            windowed[band] = segment
            record[f"fluence_{band}"] = fluence(curve.time[mask], segment)
            record[f"length_{band}"] = int(segment.size)

        prepared[trigger] = windowed
        records.append(record)

    if config.resample_dt is None and len(observed_dt) > 1:
        warnings.warn(
            "light curves in this sample have different bin widths "
            f"({sorted(observed_dt)} s). Their Fourier features sit on "
            "different frequency grids, so the feature matrix mixes "
            "timescales; set FeatureConfig.resample_dt (or --resample-dt) to "
            "put them on one grid.",
            RuntimeWarning,
        )

    metadata = pd.DataFrame.from_records(records)
    if not metadata.empty:
        metadata = metadata.set_index("trigger")
    return prepared, metadata


def _rebinned_curve(curve: LightCurve, target_dt: float) -> LightCurve:
    """A copy of ``curve`` on a ``target_dt`` grid."""
    time, rate = rebin(curve.time, curve.net_rate, target_dt)
    return LightCurve(
        trigger=curve.trigger,
        band=curve.band,
        time=time,
        net_rate=rate,
        path=curve.path,
        meta=dict(curve.meta),
    )


def _measure_duration(curve: LightCurve, denoised: np.ndarray) -> DurationResult:
    """Duration from the denoised shape, with emission located on the raw curve."""
    raw = np.asarray(curve.net_rate, dtype=float)
    return estimate_duration(
        curve.time,
        denoised,
        detection_curve=raw,
        sigma=_background_sigma(curve.time, raw, 0.0),
    )


def _duration_for(
    trigger: str,
    curve: LightCurve,
    denoised: np.ndarray,
    catalog: Optional[pd.DataFrame],
) -> DurationResult:
    """Catalogue T90 when available, otherwise measured from the light curve."""
    if catalog is not None and trigger in catalog.index:
        row = catalog.loc[trigger]
        t90 = float(row.get("t90", np.nan))
        t05 = float(row.get("t90_start", np.nan))
        if np.isfinite(t90) and t90 > 0:
            # Without a catalogue start time, anchor the window at the measured
            # 5% point so the burst stays inside it.
            if not np.isfinite(t05):
                t05 = _measure_duration(curve, denoised).t05
            return DurationResult(
                t90=t90, t05=t05, t95=t05 + t90, total_counts=np.nan,
                source="catalog",
            )
    return _measure_duration(curve, denoised)


def build_feature_matrix(
    dataset: Mapping[str, Mapping[str, LightCurve]],
    config: Optional[FeatureConfig] = None,
    denoise_config: Optional[DenoiseConfig] = None,
    catalog: Optional[pd.DataFrame] = None,
) -> FeatureMatrix:
    """Build the M x N feature matrix from a loaded dataset."""
    config = config or FeatureConfig()
    bands = list(config.bands)
    prepared, metadata = prepare_curves(dataset, config, denoise_config, catalog)
    if not prepared:
        available = sorted(
            {band for curves in dataset.values() for band in curves}
        )
        raise ValueError(
            f"no bursts left after preparation: requested bands {bands}, "
            f"available bands {available}"
        )

    if config.pad_length is not None:
        pad_length = int(config.pad_length)
    else:
        pad_length = max(
            segment.size for curves in prepared.values() for segment in curves.values()
        )
    pad_length = max(pad_length, 2)

    triggers = sorted(prepared)
    rows = []
    for trigger in triggers:
        pieces = []
        for band in bands:
            segment = pad_or_truncate(prepared[trigger][band], pad_length)
            scale = None
            if config.normalize == "fluence":
                column = f"fluence_{band}"
                if column in metadata.columns:
                    scale = float(metadata.loc[trigger, column])
            pieces.append(normalize(segment, config.normalize, scale))
        series = np.concatenate(pieces)
        if config.representation == "fft":
            rows.append(fourier_amplitudes(series, drop_dc=config.drop_dc))
        elif config.representation == "raw":
            rows.append(series)
        else:
            raise ValueError(f"unknown representation {config.representation!r}")

    X = np.vstack(rows)
    # A burst with an empty band can still produce non-finite entries; zeroing
    # them keeps the matrix usable and is equivalent to "no signal here".
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return FeatureMatrix(
        X=X,
        triggers=triggers,
        bands=bands,
        pad_length=pad_length,
        metadata=metadata.loc[list(triggers)] if not metadata.empty else metadata,
        curves=prepared,
    )

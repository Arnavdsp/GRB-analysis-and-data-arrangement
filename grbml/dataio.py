"""Discovery and loading of background-subtracted GBM light curves.

The reduction pipeline writes one CSV per burst and energy band, named like::

    bn230806168__background_subtracted_light_curve_300-900.csv
    bn231109274__background_subtracted_light_curve_231109274_300-900.csv

Both spellings (with and without the repeated trigger number) occur in this
repository and denote the same product, so the loader treats them as one and
keeps the canonical, shorter name.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TRIGGER_RE = re.compile(r"(?:^|[^0-9])(?:bn)?(?P<trigger>\d{9})(?:[^0-9]|$)")
BAND_RE = re.compile(r"(?P<low>\d{1,5})-(?P<high>\d{1,5})")

#: Column name candidates, in order of preference.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "time": ("Centroid", "centroid", "Time", "time", "TIME", "t"),
    "rate": ("Rate", "rate", "counts", "Counts"),
    "rate_err": ("Rate_uncertainty", "rate_uncertainty", "Rate_err", "rate_err"),
    "background": ("Background_Rate", "background_rate", "Background", "background"),
    "net_rate": (
        "Background_Subtracted_Rate",
        "background_subtracted_rate",
        "Net_Rate",
        "net_rate",
    ),
    "net_err": (
        "Background_Subtracted_Rate_uncertainty",
        "background_subtracted_rate_uncertainty",
        "Net_Rate_uncertainty",
        "net_err",
    ),
}


@dataclass
class LightCurve:
    """One background-subtracted light curve in one energy band."""

    trigger: str
    band: str
    time: np.ndarray
    net_rate: np.ndarray
    net_err: Optional[np.ndarray] = None
    rate: Optional[np.ndarray] = None
    background: Optional[np.ndarray] = None
    path: Optional[Path] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.net_rate = np.asarray(self.net_rate, dtype=float)
        if self.time.shape != self.net_rate.shape:
            raise ValueError(
                f"{self.trigger}/{self.band}: time and rate lengths differ "
                f"({self.time.size} vs {self.net_rate.size})"
            )
        if self.time.size < 2:
            raise ValueError(f"{self.trigger}/{self.band}: need at least two bins")

    @property
    def dt(self) -> float:
        """Median bin width in seconds (the light curves are evenly binned)."""
        return float(np.median(np.diff(self.time)))

    @property
    def n_bins(self) -> int:
        return int(self.time.size)

    def slice_time(self, start: float, stop: float) -> "LightCurve":
        """Return a copy restricted to ``start <= t < stop``."""
        mask = (self.time >= start) & (self.time < stop)
        return LightCurve(
            trigger=self.trigger,
            band=self.band,
            time=self.time[mask],
            net_rate=self.net_rate[mask],
            net_err=None if self.net_err is None else self.net_err[mask],
            rate=None if self.rate is None else self.rate[mask],
            background=None if self.background is None else self.background[mask],
            path=self.path,
            meta=dict(self.meta),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"LightCurve(trigger={self.trigger!r}, band={self.band!r}, "
            f"n_bins={self.n_bins}, dt={self.dt:g}s)"
        )


@dataclass(frozen=True)
class FileInfo:
    """A discovered CSV, keyed by the burst and band it belongs to."""

    trigger: str
    band: str
    path: Path


def parse_filename(path: str | Path) -> Optional[FileInfo]:
    """Extract ``(trigger, band)`` from a light-curve file name.

    Returns None when the name does not carry both, so callers can skip
    unrelated CSVs without special-casing them.
    """
    path = Path(path)
    stem = path.stem
    trigger_match = TRIGGER_RE.search(stem)
    band_matches = list(BAND_RE.finditer(stem))
    if trigger_match is None or not band_matches:
        return None
    # The band is the trailing "<low>-<high>" token; anything earlier is part of
    # the repeated trigger number or the fixed description.
    band_match = band_matches[-1]
    band = f"{int(band_match.group('low'))}-{int(band_match.group('high'))}"
    return FileInfo(trigger=trigger_match.group("trigger"), band=band, path=path)


def _resolve_columns(frame: pd.DataFrame) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for key, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in frame.columns:
                resolved[key] = alias
                break
    return resolved


def read_light_curve(path: str | Path, trigger: Optional[str] = None,
                     band: Optional[str] = None) -> LightCurve:
    """Read one CSV into a :class:`LightCurve`.

    ``trigger`` and ``band`` default to whatever the file name encodes.
    """
    path = Path(path)
    info = parse_filename(path)
    if trigger is None:
        trigger = info.trigger if info else path.stem
    if band is None:
        band = info.band if info else "unknown"

    frame = pd.read_csv(path)
    columns = _resolve_columns(frame)
    if "time" not in columns:
        raise ValueError(f"{path}: no recognisable time column in {list(frame.columns)}")
    if "net_rate" not in columns:
        # A file that only carries total and background rates is still usable.
        if "rate" in columns and "background" in columns:
            frame = frame.assign(
                _net=frame[columns["rate"]] - frame[columns["background"]]
            )
            columns["net_rate"] = "_net"
        else:
            raise ValueError(
                f"{path}: no background-subtracted rate column in {list(frame.columns)}"
            )

    def column(key: str) -> Optional[np.ndarray]:
        name = columns.get(key)
        return None if name is None else frame[name].to_numpy(dtype=float)

    time = frame[columns["time"]].to_numpy(dtype=float)
    order = np.argsort(time)

    def ordered(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None if values is None else values[order]

    return LightCurve(
        trigger=trigger,
        band=band,
        time=time[order],
        net_rate=ordered(column("net_rate")),
        net_err=ordered(column("net_err")),
        rate=ordered(column("rate")),
        background=ordered(column("background")),
        path=path,
    )


def _canonical(a: Path, b: Path) -> Path:
    """Pick between two files describing the same burst and band.

    Prefer the shorter name (the variant without the repeated trigger number);
    ties are broken deterministically by path so runs are reproducible.
    """
    a_key = (len(a.name), str(a))
    b_key = (len(b.name), str(b))
    return a if a_key <= b_key else b


def discover_light_curves(
    roots: Iterable[str | Path],
    bands: Optional[Sequence[str]] = None,
    pattern: str = "*.csv",
) -> Dict[str, Dict[str, Path]]:
    """Index every light-curve CSV under ``roots`` as ``{trigger: {band: path}}``.

    Duplicated products for the same burst and band are collapsed to one file.
    """
    index: Dict[str, Dict[str, Path]] = {}
    for root in roots:
        root = Path(root)
        if root.is_file():
            candidates: Iterable[Path] = [root]
        else:
            candidates = sorted(root.rglob(pattern))
        for candidate in candidates:
            info = parse_filename(candidate)
            if info is None:
                continue
            if bands is not None and info.band not in bands:
                continue
            per_band = index.setdefault(info.trigger, {})
            existing = per_band.get(info.band)
            per_band[info.band] = (
                info.path if existing is None else _canonical(existing, info.path)
            )
    return index


def load_dataset(
    roots: Iterable[str | Path],
    bands: Optional[Sequence[str]] = None,
    require_all_bands: bool = True,
    limit: Optional[int] = None,
    warn_on_error: bool = True,
) -> Dict[str, Dict[str, LightCurve]]:
    """Load light curves for every burst that has all requested bands.

    Bursts missing a band are dropped when ``require_all_bands`` is set, which
    is what the multi-band analysis needs: the bands of one burst are
    concatenated into a single feature vector, so a partial burst has no
    well-defined vector.
    """
    index = discover_light_curves(roots, bands=bands)
    wanted = tuple(bands) if bands is not None else None

    dataset: Dict[str, Dict[str, LightCurve]] = {}
    for trigger in sorted(index):
        per_band = index[trigger]
        if wanted is not None and require_all_bands:
            if any(band not in per_band for band in wanted):
                continue
        selected = wanted if wanted is not None else tuple(sorted(per_band))
        curves: Dict[str, LightCurve] = {}
        for band in selected:
            path = per_band.get(band)
            if path is None:
                continue
            try:
                curves[band] = read_light_curve(path, trigger=trigger, band=band)
            except Exception as exc:  # a malformed CSV must not sink the run
                if warn_on_error:
                    warnings.warn(f"skipping {path}: {exc}", RuntimeWarning)
        if not curves:
            continue
        if wanted is not None and require_all_bands and len(curves) != len(wanted):
            continue
        dataset[trigger] = curves
        if limit is not None and len(dataset) >= limit:
            break
    return dataset


def available_bands(roots: Iterable[str | Path]) -> List[str]:
    """List the energy bands present under ``roots``, ordered by lower edge."""
    bands = {
        band
        for per_band in discover_light_curves(roots).values()
        for band in per_band
    }
    return sorted(bands, key=lambda b: int(b.split("-")[0]))

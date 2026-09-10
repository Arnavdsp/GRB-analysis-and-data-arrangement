"""Configuration objects for the pipeline.

The thesis repeats the same analysis under a grid of choices - 16 ms vs 64 ms
binning, one vs three energy bands, with vs without PCA initialisation.  Those
choices live here so that a run is fully described by a single serialisable
object.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

# Energy bands the Fermi-GBM reduction pipeline produces, in keV.
BAND_8_50 = "8-50"
BAND_50_300 = "50-300"
BAND_300_900 = "300-900"
ALL_BANDS: Tuple[str, ...] = (BAND_8_50, BAND_50_300, BAND_300_900)


@dataclass
class DenoiseConfig:
    """Wavelet denoising of the background-subtracted light curve (sec. 3.1.2).

    ``wavelets`` and ``levels`` define the search grid; the combination with the
    lowest ``criterion`` value wins.  Setting ``enabled`` to False keeps the raw
    background-subtracted curve, which the thesis also tries as a control.
    """

    enabled: bool = True
    wavelets: Sequence[str] = ("db1", "db4", "db6", "sym4", "sym8", "coif3", "haar")
    levels: Sequence[int] = (1, 2, 3, 4, 5)
    criterion: str = "bic"  # "bic" or "aic"
    mode: str = "soft"  # thresholding mode passed to pywt.threshold
    signal_extension: str = "symmetric"


@dataclass
class FeatureConfig:
    """Standardisation, normalisation and feature extraction (sec. 3.1.3)."""

    bands: Sequence[str] = (BAND_50_300,)
    # Window applied to every light curve before padding.  "t90" reproduces the
    # thesis (0 s to T90 s); "full" keeps the whole extracted interval.
    window: str = "t90"
    window_start: float = 0.0
    # Rebin every light curve to this bin width (seconds) before anything
    # else.  The thesis fixes the binning at 16 ms or 64 ms for the whole
    # sample; a mixed-binning sample must be brought onto one grid, because
    # |FFT| features from different bin widths live on different frequency
    # grids and are not comparable column by column.  None keeps each curve as
    # delivered and warns when the sample is not uniform.
    resample_dt: Optional[float] = None
    # Common length every light curve is padded/truncated to.  None means "the
    # longest curve in the sample", which is what the thesis does.
    pad_length: Optional[int] = None
    normalize: str = "fluence"  # "fluence", "peak", "l2" or "none"
    # "fft" uses Fourier amplitudes (phase discarded to absorb trigger-time
    # offsets); "raw" feeds the padded light curve straight to the reducer.
    representation: str = "fft"
    drop_dc: bool = True


@dataclass
class ReduceConfig:
    """Dimensionality reduction (sec. 3.2)."""

    use_pca: bool = False
    # Number of principal components, or a float in (0, 1) read as the fraction
    # of variance to retain (the scree-plot criterion used in the thesis).
    pca_components: Any = 0.95
    n_neighbors: int = 5
    min_dist: float = 0.001
    n_components: int = 2
    metric: str = "euclidean"
    random_state: Optional[int] = 42
    standardize: bool = True


@dataclass
class ClusterConfig:
    """Clustering of the two-dimensional embedding (sec. 3.3)."""

    algorithm: str = "hdbscan"  # hdbscan | dbscan | kmeans | gmm
    min_cluster_size: int = 25
    min_samples: Optional[int] = None
    eps: float = 0.5  # dbscan only
    n_clusters: int = 5  # kmeans / gmm only
    random_state: Optional[int] = 42


@dataclass
class PipelineConfig:
    """A complete, reproducible description of one run."""

    data_dirs: Sequence[str] = field(default_factory=lambda: ["."])
    output_dir: str = "results"
    catalog: Optional[str] = None  # CSV with T90 / fluence per trigger
    annotations: Optional[str] = None  # CSV with GRB name -> class annotations
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    reduce: ReduceConfig = field(default_factory=ReduceConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    # Frequency range (Hz) of the power-law fit to the power spectrum, used for
    # the power-index map.  None lets the fitter use the full usable range.
    powerlaw_fmin: Optional[float] = None
    powerlaw_fmax: Optional[float] = None
    make_plots: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=list))
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        nested = {
            "denoise": DenoiseConfig,
            "features": FeatureConfig,
            "reduce": ReduceConfig,
            "cluster": ClusterConfig,
        }
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key in nested and isinstance(value, dict):
                sub = nested[key]
                sub_known = {f.name for f in fields(sub)}
                sub_unknown = set(value) - sub_known
                if sub_unknown:
                    raise ValueError(f"unknown {key} keys: {sorted(sub_unknown)}")
                kwargs[key] = sub(**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

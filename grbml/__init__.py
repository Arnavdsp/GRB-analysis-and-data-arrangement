"""grbml - clustering and classification of GRB prompt-emission light curves.

The package implements the analysis pipeline described in "Clustering and
Classification of Gamma-Ray Bursts using Machine Learning Techniques"
(Harikrishnan R, M.Sc. thesis, IIT Indore, 2025):

    background-subtracted light curves
        -> wavelet denoising (optimal wavelet/level by AIC/BIC)
        -> 0..T90 window, zero padding, fluence normalisation
        -> |FFT| feature vectors
        -> optional PCA
        -> UMAP embedding
        -> HDBSCAN clustering
        -> interpretation maps (duration, power-law index)

Every stage is importable on its own; :mod:`grbml.pipeline` wires them together.
"""

from grbml.config import (
    ClusterConfig,
    DenoiseConfig,
    FeatureConfig,
    PipelineConfig,
    ReduceConfig,
)
from grbml.dataio import LightCurve, discover_light_curves, load_dataset, read_light_curve

__version__ = "0.1.0"

__all__ = [
    "ClusterConfig",
    "DenoiseConfig",
    "FeatureConfig",
    "PipelineConfig",
    "ReduceConfig",
    "LightCurve",
    "discover_light_curves",
    "load_dataset",
    "read_light_curve",
    "__version__",
]

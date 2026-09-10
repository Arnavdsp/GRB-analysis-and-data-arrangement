"""Shared fixtures: synthetic bursts that behave like the real data."""

from __future__ import annotations

import numpy as np
import pytest

from grbml.dataio import LightCurve


def make_curve(
    trigger: str,
    band: str = "50-300",
    dt: float = 0.064,
    n_bins: int = 512,
    peak: float = 200.0,
    width: float = 2.0,
    start: float = 0.0,
    noise: float = 5.0,
    seed: int = 0,
) -> LightCurve:
    """A Gaussian burst on Gaussian background noise."""
    rng = np.random.default_rng(seed)
    time = (np.arange(n_bins) - n_bins // 4) * dt
    signal = peak * np.exp(-((time - start) ** 2) / (2.0 * width ** 2))
    return LightCurve(
        trigger=trigger,
        band=band,
        time=time,
        net_rate=signal + rng.normal(0.0, noise, n_bins),
    )


@pytest.fixture
def burst() -> LightCurve:
    return make_curve("230812790", seed=1)


@pytest.fixture
def dataset():
    """Two families of bursts - narrow and broad - in one energy band."""
    curves = {}
    for index in range(20):
        trigger = f"23081{index:04d}"
        curves[trigger] = {
            "50-300": make_curve(trigger, width=0.5, peak=300.0, seed=index)
        }
    for index in range(20, 40):
        trigger = f"23081{index:04d}"
        curves[trigger] = {
            "50-300": make_curve(trigger, width=12.0, peak=120.0, seed=index)
        }
    return curves


@pytest.fixture
def csv_tree(tmp_path):
    """A directory of light-curve CSVs named the way the pipeline writes them."""
    import pandas as pd

    root = tmp_path / "lightcurves"
    root.mkdir()
    for index in range(6):
        trigger = f"23081{index:04d}"
        curve = make_curve(trigger, seed=index)
        frame = pd.DataFrame(
            {
                "Centroid": curve.time,
                "Rate": curve.net_rate + 50.0,
                "Rate_uncertainty": np.full(curve.time.size, 5.0),
                "Background_Rate": np.full(curve.time.size, 50.0),
                "Background_Subtracted_Rate": curve.net_rate,
                "Background_Subtracted_Rate_uncertainty": np.full(curve.time.size, 5.0),
            }
        )
        name = f"bn{trigger}__background_subtracted_light_curve_50-300.csv"
        frame.to_csv(root / name, index=False)
    return root

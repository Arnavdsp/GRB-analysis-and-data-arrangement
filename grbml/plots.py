"""Figures: embeddings, cluster assignments and interpretation maps.

Reproduces the plot types the thesis leans on - the bare UMAP embedding, the
HDBSCAN cluster assignment, and the same embedding colour-coded by duration or
by power-law index, which is how the structure gets attributed to duration and
noise rather than to astrophysics.

Every function returns the Matplotlib figure and, when given ``path``, also
writes it out.  Nothing here calls ``show()``, so the module is safe in scripts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

import matplotlib

if not os.environ.get("MPLBACKEND") and not os.environ.get("DISPLAY"):
    # Headless machines (CI, a remote box) have no interactive backend.
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

#: Marker style per annotation class, following the thesis figures.
ANNOTATION_STYLES: Dict[str, Dict[str, Any]] = {
    "supernova": {"marker": "s", "color": "red", "label": "SN-associated"},
    "kilonova": {"marker": "D", "color": "blue", "label": "KN-associated"},
}


def _finish(figure, path: Optional[str | Path], dpi: int = 150):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return figure


def plot_embedding(
    embedding: np.ndarray,
    title: str = "UMAP embedding of GRB light curves",
    path: Optional[str | Path] = None,
    point_size: float = 6.0,
    ax=None,
):
    """The bare embedding: one point per burst, no colouring."""
    embedding = np.asarray(embedding, dtype=float)
    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 6))
    else:
        figure = ax.figure
    ax.scatter(embedding[:, 0], embedding[:, 1], s=point_size, c="0.25", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2)
    return _finish(figure, path)


def plot_clusters(
    embedding: np.ndarray,
    labels: Sequence[int],
    title: str = "HDBSCAN cluster assignment",
    path: Optional[str | Path] = None,
    point_size: float = 6.0,
    ax=None,
):
    """Cluster assignment, with the noise label (-1) drawn in grey."""
    embedding = np.asarray(embedding, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 6))
    else:
        figure = ax.figure

    noise = labels < 0
    if np.any(noise):
        ax.scatter(
            embedding[noise, 0],
            embedding[noise, 1],
            s=point_size,
            c="0.75",
            alpha=0.6,
            label="outliers (-1)",
        )
    real = np.unique(labels[labels >= 0])
    colors = plt.cm.viridis(np.linspace(0, 1, max(real.size, 1)))
    for color, cluster_id in zip(colors, real):
        mask = labels == cluster_id
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            color=color,
            alpha=0.85,
            label=f"cluster {cluster_id} (n={int(mask.sum())})",
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2)
    if real.size <= 12:
        ax.legend(loc="best", fontsize="small", framealpha=0.9)
    return _finish(figure, path)


def plot_value_map(
    embedding: np.ndarray,
    values: Sequence[float],
    label: str,
    title: Optional[str] = None,
    path: Optional[str | Path] = None,
    log: bool = False,
    cmap: str = "viridis",
    point_size: float = 6.0,
    ax=None,
):
    """Embedding colour-coded by a per-burst quantity.

    ``log`` takes log10 of the values first, which is how the thesis shows the
    duration map: T90 spans milliseconds to hundreds of seconds, so a linear
    scale would compress every short burst into one colour.  Bursts with a
    missing or non-positive value are drawn in grey rather than dropped, so the
    point count stays honest.
    """
    embedding = np.asarray(embedding, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.size != embedding.shape[0]:
        raise ValueError("values and embedding must have the same length")

    if ax is None:
        figure, ax = plt.subplots(figsize=(7.5, 6))
    else:
        figure = ax.figure

    shown = values
    colorbar_label = label
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            shown = np.where(values > 0, np.log10(values), np.nan)
        colorbar_label = f"log10({label})"

    missing = ~np.isfinite(shown)
    if np.any(missing):
        ax.scatter(
            embedding[missing, 0],
            embedding[missing, 1],
            s=point_size,
            c="0.8",
            alpha=0.6,
            label=f"no {label} ({int(missing.sum())})",
        )
        ax.legend(loc="best", fontsize="small")

    good = ~missing
    scatter = ax.scatter(
        embedding[good, 0],
        embedding[good, 1],
        s=point_size,
        c=shown[good],
        cmap=cmap,
        alpha=0.85,
    )
    figure.colorbar(scatter, ax=ax, label=colorbar_label)
    ax.set_title(title or f"UMAP embedding coloured by {label}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.2)
    return _finish(figure, path)


def annotate_triggers(
    ax,
    embedding: np.ndarray,
    triggers: Sequence[str],
    annotations: Mapping[str, Mapping[str, str]],
    label_points: bool = True,
    fontsize: float = 7.0,
):
    """Mark the annotated bursts on an existing axis."""
    embedding = np.asarray(embedding, dtype=float)
    positions = {trigger: index for index, trigger in enumerate(triggers)}
    used: set = set()
    for trigger, info in annotations.items():
        index = positions.get(trigger)
        if index is None:
            continue
        klass = str(info.get("class", "annotated"))
        style = ANNOTATION_STYLES.get(
            klass, {"marker": "^", "color": "black", "label": klass}
        )
        ax.scatter(
            embedding[index, 0],
            embedding[index, 1],
            marker=style["marker"],
            facecolors="none",
            edgecolors=style["color"],
            s=90,
            linewidths=1.6,
            label=style["label"] if klass not in used else None,
        )
        used.add(klass)
        if label_points:
            ax.annotate(
                str(info.get("grb", trigger)),
                (embedding[index, 0], embedding[index, 1]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=fontsize,
            )
    if used:
        ax.legend(loc="best", fontsize="small", framealpha=0.9)
    return ax


def plot_denoising(
    time: np.ndarray,
    raw: np.ndarray,
    denoised: np.ndarray,
    title: str = "Original vs denoised light curve",
    path: Optional[str | Path] = None,
):
    """Raw and denoised light curve on one axis (thesis Figure 3.2)."""
    figure, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, raw, color="0.6", lw=0.8, label="Original")
    ax.plot(time, denoised, color="tab:orange", lw=1.2, label="Denoised")
    ax.set_xlabel("Time since trigger (s)")
    ax.set_ylabel("Background-subtracted rate (counts/s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    return _finish(figure, path)


def plot_power_spectrum(
    frequency: np.ndarray,
    power: np.ndarray,
    fit: Optional[Any] = None,
    title: str = "Power spectrum",
    path: Optional[str | Path] = None,
):
    """Log-log power spectrum, optionally with the fitted power law."""
    figure, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(frequency, power, color="0.5", lw=0.7)
    if fit is not None and getattr(fit, "is_valid", False):
        ax.loglog(
            frequency,
            fit.evaluate(frequency),
            color="tab:red",
            lw=1.5,
            label=f"index = {fit.index:.2f}",
        )
        ax.legend()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.grid(alpha=0.2, which="both")
    return _finish(figure, path)


def plot_scree(
    explained_variance_ratio: np.ndarray,
    path: Optional[str | Path] = None,
    title: str = "PCA scree plot",
):
    """Explained variance per component and its cumulative sum."""
    ratios = np.asarray(explained_variance_ratio, dtype=float)
    components = np.arange(1, ratios.size + 1)
    figure, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(components, ratios, color="0.7", label="per component")
    ax.plot(components, np.cumsum(ratios), color="tab:red", marker="o", ms=3,
            label="cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    return _finish(figure, path)


def plot_light_curves(
    curves: Mapping[str, tuple],
    path: Optional[str | Path] = None,
    title: str = "Background-subtracted light curves",
    max_curves: int = 5,
):
    """Overlay a handful of light curves, ``{label: (time, rate)}``."""
    figure, ax = plt.subplots(figsize=(10, 6))
    for index, (label, (time, rate)) in enumerate(curves.items()):
        if index >= max_curves:
            break
        ax.plot(time, rate, lw=0.8, alpha=0.8, label=label)
    ax.set_xlabel("Time since trigger (s)")
    ax.set_ylabel("Background-subtracted rate (counts/s)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(alpha=0.2)
    return _finish(figure, path)

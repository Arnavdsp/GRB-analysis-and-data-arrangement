"""Burst metadata: catalogue durations and the annotated GRBs.

Two external inputs help interpret an embedding:

* a burst catalogue with T90 and fluence per trigger (the Fermi-GBM burst
  catalogue), used for the duration map, and
* a short list of GRBs with a confirmed supernova or kilonova counterpart,
  annotated on the embedding to check whether they land together.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

#: Column name candidates for a burst catalogue, in order of preference.
CATALOG_ALIASES: Dict[str, Sequence[str]] = {
    "trigger": ("trigger_name", "trigger", "name", "bcat_trigger_id", "id"),
    "t90": ("t90", "T90", "t90_duration", "duration"),
    "t90_start": ("t90_start", "T90_start", "t90start"),
    "t90_error": ("t90_error", "t90_err", "T90_error"),
    "fluence": ("fluence", "Fluence", "batse_fluence"),
    "flux_1024": ("flux_1024", "pflx_1024", "peak_flux_1024"),
}


def normalize_trigger(value: object) -> str:
    """Reduce any spelling of a trigger id to its nine digits.

    ``bn230812790``, ``GRB230812790`` and ``230812790`` all name the same
    burst; the light-curve files and most catalogues disagree about which
    spelling to use.
    """
    text = str(value).strip()
    digits = "".join(character for character in text if character.isdigit())
    return digits[-9:] if len(digits) >= 9 else digits


def load_catalog(path: str | Path) -> pd.DataFrame:
    """Load a burst catalogue as a table indexed by trigger id.

    Recognises the Fermi-GBM burst catalogue column names and a few common
    variants; unrecognised columns are kept as they are.
    """
    frame = pd.read_csv(path)
    renames: Dict[str, str] = {}
    for canonical, aliases in CATALOG_ALIASES.items():
        for alias in aliases:
            if alias in frame.columns:
                renames[alias] = canonical
                break
    frame = frame.rename(columns=renames)

    if "trigger" not in frame.columns:
        raise ValueError(
            f"{path}: no trigger column found; expected one of "
            f"{list(CATALOG_ALIASES['trigger'])}"
        )
    frame["trigger"] = frame["trigger"].map(normalize_trigger)
    frame = frame[frame["trigger"].str.len() == 9]
    for column in ("t90", "t90_start", "fluence"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.set_index("trigger")


def grb_name_date(name: str) -> str:
    """The ``YYMMDD`` part of a GRB name such as ``GRB 221009A``."""
    digits = "".join(character for character in str(name) if character.isdigit())
    return digits[:6]


def resolve_trigger(name: str, triggers: Iterable[str]) -> Optional[str]:
    """Map a GRB name to a Fermi trigger id present in ``triggers``.

    Fermi triggers are ``bnYYMMDDfff`` where ``fff`` is the fraction of the
    day, and GRB names are ``YYMMDD`` plus a letter giving the order of that
    day's bursts - so the letter picks from the same-day triggers sorted by
    time.

    The letter counts every GRB reported that day by *any* instrument, so if a
    burst of that day was not detected by GBM, or is missing from the sample,
    the ordering shifts and the answer is wrong.  This is a convenience for
    exploratory plots; put an explicit ``trigger`` column in the annotation
    file for anything that matters.
    """
    date = grb_name_date(name)
    if len(date) != 6:
        return None
    # Take the letter that follows the date, not just any letter in the string:
    # the "GRB" prefix is letters too, and "GRB 230812B" must not read as "A".
    text = str(name).upper().replace(" ", "")
    position = text.find(date)
    tail = text[position + len(date):] if position >= 0 else ""
    suffix = "".join(character for character in tail if character.isalpha())
    letter = suffix[0] if suffix else "A"
    index = ord(letter) - ord("A")
    same_day = sorted(
        trigger for trigger in triggers if normalize_trigger(trigger).startswith(date)
    )
    if not same_day or index < 0 or index >= len(same_day):
        return None
    return same_day[index]


def load_annotations(
    path: str | Path,
    triggers: Optional[Iterable[str]] = None,
    resolve: bool = True,
) -> pd.DataFrame:
    """Load the annotated-GRB table and attach trigger ids where possible.

    The file needs a ``grb`` column and a ``class`` column; a ``trigger``
    column, when present, always wins over name-based resolution.
    """
    frame = pd.read_csv(path, comment="#")
    if "grb" not in frame.columns:
        raise ValueError(f"{path}: annotation file needs a 'grb' column")
    if "class" not in frame.columns:
        frame["class"] = "annotated"
    if "trigger" not in frame.columns:
        frame["trigger"] = pd.NA
    frame["trigger"] = frame["trigger"].map(
        lambda value: normalize_trigger(value) if pd.notna(value) else pd.NA
    )

    if resolve and triggers is not None:
        available = list(triggers)
        unresolved = frame["trigger"].isna()
        if unresolved.any():
            frame.loc[unresolved, "trigger"] = frame.loc[unresolved, "grb"].map(
                lambda name: resolve_trigger(name, available)
            )
        missing = frame["trigger"].isna().sum()
        if missing:
            warnings.warn(
                f"{missing} annotated GRB(s) could not be matched to a trigger "
                "in this sample and will not be plotted",
                RuntimeWarning,
            )
    return frame


def annotation_map(annotations: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """``{trigger: {"grb": name, "class": class}}`` for the resolved rows."""
    mapping: Dict[str, Dict[str, str]] = {}
    for _, row in annotations.iterrows():
        trigger = row.get("trigger")
        if not isinstance(trigger, str) or len(trigger) != 9:
            continue
        mapping[trigger] = {
            "grb": str(row.get("grb", "")),
            "class": str(row.get("class", "annotated")),
        }
    return mapping


def merge_metadata(
    metadata: pd.DataFrame,
    catalog: Optional[pd.DataFrame] = None,
    columns: Sequence[str] = ("t90", "fluence"),
) -> pd.DataFrame:
    """Attach catalogue columns to the per-burst metadata table.

    Catalogue values are added under a ``catalog_`` prefix rather than
    overwriting the measured ones, so the two can be compared.
    """
    if catalog is None or metadata.empty:
        return metadata
    available = [column for column in columns if column in catalog.columns]
    if not available:
        return metadata
    joined = metadata.join(
        catalog[available].add_prefix("catalog_"), how="left"
    )
    return joined


def duration_class(t90: float, boundary: float = 2.0) -> str:
    """Traditional duration class - the scheme this analysis is testing."""
    if not np.isfinite(t90):
        return "unknown"
    return "long" if t90 > boundary else "short"


def summarise_clusters(
    metadata: pd.DataFrame,
    labels: Mapping[str, int] | Sequence[int],
    t90_column: str = "t90",
) -> pd.DataFrame:
    """Per-cluster summary: size, median duration and long/short split."""
    frame = metadata.copy()
    if isinstance(labels, Mapping):
        frame["cluster"] = [labels.get(index, -1) for index in frame.index]
    else:
        frame["cluster"] = list(labels)

    if t90_column in frame.columns:
        frame["duration_class"] = frame[t90_column].map(duration_class)
    else:
        frame["duration_class"] = "unknown"

    rows = []
    for cluster_id, group in frame.groupby("cluster"):
        durations = group.get(t90_column)
        # A cluster can be entirely made of bursts with no usable duration, so
        # nanmedian would warn on an all-NaN slice; report NaN directly.
        if durations is None or not bool(np.any(np.isfinite(durations.to_numpy(float)))):
            median_t90 = np.nan
        else:
            median_t90 = float(np.nanmedian(durations.to_numpy(float)))
        rows.append(
            {
                "cluster": int(cluster_id),
                "n_bursts": int(len(group)),
                "median_t90": median_t90,
                "n_long": int((group["duration_class"] == "long").sum()),
                "n_short": int((group["duration_class"] == "short").sum()),
                "n_unknown": int((group["duration_class"] == "unknown").sum()),
            }
        )
        if "t90_source" in group.columns:
            # A cluster made entirely of bursts whose duration could not be
            # measured is a warning sign: those bursts share a fallback
            # analysis window, and may be grouped by that rather than by
            # anything astrophysical.
            sources = group["t90_source"].value_counts()
            for source in ("catalog", "measured", "none"):
                rows[-1][f"n_t90_{source}"] = int(sources.get(source, 0))
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)

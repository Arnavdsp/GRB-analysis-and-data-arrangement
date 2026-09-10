# GRB Analysis and Data Arrangement

> Machine learning pipeline for **Gamma-Ray Burst (GRB)** prompt-emission light
> curves: wavelet denoising, Fourier feature extraction, UMAP dimensionality
> reduction and HDBSCAN clustering — plus the simulation controls needed to tell
> a real cluster from a preprocessing artefact.

This repository implements the analysis of *Clustering and Classification of
Gamma-Ray Bursts using Machine Learning Techniques* (Harikrishnan R, M.Sc.
thesis, DAASE, IIT Indore, May 2025) as an installable package, `grbml`, with a
command line interface and a test suite.

---

## Why

GRBs are traditionally split into "long" and "short" at T90 = 2 s, but the
boundary leaks: long bursts with kilonovae (GRB 211211A, GRB 230307A) and a
short burst with a supernova (GRB 200826A) all contradict it. Clustering bursts
by the *shape* of their light curves rather than by one number is a way out —
if the clusters mean anything. The central result of the thesis is a caution:
the structure of the embedding is driven substantially by **duration** and by
the **colour of the background noise**, and noise-driven structure has no
astrophysical significance. This code makes both effects measurable rather than
implicit.

---

## Install

```bash
git clone https://github.com/Arnavdsp/GRB-analysis-and-data-arrangement.git
cd GRB-analysis-and-data-arrangement
pip install -e .          # add [umap] if you want the UMAP extra pinned
pip install umap-learn    # required for the embedding step
```

Python ≥ 3.9. Core dependencies: numpy, scipy, pandas, matplotlib,
scikit-learn, PyWavelets. HDBSCAN comes from scikit-learn ≥ 1.3; the standalone
`hdbscan` package is used automatically if that is unavailable.

---

## Quick start

```bash
# What light curves are here?
grbml inspect --data-dir bg_sub_lightcurve1 --data-dir bg_sub_lightcurve-2

# Denoise one burst and plot before/after
grbml denoise bg_sub_lightcurve1/bn230812790__background_subtracted_light_curve_300-900.csv

# Full analysis over this repository's data
grbml run --config configs/repo_data.json

# Same, spelled out
grbml run --data-dir bg_sub_lightcurve1 --bands 300-900 --resample-dt 1.0 \
          --annotations data/annotated_grbs.csv -o results/run1

# With PCA initialisation, for the with/without comparison
grbml run --config configs/repo_data.json --pca -o results/run1_pca

# The noise control: can noise colour alone produce clusters?
grbml simulate --n-per-class 200
```

As a library:

```python
from grbml.config import FeatureConfig, PipelineConfig
from grbml.pipeline import run

config = PipelineConfig(
    data_dirs=["bg_sub_lightcurve1"],
    features=FeatureConfig(bands=("300-900",), resample_dt=1.0),
)
result = run(config)
print(result.cluster_summary)
```

---

## What a run produces

In the output directory:

| file | contents |
|---|---|
| `bursts.csv` | per burst: T90 + provenance, fluence, chosen wavelet, power-law index, cluster, UMAP coordinates |
| `cluster_summary.csv` | per cluster: size, median T90, long/short split, T90-provenance counts |
| `run_summary.json` | sample size, cluster sizes, silhouette, PCA components, missing-T90 count |
| `config.json` | the exact configuration used, re-runnable with `--config` |
| `figures/` | embedding, HDBSCAN clusters, duration map, power-index map, scree plot |

---

## Read the output critically

Two things to check before believing any cluster:

**1. The duration-provenance columns.** A cluster whose `n_t90_none` equals its
size is grouped by the analysis window its members fell back to, not by
anything about the bursts. Running the pipeline over this repository's own data
produces exactly that: of 226 bursts, 200 have no measurable T90 in the
300–900 keV band, and they land in one cluster of 185 while the 26 bursts with a
measured duration land in the other. That is a preprocessing artefact, and it is
the concrete form of the thesis's warning — *"we could misattribute physical
significance to something that may be merely a statistical artifact."* Supply a
burst catalogue (`--catalog`) so every burst gets a real T90, and the artefact
goes away.

**2. The power-index map.** If clusters line up with the power-law index rather
than with anything astrophysical, they are noise colour. `grbml simulate` shows
what that looks like in a case where the ground truth is known: simulated white,
pink and red noise — no astrophysics in them at all — separate into three clean
clusters (adjusted Rand score 1.000, reproducing the thesis's Figure 4.39).

---

## Data in this repository

`bg_sub_lightcurve1/`, `bg_sub_lightcurve-2/`, `bg_sub_lightcurve3/`,
`bg_sub_lightcurve4/`, `bg_sub_lightcurve5/` hold 226 background-subtracted
light curves (2023-08 → 2024-06), all in the **300–900 keV** band, binned at
0.5 s or 1.0 s depending on the burst.

Two consequences worth knowing:

- 300–900 keV is the hardest and faintest GBM band; most bursts have only a
  3–5σ peak there, which is why most have no measurable T90. The thesis's own
  analysis is built on the 8–50 and 50–300 keV bands, which are not in this
  repository. Add them under any directory name — the loader finds bursts by
  file name, and `--bands 8-50 50-300 300-900` concatenates them.
- The sample mixes 0.5 s and 1.0 s binning. `--resample-dt` puts everything on
  one grid; without it the pipeline warns, because Fourier features from
  different bin widths are not comparable column by column.

`data/annotated_grbs.csv` lists the SN- and KN-associated GRBs the thesis marks
on its embeddings. The trigger column is empty and is inferred from the GRB name
where possible — **fill it in from the Fermi-GBM burst catalogue before relying
on those annotations**; name-based resolution only works when every GRB of that
date is in the sample.

---

## Repository layout

```
grbml/                     the package
  config.py                run configuration (serialisable to JSON)
  dataio.py                find and load light-curve CSVs
  denoise.py               wavelet denoising, wavelet/level chosen by BIC
  features.py              T90, rebinning, padding, normalisation, |FFT|
  powerspec.py             power spectra and power-law indices
  embed.py                 PCA + UMAP
  cluster.py               HDBSCAN / DBSCAN / K-means / GMM
  simulate.py              Timmer-König and Emmanoulopoulos light curves
  plots.py                 embeddings, cluster maps, duration/index maps
  pipeline.py              the whole chain, end to end
  cli.py                   the `grbml` command
configs/                   example configurations for the thesis's comparison grid
data/annotated_grbs.csv    SN/KN-associated GRBs to annotate on embeddings
docs/methodology.md        thesis section → code, and the decisions made
tests/                     test suite (pytest)
bg_sub_lightcurve*/        background-subtracted light curves (300-900 keV)
image.png                  example of plotted and denoised light curves
```

The original standalone scripts — `Batch_csv_collector`,
`Batch_csv_collector_shell_script`, `Light Curve_denoising-code` and
`Time series plotting-Light curve` — are kept as they were. Their jobs are now
`grbml collect`, `grbml denoise` and `grbml run`.

---

## Tests

```bash
pip install pytest
pytest                 # 120 tests
pytest -m "not slow"   # skip the ones that run a full embedding
```

The tests check the parts where being wrong is quiet rather than loud: that
wavelet denoising recovers a known pulse, that T90 recovers a known Gaussian
burst and returns NaN for pure noise instead of inventing a long one, that
rebinning preserves total counts, and that simulated white/pink/red noise comes
back with power-law indices of 0, −1 and −2.

---

## Method

See [`docs/methodology.md`](docs/methodology.md) for the step-by-step mapping
from thesis sections to code, including the two places where the thesis leaves
a choice open and this implementation had to make one (T90 without a catalogue,
and mixed binning).

---

## License

MIT — see [LICENSE](LICENSE).

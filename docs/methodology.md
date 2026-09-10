# Methodology: thesis section → code

This repository implements the analysis of *Clustering and Classification of
Gamma-Ray Bursts using Machine Learning Techniques* (Harikrishnan R, M.Sc.
thesis, DAASE, IIT Indore, May 2025). This page maps each step of the thesis
onto the code that performs it, and records where the implementation had to
make a decision the thesis left open.

## Pipeline overview

```
background-subtracted light curves (one CSV per burst per energy band)
  │
  ├─ wavelet denoising, wavelet + level chosen by BIC      grbml/denoise.py
  ├─ window 0 → T90                                        grbml/features.py
  ├─ zero-pad to a common length                           grbml/features.py
  ├─ normalise by the fluence of that band                 grbml/features.py
  ├─ concatenate the bands of one burst                    grbml/features.py
  ├─ keep |FFT|, discard phases                            grbml/features.py
  │      ⇒ M × N feature matrix (M bursts, N features)
  ├─ optional PCA                                          grbml/embed.py
  ├─ UMAP → 2-D embedding                                  grbml/embed.py
  ├─ HDBSCAN → cluster labels                              grbml/cluster.py
  └─ duration map, power-index map, annotations            grbml/plots.py
```

`grbml/pipeline.py` runs the whole chain; `grbml/cli.py` exposes it as
`grbml run`.

## Section by section

### 3.1 Data reduction — *not* in this repository

Background fitting (3.1.1) is upstream of this code. It is done per burst with
the Fermi GBM Data Tools: pick the brightest detector, bin the TTE file, select
background regions before and after the trigger, fit a polynomial of order 0–4,
interpolate across the source region and subtract. The CSVs in
`bg_sub_lightcurve*/` are that step's output, and are where this code starts.

### 3.1.2 Wavelet denoising → `grbml/denoise.py`

Soft thresholding of the detail coefficients at the universal threshold
`σ·sqrt(2 ln N)`, with `σ = MAD/0.6745` estimated from the finest detail band.
The wavelet and decomposition level are chosen by searching a grid and taking
the lowest BIC, exactly as in the thesis:

    AIC = 2k + (1/σ²)·Σ(xᵢ − x̂ᵢ)²
    BIC = k·ln(n) + (1/σ²)·Σ(xᵢ − x̂ᵢ)²

with `k` the number of retained non-zero coefficients. The thesis's worked
example (GRB 150630958, db6 at level 5) is the kind of answer the search
returns.

Signals too short to decompose are returned unchanged rather than raising, so a
handful of very short bursts cannot abort a run over thousands.

### 3.1.3 Feature extraction → `grbml/features.py`

*Standardisation.* Each curve is cut to 0 → T90 and zero-padded to the length of
the longest curve in the sample.

*Normalisation.* Division by the band's fluence, so that the known
"long bursts are brighter" result cannot be what the clustering rediscovers.

*Fourier transform.* `|rfft|`, phases discarded, which absorbs the unknown
offset between when a burst went off and when the detector triggered.

Two implementation decisions the thesis does not specify:

- **T90 when there is no catalogue.** `estimate_duration` measures the 5%→95%
  points of the cumulative net counts, but only inside an emission region found
  first — the most significant contiguous block of bins above 3σ, merged across
  quiet gaps of up to 10 s, requiring a ≥5σ peak. Integrating the whole
  extracted interval instead would be wrong: clipping negative background
  residuals to zero makes pure noise accumulate counts steadily, so the 5% and
  95% points drift to the ends of the window and every faint burst gets a
  spuriously long T90. Detection runs on the raw curve (real counting noise);
  integration runs on the denoised curve (clean shape). Bursts that never reach
  5σ get `t90 = NaN` and `t90_source = "none"` rather than a guess. **Supply a
  catalogue with `--catalog` when you can** — a published T90 comes from a fit
  using the full detector response and a hand-checked source interval.

- **Mixed binning.** |FFT| features from curves of different bin widths live on
  different frequency grids and are not comparable column by column. The thesis
  sidesteps this by fixing the binning at 16 ms or 64 ms for the whole sample.
  Here a mixed-binning sample raises a warning, and `resample_dt`
  (`--resample-dt`) rebins everything onto one grid, preserving total counts.
  Rebinning to a *finer* grid is refused.

### 3.2 Dimensionality reduction → `grbml/embed.py`

UMAP with `n_neighbors=5`, `min_dist=0.001` (the thesis's appendix values), with
PCA initialisation as a switch. The number of components can be given as a
fraction of variance to retain, which is the scree-plot criterion the thesis
uses. The thesis's finding that PCA-vs-no-PCA changes the number of clusters is
the reason this is a first-class configuration option rather than a detail.

### 3.3 Clustering → `grbml/cluster.py`

HDBSCAN by default: hierarchical, density-based, no need to know the cluster
count in advance, and it labels population outliers `-1` instead of forcing them
into a group. DBSCAN, K-means and GMM are available for the
"does the algorithm change the answer?" comparison.

### 4.5 Simulations → `grbml/simulate.py`

The thesis's control experiment: simulate light curves that are *pure noise* of
a known colour, run them through the same pipeline, and see whether the colours
separate. They do — which is why clusters found in real data cannot be read as
astrophysics until the background's influence is removed.

- `timmer_koenig` — Timmer & König (1995), Gaussian, `P(f) ∝ f^-β`.
- `emmanoulopoulos` — Emmanoulopoulos et al. (2013), which also reproduces a
  chosen flux distribution, since real light curves are not Gaussian.
- `NOISE_COMBINATIONS` — the seven classes: WN, PN, RN and their mixtures.

`grbml simulate` runs it and reports the confusion matrix against the true
labels plus an adjusted Rand score.

**Reproduction status.** The three pure colours separate perfectly:

```bash
python -c "
from grbml.config import PipelineConfig, ReduceConfig, ClusterConfig
from grbml.pipeline import run_simulation
c = PipelineConfig()
c.reduce = ReduceConfig(n_neighbors=15, min_dist=0.001, random_state=0)
c.cluster = ClusterConfig(min_cluster_size=20)
out = run_simulation(n_per_class=100, n_bins=512,
                     combinations={'WN': (0.0,), 'PN': (1.0,), 'RN': (2.0,)},
                     config=c, save=False)
print(out['confusion'])"
```

gives three clusters with an adjusted Rand score of 1.000 — the thesis's
Figure 4.39. Running all seven classes instead reproduces the thesis's *other*
finding: HDBSCAN recovers fewer clusters than there are classes, because the
mixtures (PN+WN, PN+RN+WN, and to a degree RN+WN) are nearly indistinguishable.
That is the expected outcome, not a failure of the implementation.

Note that a single periodogram bin is χ²-distributed with 2 degrees of freedom,
so |FFT| features carry ~100% scatter per frequency. Separating noise colours
therefore needs enough neighbours for UMAP to average over that scatter:
`n_neighbors=5` (the thesis's value for real data, where the burst signal
dominates) under-performs on pure-noise simulations, where `n_neighbors=15` or
more is appropriate.

### 5 Interpretation → `grbml/plots.py`, `grbml/catalog.py`

The duration map (`log10 T90`) and the power-index map are what turn an
embedding into a claim about *why* points sit where they do. `plot_value_map`
draws bursts with a missing value in grey rather than dropping them, so the
point count on the figure always matches the sample.

`data/annotated_grbs.csv` lists the SN- and KN-associated GRBs the thesis marks
on its embeddings; `annotate_triggers` draws them.

## Reading the output

`grbml run` writes, per run:

| file | contents |
|---|---|
| `bursts.csv` | one row per burst: T90 and its provenance, fluence, chosen wavelet, power-law index, cluster, embedding coordinates |
| `cluster_summary.csv` | per cluster: size, median T90, long/short split, T90-provenance counts |
| `run_summary.json` | sample size, cluster sizes, silhouette score, PCA components, missing-T90 count |
| `config.json` | the exact configuration used |
| `figures/` | embedding, clusters, duration map, power-index map, scree |

**Check `cluster_summary.csv` before interpreting anything.** The
`n_t90_catalog` / `n_t90_measured` / `n_t90_none` columns exist because a
cluster made entirely of bursts with no usable duration is grouped by the
analysis window they fell back to, not by anything about the bursts. That is a
preprocessing artefact of exactly the kind the thesis's conclusion warns about,
and it is visible in a run over this repository's own data (see the README).

## References

- Timmer & König (1995), *On generating power law noise*, A&A 300, 707.
- Emmanoulopoulos, McHardy & Papadakis (2013), MNRAS 433, 907.
- McInnes, Healy & Melville (2018), *UMAP*, arXiv:1802.03426.
- Campello, Moulavi & Sander (2013), *Density-based clustering based on
  hierarchical density estimates* (HDBSCAN).
- Jespersen et al. (2020); Steinhardt et al. (2023); Dimple et al. (2023, 2024) —
  the t-SNE/UMAP GRB clustering results this work extends.

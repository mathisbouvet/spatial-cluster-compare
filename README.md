# spatial-cluster-compare

[![Tests](https://github.com/mathisbouvet/spatial-cluster-compare/actions/workflows/tests.yml/badge.svg)](https://github.com/mathisbouvet/spatial-cluster-compare/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/spatial-cluster-compare.svg)](https://pypi.org/project/spatial-cluster-compare/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Automated clustering algorithm comparison pipeline for spatial proteomics,
cytometry data, or any numerical feature table.

Originally an analysis notebook (`Comparaison_des_clusters.ipynb`), restructured

<img src="assets/figure_1.png" width="500">

## Table of contents

- [Why this package](#why-this-package)
- [Two ways to use this repo](#two-ways-to-use-this-repo)
- [What the pipeline does](#what-the-pipeline-does)
- [Expected data format](#expected-data-format)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick usage](#quick-usage)
- [Advanced usage (step by step)](#advanced-usage-step-by-step)
- [Fix applied compared to the original notebook](#fix-applied-compared-to-the-original-notebook)
- [Roadmap](#roadmap)

## Why this package

Choosing a clustering algorithm is often an arbitrary decision: KMeans gets
picked by default, parameters get tuned until the result "looks right," and
the choice is rarely backed by an objective comparison. For biological data in
particular — where clusters are expected to reflect real cell populations —
this matters: the wrong algorithm or the wrong number of clusters can produce
plausible-looking but biologically meaningless groups.

`spatial-cluster-compare` runs several clustering algorithms on the same data,
scores each one on multiple complementary metrics (including bootstrap
stability, not just a single internal index), and picks the best method
through a transparent, reproducible composite score — instead of relying on a
single, unverified default choice.

## Two ways to use this repo

- **`notebooks/Comparaison_des_clusters.ipynb`**: the full pipeline as a
  standalone notebook, no dependency on the package — handy for copy-pasting
  the code directly into your own environment without installing the library.
- **`spatial_cluster_compare` (the package)**: the same logic packaged into
  reusable, tested functions, installable via `pip`.

Detailed step-by-step documentation (formulas, biological rationale for each
step) is available in [`docs/02_comparaison_cluster.md`](docs/02_comparaison_cluster.md).

## What the pipeline does

1. **Clusterability test** (Hopkins statistic) before/after normalization
2. **Conditional normalization**: StandardScaler applied only if it improves cluster structure
3. **Dimensionality reduction** (PCA, configurable explained variance)
4. **Automatic selection of k** via a composite score (Silhouette, Davies-Bouldin, Calinski-Harabasz)
5. **Multi-algorithm benchmark**: KMeans, Agglomerative, Spectral, GMM, DBSCAN (eps auto-estimated via the elbow method)
6. **Bootstrap stability** (mean ARI) for each method
7. **Automatic selection of the best method** based on a normalized composite score
8. **Visualizations**: comparative bar chart of metrics, mean-expression heatmap per cluster

<img src="assets/figure_2.png" width="500">

## Expected data format

### Export from MACSiQView

In MACSiQView, the **"Feature Table"** tab lets you select which descriptors
to include before exporting to `.csv`. For this clustering pipeline (unlike
the morphological segmentation QC described in `macsima-qc`, which excludes
fluorescence intensities), you should instead select the **biomarker
intensity columns** (e.g. `CD8 Biomarker Exp`, `CD74 Biomarker Exp`), along
with morphological descriptors (`Cell Size`...) if you want them included in
the clustering.

### CSV structure

- **One row per cell**, one column per marker/descriptor.
- Numeric columns (`float64`/`int64`) are automatically used as clustering
  features via `select_dtypes` — no need to specify them manually.
- Non-numeric columns (ID, ROI name, text metadata) are automatically ignored
  and don't need to be dropped beforehand.
- **Caution**: if a numeric column isn't a biological marker (e.g. an integer
  `Cell_ID`, X/Y centroid coordinates), it will still be wrongly included in
  the clustering. In that case, drop it explicitly before calling
  `compare_clusters`, e.g.:
  ```python
  data = data.drop(columns=["Cell_ID", "Centroid X", "Centroid Y"])
  ```
- Stray whitespace in column names: MACSiQView can generate leading/trailing
  spaces in column names. Clean them up if needed
  (`data.columns = data.columns.str.strip()`), as in the segmentation QC
  protocol.
- **Missing values (NaN)**: not handled automatically by the pipeline. Clean
  them up beforehand (`data.dropna()`) or impute if some cells have missing
  values on one or more markers.
- **Dataset size**: plan for at least a few hundred cells so that the Hopkins
  statistic and the stability bootstrap (ARI) are reliable. Spectral
  Clustering can become slow beyond ~10-20k rows.

## Requirements

- Python 3.9+
- `numpy`, `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib` / `seaborn` (for the comparison bar chart and heatmap)

All dependencies are installed automatically with `pip install`.

## Installation

```bash
pip install spatial-cluster-compare
```

For a local/development install:

```bash
pip install -e .
```

## Quick usage

```python
import pandas as pd
from spatial_cluster_compare import compare_clusters, plot_comparison_bars, plot_cluster_heatmap

data = pd.read_csv("Cluster_ImmuneCell.csv")
result = compare_clusters(data)

print(result.results_df)
print(f"Best method: {result.best_method}")

plot_comparison_bars(result.results_df)
plot_cluster_heatmap(result.X_original, result.best_labels, method_name=result.best_method)
```

## Advanced usage (step by step)

```python
from spatial_cluster_compare import (
    auto_scale, find_best_k, run_benchmark, select_best_method
)
from sklearn.decomposition import PCA

X = data.select_dtypes(include=["float64", "int64"])
X_best, report = auto_scale(X)

pca = PCA(n_components=0.9)
X_reduced = pca.fit_transform(X_best)

optimal_k, k_results = find_best_k(X_reduced, range(2, 10))

results_df, labels_per_method = run_benchmark(X_reduced, optimal_k=optimal_k)
best_method, best_labels = select_best_method(results_df, labels_per_method)
```

## Fix applied compared to the original notebook

The original notebook referenced `best_labels` and `best_method` in the
heatmap cell without ever defining them explicitly. The `select_best_method`
function fixes this by computing a normalized composite score (Silhouette ↑,
Davies-Bouldin ↓ inverted, Calinski-Harabasz ↑, ARI stability ↑) to
objectively designate the best method.

## Roadmap

- Automatic PDF/figure export of the comparison report
- Support for multi-batch datasets (clustering comparison across runs)
- CLI (`spatial-cluster-compare run data.csv`)
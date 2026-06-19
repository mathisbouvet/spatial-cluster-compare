<h1 style="color: #333; text-align: center; padding: 15px 0; border-top: 3px double #333; border-bottom: 3px double #333; font-family: Georgia, serif; font-style: italic; font-weight: normal; margin-top: 20px; margin-bottom: 30px;">
  Comparative Clustering Analysis on Spatial Proteomics Data (MACSima)
</h1>

<div style="text-align: center; font-family: Georgia, serif; font-size: 1.2em; color: #555; margin-top: -10px; margin-bottom: 30px;">
  By <strong>Mathis BOUVET</strong> — Biologist specializing in Reproduction and Development
  <br>
  <span style="font-size: 0.8em; font-style: italic;">March 2026</span>
</div>
<br>

> **Important note**
> : This document contains no real data

<br>

<div style="border: 1px solid #569cd6; border-radius: 10px; padding: 20px; background-color: rgba(86, 156, 214, 0.1); color: #000000;">
  <strong>Clustering</strong><br>
Clustering is an unsupervised learning method aimed at grouping entities (in this case, cells) exhibiting similar protein expression profiles. In the context of cyclic imaging, this technique allows for the agnostic identification of cell populations without biological a priori to reveal phenotypic heterogeneity within a tissue. The challenge lies in partitioning the multidimensional marker space to define distinct cellular signatures.
</div>
<br>
<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
  Objective
</h2>


The goal of this pipeline is to automate the selection of the most efficient partitioning algorithm for a given dataset. The process is broken down into three steps: structure validation, k optimization, and domedele benchmarking.

<div style="border: 1px solid #d65323; border-radius: 10px; padding: 20px; background-color: rgba(213, 101, 45, 0.1); color: #000000;">
  <strong>Importing libraries</strong><br>

<details>
  <summary><b>Show/hide configuration code</b></summary>

```python
import importlib
import subprocess
import sys

# Library Dictionary
required_packages = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "minisom": "minisom",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "scipy": "scipy"
}

def install_and_import():
    print("Analysis of the work environment...")
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} is already ready.")
        except ImportError:
            print(f"⚠️ {module_name} lack. Installation of {package_name} in progress...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"{package_name} installed successfully !")
            except Exception as e:
                print(f"❌ Error during installation of {package_name} : {e}")

install_and_import()


# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from minisom import MiniSom  

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

warnings.filterwarnings('ignore')

print("\n All clustering and analysis libraries are ready!")
```
</details>
</div>

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
1. Preprocessing and Data Quality (Hopkins Score)
</h2>

### 1.a Hopkins Score

The Hopkins statistic (or score) generates $m$ uniformly random points $W$ and selects $m$ real data points $U$.

```math
H = \frac{\sum_{i=1}^{m} w_i^d}{\sum_{i=1}^{m} u_i^d + \sum_{i=1}^{m} w_i^d}
```
If the $H$ score is greater than 0.75, the data is considered to have a clustering tendency. Around 0.5, the data is distributed randomly.

```python
def hopkins_statistic(X, m_ratio=0.1):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    d = X.shape[1]
    n = len(X)
    m = int(m_ratio * n)

    rand_X = X.sample(m)
    neigh = NearestNeighbors(n_neighbors=1).fit(X)

    u_dist, _ = neigh.kneighbors(rand_X)

    min_vals, max_vals = X.min(), X.max()
    rand_uniform = np.random.uniform(low=min_vals, high=max_vals, size=(m, d))
    w_dist, _ = neigh.kneighbors(rand_uniform)

    return np.sum(w_dist) / (np.sum(u_dist) + np.sum(w_dist))
```
### 1.b Evaluation metric


| Metric | Description | Formula | Interpretation |
| :--- | :--- | :--- | :--- |
| **Silhouette (S)** | Measures internal cohesion and separation from neighbors. | $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$ | **Close to 1**: Excellent<br>**Close to 0**: Overlapping clusters<br>**Negative**: Assignment error |
| **Davies-Bouldin (DB)** | Ratio of intra-cluster dispersion to inter-cluster distance. | $$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{s_i + s_j}{d_{ij}} \right)$$ | **Lower is better**<br>Indicates dense and well-spaced clusters. |
| **Calinski-Harabasz (CH)** | Ratio of between-cluster variance to within-cluster variance. | $$CH = \frac{SS_B}{SS_W} \times \frac{N - k}{k - 1}$$ | **Higher is better**<br>Favors clear separation and compactness. |
<br>

> **Technical note** : To calculate the final Global Score in the benchmark, the Davies-Bouldin index is inverted. This harmonizes the criteria so that, across all three metrics, a higher value consistently signifies better partitioning. Additionally, DBSCAN may label points as noise (index -1) — these are excluded before computing any metric, as the indices require at least two valid clusters.

```python
def evaluate_clustering(X, labels):
    unique = set(labels) - {-1}
    if len(unique) <= 1:
        return np.nan, np.nan, np.nan

    return (
        silhouette_score(X, labels),
        davies_bouldin_score(X, labels),
        calinski_harabasz_score(X, labels)
    )
```

#### Stability (ARI)

The data is resampled using 80% bootstrapping across multiple runs. Rather than naively truncating label arrays, which would compare labels from different cells, each run tracks the indices of the sampled cells. The ARI is then computed only on cells that appear in both runs, ensuring a meaningful comparison between partitionings while adjusting for chance.

```python
def compute_stability(X, model, n_runs=5):
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    n = len(X)
    labels_list = []
    indices_list = []

    for i in range(n_runs):
        idx = resample(np.arange(n), n_samples=int(0.8 * n), random_state=42 + i, replace=False)
        X_sample = X[idx]

        if isinstance(model, GaussianMixture):
            labels = model.fit(X_sample).predict(X_sample)
        else:
            labels = model.fit_predict(X_sample)

        labels_list.append(labels)
        indices_list.append(idx)

    ari_scores = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            common = np.intersect1d(indices_list[i], indices_list[j])
            if len(common) < 2:
                continue
            pos_i = np.where(np.isin(indices_list[i], common))[0]
            pos_j = np.where(np.isin(indices_list[j], common))[0]
            ari = adjusted_rand_score(
                labels_list[i][pos_i],
                labels_list[j][pos_j]
            )
            ari_scores.append(ari)

    return np.mean(ari_scores) if ari_scores else np.nan
```

### 1.c Automatic K calculation

We use a `KMeans` algorithm for each possible value of k.

For each $k$, the three metrics are calculated. Since the score results are on different scales, the Davies-Bouldin index is inverted. We multiply it by $-1$ because for Silhouette and CH, 'higher is better,' whereas for DB, 'lower is better.' By inverting it, the metrics are harmonized: a higher value becomes better for all three. Finally, all values are normalized to a range between 0 and 1. This ensures that each index carries the same weight in the final decision.

We calculate the average of the 3 normalized indices and retrieve the corresponding value of k.

```python
def find_best_k(X, k_range):
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        sil, db, ch = evaluate_clustering(X, labels)

        results.append([k, sil, db, ch])

    df = pd.DataFrame(results, columns=["k", "sil", "db", "ch"])

    # Normalisation
    scaler = MinMaxScaler()
    scores = df[["sil", "db", "ch"]].copy()
    scores["db"] = -scores["db"]
    scores_scaled = scaler.fit_transform(scores)

    df["score"] = scores_scaled.mean(axis=1)

    best_k = df.loc[df["score"].idxmax(), "k"]

    return int(best_k), df
```

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
2. Data Application
</h2>


### 2.a Data loading and normalization

```python
data = pd.read_csv("[Data].csv")
X = data.select_dtypes(include=['float64', 'int64'])

hopkins_before = hopkins_statistic(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

hopkins_after = hopkins_statistic(pd.DataFrame(X_scaled, columns=X.columns))

print(f'Hopkins avant : {hopkins_before:.3f}')
print(f'Hopkins après : {hopkins_after:.3f}')

X_best = X_scaled if hopkins_after > hopkins_before else X.values
```
### 2.b Data reduction (PCA) and k selection

Once the Hopkins test has validated the presence of clusters, the data is simplified. We compress the markers to retain only those that explain 90% of the variance. This allows us to eliminate background sensor noise before proceeding to computationally intensive models.

```python
# DIMENSION REDUCTION
pca = PCA(n_components=0.9)
X_reduced = pca.fit_transform(X_best)

print(f"📐 PCA components retained : {pca.n_components_} (90% of variance explained)")

# K SELECTION
k_range = range(2, 10)
optimal_k, k_results = find_best_k(X_reduced, k_range)

print(f"\n👉 Optimal k detected : {optimal_k}")
```

### 2.c Benchmarking of models

We benchmark five clustering algorithms, each with a fundamentally different approach to partitioning the data:

- **KMeans** assigns each cell to the nearest centroid by minimizing intra-cluster variance. Fast and effective on globular, well-separated clusters.
- **Agglomerative** builds a hierarchy of clusters bottom-up by successively merging the closest pairs. Well-suited for non-spherical structures and does not require initialization.
- **Spectral** projects the data into a lower-dimensional graph-based representation before clustering. Particularly effective when clusters are non-convex or entangled.
- **GMM** (Gaussian Mixture Model) models the data as a mixture of Gaussian distributions, assigning probabilistic memberships to each cell. More flexible than KMeans as it handles ellipsoidal cluster shapes.
- **DBSCAN** is a density-based algorithm that identifies clusters as high-density regions separated by low-density areas. Unlike the others, it does not require specifying k and naturally handles noise by labeling low-density points as outliers. Its search radius (`eps`) is estimated automatically from the data after PCA reduction to avoid scale artifacts.

```python
# Automatic eps estimation for DBSCAN using k-distance elbow method
neighbors = NearestNeighbors(n_neighbors=5).fit(X_reduced)
distances, _ = neighbors.kneighbors(X_reduced)
k_distances = np.sort(distances[:, -1])
eps_auto = float(np.percentile(k_distances, 90))

print(f"📐 DBSCAN eps estimated : {eps_auto:.3f}")

clustering_methods = {
    'KMeans': KMeans(n_clusters=optimal_k, random_state=42),
    'Agglomerative': AgglomerativeClustering(n_clusters=optimal_k),
    'Spectral': SpectralClustering(n_clusters=optimal_k, affinity='nearest_neighbors', random_state=42),
    'GMM': GaussianMixture(n_components=optimal_k, random_state=42),
    'DBSCAN': DBSCAN(eps=eps_auto, min_samples=5)
}
```
<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
3. Evaluation of results
</h2>


```python
results = []

for name, model in clustering_methods.items():
    print(f"⏳ Running {name}...")

    try:
        if name == "GMM":
            labels = model.fit(X_reduced).predict(X_reduced)
        else:
            labels = model.fit_predict(X_reduced)

        sil, db, ch = evaluate_clustering(X_reduced, labels)
        stability = compute_stability(X_reduced, model)
        print(f"✅ {name} done.")

    except Exception as e:
        print(f"❌ Error {name}: {e}")
        sil, db, ch, stability = np.nan, np.nan, np.nan, np.nan

    results.append({
        "Method": name,
        "Silhouette": sil,
        "Davies-Bouldin": db,
        "Calinski-Harabasz": ch,
        "Stability (ARI)": stability
    })

results_df = pd.DataFrame(results)
```
<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
4. Visualization and optimal algorithm
</h2>

```python
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

methods = results_df['Method']
x = np.arange(len(methods))
width = 0.25

fig, ax1 = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#000000')
ax1.set_facecolor('#050505')

# First Y axis : Calinski-Harabasz
ch_values = results_df['Calinski-Harabasz'].fillna(0)
rects1 = ax1.bar(x - width, ch_values, width,
                 label='Calinski-Harabasz', color='#31905e', alpha=0.8)

ax1.set_ylabel('Calinski-Harabasz (↑ better)', color='white', fontsize=12)
ax1.tick_params(axis='y', labelcolor='white')
ax1.set_ylim(0, ch_values.max() * 1.2 if ch_values.max() > 0 else 100)

# Second Y axis : Silhouette & Davies-Bouldin
ax2 = ax1.twinx()
sil_values = results_df['Silhouette'].fillna(0)
db_values = results_df['Davies-Bouldin'].fillna(0)

rects2 = ax2.bar(x, sil_values, width,
                 label='Silhouette', color='#9381cf', alpha=0.8)
rects3 = ax2.bar(x + width, db_values, width,
                 label='Davies-Bouldin', color='#d67b6f', hatch='//', alpha=0.8)

ax2.set_ylabel('Silhouette (↑) & Davies-Bouldin (↓)', color='white', fontsize=12)
ax2.tick_params(axis='y', labelcolor='white')
ax2.set_ylim(0, max(sil_values.max(), db_values.max()) * 1.2 if max(sil_values.max(), db_values.max()) > 0 else 1.5)

plt.title("Comparison of Clustering Scores", color='white', fontsize=15, pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, color='white')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, facecolor='#222')

plt.grid(axis='y', linestyle='--', alpha=0.2)
plt.tight_layout()
plt.show()

scaler = MinMaxScaler()

scores_to_rank = results_df[['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Stability (ARI)']].copy()
scores_to_rank = scores_to_rank.fillna(0)

# DB is inverted so that higher is consistently better across all metrics
scores_to_rank['Davies-Bouldin'] = -scores_to_rank['Davies-Bouldin']

# Normalization between 0 and 1 so each metric carries equal weight
scores_scaled = scaler.fit_transform(scores_to_rank)
results_df['Global_Score'] = scores_scaled.mean(axis=1)

best_method = results_df.loc[results_df['Global_Score'].idxmax(), 'Method']

print("\n--- Summary table ---")
print(results_df[['Method', 'Silhouette', 'Davies-Bouldin', 'Global_Score']].to_string(index=False))

print(f"\n🏆 The recommended algorithm for your MACSima data is : {best_method}")

plt.style.use('default')
```

The absence of data for the DBSCAN algorithm is due to its fundamental operational difference compared to centroid based models (such as KMeans). Unlike the latter, which force every point into a cluster, DBSCAN is a density based algorithm. It relies on two crucial parameters: the search radius (eps) and the minimum number of points (min_samples).
If these parameters are too restrictive relative to the spatial distribution of the data especially following Dimensionality Reduction (PCA), which alters distances the algorithm may classify all points as 'noise' (outliers). Mathematically, if no clusters are formed or if only a single global group is identified, validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) cannot be calculated, as they require the comparison of at least two distinct partitions. In the context of tissue biology, this frequently occurs when cell density is too heterogeneous to be captured by a single, fixed search radius.

<h2 style="color: #000000; border-bottom: 1px solid #333; font-family: Georgia, serif;  font-weight: normal; padding-bottom: 5px; margin-top: 35px;">
5. Marker Expression Heatmap
</h2>

Once the optimal clustering algorithm has been identified, a heatmap is generated to interpret the biological meaning of each cluster. Each row represents a marker (protein) and each column a cluster. The values displayed correspond to the mean expression level of each marker within each cluster, computed on the original (non-PCA) data to preserve biological interpretability.
This visualization allows for the identification of cluster-specific protein signatures a key step in phenotyping cell populations from MACSima cyclic imaging data.

```python
# ------------------- HEATMAP -------------------
import seaborn as sns

# Ajouter les labels au DataFrame original
df_heatmap = X.copy()
df_heatmap['Cluster'] = best_labels

# Exclure le bruit DBSCAN si présent
df_heatmap = df_heatmap[df_heatmap['Cluster'] != -1]

# Moyenne des marqueurs par cluster
cluster_means = df_heatmap.groupby('Cluster').mean()

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#000000')
ax.set_facecolor('#050505')

sns.heatmap(
    cluster_means.T,
    ax=ax,
    cmap='magma',
    annot=True,
    fmt='.2f',
    linewidths=0.5,
    linecolor='#222',
    cbar_kws={'label': 'Mean Expression'}
)

ax.set_title(f"Marker Expression per Cluster — {best_method}", 
             color='white', fontsize=14, pad=15)
ax.set_xlabel('Cluster', color='white', fontsize=12)
ax.set_ylabel('Marker', color='white', fontsize=12)
ax.tick_params(colors='white')
plt.colorbar(ax.collections[0]).set_label('Mean Expression', color='white')
plt.tight_layout()
plt.show()

plt.style.use('default')
````

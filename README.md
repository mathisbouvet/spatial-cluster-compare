# spatial-cluster-compare

[![Tests](https://github.com/mathisbouvet/spatial-cluster-compare/actions/workflows/tests.yml/badge.svg)](https://github.com/mathisbouvet/spatial-cluster-compare/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/spatial-cluster-compare.svg)](https://pypi.org/project/spatial-cluster-compare/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Pipeline de comparaison automatisée d'algorithmes de clustering pour données de
spatial proteomics, cytométrie ou tout tableau de features numériques.

Issu d'un notebook d'analyse (`Comparaison_des_clusters.ipynb`) et restructuré
en package réutilisable, dans le même esprit que [`macsima-qc`](https://pypi.org/project/macsima-qc/).

## Deux façons d'utiliser ce repo

- **`notebooks/Comparaison_des_clusters.ipynb`** : le pipeline complet en notebook
  autonome, sans dépendance au package — pratique pour copier-coller le code directement
  dans ton propre environnement sans installer la lib.
- **`spatial_cluster_compare` (le package)** : la même logique encapsulée en fonctions
  réutilisables, testées, et installables via `pip`.

La documentation détaillée pas-à-pas (formules, justification biologique de chaque étape)
est disponible dans [`docs/02_comparaison_cluster.md`](docs/02_comparaison_cluster.md).

## Ce que fait le pipeline

1. **Test de clusterabilité** (statistique de Hopkins) avant/après normalisation
2. **Normalisation conditionnelle** : StandardScaler appliqué uniquement si cela améliore la structure de clusters
3. **Réduction de dimension** (PCA, variance expliquée paramétrable)
4. **Sélection automatique de k** via un score composite (Silhouette, Davies-Bouldin, Calinski-Harabasz)
5. **Benchmark multi-algorithmes** : KMeans, Agglomerative, Spectral, GMM, DBSCAN (eps estimé automatiquement par la méthode du coude)
6. **Stabilité bootstrap** (ARI moyen) pour chaque méthode
7. **Sélection automatique de la meilleure méthode** selon un score composite normalisé
8. **Visualisations** : bar chart comparatif des métriques, heatmap d'expression moyenne par cluster

## Format des données attendues

### Export depuis MACSiQView

Dans MACSiQView, l'onglet **"Feature Table"** permet de sélectionner les descripteurs
à inclure avant export `.csv`. Pour ce pipeline de clustering (contrairement à la QC de
segmentation morphologique décrite dans `macsima-qc`, qui exclut les intensités de
fluorescence), il faut au contraire sélectionner les **intensités de biomarqueurs**
(colonnes du type `CD8 Biomarker Exp`, `CD74 Biomarker Exp`, etc.), à l'exclusion ou non
des descripteurs morphologiques (`Cell Size`...) selon que tu veuilles les inclure dans
le clustering.

### Structure du CSV

- **Une ligne par cellule**, une colonne par marqueur/descripteur.
- Les colonnes numériques (`float64`/`int64`) sont automatiquement utilisées comme
  features de clustering via `select_dtypes` — pas besoin de les spécifier manuellement.
- Les colonnes non numériques (ID, nom de ROI, métadonnées texte) sont ignorées
  automatiquement et n'ont pas besoin d'être supprimées au préalable.
- **Attention** : si une colonne numérique n'est pas un marqueur biologique (ex: un
  `Cell_ID` entier, des coordonnées X/Y de centroïde), elle sera quand même incluse à
  tort dans le clustering. Dans ce cas, retire-la explicitement avant d'appeler
  `compare_clusters`, par exemple :
  ```python
  data = data.drop(columns=["Cell_ID", "Centroid X", "Centroid Y"])
  ```
- Espaces parasites dans les noms de colonnes : MACSiQView peut générer des espaces en
  début/fin de nom de colonne. Pense à nettoyer si besoin (`data.columns = data.columns.str.strip()`),
  comme dans le protocole de QC de segmentation.
- **Valeurs manquantes (NaN)** : non gérées automatiquement par le pipeline. Nettoyer
  avant (`data.dropna()`) ou imputer si certaines cellules ont des valeurs manquantes
  sur un ou plusieurs marqueurs.
- **Taille du jeu de données** : prévoir au moins quelques centaines de cellules pour
  que la statistique de Hopkins et le bootstrap de stabilité (ARI) soient fiables.
  Spectral Clustering peut devenir lent au-delà de ~10-20k lignes.

## Installation

```bash
pip install -e .
```

(Publication sur PyPI à faire plus tard, comme pour `macsima-qc`.)

## Usage rapide

```python
import pandas as pd
from spatial_cluster_compare import compare_clusters, plot_comparison_bars, plot_cluster_heatmap

data = pd.read_csv("Cluster_ImmuneCell.csv")
result = compare_clusters(data)

print(result.results_df)
print(f"Meilleure méthode : {result.best_method}")

plot_comparison_bars(result.results_df)
plot_cluster_heatmap(result.X_original, result.best_labels, method_name=result.best_method)
```

## Usage avancé (étape par étape)

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

## Correction apportée par rapport au notebook d'origine

Le notebook original référençait `best_labels` et `best_method` dans la
cellule de heatmap sans jamais les définir explicitement. La fonction
`select_best_method` corrige ce point en calculant un score composite
normalisé (Silhouette ↑, Davies-Bouldin ↓ inversé, Calinski-Harabasz ↑,
Stabilité ARI ↑) pour désigner objectivement la meilleure méthode.

## Prochaines pistes

- Export PDF/figure automatique du rapport de comparaison
- Support de jeux de données multi-batch (comparaison de clustering entre runs)
- CLI (`spatial-cluster-compare run data.csv`)

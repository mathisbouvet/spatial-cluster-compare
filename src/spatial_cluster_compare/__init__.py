"""spatial_cluster_compare

Pipeline de comparaison automatisée d'algorithmes de clustering pour données
de spatial proteomics / cytométrie (et plus généralement tout tableau de
features numériques) : test de clusterabilité, normalisation conditionnelle,
réduction de dimension, sélection automatique de k, benchmark multi-algorithmes
(KMeans, Agglomerative, Spectral, GMM, DBSCAN) avec stabilité bootstrap (ARI),
et visualisations prêtes à l'emploi.

Usage rapide
------------
    from spatial_cluster_compare import compare_clusters, plot_comparison_bars, plot_cluster_heatmap
    import pandas as pd

    data = pd.read_csv("Cluster_ImmuneCell.csv")
    result = compare_clusters(data)

    plot_comparison_bars(result.results_df)
    plot_cluster_heatmap(result.X_original, result.best_labels, method_name=result.best_method)
"""

from .api import ClusterComparisonResult, compare_clusters
from .benchmark import build_default_methods, run_benchmark, select_best_method
from .metrics import compute_stability, evaluate_clustering
from .preprocessing import auto_scale, hopkins_statistic
from .selection import estimate_dbscan_eps, find_best_k
from .viz import plot_cluster_heatmap, plot_comparison_bars

__version__ = "0.1.0"

__all__ = [
    "compare_clusters",
    "ClusterComparisonResult",
    "build_default_methods",
    "run_benchmark",
    "select_best_method",
    "compute_stability",
    "evaluate_clustering",
    "auto_scale",
    "hopkins_statistic",
    "estimate_dbscan_eps",
    "find_best_k",
    "plot_cluster_heatmap",
    "plot_comparison_bars",
]

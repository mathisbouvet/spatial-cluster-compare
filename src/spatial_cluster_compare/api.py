"""API haut niveau : pipeline complet de comparaison de clustering en un appel."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .benchmark import run_benchmark, select_best_method
from .preprocessing import auto_scale
from .selection import find_best_k


@dataclass
class ClusterComparisonResult:
    """Conteneur pour les résultats complets du pipeline."""

    X_original: pd.DataFrame
    X_reduced: np.ndarray
    optimal_k: int
    k_search_results: pd.DataFrame
    results_df: pd.DataFrame
    labels_per_method: dict
    best_method: str
    best_labels: np.ndarray
    preprocessing_report: dict = field(default_factory=dict)
    pca_variance_ratio: float = 0.9
    n_pca_components: int = 0


def compare_clusters(
    data: pd.DataFrame,
    k_range: range = range(2, 10),
    pca_variance: float = 0.9,
    compute_stability_scores: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> ClusterComparisonResult:
    """Pipeline complet : Hopkins -> normalisation conditionnelle -> PCA ->
    sélection de k -> benchmark multi-algorithmes -> sélection du meilleur.

    Parameters
    ----------
    data : pd.DataFrame
        Données brutes (colonnes numériques = marqueurs/features).
        Les colonnes non numériques sont automatiquement ignorées pour le clustering.
    k_range : range
        Plage de k testée pour le KMeans de référence.
    pca_variance : float
        Variance cumulée à conserver lors de la PCA (ex: 0.9 = 90%).
    compute_stability_scores : bool
        Calcule l'ARI bootstrap pour chaque méthode (plus lent, mettre False pour itérer vite).

    Returns
    -------
    ClusterComparisonResult
    """
    X = data.select_dtypes(include=["float64", "int64"])

    X_best, preprocessing_report = auto_scale(X, random_state=random_state)
    if verbose:
        print(f"Hopkins avant : {preprocessing_report['hopkins_before']:.3f}")
        print(f"Hopkins après : {preprocessing_report['hopkins_after']:.3f}")

    pca = PCA(n_components=pca_variance)
    X_reduced = pca.fit_transform(X_best)
    if verbose:
        print(f"📐 PCA components retained: {pca.n_components_} ({int(pca_variance * 100)}% of variance explained)")

    optimal_k, k_results = find_best_k(X_reduced, k_range, random_state=random_state)
    if verbose:
        print(f"👉 Optimal k detected: {optimal_k}")

    results_df, labels_per_method = run_benchmark(
        X_reduced,
        optimal_k=optimal_k,
        compute_stability_scores=compute_stability_scores,
        random_state=random_state,
        verbose=verbose,
    )

    best_method, best_labels = select_best_method(results_df, labels_per_method)
    if verbose:
        print(f"🏆 Best method: {best_method}")

    return ClusterComparisonResult(
        X_original=X,
        X_reduced=X_reduced,
        optimal_k=optimal_k,
        k_search_results=k_results,
        results_df=results_df,
        labels_per_method=labels_per_method,
        best_method=best_method,
        best_labels=best_labels,
        preprocessing_report=preprocessing_report,
        pca_variance_ratio=pca_variance,
        n_pca_components=pca.n_components_,
    )

"""Sélection automatique des hyperparamètres de clustering : nombre de
clusters optimal (k) et rayon de voisinage (eps) pour DBSCAN."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from .metrics import evaluate_clustering


def find_best_k(X: np.ndarray, k_range: range, random_state: int = 42) -> tuple[int, pd.DataFrame]:
    """Teste plusieurs valeurs de k avec KMeans et sélectionne la meilleure
    selon un score composite (Silhouette + Davies-Bouldin inversé + Calinski-Harabasz).

    Returns
    -------
    optimal_k : int
    results : pd.DataFrame
        Colonnes : k, sil, db, ch, score (score composite normalisé).
    """
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)

        sil, db, ch = evaluate_clustering(X, labels)
        results.append([k, sil, db, ch])

    df = pd.DataFrame(results, columns=["k", "sil", "db", "ch"])

    scaler = MinMaxScaler()
    scores = df[["sil", "db", "ch"]].copy()
    scores["db"] = -scores["db"]  # plus bas = mieux -> on inverse pour que "plus haut = mieux"
    scores_scaled = scaler.fit_transform(scores.fillna(scores.min()))

    df["score"] = scores_scaled.mean(axis=1)
    optimal_k = int(df.loc[df["score"].idxmax(), "k"])

    return optimal_k, df


def estimate_dbscan_eps(X: np.ndarray, n_neighbors: int = 5, percentile: float = 90.0) -> float:
    """Estime automatiquement le paramètre eps de DBSCAN via la méthode du
    coude sur les k-distances (k-distance elbow method).

    Parameters
    ----------
    X : np.ndarray
    n_neighbors : int
        Nombre de voisins utilisé pour le calcul des k-distances.
    percentile : float
        Percentile des k-distances triées utilisé comme estimation d'eps.
    """
    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, _ = neighbors.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    return float(np.percentile(k_distances, percentile))

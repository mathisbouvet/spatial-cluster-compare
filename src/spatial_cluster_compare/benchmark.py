"""Benchmark de plusieurs algorithmes de clustering et sélection du meilleur."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from .metrics import compute_stability, evaluate_clustering
from .selection import estimate_dbscan_eps


DEFAULT_METHODS = ("KMeans", "Agglomerative", "Spectral", "GMM", "DBSCAN")


def build_default_methods(
    optimal_k: int,
    X_reduced: np.ndarray,
    random_state: int = 42,
    dbscan_min_samples: int = 5,
) -> dict:
    """Construit le dictionnaire de modèles par défaut, avec eps DBSCAN estimé
    automatiquement à partir des données.
    """
    eps_auto = estimate_dbscan_eps(X_reduced, n_neighbors=dbscan_min_samples)

    return {
        "KMeans": KMeans(n_clusters=optimal_k, random_state=random_state),
        "Agglomerative": AgglomerativeClustering(n_clusters=optimal_k),
        "Spectral": SpectralClustering(
            n_clusters=optimal_k, affinity="nearest_neighbors", random_state=random_state
        ),
        "GMM": GaussianMixture(n_components=optimal_k, random_state=random_state),
        "DBSCAN": DBSCAN(eps=eps_auto, min_samples=dbscan_min_samples),
    }


def run_benchmark(
    X_reduced: np.ndarray,
    methods: dict | None = None,
    optimal_k: int | None = None,
    compute_stability_scores: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Exécute plusieurs algorithmes de clustering et calcule leurs métriques.

    Parameters
    ----------
    X_reduced : np.ndarray
        Données (typiquement après PCA).
    methods : dict, optionnel
        Dictionnaire {nom: estimator}. Si None, utilise `build_default_methods`
        (nécessite `optimal_k`).
    optimal_k : int, optionnel
        Requis si `methods` n'est pas fourni.
    compute_stability_scores : bool
        Si True, calcule l'ARI bootstrap pour chaque méthode (plus lent).

    Returns
    -------
    results_df : pd.DataFrame
        Colonnes : Method, Silhouette, Davies-Bouldin, Calinski-Harabasz, Stability (ARI).
    labels_per_method : dict
        {nom_méthode: labels_array} pour récupération ultérieure (ex: heatmap).
    """
    if methods is None:
        if optimal_k is None:
            raise ValueError("Fournir `methods` ou `optimal_k` pour construire les modèles par défaut.")
        methods = build_default_methods(optimal_k, X_reduced, random_state=random_state)

    results = []
    labels_per_method = {}

    for name, model in methods.items():
        if verbose:
            print(f"⏳ Running {name}...")

        try:
            if hasattr(model, "fit_predict"):
                labels = model.fit_predict(X_reduced)
            else:
                labels = model.fit(X_reduced).predict(X_reduced)

            sil, db, ch = evaluate_clustering(X_reduced, labels)
            stability = (
                compute_stability(X_reduced, model, random_state=random_state)
                if compute_stability_scores
                else np.nan
            )
            labels_per_method[name] = labels
            if verbose:
                print(f"✅ {name} done.")

        except Exception as e:
            if verbose:
                print(f"❌ Error {name}: {e}")
            sil, db, ch, stability = np.nan, np.nan, np.nan, np.nan
            labels_per_method[name] = None

        results.append(
            {
                "Method": name,
                "Silhouette": sil,
                "Davies-Bouldin": db,
                "Calinski-Harabasz": ch,
                "Stability (ARI)": stability,
            }
        )

    return pd.DataFrame(results), labels_per_method


def select_best_method(
    results_df: pd.DataFrame,
    labels_per_method: dict,
    weights: dict | None = None,
) -> tuple[str, np.ndarray]:
    """Sélectionne automatiquement la meilleure méthode de clustering à partir
    d'un score composite normalisé (corrige le `best_labels`/`best_method`
    manquants dans le notebook d'origine).

    Parameters
    ----------
    results_df : pd.DataFrame
        Sortie de `run_benchmark`.
    labels_per_method : dict
        Sortie de `run_benchmark`.
    weights : dict, optionnel
        Poids par métrique, ex: {"Silhouette": 1, "Davies-Bouldin": 1,
        "Calinski-Harabasz": 1, "Stability (ARI)": 1}. Par défaut tout à 1.

    Returns
    -------
    best_method : str
    best_labels : np.ndarray
    """
    df = results_df.copy()
    available = df.dropna(subset=["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"], how="all")

    if available.empty:
        raise ValueError("Aucune méthode n'a produit de métriques valides (clustering échoué partout).")

    default_weights = {
        "Silhouette": 1.0,
        "Davies-Bouldin": 1.0,
        "Calinski-Harabasz": 1.0,
        "Stability (ARI)": 1.0,
    }
    weights = weights or default_weights

    norm = available.copy()
    for col in ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Stability (ARI)"]:
        if col not in norm:
            continue
        vals = norm[col]
        rng = vals.max() - vals.min()
        if rng == 0 or vals.isna().all():
            norm[col] = 0.0
        else:
            norm[col] = (vals - vals.min()) / rng

    # Davies-Bouldin : plus bas = mieux -> on inverse
    if "Davies-Bouldin" in norm:
        norm["Davies-Bouldin"] = 1 - norm["Davies-Bouldin"]

    norm = norm.fillna(0.0)

    score = sum(norm[col] * w for col, w in weights.items() if col in norm)
    best_idx = score.idxmax()
    best_method = available.loc[best_idx, "Method"]
    best_labels = labels_per_method[best_method]

    return best_method, best_labels

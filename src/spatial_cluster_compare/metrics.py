"""Métriques de qualité et de stabilité pour le clustering."""

from __future__ import annotations

import copy

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Calcule Silhouette, Davies-Bouldin et Calinski-Harabasz pour un clustering.

    Les points de bruit DBSCAN (label -1) sont exclus. Retourne (nan, nan, nan)
    si moins de 2 clusters distincts sont présents (métriques non définies).
    """
    unique = set(labels) - {-1}
    if len(unique) <= 1:
        return np.nan, np.nan, np.nan

    mask = labels != -1
    X_eval, labels_eval = X[mask], labels[mask]

    return (
        silhouette_score(X_eval, labels_eval),
        davies_bouldin_score(X_eval, labels_eval),
        calinski_harabasz_score(X_eval, labels_eval),
    )


def compute_stability(
    X: np.ndarray,
    model,
    n_bootstrap: int = 10,
    sample_frac: float = 0.8,
    random_state: int = 42,
) -> float:
    """Évalue la stabilité d'un clustering par bootstrap (ARI moyen entre le
    clustering de référence et des clusterings sur des sous-échantillons).

    Parameters
    ----------
    X : np.ndarray
        Données déjà réduites/normalisées.
    model : estimator scikit-learn (non fit)
        Doit implémenter fit_predict ou fit+predict (ex: GaussianMixture).
    n_bootstrap : int
        Nombre de répétitions bootstrap.
    sample_frac : float
        Fraction des points échantillonnés à chaque répétition.

    Returns
    -------
    float
        ARI moyen (1.0 = parfaitement stable, 0 = stabilité aléatoire).
    """
    rng = np.random.RandomState(random_state)
    n = len(X)

    reference_model = copy.deepcopy(model)
    if hasattr(reference_model, "fit_predict"):
        reference_labels = reference_model.fit_predict(X)
    else:
        reference_labels = reference_model.fit(X).predict(X)

    aris = []
    for i in range(n_bootstrap):
        sample_idx = resample(
            np.arange(n), n_samples=int(sample_frac * n), random_state=rng.randint(0, 1_000_000)
        )
        X_sample = X[sample_idx]

        boot_model = copy.deepcopy(model)
        try:
            if hasattr(boot_model, "fit_predict"):
                boot_labels = boot_model.fit_predict(X_sample)
            else:
                boot_labels = boot_model.fit(X_sample).predict(X_sample)
        except Exception:
            continue

        ari = adjusted_rand_score(reference_labels[sample_idx], boot_labels)
        aris.append(ari)

    if not aris:
        return np.nan

    return float(np.mean(aris))

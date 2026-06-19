"""Préparation des données avant clustering : test de clusterabilité (Hopkins)
et normalisation conditionnelle."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def hopkins_statistic(X: pd.DataFrame | np.ndarray, m_ratio: float = 0.1, random_state: int | None = None) -> float:
    """Calcule la statistique de Hopkins pour évaluer la tendance au clustering.

    Une valeur proche de 0.5 indique des données distribuées aléatoirement
    (peu de structure de clusters), tandis qu'une valeur proche de 1 indique
    une forte structure de clusters.

    Parameters
    ----------
    X : DataFrame ou array
        Données (échantillons en lignes, variables en colonnes).
    m_ratio : float
        Proportion de points échantillonnés pour le test (défaut 10%).
    random_state : int, optionnel
        Graine pour la reproductibilité de l'échantillonnage.

    Returns
    -------
    float
        Statistique de Hopkins (entre 0 et 1).
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    rng = np.random.RandomState(random_state)

    d = X.shape[1]
    n = len(X)
    m = max(1, int(m_ratio * n))

    rand_X = X.sample(m, random_state=random_state)
    neigh = NearestNeighbors(n_neighbors=1).fit(X)

    u_dist, _ = neigh.kneighbors(rand_X)

    min_vals, max_vals = X.min(), X.max()
    rand_uniform = rng.uniform(low=min_vals, high=max_vals, size=(m, d))
    w_dist, _ = neigh.kneighbors(rand_uniform)

    denom = np.sum(w_dist) + np.sum(u_dist)
    if denom == 0:
        return np.nan

    return float(np.sum(w_dist) / denom)


def auto_scale(
    X: pd.DataFrame,
    m_ratio: float = 0.1,
    random_state: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Standardise les données uniquement si cela améliore la structure de clusters.

    Compare la statistique de Hopkins avant/après StandardScaler et retourne
    la version la plus favorable au clustering.

    Returns
    -------
    X_best : np.ndarray
        Données choisies (brutes ou standardisées).
    report : dict
        Contient 'hopkins_before', 'hopkins_after', et 'scaled' (bool).
    """
    hopkins_before = hopkins_statistic(X, m_ratio=m_ratio, random_state=random_state)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    hopkins_after = hopkins_statistic(
        pd.DataFrame(X_scaled, columns=X.columns), m_ratio=m_ratio, random_state=random_state
    )

    scaled = hopkins_after > hopkins_before
    X_best = X_scaled if scaled else X.values

    report = {
        "hopkins_before": hopkins_before,
        "hopkins_after": hopkins_after,
        "scaled": scaled,
    }

    return X_best, report

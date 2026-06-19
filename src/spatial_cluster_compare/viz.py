"""Visualisations pour la comparaison de clustering : bar chart double axe et
heatmap d'expression moyenne par cluster."""

from __future__ import annotations

import numpy as np
import pandas as pd


def plot_comparison_bars(results_df: pd.DataFrame, dark_theme: bool = True, ax=None):
    """Bar chart comparatif des méthodes de clustering (Calinski-Harabasz sur
    un axe, Silhouette + Davies-Bouldin sur l'autre).

    Parameters
    ----------
    results_df : pd.DataFrame
        Sortie de `run_benchmark`.
    dark_theme : bool
        Applique le thème sombre (par défaut). Mettre False pour le thème matplotlib standard.
    ax : matplotlib Axes, optionnel
        Si fourni, dessine sur cet axe au lieu d'en créer un nouveau.

    Returns
    -------
    fig, ax1, ax2 : Figure et les deux axes matplotlib.
    """
    import matplotlib.pyplot as plt

    if dark_theme:
        plt.style.use("dark_background")

    methods = results_df["Method"]
    x = np.arange(len(methods))
    width = 0.25

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(12, 7))
    else:
        ax1 = ax
        fig = ax1.figure

    if dark_theme:
        fig.patch.set_facecolor("#000000")
        ax1.set_facecolor("#050505")

    ch_values = results_df["Calinski-Harabasz"].fillna(0)
    ax1.bar(x - width, ch_values, width, label="Calinski-Harabasz", color="#31905e", alpha=0.8)
    ax1.set_ylabel("Calinski-Harabasz (↑ better)", color="white" if dark_theme else "black", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="white" if dark_theme else "black")
    ax1.set_ylim(0, ch_values.max() * 1.2 if ch_values.max() > 0 else 100)

    ax2 = ax1.twinx()
    sil_values = results_df["Silhouette"].fillna(0)
    db_values = results_df["Davies-Bouldin"].fillna(0)

    ax2.bar(x, sil_values, width, label="Silhouette", color="#9381cf", alpha=0.8)
    ax2.bar(x + width, db_values, width, label="Davies-Bouldin", color="#d67b6f", hatch="//", alpha=0.8)

    ax2.set_ylabel(
        "Silhouette (↑) & Davies-Bouldin (↓)", color="white" if dark_theme else "black", fontsize=12
    )
    ax2.tick_params(axis="y", labelcolor="white" if dark_theme else "black")
    max_val = max(sil_values.max(), db_values.max())
    ax2.set_ylim(0, max_val * 1.2 if max_val > 0 else 1.5)

    text_color = "white" if dark_theme else "black"
    ax1.set_title("Comparison of Clustering Scores", color=text_color, fontsize=15, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, color=text_color)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    fig.tight_layout()

    if dark_theme:
        plt.style.use("default")

    return fig, ax1, ax2


def plot_cluster_heatmap(
    X: pd.DataFrame,
    labels: np.ndarray,
    method_name: str = "",
    dark_theme: bool = True,
    ax=None,
    normalize: str | None = "zscore",
    annot_raw: bool = True,
):
    """Heatmap de l'expression moyenne de chaque marqueur par cluster.

    Parameters
    ----------
    X : pd.DataFrame
        Données originales (marqueurs en colonnes), non normalisées de préférence.
    labels : np.ndarray
        Labels de cluster (mêmes index que X). Les points DBSCAN bruit (-1) sont exclus.
    method_name : str
        Nom de la méthode, affiché dans le titre.
    normalize : {"zscore", "minmax", None}
        Normalisation appliquée par marqueur (par ligne) avant affichage de la couleur.
        Indispensable quand les marqueurs ont des échelles très différentes (ex: un
        marqueur à ~300 et un autre à ~30000), sinon la colorbar globale écrase les
        marqueurs à faible amplitude. "zscore" (défaut) centre-réduit chaque marqueur ;
        "minmax" ramène chaque marqueur entre 0 et 1 ; None désactive (comportement brut).
    annot_raw : bool
        Si True (défaut) et `normalize` est actif, les valeurs annotées dans les cellules
        restent les moyennes brutes (plus lisibles) même si la couleur, elle, est normalisée.
        Si False, annote directement les valeurs normalisées.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_heatmap = X.copy()
    df_heatmap["Cluster"] = labels
    df_heatmap = df_heatmap[df_heatmap["Cluster"] != -1]

    cluster_means = df_heatmap.groupby("Cluster").mean()

    cluster_means_raw = cluster_means.copy()

    if normalize == "zscore":
        col_mean = cluster_means.mean(axis=0)
        col_std = cluster_means.std(axis=0).replace(0, 1)
        cluster_means = cluster_means.sub(col_mean, axis=1).div(col_std, axis=1)
    elif normalize == "minmax":
        col_min = cluster_means.min(axis=0)
        col_max = cluster_means.max(axis=0)
        col_range = (col_max - col_min).replace(0, 1)
        cluster_means = cluster_means.sub(col_min, axis=1).div(col_range, axis=1)
    elif normalize is not None:
        raise ValueError("normalize doit être 'zscore', 'minmax' ou None.")

    annot_data = cluster_means_raw.T if (normalize and annot_raw) else cluster_means.T
    cbar_label = {
        "zscore": "Expression (z-score par marqueur)",
        "minmax": "Expression (normalisée 0-1 par marqueur)",
        None: "Mean Expression",
    }[normalize]

    if dark_theme:
        plt.style.use("dark_background")

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        fig = ax.figure

    if dark_theme:
        fig.patch.set_facecolor("#000000")
        ax.set_facecolor("#050505")

    text_color = "white" if dark_theme else "black"

    sns.heatmap(
        cluster_means.T,
        ax=ax,
        cmap="magma",
        annot=annot_data,
        fmt=".2f",
        linewidths=0.5,
        linecolor="#222" if dark_theme else "#ddd",
        cbar_kws={"label": cbar_label},
    )

    title = f"Marker Expression per Cluster — {method_name}" if method_name else "Marker Expression per Cluster"
    ax.set_title(title, color=text_color, fontsize=14, pad=15)
    ax.set_xlabel("Cluster", color=text_color, fontsize=12)
    ax.set_ylabel("Marker", color=text_color, fontsize=12)
    ax.tick_params(colors=text_color)

    fig.tight_layout()

    if dark_theme:
        plt.style.use("default")

    return fig, ax

import numpy as np
import pandas as pd
import pytest

from spatial_cluster_compare import (
    compare_clusters,
    estimate_dbscan_eps,
    find_best_k,
    hopkins_statistic,
)


@pytest.fixture
def synthetic_blobs():
    rng = np.random.RandomState(0)
    cluster_a = rng.normal(loc=0, scale=0.5, size=(50, 4))
    cluster_b = rng.normal(loc=8, scale=0.5, size=(50, 4))
    cluster_c = rng.normal(loc=-8, scale=0.5, size=(50, 4))
    X = np.vstack([cluster_a, cluster_b, cluster_c])
    return pd.DataFrame(X, columns=["marker_1", "marker_2", "marker_3", "marker_4"])


def test_hopkins_statistic_detects_structure(synthetic_blobs):
    score = hopkins_statistic(synthetic_blobs, random_state=0)
    assert 0.7 < score <= 1.0


def test_find_best_k(synthetic_blobs):
    optimal_k, results = find_best_k(synthetic_blobs.values, range(2, 6))
    assert optimal_k == 3
    assert "score" in results.columns


def test_estimate_dbscan_eps_positive(synthetic_blobs):
    eps = estimate_dbscan_eps(synthetic_blobs.values)
    assert eps > 0


def test_compare_clusters_end_to_end(synthetic_blobs):
    result = compare_clusters(
        synthetic_blobs,
        k_range=range(2, 6),
        compute_stability_scores=False,
        verbose=False,
    )
    assert result.optimal_k == 3
    assert result.best_method in result.results_df["Method"].values
    assert len(result.best_labels) == len(synthetic_blobs)

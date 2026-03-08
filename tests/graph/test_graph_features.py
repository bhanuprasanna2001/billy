"""Tests for graph feature engineering (NetworkX local backend)."""

from pathlib import Path

import pytest

from credit_domino.graph.features import compute_graph_features_local

CSV_PATH = Path("data/credit_risk_dataset.csv")
DATA_DIR = Path("data")


@pytest.fixture(scope="module")
def graph_features():
    return compute_graph_features_local(data_dir=DATA_DIR, seed=42, n_customers=200)


def test_graph_features_columns(graph_features):
    expected = {
        "customer_id",
        "degree",
        "in_degree",
        "out_degree",
        "norm_in_degree",
        "norm_out_degree",
        "pagerank",
        "distance_to_prior_default",
        "clustering_coefficient",
        "neighbor_default_frac",
        "neighbor_default_frac_2hop",
    }
    assert expected.issubset(graph_features.columns)


def test_graph_features_row_count(graph_features):
    assert len(graph_features) == 200


def test_graph_features_degree_is_numeric(graph_features):
    assert graph_features["degree"].dtype in ("int64", "float64")
    assert (graph_features["degree"] >= 0).all()


def test_graph_features_pagerank_is_float(graph_features):
    assert graph_features["pagerank"].dtype == "float64"
    assert (graph_features["pagerank"] >= 0).all()
    assert abs(graph_features["pagerank"].sum() - 1.0) < 0.01  # pagerank sums to ~1


def test_graph_features_distance_values(graph_features):
    dist = graph_features["distance_to_prior_default"]
    # All values should be -1 (unreachable) or >= 0 (reachable)
    assert ((dist == -1) | (dist >= 0)).all()
    # Default nodes themselves should have distance 0
    assert (dist == 0).any(), "Expected some nodes at distance 0 (defaulted customers)"


def test_graph_features_no_null_keys(graph_features):
    assert graph_features["customer_id"].isna().sum() == 0


def test_graph_features_deterministic():
    df1 = compute_graph_features_local(data_dir=DATA_DIR, seed=42, n_customers=50)
    df2 = compute_graph_features_local(data_dir=DATA_DIR, seed=42, n_customers=50)
    assert df1.equals(df2)

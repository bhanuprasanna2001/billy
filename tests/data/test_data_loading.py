"""Tests for data loading and synthetic relationship graph generation."""

from pathlib import Path

import pytest

from credit_domino.data.loaders import generate_relationship_graph, load_credit_data

CSV_PATH = Path("data/credit_risk_dataset.csv")


@pytest.fixture(scope="module")
def credit_data():
    return load_credit_data(CSV_PATH)


@pytest.fixture(scope="module")
def relationships(credit_data):
    return generate_relationship_graph(credit_data, seed=42)


# ── load_credit_data tests ─────────────────────────────────────────


def test_load_has_expected_columns(credit_data):
    expected = {
        "customer_id",
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_grade",
        "loan_amnt",
        "loan_int_rate",
        "loan_status",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length",
        "is_recent_default",
    }
    assert expected.issubset(credit_data.columns)


def test_load_preserves_row_count(credit_data):
    assert len(credit_data) == 32581, "Unexpected row count — is the CSV correct?"


def test_outliers_capped(credit_data):
    assert credit_data["person_age"].max() <= 100
    assert credit_data["person_income"].max() <= 500_000


def test_nulls_filled(credit_data):
    assert credit_data["person_emp_length"].isna().sum() == 0
    assert credit_data["loan_int_rate"].isna().sum() == 0


def test_customer_id_unique_and_formatted(credit_data):
    assert credit_data["customer_id"].is_unique
    assert credit_data["customer_id"].str.startswith("CUST_").all()


def test_cb_default_encoded_as_int(credit_data):
    assert set(credit_data["cb_person_default_on_file"].unique()).issubset({0, 1})


def test_is_recent_default_matches_loan_status(credit_data):
    assert (credit_data["is_recent_default"] == (credit_data["loan_status"] == 1)).all()


# ── generate_relationship_graph tests ──────────────────────────────


def test_relationship_graph_structure(relationships):
    assert {"src_customer_id", "dst_customer_id", "edge_type"}.issubset(relationships.columns)
    assert set(relationships["edge_type"].unique()).issubset(
        {"co-borrower", "guarantor", "employer"}
    )


def test_relationship_graph_valid_customers(credit_data, relationships):
    valid_ids = set(credit_data["customer_id"])
    assert set(relationships["src_customer_id"]).issubset(valid_ids)
    assert set(relationships["dst_customer_id"]).issubset(valid_ids)


def test_relationship_graph_no_self_loops(relationships):
    assert (relationships["src_customer_id"] != relationships["dst_customer_id"]).all()


def test_relationship_graph_reasonable_size(credit_data, relationships):
    n = len(credit_data)
    # Expect roughly 1–3 edges per node on average
    assert len(relationships) > n * 0.5
    assert len(relationships) < n * 5


def test_relationship_graph_deterministic(credit_data):
    e1 = generate_relationship_graph(credit_data, seed=42)
    e2 = generate_relationship_graph(credit_data, seed=42)
    assert e1.equals(e2)

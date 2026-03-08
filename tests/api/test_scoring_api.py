"""API tests — uses TestClient (no server needed), mocks external dependencies."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def mock_model():
    """Create a fake model that returns predictable probabilities."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    model.get_booster.return_value = MagicMock()
    return model


@pytest.fixture(scope="module")
def mock_encoders():
    from sklearn.preprocessing import LabelEncoder

    encoders = {}
    for col, classes in [
        ("person_home_ownership", ["MORTGAGE", "OTHER", "OWN", "RENT"]),
        (
            "loan_intent",
            ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"],
        ),
        ("loan_grade", ["A", "B", "C", "D", "E", "F", "G"]),
    ]:
        le = LabelEncoder()
        le.fit(classes)
        encoders[col] = le
    return encoders


@pytest.fixture(scope="module")
def client(mock_model, mock_encoders):
    """TestClient with pre-loaded mock model (no MLflow/DB needed)."""
    from credit_domino.api import app, state

    state.model = mock_model
    state.encoders = mock_encoders
    state.model_version = "test"
    state.pg_engine = None  # Skip DB writes in tests
    state.ch_client = None

    return TestClient(app)


SAMPLE_REQUEST = {
    "customer_id": "CUST_0",
    "person_age": 30,
    "person_income": 60000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5.0,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.17,
    "cb_person_default_on_file": 0,
    "cb_person_cred_hist_length": 4,
    "degree": 3,
    "pagerank": 0.001,
    "distance_to_prior_default": 2,
    "clustering_coefficient": 0.1,
}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test"


def test_ready(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


@patch("credit_domino.api.get_top_factors")
def test_score_success(mock_shap, client):
    mock_shap.return_value = [
        {"feature": "loan_amnt", "shap_value": 0.42},
        {"feature": "person_income", "shap_value": -0.15},
    ]
    resp = client.post("/score", json=SAMPLE_REQUEST)
    assert resp.status_code == 200
    body = resp.json()
    assert body["customer_id"] == "CUST_0"
    assert body["risk_score"] == 0.7
    assert body["decision_band"] == "high"
    assert len(body["top_factors"]) == 2
    assert body["scoring_event_id"]  # UUID present
    assert body["scored_at"]


@patch("credit_domino.api.get_top_factors")
def test_score_band_low(mock_shap, client, mock_model):
    mock_shap.return_value = []
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])
    resp = client.post("/score", json=SAMPLE_REQUEST)
    assert resp.json()["decision_band"] == "low"
    # Reset
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])


@patch("credit_domino.api.get_top_factors")
def test_score_band_medium(mock_shap, client, mock_model):
    mock_shap.return_value = []
    mock_model.predict_proba.return_value = np.array([[0.65, 0.35]])
    resp = client.post("/score", json=SAMPLE_REQUEST)
    assert resp.json()["decision_band"] == "medium"
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])


def test_score_unknown_category(client):
    bad_req = {**SAMPLE_REQUEST, "person_home_ownership": "SPACESHIP"}
    resp = client.post("/score", json=bad_req)
    assert resp.status_code == 422


def test_score_missing_field(client):
    bad_req = {k: v for k, v in SAMPLE_REQUEST.items() if k != "customer_id"}
    resp = client.post("/score", json=bad_req)
    assert resp.status_code == 422


def test_score_negative_income(client):
    bad_req = {**SAMPLE_REQUEST, "person_income": -1000}
    resp = client.post("/score", json=bad_req)
    assert resp.status_code == 422

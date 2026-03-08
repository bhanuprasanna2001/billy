"""Tests for model training pipeline — fast (200 customers), no MLflow server needed."""

from pathlib import Path

import numpy as np
import pytest

CSV_PATH = Path("data/credit_risk_dataset.csv")
DATA_DIR = Path("data")
N_CUSTOMERS = 200  # small for speed


@pytest.fixture(scope="module")
def artifacts():
    """Train a small model once for all tests in this module."""
    from credit_domino.modeling.train import train_model

    return train_model(DATA_DIR, seed=42, n_customers=N_CUSTOMERS)


class TestFeatureAssembly:
    def test_feature_columns_present(self, artifacts):
        expected = [
            "person_age",
            "person_income",
            "person_home_ownership",
            "person_emp_length",
            "loan_intent",
            "loan_grade",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_default_on_file",
            "cb_person_cred_hist_length",
            "crisis_exposure",
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
            "dti",
            "norm_in_x_dti",
            "norm_in_x_nbr_def",
            "dti_x_cb_default",
            "int_rate_x_dti",
            "degree_ratio",
            "income_log",
        ]
        assert list(artifacts["X_train"].columns) == expected

    def test_no_nulls_in_features(self, artifacts):
        assert artifacts["X_train"].isnull().sum().sum() == 0
        assert artifacts["X_test"].isnull().sum().sum() == 0

    def test_target_is_binary(self, artifacts):
        assert set(artifacts["y_train"].unique()).issubset({0, 1})

    def test_train_test_split_size(self, artifacts):
        total = len(artifacts["X_train"]) + len(artifacts["X_test"])
        assert total == N_CUSTOMERS
        # ~80/20 split
        assert len(artifacts["X_test"]) == pytest.approx(N_CUSTOMERS * 0.2, abs=5)

    def test_categoricals_are_encoded(self, artifacts):
        for col in ["person_home_ownership", "loan_intent", "loan_grade"]:
            assert artifacts["X_train"][col].dtype in (np.int32, np.int64, np.intp)

    def test_encoders_present(self, artifacts):
        assert set(artifacts["encoders"].keys()) == {
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
        }


class TestModelTraining:
    def test_model_type(self, artifacts):
        import xgboost as xgb

        assert isinstance(artifacts["model"], xgb.XGBClassifier)

    def test_predictions_are_binary(self, artifacts):
        preds = artifacts["model"].predict(artifacts["X_test"])
        assert set(np.unique(preds)).issubset({0, 1})

    def test_probabilities_in_range(self, artifacts):
        proba = artifacts["model"].predict_proba(artifacts["X_test"])[:, 1]
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0


class TestEvaluation:
    def test_metrics_keys(self, artifacts):
        from credit_domino.modeling.evaluate import evaluate_model

        metrics, threshold = evaluate_model(
            artifacts["model"],
            artifacts["X_test"],
            artifacts["y_test"],
        )
        assert set(metrics.keys()) == {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "optimal_threshold",
        }

    def test_metrics_in_range(self, artifacts):
        from credit_domino.modeling.evaluate import evaluate_model

        metrics, threshold = evaluate_model(
            artifacts["model"],
            artifacts["X_test"],
            artifacts["y_test"],
        )
        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{name} = {value} out of range"
        assert 0.0 < threshold < 1.0

    def test_roc_auc_above_baseline(self, artifacts):
        """Model should beat random (AUC > 0.5) even on 200 samples."""
        from credit_domino.modeling.evaluate import evaluate_model

        metrics, _ = evaluate_model(
            artifacts["model"],
            artifacts["X_test"],
            artifacts["y_test"],
        )
        assert metrics["roc_auc"] > 0.5


class TestSHAP:
    def test_shap_values_shape(self, artifacts):
        from credit_domino.modeling.evaluate import compute_shap_values

        sv = compute_shap_values(artifacts["model"], artifacts["X_test"])
        assert sv.values.shape == artifacts["X_test"].shape

    def test_top_factors_structure(self, artifacts):
        from credit_domino.modeling.evaluate import get_top_factors

        X_single = artifacts["X_test"].iloc[[0]]
        factors = get_top_factors(artifacts["model"], X_single, top_n=3)
        assert len(factors) == 3
        assert all("feature" in f and "shap_value" in f for f in factors)

    def test_top_factors_sorted_by_importance(self, artifacts):
        from credit_domino.modeling.evaluate import get_top_factors

        X_single = artifacts["X_test"].iloc[[0]]
        factors = get_top_factors(artifacts["model"], X_single, top_n=5)
        abs_vals = [abs(f["shap_value"]) for f in factors]
        assert abs_vals == sorted(abs_vals, reverse=True)

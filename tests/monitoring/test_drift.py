"""Tests for Evidently drift detection."""

from pathlib import Path

from credit_domino.monitoring.drift import run_drift_report

CSV_PATH = Path("data/credit_risk_dataset.csv")


def test_drift_report_structure():
    result = run_drift_report(CSV_PATH)
    assert "drift_detected" in result
    assert isinstance(result["drift_detected"], bool)
    assert "n_drifted_columns" in result
    assert "drift_share" in result
    assert "drift_by_column" in result


def test_drift_report_columns():
    result = run_drift_report(CSV_PATH)
    expected_cols = {
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    }
    assert set(result["drift_by_column"].keys()) == expected_cols

import subprocess
from pathlib import Path

import pytest


def test_dbt_project_compiles():
    """dbt compile must succeed — proves SQL is valid and refs resolve."""
    result = subprocess.run(
        ["dbt", "compile", "--project-dir", "dbt", "--profiles-dir", "dbt"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if "Connection refused" in result.stdout or "Connection refused" in result.stderr:
        pytest.skip("Postgres not running — skipping dbt compile test")
    assert result.returncode == 0, f"dbt compile failed:\n{result.stderr}"


def test_fct_credit_features_has_expected_columns():
    """Compiled SQL for the mart must SELECT the columns we need downstream."""
    compiled = Path("dbt/target/compiled/credit_domino/models/marts/fct_credit_features.sql")
    assert compiled.exists(), "Run `dbt compile` first"
    sql = compiled.read_text().lower()
    expected = [
        "customer_id",
        "person_income",
        "loan_percent_income",
        "person_emp_length",
        "loan_status",
    ]
    for col in expected:
        assert col in sql, f"Missing column {col} in compiled fct_credit_features"


def test_fct_scoring_log_is_incremental():
    """The scoring log model must use incremental materialization."""
    model_sql = Path("dbt/models/marts/fct_scoring_log.sql").read_text().lower()
    assert "incremental" in model_sql


def test_schema_has_tests():
    """schema.yml must define at least unique + not_null tests."""
    schema = Path("dbt/models/schema.yml").read_text().lower()
    assert "unique" in schema
    assert "not_null" in schema

"""Data drift detection using Evidently 0.7.x."""

from pathlib import Path

from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift

from credit_domino.data.loaders import load_credit_data

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]


def run_drift_report(
    csv_path: Path,
    reference_frac: float = 0.8,
) -> dict:
    """Run Evidently drift report comparing reference vs current data.

    Returns dict with drift_detected (bool) and per-column drift results.
    """
    df = load_credit_data(csv_path)
    split = int(len(df) * reference_frac)
    reference = df.iloc[:split]
    current = df.iloc[split:]

    metrics = [DriftedColumnsCount(columns=NUMERIC_FEATURES)]
    for col in NUMERIC_FEATURES:
        metrics.append(ValueDrift(column=col))

    report = Report(metrics=metrics)
    snapshot = report.run(reference_data=reference, current_data=current)
    result = snapshot.dict()

    # Extract DriftedColumnsCount result
    drift_count_metric = result["metrics"][0]
    count = drift_count_metric["value"]["count"]
    share = drift_count_metric["value"]["share"]
    drift_detected = count > 0

    # Extract per-column ValueDrift results
    drift_by_column = {}
    for i, col in enumerate(NUMERIC_FEATURES):
        col_metric = result["metrics"][i + 1]
        col_drift_val = col_metric["value"]
        # ValueDrift threshold default is 0.05 for K-S p-value
        # drift_detected when p_value < threshold (i.e., value < 0.05)
        drift_by_column[col] = float(col_drift_val) < 0.05

    return {
        "drift_detected": drift_detected,
        "n_drifted_columns": int(count),
        "drift_share": float(share),
        "drift_by_column": drift_by_column,
    }


def generate_drift_html(csv_path: Path, output_path: Path) -> Path:
    """Generate Evidently HTML report and save to disk."""
    df = load_credit_data(csv_path)
    split = int(len(df) * 0.8)
    reference = df.iloc[:split]
    current = df.iloc[split:]

    metrics = [DriftedColumnsCount(columns=NUMERIC_FEATURES)]
    for col in NUMERIC_FEATURES:
        metrics.append(ValueDrift(column=col))

    report = Report(metrics=metrics)
    snapshot = report.run(reference_data=reference, current_data=current)
    snapshot.save_html(str(output_path))
    return output_path

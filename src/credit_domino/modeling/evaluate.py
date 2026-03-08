"""Model evaluation: classification metrics + SHAP explanations."""

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_optimal_threshold(y_true, y_proba) -> float:
    """Find the threshold that maximizes F1 on the precision-recall curve.

    For credit risk, this is better than the default 0.5 because class
    imbalance makes 0.5 overly conservative (misses most defaults).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision and recall have len(thresholds) + 1 elements; drop the last
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], float]:
    """Compute classification metrics with optimal threshold.

    Returns (metrics_dict, optimal_threshold).
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    optimal_threshold = find_optimal_threshold(y_test, y_proba)
    y_pred = (y_proba >= optimal_threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "optimal_threshold": optimal_threshold,
    }
    return metrics, optimal_threshold


def compute_shap_values(model, X_test: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values for the test set using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    return shap_values


_cached_explainer = None
_cached_explainer_model_id = None


def get_top_factors(
    model,
    X_single: pd.DataFrame,
    top_n: int = 5,
) -> list[dict[str, object]]:
    """Get top SHAP factors for a single prediction (used in API scoring).

    Returns list of dicts: [{"feature": "loan_amnt", "shap_value": 0.42}, ...]
    """
    global _cached_explainer, _cached_explainer_model_id
    if _cached_explainer is None or _cached_explainer_model_id is not id(model):
        _cached_explainer = shap.TreeExplainer(model)
        _cached_explainer_model_id = id(model)
    sv = _cached_explainer(X_single)
    values = sv.values[0]
    features = list(X_single.columns)

    # Sort by absolute SHAP value descending
    indices = np.argsort(np.abs(values))[::-1][:top_n]
    return [{"feature": features[i], "shap_value": float(values[i])} for i in indices]

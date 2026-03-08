"""Model training pipeline: feature assembly, XGBoost training, SHAP explanations."""

from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from credit_domino.data.loaders import load_data
from credit_domino.graph.features import compute_graph_features

# Categorical columns that need label encoding
_CAT_COLS = ["person_home_ownership", "loan_intent", "loan_grade"]

# Tabular feature columns (post-encoding)
_TABULAR_FEATURES = [
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
]

# Graph feature columns merged from graph module
_GRAPH_FEATURES = [
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
]

TARGET = "loan_status"


def assemble_features(
    data_dir: Path = Path("data"),
    seed: int = 42,
    n_customers: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
    """Load data, compute graph features, encode categoricals, return X, y, encoders."""
    df, edges = load_data(data_dir, seed=seed, n_customers=n_customers)

    # Compute graph features and merge (uses pre-loaded data, no second load)
    graph_df = compute_graph_features(df, edges)
    df = df.merge(graph_df, on="customer_id", how="left")

    # Label-encode categoricals
    encoders: dict[str, LabelEncoder] = {}
    for col in _CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = _TABULAR_FEATURES + _GRAPH_FEATURES
    X = df[feature_cols].copy()
    # Add derived feature: unclipped DTI for stronger signal
    # (loan_percent_income is clipped at 0.8, but the target uses raw ratio)
    X["dti"] = df["loan_amnt"] / df["person_income"]
    # Interaction features that capture compound risk signals
    X["norm_in_x_dti"] = X["norm_in_degree"] * X["dti"]
    X["norm_in_x_nbr_def"] = X["norm_in_degree"] * X["neighbor_default_frac"]
    X["dti_x_cb_default"] = X["dti"] * X["cb_person_default_on_file"]
    X["int_rate_x_dti"] = X["loan_int_rate"] * X["dti"]
    X["degree_ratio"] = X["in_degree"] / (X["out_degree"] + 1)
    X["income_log"] = df["person_income"].apply(lambda v: __import__("math").log1p(v))
    y = df[TARGET].copy()

    return X, y, encoders


def train_model(
    data_dir: Path = Path("data"),
    seed: int = 42,
    test_size: float = 0.2,
    n_customers: int | None = None,
    xgb_params: dict | None = None,
) -> dict:
    """Train XGBoost classifier and return model artifacts.

    Returns dict with keys:
      - model: trained xgb.XGBClassifier
      - X_train, X_test, y_train, y_test: split DataFrames
      - encoders: dict of LabelEncoders
      - feature_names: list of feature column names
    """
    X, y, encoders = assemble_features(data_dir, seed=seed, n_customers=n_customers)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # scale_pos_weight: counteracts class imbalance (~80/20 split)
    # Without it, XGBoost optimizes logloss by predicting majority class,
    # giving ~1% recall — effectively blind to defaults.
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    spw = neg_count / pos_count if pos_count > 0 else 1.0

    params = {
        "n_estimators": 1500,
        "max_depth": 7,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 3,
        "gamma": 0.05,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": spw,
        "eval_metric": "logloss",
        "random_state": seed,
        "early_stopping_rounds": 50,
    }
    if xgb_params:
        params.update(xgb_params)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoders": encoders,
        "feature_names": list(X.columns),
    }

"""MLflow experiment tracking and model registry with 3.x alias-based promotion."""

import json
import pickle
import tempfile
from pathlib import Path

import mlflow
import shap
from matplotlib import pyplot as plt

from credit_domino.config import Settings
from credit_domino.modeling.evaluate import compute_shap_values, evaluate_model
from credit_domino.modeling.train import train_model

MODEL_NAME = "credit-domino-xgb"


def run_experiment(
    data_dir: Path = Path("data"),
    seed: int = 42,
    n_customers: int | None = None,
    promote: bool = True,
) -> str:
    """Full training run: train, evaluate, log to MLflow, register, optionally promote.

    Returns the MLflow run_id.
    """
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("credit-domino-v1")

    artifacts = train_model(data_dir, seed=seed, n_customers=n_customers)
    model = artifacts["model"]
    metrics, optimal_threshold = evaluate_model(model, artifacts["X_test"], artifacts["y_test"])

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(
            {
                "seed": seed,
                "n_customers": n_customers or "all",
                "n_features": len(artifacts["feature_names"]),
                "test_size": 0.2,
                "optimal_threshold": round(optimal_threshold, 4),
                **{
                    f"xgb_{k}": v
                    for k, v in model.get_params().items()
                    if k
                    in (
                        "n_estimators",
                        "max_depth",
                        "learning_rate",
                        "subsample",
                        "scale_pos_weight",
                        "min_child_weight",
                        "gamma",
                    )
                },
            }
        )

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log feature names
        mlflow.log_text(
            json.dumps(artifacts["feature_names"], indent=2),
            "feature_names.json",
        )

        # Log SHAP summary plot
        shap_values = compute_shap_values(model, artifacts["X_test"])
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, artifacts["X_test"], show=False)
            shap_path = Path(tmpdir) / "shap_summary.png"
            plt.savefig(shap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(shap_path))

            # Save label encoders for API serving (loaded from MLflow, not CSV)
            encoder_path = Path(tmpdir) / "label_encoders.pkl"
            with open(encoder_path, "wb") as f:
                pickle.dump(artifacts["encoders"], f)
            mlflow.log_artifact(str(encoder_path))

            # Save optimal decision threshold for API scoring
            threshold_path = Path(tmpdir) / "optimal_threshold.json"
            with open(threshold_path, "w") as f:
                json.dump({"optimal_threshold": optimal_threshold}, f)
            mlflow.log_artifact(str(threshold_path))

        # Log model with XGBoost flavor
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        run_id = run.info.run_id

    # Promote to champion alias (MLflow 3.x — no stages)
    if promote:
        client = mlflow.MlflowClient()
        # Get latest version for this model
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest_version = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(MODEL_NAME, "champion", latest_version.version)

    return run_id

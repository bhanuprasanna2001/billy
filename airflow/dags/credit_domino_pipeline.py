"""Credit Domino Pipeline DAG — orchestrates data → dbt → graph → train → evaluate → register."""

from datetime import datetime, timedelta

import pandas as pd

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator

default_args = {
    "owner": "credit-domino",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

_AUC_THRESHOLD = 0.72


def _copy_df_to_postgres(df, table_name, engine):
    """Bulk-load a DataFrame into Postgres using COPY FROM (10-50× faster than INSERT).

    Uses DROP CASCADE to clear dependent dbt views before recreating the table.
    dbt_build runs after this and recreates all views, so CASCADE is safe.
    """
    import io

    from sqlalchemy import text

    # DROP CASCADE removes dependent dbt views (stg_*) — dbt_build recreates them
    with engine.connect() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
        conn.commit()

    # Create empty table with correct schema, then bulk-load via COPY
    df.head(0).to_sql(table_name, engine, if_exists="fail", index=False)
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    quoted_cols = ", ".join(f'"{c}"' for c in df.columns)
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.copy_expert(
                f'COPY "{table_name}"({quoted_cols}) FROM STDIN WITH CSV',
                buf,
            )
        raw_conn.commit()
    finally:
        raw_conn.close()


_DATA_CACHE_DIR = "/tmp/credit_domino_cache"


def _load_and_prepare_data(**context):
    """Load Prosper/synthetic data, compute graph features, write to Postgres.

    Caches DataFrames to parquet so downstream tasks (Neo4j, features) avoid
    re-running the expensive synthetic target generation.
    """
    import os
    from pathlib import Path

    from sqlalchemy import create_engine

    from credit_domino.config import Settings
    from credit_domino.data.loaders import load_data
    from credit_domino.graph.features import compute_graph_features

    settings = Settings()
    data_dir = Path("/opt/airflow/data")
    df, edges = load_data(data_dir, seed=42)

    # Cache to parquet for downstream tasks
    os.makedirs(_DATA_CACHE_DIR, exist_ok=True)
    df.to_parquet(f"{_DATA_CACHE_DIR}/customers.parquet", index=False)
    edges.to_parquet(f"{_DATA_CACHE_DIR}/edges.parquet", index=False)

    # Compute graph features (NetworkX — fast for 89K nodes)
    graph_df = compute_graph_features(df, edges)

    engine = create_engine(settings.postgres_dsn)
    _copy_df_to_postgres(df, "customers", engine)
    _copy_df_to_postgres(edges, "relationships", engine)
    _copy_df_to_postgres(graph_df, "graph_features", engine)
    engine.dispose()

    context["ti"].xcom_push(key="n_customers", value=len(df))
    context["ti"].xcom_push(key="n_edges", value=len(edges))
    context["ti"].xcom_push(key="n_graph_features", value=len(graph_df))


def _load_cached_data():
    """Load cached DataFrames from parquet (written by _load_and_prepare_data)."""
    return (
        pd.read_parquet(f"{_DATA_CACHE_DIR}/customers.parquet"),
        pd.read_parquet(f"{_DATA_CACHE_DIR}/edges.parquet"),
    )


def _load_neo4j(**context):
    """Load customer nodes + relationship edges into Neo4j."""
    from credit_domino.graph.loader import get_neo4j_driver, load_graph_to_neo4j

    df, edges = _load_cached_data()

    driver = get_neo4j_driver()
    result = load_graph_to_neo4j(df, edges, driver)
    driver.close()
    context["ti"].xcom_push(key="neo4j_nodes", value=result["nodes_loaded"])
    context["ti"].xcom_push(key="neo4j_edges", value=result["edges_loaded"])


def _compute_neo4j_features(**context):
    """Compute graph features via Neo4j GDS (pagerank) + NetworkX (BFS), write to Postgres.

    Production path: Neo4j GDS provides scalable pagerank on 89K+ nodes.
    BFS distance uses NetworkX multi-source BFS (O(N+M)) because Cypher
    shortestPath is O(N×(N+M)) and unusable at this scale.
    """
    from sqlalchemy import create_engine

    from credit_domino.config import Settings
    from credit_domino.graph.features import compute_graph_features_neo4j
    from credit_domino.graph.loader import get_neo4j_driver

    settings = Settings()
    df, edges = _load_cached_data()

    driver = get_neo4j_driver(settings)
    graph_df = compute_graph_features_neo4j(driver, customers_df=df, edges_df=edges)
    driver.close()

    engine = create_engine(settings.postgres_dsn)
    _copy_df_to_postgres(graph_df, "graph_features", engine)
    engine.dispose()
    context["ti"].xcom_push(key="neo4j_graph_features", value=len(graph_df))


def _init_clickhouse_tables(**context):
    """Create ClickHouse scoring_events table and materialized view."""
    import clickhouse_connect

    from credit_domino.config import Settings

    settings = Settings()
    client = clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        database=settings.clickhouse_db,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
    )
    client.command("""
        CREATE TABLE IF NOT EXISTS scoring_events (
            scoring_event_id String,
            customer_id String,
            risk_score Float64,
            decision_band String,
            top_factors String,
            scored_at DateTime64(3)
        ) ENGINE = MergeTree()
        ORDER BY (scored_at, customer_id)
    """)
    client.command("""
        CREATE TABLE IF NOT EXISTS scoring_hourly (
            hour DateTime,
            decision_band String,
            event_count UInt64,
            risk_sum Float64
        ) ENGINE = SummingMergeTree()
        ORDER BY (hour, decision_band)
    """)
    client.command("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS scoring_hourly_mv TO scoring_hourly
        AS SELECT
            toStartOfHour(scored_at) AS hour,
            decision_band,
            count() AS event_count,
            sum(risk_score) AS risk_sum
        FROM scoring_events
        GROUP BY hour, decision_band
    """)
    client.close()


def _train_model(**context):
    """Train XGBoost and log to MLflow (without promoting to champion)."""
    from pathlib import Path

    from credit_domino.modeling.register import run_experiment

    data_dir = Path("/opt/airflow/data")
    run_id = run_experiment(data_dir, seed=42, promote=False)
    context["ti"].xcom_push(key="mlflow_run_id", value=run_id)


def _evaluate_model(**context):
    """Pull metrics from the training run for downstream quality gate."""
    import mlflow

    from credit_domino.config import Settings

    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    run_id = context["ti"].xcom_pull(task_ids="train_model", key="mlflow_run_id")
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    metrics = dict(run.data.metrics)

    context["ti"].xcom_push(key="roc_auc", value=metrics.get("roc_auc", 0.0))
    context["ti"].xcom_push(key="metrics", value=metrics)


def _check_model_quality(**context):
    """Branch: register model if AUC exceeds threshold, else skip."""
    auc = context["ti"].xcom_pull(task_ids="evaluate_model", key="roc_auc")
    if auc and auc >= _AUC_THRESHOLD:
        return "register_model_if_good"
    return "skip_registration"


def _register_model(**context):
    """Promote the trained model to champion alias and tell the API to reload."""
    import mlflow

    from credit_domino.config import Settings
    from credit_domino.modeling.register import MODEL_NAME

    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)

    # Tell the running API to hot-reload the newly promoted champion model
    import httpx

    try:
        resp = httpx.post("http://api:8000/reload-model", timeout=30)
        print(f"API model reload: {resp.status_code} — {resp.json()}")
    except Exception as exc:
        print(f"API reload request failed (non-fatal): {exc}")


def _skip_registration(**context):
    """No-op: model AUC below threshold."""
    auc = context["ti"].xcom_pull(task_ids="evaluate_model", key="roc_auc")
    print(f"Model AUC {auc:.4f} < {_AUC_THRESHOLD} — skipping registration.")


def _run_drift_check(**context):
    """Run Evidently drift detection and push results to API Prometheus gauges."""
    from pathlib import Path

    import httpx

    from credit_domino.monitoring.drift import run_drift_report

    csv_path = Path("/opt/airflow/data/credit_risk_dataset.csv")
    result = run_drift_report(csv_path)
    context["ti"].xcom_push(key="drift_detected", value=result["drift_detected"])
    context["ti"].xcom_push(key="n_drifted_columns", value=result["n_drifted_columns"])

    # Push pre-computed drift metrics to API → Prometheus gauges
    try:
        httpx.post(
            "http://api:8000/monitoring/drift",
            params={
                "n_drifted_columns": result["n_drifted_columns"],
                "drift_share": result["drift_share"],
            },
            timeout=10,
        )
    except Exception:
        print("API unreachable — drift metrics not pushed to Prometheus")

    print(
        f"Drift check: {result['n_drifted_columns']}/{len(result['drift_by_column'])} "
        f"columns drifted (share={result['drift_share']:.2%})"
    )


def _push_model_metrics(**context):
    """Push model ROC-AUC to API Prometheus gauges for Grafana."""
    import httpx

    auc = context["ti"].xcom_pull(task_ids="evaluate_model", key="roc_auc")
    if auc is None:
        return
    try:
        httpx.post(
            "http://api:8000/monitoring/model-metrics",
            params={"roc_auc": auc},
            timeout=10,
        )
    except Exception:
        print("API unreachable — model metrics not pushed to Prometheus")


def _notify(**context):
    """Log pipeline completion summary."""
    ti = context["ti"]
    n_customers = ti.xcom_pull(task_ids="load_and_prepare_data", key="n_customers")
    n_edges = ti.xcom_pull(task_ids="load_and_prepare_data", key="n_edges")
    metrics = ti.xcom_pull(task_ids="evaluate_model", key="metrics") or {}
    auc = metrics.get("roc_auc", "N/A")
    print(f"Pipeline complete: {n_customers} customers, {n_edges} edges, AUC={auc}")


with DAG(
    dag_id="credit_domino_pipeline",
    default_args=default_args,
    description="Credit Domino: data → dbt → graph → train → evaluate → register",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["credit-domino", "ml-pipeline"],
) as dag:
    load_and_prepare_data = PythonOperator(
        task_id="load_and_prepare_data",
        python_callable=_load_and_prepare_data,
    )

    dbt_build = BashOperator(
        task_id="dbt_build",
        bash_command="dbt build --project-dir /opt/airflow/dbt --profiles-dir /opt/airflow/dbt",
    )

    load_neo4j = PythonOperator(
        task_id="load_neo4j",
        python_callable=_load_neo4j,
        execution_timeout=timedelta(minutes=30),
    )

    compute_neo4j_features = PythonOperator(
        task_id="compute_neo4j_features",
        python_callable=_compute_neo4j_features,
        execution_timeout=timedelta(minutes=15),
    )

    init_clickhouse_tables = PythonOperator(
        task_id="init_clickhouse_tables",
        python_callable=_init_clickhouse_tables,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
    )

    check_model_quality = BranchPythonOperator(
        task_id="check_model_quality",
        python_callable=_check_model_quality,
    )

    register_model_if_good = PythonOperator(
        task_id="register_model_if_good",
        python_callable=_register_model,
    )

    skip_registration = PythonOperator(
        task_id="skip_registration",
        python_callable=_skip_registration,
    )

    notify = PythonOperator(
        task_id="notify",
        python_callable=_notify,
        trigger_rule="none_failed_min_one_success",
    )

    run_drift_check = PythonOperator(
        task_id="run_drift_check",
        python_callable=_run_drift_check,
    )

    push_model_metrics = PythonOperator(
        task_id="push_model_metrics",
        python_callable=_push_model_metrics,
    )

    # DAG structure:
    # load_and_prepare_data → [load_neo4j ‖ init_clickhouse_tables] (fan-out)
    #   load_neo4j → compute_neo4j_features (GDS pagerank → Postgres)
    #   [compute_neo4j_features, init_clickhouse] → dbt_build (views on final data)
    #   → train_model → evaluate_model → [check_model_quality, run_drift_check, push_model_metrics]
    #   check_model_quality → [register_model_if_good | skip_registration] → notify
    #
    # dbt_build runs AFTER compute_neo4j_features so that graph_features has
    # Neo4j-computed values before dbt creates views on top of it.
    # (Previous ordering had dbt_build before Neo4j, so DROP CASCADE destroyed views.)
    load_and_prepare_data >> [load_neo4j, init_clickhouse_tables]
    load_neo4j >> compute_neo4j_features
    [compute_neo4j_features, init_clickhouse_tables] >> dbt_build >> train_model >> evaluate_model
    evaluate_model >> [check_model_quality, run_drift_check, push_model_metrics]
    check_model_quality >> [register_model_if_good, skip_registration]
    [register_model_if_good, skip_registration] >> notify

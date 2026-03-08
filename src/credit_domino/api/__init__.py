"""FastAPI scoring API with ClickHouse + Postgres dual-write."""

import pickle
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import clickhouse_connect
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from credit_domino.config import Settings
from credit_domino.modeling.evaluate import get_top_factors
from credit_domino.modeling.register import MODEL_NAME
from credit_domino.modeling.train import _CAT_COLS, _GRAPH_FEATURES, _TABULAR_FEATURES

# ── Request / Response schemas ──────────────────────────────────────────────


class ScoreRequest(BaseModel):
    customer_id: str
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0)
    person_home_ownership: str
    person_emp_length: float = Field(ge=0)
    loan_intent: str
    loan_grade: str
    loan_amnt: float = Field(gt=0)
    loan_int_rate: float = Field(ge=0)
    loan_percent_income: float = Field(ge=0)
    cb_person_default_on_file: int = Field(ge=0, le=1)
    cb_person_cred_hist_length: int = Field(ge=0)
    crisis_exposure: float = 0.0
    # Graph features (pre-computed or defaults)
    degree: int = 0
    in_degree: int = 0
    out_degree: int = 0
    norm_in_degree: float = 0.0
    norm_out_degree: float = 0.0
    pagerank: float = 0.0
    distance_to_prior_default: int = -1
    clustering_coefficient: float = 0.0
    neighbor_default_frac: float = 0.0
    neighbor_default_frac_2hop: float = 0.0


class ScoreResponse(BaseModel):
    scoring_event_id: str
    customer_id: str
    risk_score: float
    decision_band: str
    top_factors: list[dict]
    scored_at: str


class HealthResponse(BaseModel):
    status: str
    model_version: str | None = None


# ── Application state ──────────────────────────────────────────────────────


class AppState:
    model = None
    encoders: dict = {}
    model_version: str | None = None
    optimal_threshold: float = 0.5
    pg_engine = None
    ch_client = None


state = AppState()


def _score_to_band(score: float, threshold: float = 0.5) -> str:
    """Map risk probability to decision band using model's optimal threshold."""
    low_bound = threshold * 0.5
    if score < low_bound:
        return "low"
    elif score < threshold:
        return "medium"
    return "high"


def _load_model(settings: Settings) -> None:
    """Load champion model and label encoders from MLflow registry."""
    import logging

    logger = logging.getLogger(__name__)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except Exception:
        logger.warning("No champion model registered yet — API will start without a model.")
        return

    state.model_version = mv.version

    model_uri = f"models:/{MODEL_NAME}@champion"
    state.model = mlflow.xgboost.load_model(model_uri)

    # Load label encoders from MLflow artifacts (saved during training)
    run_id = mv.run_id
    try:
        local_path = client.download_artifacts(run_id, "label_encoders.pkl")
        with open(local_path, "rb") as f:
            state.encoders = pickle.load(f)  # noqa: S301
    except Exception:
        # Fallback: re-fit from CSV if artifact not available (legacy models)
        from sklearn.preprocessing import LabelEncoder

        from credit_domino.data.loaders import load_data

        try:
            df, _ = load_data(Path("data"))
        except FileNotFoundError:
            df = None
        if df is not None:
            for col in _CAT_COLS:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                state.encoders[col] = le

    # Load optimal decision threshold
    try:
        threshold_path = client.download_artifacts(run_id, "optimal_threshold.json")
        import json as _json

        with open(threshold_path) as f:
            state.optimal_threshold = _json.load(f).get("optimal_threshold", 0.5)
        logger.info("Loaded optimal threshold: %.4f", state.optimal_threshold)
    except Exception:
        logger.warning("No optimal_threshold artifact found — using default 0.5")
        state.optimal_threshold = 0.5

    # Restore Prometheus MODEL_AUC gauge from MLflow run metrics so it
    # survives API container restarts without waiting for Airflow to re-push.
    try:
        run = client.get_run(run_id)
        auc = run.data.metrics.get("roc_auc")
        if auc is not None:
            MODEL_AUC.set(auc)
            logger.info("Restored MODEL_AUC gauge: %.4f", auc)
    except Exception:
        logger.warning("Could not restore MODEL_AUC from MLflow run")


def _init_clickhouse(settings: Settings) -> None:
    """Create ClickHouse scoring_events table and materialized view."""
    state.ch_client = clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        database=settings.clickhouse_db,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
    )
    state.ch_client.command("""
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
    # Pre-aggregated hourly analytics (SummingMergeTree for fast dashboard queries)
    state.ch_client.command("""
        CREATE TABLE IF NOT EXISTS scoring_hourly (
            hour DateTime,
            decision_band String,
            event_count UInt64,
            risk_sum Float64
        ) ENGINE = SummingMergeTree()
        ORDER BY (hour, decision_band)
    """)
    state.ch_client.command("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS scoring_hourly_mv TO scoring_hourly
        AS SELECT
            toStartOfHour(scored_at) AS hour,
            decision_band,
            count() AS event_count,
            sum(risk_score) AS risk_sum
        FROM scoring_events
        GROUP BY hour, decision_band
    """)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging

    logger = logging.getLogger(__name__)
    settings = Settings()
    _load_model(settings)
    state.pg_engine = create_engine(settings.postgres_dsn)
    try:
        _init_clickhouse(settings)
    except Exception:
        logger.warning("ClickHouse unavailable — scoring will write to Postgres only.")
    yield
    if state.pg_engine:
        state.pg_engine.dispose()
    if state.ch_client:
        state.ch_client.close()


app = FastAPI(
    title="Credit Domino Scoring API",
    version="0.1.0",
    lifespan=lifespan,
)

# Prometheus metrics instrumentation
Instrumentator().instrument(app).expose(app)

# Custom business metrics for Grafana dashboard
SCORE_COUNTER = Counter(
    "credit_domino_scores_total",
    "Total credit scoring events by decision band",
    ["decision_band"],
)
MODEL_AUC = Gauge(
    "credit_domino_model_auc",
    "Latest model ROC-AUC from training evaluation",
)
DRIFT_COLUMNS = Gauge(
    "credit_domino_drift_columns",
    "Number of features with detected distribution drift",
)
DRIFT_SHARE = Gauge(
    "credit_domino_drift_share",
    "Fraction of monitored features that drifted",
)


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if state.model is not None else "loading",
        model_version=state.model_version,
    )


@app.get("/ready")
async def ready():
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"ready": True}


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build feature vector
    features = {}
    for col in _TABULAR_FEATURES:
        val = getattr(req, col)
        if col in _CAT_COLS:
            le = state.encoders.get(col)
            if le is None:
                raise HTTPException(status_code=500, detail=f"Encoder missing for {col}")
            try:
                val = int(le.transform([str(val)])[0])
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown category for {col}",
                )
        features[col] = val

    for col in _GRAPH_FEATURES:
        features[col] = getattr(req, col)

    # Auto-lookup graph features from Postgres if caller sent defaults
    if (
        req.degree == 0
        and req.pagerank == 0.0
        and req.distance_to_prior_default == -1
        and req.clustering_coefficient == 0.0
    ):
        if state.pg_engine:
            with state.pg_engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT degree, in_degree, out_degree, "
                        "norm_in_degree, norm_out_degree, "
                        "pagerank, distance_to_prior_default, "
                        "clustering_coefficient, neighbor_default_frac, "
                        "neighbor_default_frac_2hop "
                        "FROM graph_features WHERE customer_id = :cid"
                    ),
                    {"cid": req.customer_id},
                ).fetchone()
            if row:
                features["degree"] = row[0]
                features["in_degree"] = row[1]
                features["out_degree"] = row[2]
                features["norm_in_degree"] = row[3]
                features["norm_out_degree"] = row[4]
                features["pagerank"] = row[5]
                features["distance_to_prior_default"] = row[6]
                features["clustering_coefficient"] = row[7]
                features["neighbor_default_frac"] = row[8]
                features["neighbor_default_frac_2hop"] = row[9]

    # Derived features (must match assemble_features in train.py)
    features["dti"] = req.loan_amnt / req.person_income
    features["norm_in_x_dti"] = features["norm_in_degree"] * features["dti"]
    features["norm_in_x_nbr_def"] = features["norm_in_degree"] * features["neighbor_default_frac"]
    features["dti_x_cb_default"] = features["dti"] * features["cb_person_default_on_file"]
    features["int_rate_x_dti"] = features["loan_int_rate"] * features["dti"]
    features["degree_ratio"] = features["in_degree"] / (features["out_degree"] + 1)
    import math as _math

    features["income_log"] = _math.log1p(req.person_income)

    X = pd.DataFrame([features])
    risk_score = float(state.model.predict_proba(X)[:, 1][0])
    decision_band = _score_to_band(risk_score, state.optimal_threshold)
    top_factors = get_top_factors(state.model, X, top_n=5)

    # Track business metric for Grafana
    SCORE_COUNTER.labels(decision_band=decision_band).inc()

    scoring_event_id = str(uuid.uuid4())
    scored_at = datetime.now(UTC)

    # Dual-write: Postgres + ClickHouse
    import json

    # Postgres write
    if state.pg_engine:
        with state.pg_engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO raw_scoring_events "
                    "(scoring_event_id, customer_id, risk_score, decision_band, "
                    "requested_amount, top_factors, scored_at) "
                    "VALUES (:eid, :cid, :rs, :db, :ra, :tf, :sa)"
                ),
                {
                    "eid": scoring_event_id,
                    "cid": req.customer_id,
                    "rs": risk_score,
                    "db": decision_band,
                    "ra": req.loan_amnt,
                    "tf": json.dumps(top_factors),
                    "sa": scored_at,
                },
            )
            conn.commit()

    # ClickHouse write (skip if unavailable)
    if state.ch_client is not None:
        state.ch_client.insert(
            "scoring_events",
            [
                [
                    scoring_event_id,
                    req.customer_id,
                    risk_score,
                    decision_band,
                    json.dumps(top_factors),
                    scored_at,
                ]
            ],
            column_names=[
                "scoring_event_id",
                "customer_id",
                "risk_score",
                "decision_band",
                "top_factors",
                "scored_at",
            ],
        )

    return ScoreResponse(
        scoring_event_id=scoring_event_id,
        customer_id=req.customer_id,
        risk_score=round(risk_score, 4),
        decision_band=decision_band,
        top_factors=top_factors,
        scored_at=scored_at.isoformat(),
    )


# ── Domino simulation endpoint ─────────────────────────────────────────────


class DominoRequest(BaseModel):
    trigger_customer_id: str
    initial_shock: float = Field(default=1.0, ge=0.0, le=1.0)
    decay: float = Field(default=0.6, ge=0.0, le=1.0)
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_hops: int = Field(default=5, ge=1, le=10)
    n_customers: int | None = Field(default=500, description="Limit graph size for speed")


class DominoResponse(BaseModel):
    trigger_customer_id: str
    total_affected: int
    total_fallen: int
    max_hop: int
    cascade: list[dict]


@app.post("/simulate-domino", response_model=DominoResponse)
async def simulate_domino_endpoint(req: DominoRequest):
    from credit_domino.simulation.domino import build_graph, simulate_domino

    data_dir = Path("data")
    G, _ = build_graph(data_dir, n_customers=req.n_customers)

    if req.trigger_customer_id not in G:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {req.trigger_customer_id} not found in graph",
        )

    cascade = simulate_domino(
        G,
        trigger_node=req.trigger_customer_id,
        initial_shock=req.initial_shock,
        decay=req.decay,
        threshold=req.threshold,
        max_hops=req.max_hops,
    )

    fallen = [c for c in cascade if c["fallen"]]
    return DominoResponse(
        trigger_customer_id=req.trigger_customer_id,
        total_affected=len(cascade),
        total_fallen=len(fallen),
        max_hop=max(c["hop"] for c in cascade) if cascade else 0,
        cascade=cascade,
    )


# ── Monitoring endpoints ───────────────────────────────────────────────────


@app.post("/monitoring/drift")
async def run_drift_check(
    n_drifted_columns: int | None = None,
    drift_share: float | None = None,
):
    """Update Prometheus drift gauges.

    When called with parameters (from Airflow): uses pre-computed values.
    When called without parameters (on-demand): runs Evidently locally.
    """
    if n_drifted_columns is not None and drift_share is not None:
        DRIFT_COLUMNS.set(n_drifted_columns)
        DRIFT_SHARE.set(drift_share)
        return {
            "n_drifted_columns": n_drifted_columns,
            "drift_share": drift_share,
        }

    from credit_domino.monitoring.drift import run_drift_report

    csv_path = Path("data/credit_risk_dataset.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    result = run_drift_report(csv_path)
    DRIFT_COLUMNS.set(result["n_drifted_columns"])
    DRIFT_SHARE.set(result["drift_share"])
    return result


@app.post("/monitoring/model-metrics")
async def update_model_metrics(roc_auc: float):
    """Update Prometheus gauge for model quality (called by Airflow)."""
    MODEL_AUC.set(roc_auc)
    return {"status": "ok", "roc_auc": roc_auc}


@app.post("/reload-model")
async def reload_model():
    """Hot-reload the champion model from MLflow (called by Airflow after promotion)."""
    settings = Settings()
    _load_model(settings)
    if state.model is None:
        raise HTTPException(status_code=503, detail="No champion model found in registry")
    return {"status": "ok", "model_version": state.model_version}

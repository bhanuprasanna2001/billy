-- Create separate databases for MLflow and Airflow to avoid schema conflicts
CREATE DATABASE mlflow;
CREATE DATABASE airflow;

-- Scoring events table (populated by FastAPI, consumed by dbt fct_scoring_log)
CREATE TABLE IF NOT EXISTS raw_scoring_events (
    scoring_event_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    decision_band TEXT NOT NULL,
    requested_amount DOUBLE PRECISION,
    top_factors JSONB,
    scored_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_scoring_events_customer ON raw_scoring_events (customer_id);
CREATE INDEX IF NOT EXISTS idx_scoring_events_scored_at ON raw_scoring_events (scored_at);

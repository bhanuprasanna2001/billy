#!/usr/bin/env python3
"""Prepare credit data: load (Prosper or synthetic), compute graph features, load Postgres."""

import sys
from pathlib import Path

from sqlalchemy import create_engine, text

from credit_domino.config import Settings
from credit_domino.data.loaders import load_data
from credit_domino.graph.features import compute_graph_features


def main() -> None:
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        sys.exit(1)

    print("Loading data (auto-detecting source)...")
    customers, relationships = load_data(data_dir, seed=42)
    print(f"  {len(customers):,} customers, {len(relationships):,} edges")

    print("Computing graph features (degree, pagerank, distance_to_prior_default)...")
    graph_features = compute_graph_features(customers, relationships)
    print(f"  {len(graph_features):,} graph feature rows")

    # Persist to data/raw/
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    customers.to_csv(raw_dir / "customers_cleaned.csv", index=False)
    relationships.to_csv(raw_dir / "relationships.csv", index=False)
    graph_features.to_csv(raw_dir / "graph_features.csv", index=False)
    print(f"  Saved CSVs to {raw_dir}/")

    # Load into Postgres
    settings = Settings()
    print(f"Loading into Postgres ({settings.postgres_host}:{settings.postgres_port})...")
    engine = create_engine(settings.postgres_dsn)

    # Drop dbt views that depend on source tables so to_sql(replace) can work
    with engine.connect() as conn:
        conn.execute(text("DROP VIEW IF EXISTS stg_customers CASCADE"))
        conn.execute(text("DROP VIEW IF EXISTS stg_relationships CASCADE"))
        conn.execute(text("DROP VIEW IF EXISTS stg_graph_features CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS fct_credit_features CASCADE"))
        conn.commit()

    customers.to_sql(
        "customers", engine, if_exists="replace", index=False, method="multi", chunksize=1000
    )
    relationships.to_sql(
        "relationships", engine, if_exists="replace", index=False, method="multi", chunksize=5000
    )
    graph_features.to_sql(
        "graph_features", engine, if_exists="replace", index=False, method="multi", chunksize=1000
    )

    with engine.connect() as conn:
        c = conn.execute(text("SELECT count(*) FROM customers")).scalar()
        r = conn.execute(text("SELECT count(*) FROM relationships")).scalar()
        g = conn.execute(text("SELECT count(*) FROM graph_features")).scalar()
    print(f"  Loaded: {c:,} customers, {r:,} relationships, {g:,} graph features")
    print("Done!")


if __name__ == "__main__":
    main()

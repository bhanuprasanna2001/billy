"""Data loading and synthetic relationship graph generation for Credit Domino."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_credit_data(csv_path: Path) -> pd.DataFrame:
    """Load Kaggle credit risk CSV, clean outliers, fill nulls, add identifiers.

    Transformations applied (in order):
      1. Add customer_id column (CUST_0, CUST_1, ...)
      2. Fill nulls: person_emp_length and loan_int_rate → median
      3. Cap outliers: person_age ≤ 100, person_income ≤ 500,000
      4. Encode cb_person_default_on_file: Y → 1, N → 0
      5. Add is_recent_default flag (True where loan_status == 1)
    """
    df = pd.read_csv(csv_path)

    df.insert(0, "customer_id", [f"CUST_{i}" for i in range(len(df))])

    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    df["person_age"] = df["person_age"].clip(upper=100)
    df["person_income"] = df["person_income"].clip(upper=500_000)

    df["cb_person_default_on_file"] = (df["cb_person_default_on_file"] == "Y").astype(int)

    df["is_recent_default"] = df["loan_status"] == 1

    # No crisis concept for synthetic data — zero-fill for schema consistency
    df["crisis_exposure"] = 0.0

    return df


def generate_relationship_graph(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic co-borrower/guarantor/employer relationship edges.

    Edge generation strategy:
      - Customers bucketed by (loan_intent, income_quartile)
      - 70% probability of same-bucket neighbor, 30% cross-bucket
      - 1–4 edges per node (weighted toward 2)
      - Edge type by attribute overlap:
          same home_ownership → co-borrower
          same loan_intent    → employer
          otherwise           → guarantor
      - No self-loops, no duplicate edges, deterministic given seed
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    cust_ids = df["customer_id"].values
    intents = df["loan_intent"].values
    homes = df["person_home_ownership"].values

    # Bucket by (loan_intent, income_quartile) for similarity-based sampling
    income_q = pd.qcut(df["person_income"], q=4, labels=False, duplicates="drop")
    income_vals = income_q.fillna(0).astype(int).values

    buckets: dict[tuple, list[int]] = {}
    bucket_of: list[tuple] = []
    for i in range(n):
        key = (str(intents[i]), int(income_vals[i]))
        buckets.setdefault(key, []).append(i)
        bucket_of.append(key)

    # Sample edges: 1–4 per node, biased toward same-bucket neighbors
    edges: set[tuple[int, int]] = set()
    num_per_node = rng.choice([1, 2, 3, 4], size=n, p=[0.2, 0.4, 0.3, 0.1])

    for i in range(n):
        same = buckets[bucket_of[i]]
        for _ in range(num_per_node[i]):
            if rng.random() < 0.7 and len(same) > 1:
                j = same[rng.integers(len(same))]
                if j == i:
                    j = same[rng.integers(len(same))]
                if j == i:
                    continue
            else:
                j = rng.integers(n)
                if j == i:
                    j = (j + 1) % n
            edges.add((min(i, j), max(i, j)))

    # Assign edge types by attribute similarity
    rows = []
    for src, dst in sorted(edges):
        if homes[src] == homes[dst]:
            edge_type = "co-borrower"
        elif intents[src] == intents[dst]:
            edge_type = "employer"
        else:
            edge_type = "guarantor"
        rows.append(
            {
                "src_customer_id": cust_ids[src],
                "dst_customer_id": cust_ids[dst],
                "edge_type": edge_type,
            }
        )

    return pd.DataFrame(rows)


def load_data(
    data_dir: Path = Path("data"),
    seed: int = 42,
    n_customers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load customer data + relationship edges, auto-detecting the data source.

    Checks for the Prosper P2P lending dataset first (real graph topology).
    Falls back to the synthetic credit_risk_dataset.csv if Prosper is absent.

    Returns (customers_df, edges_df) with a consistent schema regardless of source.
    """
    prosper_dir = data_dir / "prosper_user_loans_2016"
    if prosper_dir.exists() and (prosper_dir / "edges.csv").exists():
        from credit_domino.data.prosper_loader import load_prosper_data

        return load_prosper_data(prosper_dir, seed=seed, n_customers=n_customers)

    csv_path = data_dir / "credit_risk_dataset.csv"
    df = load_credit_data(csv_path)
    if n_customers is not None:
        df = df.head(n_customers).reset_index(drop=True)
    edges = generate_relationship_graph(df, seed=seed)
    return df, edges

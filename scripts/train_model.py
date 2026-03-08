#!/usr/bin/env python
"""CLI: train model, log to MLflow, register with champion alias."""

import argparse
from pathlib import Path

from credit_domino.modeling.register import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit-domino XGBoost model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory (auto-detects Prosper or synthetic)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-customers",
        type=int,
        default=None,
        help="Limit to first N customers (for testing)",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Skip promoting to champion alias",
    )
    args = parser.parse_args()

    run_id = run_experiment(
        data_dir=args.data_dir,
        seed=args.seed,
        n_customers=args.n_customers,
        promote=not args.no_promote,
    )
    print(f"Training complete. MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()

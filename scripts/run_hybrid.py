#!/usr/bin/env python3
"""Run the improved spectral hybrid pipeline and show results."""

from pathlib import Path

from credit_domino.modeling.gnn import train_hybrid


def main():
    print("=" * 70)
    print("IMPROVED HYBRID PIPELINE: Spectral Embeddings + XGBoost")
    print("=" * 70)

    result = train_hybrid(data_dir=Path("data"), method="spectral", seed=42)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    h = result["hybrid_metrics"]
    v = result["vanilla_metrics"]
    print(f"  Hybrid AUC:       {h['roc_auc']:.4f}")
    print(f"  Hybrid Accuracy:  {h['accuracy']:.4f}")
    print(f"  Hybrid Precision: {h['precision']:.4f}")
    print(f"  Hybrid Recall:    {h['recall']:.4f}")
    print(f"  Hybrid F1:        {h['f1']:.4f}")
    print()
    print(f"  Vanilla AUC:      {v['roc_auc']:.4f}")
    print(f"  Vanilla Accuracy: {v['accuracy']:.4f}")
    print(f"  Vanilla Precision:{v['precision']:.4f}")
    print(f"  Vanilla Recall:   {v['recall']:.4f}")
    print(f"  Vanilla F1:       {v['f1']:.4f}")
    print()
    print(f"  AUC boost from graph embeddings: {h['roc_auc'] - v['roc_auc']:+.4f}")


if __name__ == "__main__":
    main()

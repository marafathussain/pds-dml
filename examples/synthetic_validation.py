"""
Run CIA and baseline methods on synthetic data with known causal structure.

Usage:
    python examples/synthetic_validation.py
    python examples/synthetic_validation.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_inference import (
    ConditionalInterventionAnalysis,
    DoubleMachineLearning,
    generate_synthetic_data,
)
from causal_inference.baselines import (
    ancova_adjusted_ranking,
    conditional_permutation_importance_ranking,
    correlation_importance,
    evaluate_ranking,
    linear_dml_importance_ranking,
    permutation_importance_ranking,
)

DEFAULT_OUTPUT = ROOT / "outputs" / "synthetic"


def run_synthetic_validation(output_dir: Path, quick: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = 200 if quick else 500
    n_features = 12
    n_causal = 3
    n_samples = 15 if quick else 30
    n_folds = 3 if quick else 5

    print("\n" + "=" * 72)
    print("  SYNTHETIC VALIDATION")
    print("=" * 72)
    print(f"  Subjects: {n_subjects}, features: {n_features}, true causal: {n_causal}")

    X, y, true_causal = generate_synthetic_data(
        n_subjects=n_subjects,
        n_features=n_features,
        n_causal=n_causal,
        noise_level=0.5,
        task_type="regression",
        random_state=42,
    )
    print(f"  Ground truth causal features: {true_causal}")

    cia = ConditionalInterventionAnalysis(
        task_type="regression",
        n_samples=n_samples,
        n_folds=n_folds,
        model_name="consensus_fast" if n_subjects >= 300 else "consensus",
        conditional_model="consensus_fast" if n_subjects >= 300 else "consensus",
        max_subjects=150 if quick else None,
        random_state=42,
    )

    print("\n  Running CIA...")
    t0 = time.time()
    cia.fit(X, y)
    cia_elapsed = time.time() - t0
    cia_results = cia.get_results()
    cia_results.to_csv(output_dir / "synthetic_cia_results.csv", index=False)

    k = len(true_causal)
    cia_top_k = cia_results.sort_values("causal_effect", key=abs, ascending=False)["feature"].tolist()[:k]
    cia_top_metrics = evaluate_ranking(cia_top_k, true_causal)
    print(f"  CIA elapsed: {cia_elapsed:.1f}s")
    print(f"  CIA top-{k}: precision={cia_top_metrics['precision']:.2f}, recall={cia_top_metrics['recall']:.2f}")

    print("\n  Running DML (ridge nuisances, pooled residuals)...")
    t0 = time.time()
    dml = DoubleMachineLearning(
        n_folds=n_folds,
        nuisance_model="ridge",
        n_bootstrap=50 if quick else 200,
        random_state=42,
    )
    dml.fit(X, y)
    dml_elapsed = time.time() - t0
    dml_results = dml.get_results()
    dml_results.to_csv(output_dir / "synthetic_dml_results.csv", index=False)

    dml_top_k = dml_results.sort_values("causal_effect", key=abs, ascending=False)["feature"].tolist()[:k]
    dml_top_metrics = evaluate_ranking(dml_top_k, true_causal)
    print(f"  DML elapsed: {dml_elapsed:.1f}s")
    print(f"  DML top-{k}: precision={dml_top_metrics['precision']:.2f}, recall={dml_top_metrics['recall']:.2f}")

    print("\n  Running baselines...")
    baselines = [
        ("correlation", correlation_importance(X, y)),
        ("ANCOVA-style regression", ancova_adjusted_ranking(X, y)),
        ("permutation_importance", permutation_importance_ranking(X, y, n_repeats=5 if quick else 10)),
        (
            "conditional_permutation_importance",
            conditional_permutation_importance_ranking(X, y, n_repeats=5 if quick else 10),
        ),
        ("linear_dml", linear_dml_importance_ranking(X, y, n_folds=n_folds)),
    ]

    rows = [
        {
            "method": "CIA (top-3 by |effect|)",
            "precision": cia_top_metrics["precision"],
            "recall": cia_top_metrics["recall"],
            "top_k_features": cia_top_k,
        },
        {
            "method": "DML ridge (top-3 by |theta|)",
            "precision": dml_top_metrics["precision"],
            "recall": dml_top_metrics["recall"],
            "top_k_features": dml_top_k,
        },
    ]

    for name, ranked in baselines:
        top_k = ranked["feature"].tolist()[:k]
        m = evaluate_ranking(top_k, true_causal)
        rows.append({"method": name, "precision": m["precision"], "recall": m["recall"], "top_k_features": top_k})
        print(f"  {name:32s} precision={m['precision']:.2f}, recall={m['recall']:.2f}")

    pd.DataFrame(rows).to_csv(output_dir / "synthetic_method_comparison.csv", index=False)

    summary = {
        "n_subjects": n_subjects,
        "n_features": n_features,
        "n_causal": n_causal,
        "true_causal": true_causal,
        "cia_elapsed_seconds": cia_elapsed,
        "dml_elapsed_seconds": dml_elapsed,
        "baseline_comparison": rows,
    }
    with open(output_dir / "synthetic_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved results to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Synthetic CIA vs baseline comparison")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_synthetic_validation(args.output_dir, quick=args.quick)


if __name__ == "__main__":
    main()

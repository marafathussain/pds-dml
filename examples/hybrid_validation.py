"""
Benchmark the CIA-DML two-stage hybrid on synthetic data.

Usage:
    python examples/hybrid_validation.py
    python examples/hybrid_validation.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_inference import CIA_DML_Pipeline, generate_synthetic_data
from causal_inference.baselines import evaluate_ranking

DEFAULT_OUTPUT = ROOT / "outputs" / "hybrid"


def run_hybrid_validation(output_dir: Path, quick: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = 200 if quick else 500
    n_features = 12
    n_causal = 3
    n_samples = 15 if quick else 30
    n_folds = 3 if quick else 5

    print("\n" + "=" * 72)
    print("  CIA-DML HYBRID VALIDATION")
    print("=" * 72)

    X, y, true_causal = generate_synthetic_data(
        n_subjects=n_subjects,
        n_features=n_features,
        n_causal=n_causal,
        noise_level=0.5,
        task_type="regression",
        random_state=42,
    )
    print(f"  Subjects: {n_subjects}, features: {n_features}, true causal: {true_causal}")

    pipeline = CIA_DML_Pipeline(
        screen_alpha=0.05,
        screen_top_k=n_causal,
        confirm_alpha=0.05,
        fallback_top_k=n_causal,
        n_samples=n_samples,
        n_folds=n_folds,
        model_name="consensus_fast" if n_subjects >= 300 else "consensus",
        conditional_model="consensus_fast" if n_subjects >= 300 else "consensus",
        max_subjects=150 if quick else None,
        nuisance_model="ridge",
        n_bootstrap=50 if quick else 200,
        random_state=42,
    )

    t0 = time.time()
    pipeline.fit(X, y)
    elapsed = time.time() - t0

    results = pipeline.get_results()
    results.to_csv(output_dir / "hybrid_results.csv", index=False)
    pipeline.summary()

    k = len(true_causal)
    confirmed = pipeline.get_confirmed_features()
    screened = pipeline.get_screened_features()

    confirmed_metrics = evaluate_ranking(confirmed, true_causal)
    screened_metrics = evaluate_ranking(screened, true_causal, top_k=k)

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Screened top-{k}: P={screened_metrics['precision']:.2f}, R={screened_metrics['recall']:.2f}")
    print(f"  Confirmed set:     P={confirmed_metrics['precision']:.2f}, R={confirmed_metrics['recall']:.2f}")
    print(f"  True causal recovered: {confirmed_metrics['true_positives']}")

    summary = {
        "n_subjects": n_subjects,
        "true_causal": true_causal,
        "screened_features": screened,
        "confirmed_features": confirmed,
        "screened_metrics": screened_metrics,
        "confirmed_metrics": confirmed_metrics,
        "elapsed_seconds": elapsed,
    }
    with open(output_dir / "hybrid_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="CIA-DML hybrid synthetic validation")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_hybrid_validation(args.output_dir, quick=args.quick)


if __name__ == "__main__":
    main()

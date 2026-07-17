"""
ABCD analysis with baseline Chernozhukov-style DML
(ridge nuisances, all-except-i adjustment, residual SE).

Matched to PDS-DML for fair comparison:
  410 predictors -> top-K by correlation -> baseline DML per candidate

Usage:
    python examples/abcd_chernozhukov_dml.py --three-outcomes
    python examples/abcd_chernozhukov_dml.py --three-outcomes --top-k 50 --n-folds 3
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

from causal_inference.abcd_data import (
    ABCD_OUTCOME_LABELS,
    ABCD_THREE_OUTCOMES,
    prepare_all_features,
    select_top_correlated_features,
)
from causal_inference.chernozhukov_dml import (
    apply_study_bonferroni,
    estimate_vanilla_dml,
)
DEFAULT_CSV = ROOT / "data" / "ABCD_ML_ready_reduced_final_with_parent_ed.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "abcd_chernozhukov_dml"


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")[:120]


def run_vanilla_on_candidates(
    X_pool: pd.DataFrame,
    y: pd.Series,
    candidate_features: list[str],
    n_folds: int = 3,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    pool_names = list(X_pool.columns)
    X_arr = X_pool.values.astype(float)
    y_arr = y.values.astype(float)

    rows = []
    for i, feat in enumerate(candidate_features):
        if feat not in pool_names:
            continue
        if verbose:
            print(f"  [{i + 1}/{len(candidate_features)}] {feat[:70]}...")
        rows.append(
            estimate_vanilla_dml(
                X_arr, y_arr, pool_names, feat, n_folds=n_folds, random_state=random_state
            )
        )

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("p_value", ascending=True, na_position="last").reset_index(drop=True)
    return df


def run_outcome(
    outcome: str,
    csv_path: Path,
    output_dir: Path,
    top_k: int,
    n_folds: int,
) -> pd.DataFrame:
    print(f"\n{'=' * 72}")
    print(f"  {ABCD_OUTCOME_LABELS.get(outcome, outcome)}")
    print(f"  Outcome column: {outcome}")
    print(f"{'=' * 72}")

    df = pd.read_csv(csv_path)
    all_features, prep = prepare_all_features(df.columns.tolist(), outcome)
    candidates, ranking, prefilter = select_top_correlated_features(
        df, outcome, all_features, top_k=top_k
    )

    use_cols = sorted(set(all_features) | {outcome})
    subset = df[use_cols].dropna()
    y = subset[outcome]
    X_pool = subset[all_features]

    print(f"  Subjects: {len(y)}")
    print(f"  Full pool: {len(all_features)} predictors")
    print(f"  DML candidates (top-{top_k} correlation): {len(candidates)}")
    print("  Method: vanilla ridge DML, all-except-i adjustment")

    t0 = time.time()
    results = run_vanilla_on_candidates(
        X_pool, y, candidates, n_folds=n_folds, random_state=42, verbose=True
    )
    elapsed = time.time() - t0

    out_dir = output_dir / _safe_name(outcome)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "chernozhukov_dml_results.csv", index=False)
    ranking.to_csv(out_dir / "correlation_ranking.csv", index=False)

    meta = {
        "outcome": outcome,
        "outcome_label": ABCD_OUTCOME_LABELS.get(outcome, outcome),
        "method": "chernozhukov_dml",
        "n_subjects": len(y),
        "n_pool": len(all_features),
        "top_k": top_k,
        "n_candidates": len(candidates),
        "n_folds": n_folds,
        "elapsed_seconds": elapsed,
        "preparation": prep,
        "prefilter": prefilter,
        "n_significant_05": int(results["significant_05"].sum()) if len(results) else 0,
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Done in {elapsed / 60:.1f} min")
    print(f"  Nominal p<0.05: {meta['n_significant_05']}/{len(results)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="ABCD vanilla DML (matched to PDS-DML)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--three-outcomes", action="store_true")
    parser.add_argument("--outcome", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--n-folds", type=int, default=3)
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}")
        sys.exit(1)

    outcomes = ABCD_THREE_OUTCOMES if (args.three_outcomes or not args.outcome) else [args.outcome]

    print("=" * 72)
    print("  VANILLA DML — ABCD")
    print("=" * 72)
    print(f"  Outcomes: {len(outcomes)}")
    print(f"  top-K prefilter: {args.top_k}")
    print(f"  Cross-fitting folds: {args.n_folds}")

    results_by_outcome = {}
    for outcome in outcomes:
        results_by_outcome[outcome] = run_outcome(
            outcome, args.csv, args.output_dir, args.top_k, args.n_folds
        )

    bonf = apply_study_bonferroni(results_by_outcome)
    for outcome, df in bonf["results"].items():
        out_dir = args.output_dir / _safe_name(outcome)
        df.to_csv(out_dir / "chernozhukov_dml_results.csv", index=False)
        n_bonf = int(df["significant_global_bonferroni"].sum())
        print(f"\n  {ABCD_OUTCOME_LABELS.get(outcome, outcome)}: Bonferroni-significant = {n_bonf}")

    summary = {
        "method": "chernozhukov_dml",
        "outcomes": outcomes,
        "top_k": args.top_k,
        "n_folds": args.n_folds,
        "bonferroni": {
            "n_tests_total": bonf["n_tests_total"],
            "threshold": bonf["bonferroni_threshold"],
            "alpha": bonf["alpha"],
        },
        "per_outcome": {},
    }
    for outcome, df in bonf["results"].items():
        sig = df[df["significant_global_bonferroni"]]
        summary["per_outcome"][outcome] = {
            "label": ABCD_OUTCOME_LABELS.get(outcome, outcome),
            "n_tested": len(df),
            "n_nominal_05": int(df["significant_05"].sum()),
            "n_bonferroni": int(sig.shape[0]),
            "bonferroni_features": sig["feature"].tolist(),
        }

    summary_path = args.output_dir / "study_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Global Bonferroni: {bonf['n_tests_total']} tests, threshold p < {bonf['bonferroni_threshold']:.6f}")
    print(f"  Summary -> {summary_path}")


if __name__ == "__main__":
    main()

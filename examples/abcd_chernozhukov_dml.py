"""
ABCD analysis with Chernozhukov PLR-DML (6 improvements).

Pipeline per outcome:
  410 predictors -> top-K by correlation -> post-double-selection PLR-DML
  -> influence-function SE -> global Bonferroni -> sensitivity metrics

Usage:
    python examples/abcd_chernozhukov_dml.py --mirasol-three
    python examples/abcd_chernozhukov_dml.py --mirasol-three --top-k 50 --n-folds 3
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
    prepare_all_cia_features,
    select_top_correlated_features,
)
from causal_inference.chernozhukov_dml import (
    ESTIMAND_DESCRIPTION,
    apply_study_bonferroni,
    run_plr_dml_on_candidates,
)
from examples.abcd_hybrid_analysis import MIRASOL_THREE_OUTCOMES, MIRASOL_OUTCOME_LABELS

DEFAULT_CSV = ROOT / "data" / "ABCD_ML_ready_reduced_final_with_parent_ed.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "abcd_chernozhukov_dml"


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")[:120]


def run_outcome(
    outcome: str,
    csv_path: Path,
    output_dir: Path,
    top_k: int,
    n_folds: int,
) -> pd.DataFrame:
    print(f"\n{'=' * 72}")
    print(f"  {MIRASOL_OUTCOME_LABELS.get(outcome, outcome)}")
    print(f"  Outcome column: {outcome}")
    print(f"{'=' * 72}")

    df = pd.read_csv(csv_path)
    all_features, prep = prepare_all_cia_features(df.columns.tolist(), outcome)
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
    print(f"  Estimand: {ESTIMAND_DESCRIPTION[:80]}...")

    t0 = time.time()
    results = run_plr_dml_on_candidates(
        X_pool, y, candidates, n_folds=n_folds, random_state=42, verbose=True
    )
    elapsed = time.time() - t0

    out_dir = output_dir / _safe_name(outcome)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "chernozhukov_dml_results.csv", index=False)
    ranking.to_csv(out_dir / "correlation_ranking.csv", index=False)

    meta = {
        "outcome": outcome,
        "outcome_label": MIRASOL_OUTCOME_LABELS.get(outcome, outcome),
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
    parser = argparse.ArgumentParser(description="ABCD Chernozhukov PLR-DML")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mirasol-three", action="store_true")
    parser.add_argument("--outcome", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=50, help="Correlation prefilter (default 50 for speed)")
    parser.add_argument("--n-folds", type=int, default=3)
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}")
        sys.exit(1)

    if args.mirasol_three:
        outcomes = MIRASOL_THREE_OUTCOMES
    elif args.outcome:
        outcomes = [args.outcome]
    else:
        outcomes = MIRASOL_THREE_OUTCOMES

    print("=" * 72)
    print("  CHERNOZHUKOV PLR-DML — ABCD")
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
        print(f"\n  {MIRASOL_OUTCOME_LABELS.get(outcome, outcome)}: Bonferroni-significant = {n_bonf}")

    summary = {
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
            "label": MIRASOL_OUTCOME_LABELS.get(outcome, outcome),
            "n_tested": len(df),
            "n_nominal_05": int(df["significant_05"].sum()),
            "n_bonferroni": int(sig.shape[0]),
            "bonferroni_features": sig["feature"].tolist(),
            "top_theta": df.nlargest(5, "theta", keep="all")[["feature", "theta", "p_value"]].to_dict("records"),
        }

    summary_path = args.output_dir / "study_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Global Bonferroni: {bonf['n_tests_total']} tests, threshold p < {bonf['bonferroni_threshold']:.6f}")
    print(f"  Summary -> {summary_path}")


if __name__ == "__main__":
    main()

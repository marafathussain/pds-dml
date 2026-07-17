"""
ABCD cohort analysis with the CIA-DML hybrid pipeline.

Modes
-----
fibe-full-dml (recommended)
    CIA screens FIBE-selected features (~10-17); DML confirms using the full
    predictor pool (~410) for nuisance adjustment (mentor design).

top100-full-dml
    CIA screens top-100 correlation features; DML uses full pool for nuisances.

fibe-fibe-dml
    Both stages on FIBE features only (faster baseline comparison).

Usage
-----
    # MIRASOL: 3 outcomes, top-100 correlation -> CIA -> DML (full 410 nuisances)
    python examples/abcd_hybrid_analysis.py --mirasol-three
    python examples/abcd_hybrid_analysis.py --mirasol-three --skip-existing

    python examples/abcd_hybrid_analysis.py --outcome "Cognition Total Composite Score Age-Corrected Standard Score"
    python examples/abcd_hybrid_analysis.py --all-outcomes --mode fibe-full-dml
    python examples/abcd_hybrid_analysis.py --outcome "Cognition Fluid Composite Age-Corrected Standard Score" --quick
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

from causal_inference import CIA_DML_Pipeline
from causal_inference.abcd_data import (
    PARENT_EDUCATION_COLUMN,
    list_outcome_json_files,
    load_fibe_selected_features,
    outcome_name_from_json,
    prepare_all_cia_features,
    prepare_cia_features,
    select_top_correlated_features,
)

DEFAULT_CSV = ROOT / "data" / "ABCD_ML_ready_reduced_final_with_parent_ed.csv"
DEFAULT_FEATURES_DIR = ROOT / "data" / "final_features"
DEFAULT_OUTPUT = ROOT / "outputs" / "abcd_hybrid"

# MIRASOL paper: 3 targeted outcomes (IQ proxy + 2 executive function measures)
MIRASOL_THREE_OUTCOMES = [
    "Cognition Total Composite Score Age-Corrected Standard Score",
    "NIH Toolbox Flanker Inhibitory Control and Attention Test Ages 8-11 v2.0 Age-Corrected Standard Score",
    "NIH Toolbox Dimensional Change Card Sort Test Ages 8-11 v2.0 Age-Corrected Standard Score",
]
MIRASOL_OUTCOME_LABELS = {
    MIRASOL_THREE_OUTCOMES[0]: "Total Composite (IQ proxy)",
    MIRASOL_THREE_OUTCOMES[1]: "Flanker (EF: inhibition)",
    MIRASOL_THREE_OUTCOMES[2]: "DS Card Sort (EF: flexibility)",
}


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")[:120]


def _load_cohort(csv_path: Path, outcome: str, cia_features: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load aligned (X_cia, y, X_full) after dropna on outcome + all predictors."""
    df = pd.read_csv(csv_path)
    all_features, _ = prepare_all_cia_features(df.columns.tolist(), outcome)

    use_cols = sorted(set(cia_features) | set(all_features))
    subset = df[[outcome] + use_cols].dropna()
    y = subset[outcome]
    X_cia = subset[cia_features]
    X_full = subset[all_features]
    return X_cia, y, X_full


def run_one_outcome(
    outcome: str,
    csv_path: Path,
    features_dir: Path,
    output_dir: Path,
    mode: str = "fibe-full-dml",
    top_k: int = 100,
    quick: bool = False,
) -> dict:
    df_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    all_features, full_prep = prepare_all_cia_features(df_cols, outcome)

    correlation_ranking = None
    if mode.startswith("fibe"):
        json_path = features_dir / f"{outcome}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"FIBE JSON not found: {json_path}")
        fibe_features = load_fibe_selected_features(json_path)
        cia_features, cia_prep = prepare_cia_features(fibe_features, df_cols)
        cia_label = "FIBE-selected"
    elif mode.startswith("top"):
        df = pd.read_csv(csv_path)
        cia_features, correlation_ranking, prefilter = select_top_correlated_features(
            df, outcome, all_features, top_k=top_k
        )
        cia_prep = {"prefilter": prefilter, "top_k": top_k}
        cia_label = f"top-{top_k} correlation"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    use_full_dml = mode.endswith("full-dml")

    print(f"\n  Outcome: {outcome}")
    print(f"  Mode: {mode}")
    print(f"  CIA features ({cia_label}): {len(cia_features)}")
    print(f"  Full predictor pool: {full_prep['n_final_features']}")
    print(f"  DML nuisances: {'full pool' if use_full_dml else 'CIA subset only'}")

    X_cia, y, X_full = _load_cohort(csv_path, outcome, cia_features)
    print(f"  Subjects after dropna: {len(y)}")

    n_samples = 10 if quick else 20
    n_folds = 3 if quick else 5
    max_subjects = 100 if quick else 300
    n_bootstrap = 50 if quick else 100

    # Top-100 mode: screen by |CIA effect| rank (not p-value alone)
    screen_top_k = 20 if mode.startswith("top") else min(10, len(cia_features))
    fallback_top_k = 20 if mode.startswith("top") else min(5, len(cia_features))

    pipeline = CIA_DML_Pipeline(
        screen_alpha=0.05,
        screen_top_k=screen_top_k,
        confirm_alpha=0.05,
        fallback_top_k=fallback_top_k,
        n_samples=n_samples,
        n_folds=n_folds,
        model_name="consensus_fast",
        conditional_model="consensus_fast",
        max_subjects=max_subjects,
        nuisance_model="ridge",
        n_bootstrap=n_bootstrap,
        random_state=42,
    )

    t0 = time.time()
    if use_full_dml:
        pipeline.fit(X_cia, y, dml_X=X_full, dml_feature_names=list(X_full.columns))
    else:
        pipeline.fit(X_cia, y)
    elapsed = time.time() - t0

    results = pipeline.get_results()
    out_subdir = output_dir / mode / _safe_name(outcome)
    out_subdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_subdir / "hybrid_results.csv", index=False)
    if correlation_ranking is not None:
        correlation_ranking.to_csv(out_subdir / "correlation_ranking.csv", index=False)

    label = MIRASOL_OUTCOME_LABELS.get(outcome, outcome)
    meta = {
        "outcome": outcome,
        "outcome_label": label,
        "mode": mode,
        "n_subjects": len(y),
        "n_cia_features": len(cia_features),
        "n_full_predictors": len(X_full.columns),
        "dml_full_pool": use_full_dml,
        "cia_preparation": cia_prep,
        "full_preparation": full_prep,
        "screened_features": pipeline.get_screened_features(),
        "confirmed_features": pipeline.get_confirmed_features(),
        "elapsed_seconds": elapsed,
    }
    with open(out_subdir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Elapsed: {elapsed/60:.1f} min")
    print(f"  Screened: {len(meta['screened_features'])}, Confirmed: {len(meta['confirmed_features'])}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="ABCD CIA-DML hybrid analysis")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outcome", type=str, default=None)
    parser.add_argument("--all-outcomes", action="store_true")
    parser.add_argument(
        "--mirasol-three",
        action="store_true",
        help="Run 3 MIRASOL outcomes (Total Composite, Flanker, DS Card Sort)",
    )
    parser.add_argument(
        "--mode",
        choices=["fibe-full-dml", "fibe-fibe-dml", "top100-full-dml"],
        default="top100-full-dml",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ABCD CSV not found: {args.csv}")
        print("Place data under data/ or pass --csv.")
        sys.exit(1)

    if args.mirasol_three:
        outcomes = MIRASOL_THREE_OUTCOMES
        if args.mode == "fibe-full-dml":
            args.mode = "top100-full-dml"
    elif args.all_outcomes:
        outcomes = [outcome_name_from_json(p) for p in list_outcome_json_files(args.features_dir)]
    elif args.outcome:
        outcomes = [args.outcome]
    else:
        outcomes = [MIRASOL_THREE_OUTCOMES[0]]

    print("=" * 72)
    print("  ABCD CIA-DML HYBRID")
    print("=" * 72)
    print(f"  CSV: {args.csv}")
    print(f"  Mode: {args.mode}")
    print(f"  Outcomes: {len(outcomes)}")

    summaries = []
    for outcome in outcomes:
        out_csv = args.output_dir / args.mode / _safe_name(outcome) / "hybrid_results.csv"
        if args.skip_existing and out_csv.exists():
            print(f"\n  Skipping (exists): {outcome}")
            continue
        try:
            summaries.append(
                run_one_outcome(
                    outcome,
                    args.csv,
                    args.features_dir,
                    args.output_dir,
                    mode=args.mode,
                    top_k=args.top_k,
                    quick=args.quick,
                )
            )
        except Exception as exc:
            print(f"  ERROR: {outcome}: {exc}", file=sys.stderr)
            summaries.append({"outcome": outcome, "error": str(exc)})

    summary_path = args.output_dir / f"{args.mode}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n  Summary -> {summary_path}")


if __name__ == "__main__":
    main()

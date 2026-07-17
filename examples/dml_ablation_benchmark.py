"""
Six-improvement DML ablation benchmark on purpose-built synthetic data.

Each scenario isolates one Chernozhukov improvement over vanilla DML.

Usage:
    python examples/dml_ablation_benchmark.py
    python examples/dml_ablation_benchmark.py --quick
    python examples/dml_ablation_benchmark.py --mc-reps 80
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from causal_inference.chernozhukov_dml import (
    apply_study_bonferroni,
    estimate_plr_dml,
    estimate_vanilla_dml,
)
from causal_inference.synthetic_dml_ablation import (
    SyntheticDMLScenario,
    all_ablation_scenarios,
    generate_heteroskedastic_dgp,
)

DEFAULT_OUTPUT = ROOT / "outputs" / "dml_ablation"


def _estimate(
    scenario: SyntheticDMLScenario,
    method: str,
    n_folds: int,
    seed: int,
) -> dict:
    X = scenario.X.values.astype(float)
    y = scenario.y.values.astype(float)
    names = list(scenario.X.columns)

    if method == "vanilla":
        return estimate_vanilla_dml(X, y, names, scenario.treatment, n_folds=n_folds, random_state=seed)

    configs = {
        "improved_full": dict(
            adjustment_mode="post_double_selection",
            nuisance_mode="post_lasso_ols",
            se_mode="influence_function",
        ),
        "ablation_1_pds": dict(
            adjustment_mode="post_double_selection",
            nuisance_mode="ridge",
            se_mode="residual",
        ),
        "ablation_2_lasso": dict(
            adjustment_mode="post_double_selection",
            nuisance_mode="post_lasso_ols",
            se_mode="residual",
        ),
        "ablation_3_ifse": dict(
            adjustment_mode="post_double_selection",
            nuisance_mode="post_lasso_ols",
            se_mode="influence_function",
        ),
        "ablation_4_estimand": dict(
            adjustment_mode="post_double_selection",
            nuisance_mode="post_lasso_ols",
            se_mode="influence_function",
        ),
        "vanilla_all_except": dict(
            adjustment_mode="all_except_i",
            nuisance_mode="ridge",
            se_mode="residual",
        ),
    }
    cfg = configs[method]
    return estimate_plr_dml(
        X,
        y,
        names,
        scenario.treatment,
        n_folds=n_folds,
        mandatory=[],
        random_state=seed,
        **cfg,
    )


def _theta_error(result: dict, theta_true: float) -> float:
    return abs(float(result["theta"]) - theta_true)


def _ci_covers(result: dict, theta_true: float) -> bool:
    lo, hi = result["ci_lower"], result["ci_upper"]
    if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
        return False
    return lo <= theta_true <= hi


def run_ablation_1(scenario: SyntheticDMLScenario, n_folds: int, mc_reps: int = 20, seed: int = 42) -> dict:
    from causal_inference.synthetic_dml_ablation import generate_high_dimensional_sparse_dgp

    vanilla_errors = []
    improved_errors = []
    for rep in range(mc_reps):
        sc = scenario if rep == 0 else generate_high_dimensional_sparse_dgp(random_state=seed + rep)
        vanilla = _estimate(sc, "vanilla", n_folds, seed + rep)
        improved = _estimate(sc, "improved_full", n_folds, seed + rep)
        vanilla_errors.append(_theta_error(vanilla, sc.theta_true))
        improved_errors.append(_theta_error(improved, sc.theta_true))

    vanilla = _estimate(scenario, "vanilla", n_folds, 42)
    improved = _estimate(scenario, "improved_full", n_folds, 42)
    return {
        "improvement_id": 1,
        "improvement": scenario.improvement_label,
        "theta_true": scenario.theta_true,
        "mc_reps": mc_reps,
        "vanilla_mean_abs_error": float(np.mean(vanilla_errors)),
        "improved_mean_abs_error": float(np.mean(improved_errors)),
        "vanilla_theta": vanilla["theta"],
        "improved_theta": improved["theta"],
        "vanilla_n_adjustment": vanilla["n_adjustment_features"],
        "improved_n_adjustment": improved["n_adjustment_features"],
        "improved_wins": float(np.mean(improved_errors)) + 1e-4 < float(np.mean(vanilla_errors)),
    }


def run_ablation_2(scenario: SyntheticDMLScenario, n_folds: int, mc_reps: int = 20, seed: int = 7) -> dict:
    from causal_inference.chernozhukov_dml import nuisance_crossfit_mse
    from causal_inference.synthetic_dml_ablation import generate_sparse_nuisance_dgp

    ridge_mses = []
    lasso_mses = []
    ridge_errors = []
    lasso_errors = []
    for rep in range(mc_reps):
        sc = scenario if rep == 0 else generate_sparse_nuisance_dgp(random_state=seed + rep)
        adj_cols = [c for c in sc.X.columns if c != sc.treatment]
        reduced = sc.X[adj_cols + [sc.treatment]]
        X = reduced.values.astype(float)
        y = sc.y.values.astype(float)
        names = list(reduced.columns)

        ridge_mse = nuisance_crossfit_mse(
            X, y, names, sc.treatment, n_folds=n_folds,
            nuisance_mode="ridge", adjustment_mode="all_except_i",
            ridge_alpha=10.0, random_state=seed + rep,
        )
        lasso_mse = nuisance_crossfit_mse(
            X, y, names, sc.treatment, n_folds=n_folds,
            nuisance_mode="post_lasso_ols", adjustment_mode="all_except_i",
            random_state=seed + rep,
        )
        ridge_mses.append(ridge_mse["total_nuisance_mse"])
        lasso_mses.append(lasso_mse["total_nuisance_mse"])

        ridge = estimate_plr_dml(
            X, y, names, sc.treatment, n_folds=n_folds, mandatory=[], random_state=seed + rep,
            adjustment_mode="all_except_i", nuisance_mode="ridge", se_mode="residual", ridge_alpha=10.0,
        )
        lasso = estimate_plr_dml(
            X, y, names, sc.treatment, n_folds=n_folds, mandatory=[], random_state=seed + rep,
            adjustment_mode="all_except_i", nuisance_mode="post_lasso_ols", se_mode="residual",
        )
        ridge_errors.append(_theta_error(ridge, sc.theta_true))
        lasso_errors.append(_theta_error(lasso, sc.theta_true))

    return {
        "improvement_id": 2,
        "improvement": scenario.improvement_label,
        "theta_true": scenario.theta_true,
        "mc_reps": mc_reps,
        "n_adjustment_features": len(scenario.X.columns) - 1,
        "ridge_mean_nuisance_mse": float(np.mean(ridge_mses)),
        "post_lasso_mean_nuisance_mse": float(np.mean(lasso_mses)),
        "ridge_mean_abs_error": float(np.mean(ridge_errors)),
        "post_lasso_mean_abs_error": float(np.mean(lasso_errors)),
        "improved_wins": float(np.mean(lasso_mses)) + 1e-4 < float(np.mean(ridge_mses)),
    }


def run_ablation_3(mc_reps: int, n_folds: int, seed: int) -> dict:
    vanilla_cover = []
    if_cover = []
    vanilla_width = []
    if_width = []

    for rep in range(mc_reps):
        scenario = generate_heteroskedastic_dgp(random_state=seed + rep)
        vanilla = _estimate(scenario, "vanilla", n_folds, seed + rep)
        improved = _estimate(scenario, "ablation_3_ifse", n_folds, seed + rep)

        vanilla_cover.append(_ci_covers(vanilla, scenario.theta_true))
        if_cover.append(_ci_covers(improved, scenario.theta_true))
        vanilla_width.append(vanilla["ci_upper"] - vanilla["ci_lower"])
        if_width.append(improved["ci_upper"] - improved["ci_lower"])

    vanilla_cov = float(np.mean(vanilla_cover))
    if_cov = float(np.mean(if_cover))
    return {
        "improvement_id": 3,
        "improvement": "Influence-function standard errors",
        "theta_true": 2.2,
        "mc_reps": mc_reps,
        "target_coverage": 0.95,
        "vanilla_coverage": vanilla_cov,
        "if_coverage": if_cov,
        "vanilla_mean_ci_width": float(np.mean(vanilla_width)),
        "if_mean_ci_width": float(np.mean(if_width)),
        "vanilla_coverage_error": abs(vanilla_cov - 0.95),
        "if_coverage_error": abs(if_cov - 0.95),
        "improved_wins": abs(if_cov - 0.95) + 1e-6 < abs(vanilla_cov - 0.95),
    }


def run_ablation_4(scenario: SyntheticDMLScenario, n_folds: int) -> dict:
    vanilla = _estimate(scenario, "vanilla", n_folds, 42)
    improved = _estimate(scenario, "ablation_4_estimand", n_folds, 42)
    vanilla_err = _theta_error(vanilla, scenario.theta_true)
    improved_err = _theta_error(improved, scenario.theta_true)
    return {
        "improvement_id": 4,
        "improvement": scenario.improvement_label,
        "theta_true_partial_effect": scenario.theta_true,
        "vanilla_theta": vanilla["theta"],
        "vanilla_abs_error": vanilla_err,
        "improved_theta": improved["theta"],
        "improved_abs_error": improved_err,
        "vanilla_estimand": vanilla["estimand"],
        "improved_estimand": improved["estimand"],
        "improved_wins": improved_err + 1e-6 < vanilla_err,
    }


def run_ablation_5(scenario: SyntheticDMLScenario, n_folds: int, mc_reps: int, seed: int) -> dict:
    vanilla_fps = []
    bonf_fps = []

    for rep in range(mc_reps):
        sc = scenario
        if rep > 0:
            from causal_inference.synthetic_dml_ablation import generate_multiplicity_dgp

            sc = generate_multiplicity_dgp(random_state=seed + rep)

        X = sc.X.values.astype(float)
        y = sc.y.values.astype(float)
        names = list(sc.X.columns)

        rows = []
        for feat in sc.candidate_features:
            rows.append(estimate_vanilla_dml(X, y, names, feat, n_folds=n_folds, random_state=seed + rep))
        df = pd.DataFrame(rows)
        vanilla_fps.append(int((df["p_value"] < 0.05).sum() - int(df.loc[df["feature"] == sc.treatment, "p_value"].iloc[0] < 0.05)))

        bonf = apply_study_bonferroni({"outcome": df})
        bonf_df = bonf["results"]["outcome"]
        bonf_fps.append(int(bonf_df["significant_global_bonferroni"].sum() - int(bonf_df.loc[bonf_df["feature"] == sc.treatment, "significant_global_bonferroni"].iloc[0])))

    return {
        "improvement_id": 5,
        "improvement": scenario.improvement_label,
        "n_candidates": len(scenario.candidate_features),
        "mc_reps": mc_reps,
        "mean_false_positives_nominal": float(np.mean(vanilla_fps)),
        "mean_false_positives_bonferroni": float(np.mean(bonf_fps)),
        "bonferroni_threshold": global_bonferroni_threshold(len(scenario.candidate_features)),
        "improved_wins": float(np.mean(bonf_fps)) < float(np.mean(vanilla_fps)),
    }


def run_ablation_6(scenario: SyntheticDMLScenario, n_folds: int) -> dict:
    X = scenario.X.values.astype(float)
    y = scenario.y.values.astype(float)
    names = list(scenario.X.columns)

    true_res = estimate_plr_dml(
        X,
        y,
        names,
        scenario.treatment,
        n_folds=n_folds,
        mandatory=[],
        random_state=42,
    )
    null_e = []
    for feat in scenario.candidate_features:
        if feat == scenario.treatment:
            continue
        res = estimate_plr_dml(X, y, names, feat, n_folds=n_folds, mandatory=[], random_state=42)
        null_e.append(res["e_value_approx"])

    return {
        "improvement_id": 6,
        "improvement": scenario.improvement_label,
        "true_effect_e_value": true_res["e_value_approx"],
        "null_e_value_mean": float(np.mean(null_e)),
        "null_e_value_max": float(np.max(null_e)),
        "e_value_gap": float(true_res["e_value_approx"] - np.mean(null_e)),
        "vanilla_has_e_value": False,
        "improved_wins": true_res["e_value_approx"] > 2.0 and float(np.max(null_e)) < 2.0,
    }


def global_bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    from causal_inference.multiple_testing import global_bonferroni_threshold as _thresh

    return _thresh(n_tests, alpha)


def run_all_ablations(quick: bool = False, mc_reps: int | None = None) -> dict:
    n_folds = 3 if quick else 5
    if mc_reps is None:
        mc_reps = 30 if quick else 80

    scenarios = all_ablation_scenarios()
    by_id = {s.improvement_id: s for s in scenarios}

    results = []
    results.append(run_ablation_1(by_id[1], n_folds, mc_reps=min(mc_reps, 20)))
    results.append(run_ablation_2(by_id[2], n_folds, mc_reps=min(mc_reps, 20)))
    results.append(run_ablation_3(mc_reps=mc_reps, n_folds=n_folds, seed=1000))
    results.append(run_ablation_4(by_id[4], n_folds))
    results.append(run_ablation_5(by_id[5], n_folds=n_folds, mc_reps=mc_reps, seed=2000))
    results.append(run_ablation_6(by_id[6], n_folds))

    n_wins = sum(1 for r in results if r.get("improved_wins"))
    return {
        "n_folds": n_folds,
        "mc_reps": mc_reps,
        "n_improvements_demonstrated": n_wins,
        "all_improvements_win": n_wins == 6,
        "ablations": results,
        "scenarios": [
            {
                "improvement_id": s.improvement_id,
                "name": s.name,
                "description": s.description,
                "theta_true": s.theta_true,
                "n_features": s.X.shape[1],
                "n_subjects": s.X.shape[0],
            }
            for s in scenarios
        ],
    }


def _format_table(summary: dict) -> str:
    lines = [
        "",
        "=" * 88,
        "  DML ABLATION BENCHMARK — VANILLA VS CHERNOZHUKOV IMPROVEMENTS",
        "=" * 88,
    ]
    for row in summary["ablations"]:
        i = row["improvement_id"]
        label = row["improvement"]
        win = "YES" if row.get("improved_wins") else "NO"
        lines.append(f"\n  [{i}] {label}  -> improved wins: {win}")
        for k, v in row.items():
            if k in {"improvement_id", "improvement", "improved_wins"}:
                continue
            if isinstance(v, float):
                lines.append(f"      {k}: {v:.4f}")
            else:
                lines.append(f"      {k}: {v}")
    lines.append(
        f"\n  Summary: {summary['n_improvements_demonstrated']}/6 improvements clearly beat vanilla"
    )
    lines.append("=" * 88)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Six-improvement DML ablation benchmark")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mc-reps", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Running DML ablation benchmark...")
    t0 = time.time()
    summary = run_all_ablations(quick=args.quick, mc_reps=args.mc_reps)
    elapsed = time.time() - t0

    summary["elapsed_seconds"] = elapsed
    table = _format_table(summary)
    print(table)

    pd.DataFrame(summary["ablations"]).to_csv(args.output_dir / "ablation_results.csv", index=False)
    with open(args.output_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(args.output_dir / "ablation_report.txt", "w", encoding="utf-8") as f:
        f.write(table)

    print(f"\nSaved to {args.output_dir} ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()

"""Multiple-testing adjustments for CIA result tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

FIBE_RESULTS_SKIP_DIRS = {"top100_correlation", "synthetic"}


def list_fibe_result_csvs(results_dir: Path) -> list[Path]:
    """Return cia_results.csv paths for standard FIBE outcome folders."""
    paths = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir() or subdir.name in FIBE_RESULTS_SKIP_DIRS:
            continue
        csv_path = subdir / "cia_results.csv"
        if csv_path.exists():
            paths.append(csv_path)
    return paths


def global_bonferroni_threshold(n_tests: int, alpha: float = 0.05) -> float:
    """Uncorrected p-value cutoff for global Bonferroni at level alpha."""
    if n_tests <= 0:
        raise ValueError("n_tests must be positive")
    return alpha / n_tests


def apply_global_bonferroni_to_csv(
    csv_path: Path,
    n_tests_total: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add global Bonferroni columns to one outcome results table."""
    df = pd.read_csv(csv_path)
    thresh = global_bonferroni_threshold(n_tests_total, alpha)
    df["p_value_global_bonferroni"] = (df["p_value"] * n_tests_total).clip(upper=1.0)
    df["significant_global_bonferroni"] = df["p_value"] < thresh
    return df


def apply_global_bonferroni_across_study(
    results_dir: Path,
    alpha: float = 0.05,
    csv_paths: Iterable[Path] | None = None,
) -> dict:
    """
    Apply one global Bonferroni correction over all feature-outcome tests.

    Writes updated cia_results.csv files with columns:
      p_value_global_bonferroni, significant_global_bonferroni
    """
    paths = list(csv_paths) if csv_paths is not None else list_fibe_result_csvs(results_dir)
    if not paths:
        raise FileNotFoundError(f"No FIBE cia_results.csv files under {results_dir}")

    n_tests_total = sum(len(pd.read_csv(p)) for p in paths)
    thresh = global_bonferroni_threshold(n_tests_total, alpha)

    per_outcome = []
    total_sig = 0
    for csv_path in paths:
        df = apply_global_bonferroni_to_csv(csv_path, n_tests_total, alpha)
        df.to_csv(csv_path, index=False)
        n_sig = int(df["significant_global_bonferroni"].sum())
        total_sig += n_sig
        per_outcome.append(
            {
                "csv": str(csv_path),
                "outcome_dir": csv_path.parent.name,
                "n_features": len(df),
                "n_significant_global_bonferroni": n_sig,
            }
        )

    summary = {
        "alpha": alpha,
        "n_tests_total": n_tests_total,
        "uncorrected_threshold": thresh,
        "n_significant_global_bonferroni": total_sig,
        "per_outcome": per_outcome,
    }
    meta_path = results_dir / "global_bonferroni_summary.json"
    import json

    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

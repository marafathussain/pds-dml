"""
Utilities for loading ABCD data and FIBE-selected features for CIA analysis.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Parent education was added after FIBE; always include in CIA analyses.
PARENT_EDUCATION_COLUMN = "Parent/caregiver highest education level (demo_prnt_ed_v2)"

# Ancestry proportion features are excluded from CIA (sensitive / backlash risk).
ANCESTRY_KEYWORDS = (
    "proportion of african ancestry",
    "proportion of european ancestry",
    "proportion of east asian ancestry",
    "proportion of american ancestry",
)


def is_ancestry_feature(feature_name: str) -> bool:
    """Return True if the feature is an ancestry proportion column."""
    name_lower = feature_name.strip().lower()
    return any(kw in name_lower for kw in ANCESTRY_KEYWORDS)


def load_fibe_selected_features(json_path: str | Path) -> List[str]:
    """Load FIBE final feature list from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        features = json.load(f)
    if not isinstance(features, list):
        raise ValueError(f"Expected a JSON list in {json_path}")
    return features


def prepare_cia_features(
    fibe_features: List[str],
    data_columns: List[str],
    always_include: Optional[List[str]] = None,
    exclude_ancestry: bool = True,
) -> Tuple[List[str], dict]:
    """
    Build the feature list for CIA from FIBE-selected features.

    Rules:
    1. Remove ancestry proportion features (if exclude_ancestry=True).
    2. Always include columns in always_include (default: parent education).
    3. Drop features not present in the data CSV.

    Returns
    -------
    feature_list : list of str
    report : dict with counts and removed/added feature names
    """
    if always_include is None:
        always_include = [PARENT_EDUCATION_COLUMN]

    col_set = set(data_columns)
    removed_ancestry = []
    kept = []

    for feat in fibe_features:
        if exclude_ancestry and is_ancestry_feature(feat):
            removed_ancestry.append(feat)
            continue
        if feat in col_set:
            kept.append(feat)

    added_mandatory = []
    for feat in always_include:
        if feat not in kept and feat in col_set:
            kept.append(feat)
            added_mandatory.append(feat)

    missing = [f for f in fibe_features if f not in col_set and not is_ancestry_feature(f)]

    report = {
        "n_fibe_input": len(fibe_features),
        "n_ancestry_removed": len(removed_ancestry),
        "ancestry_removed": removed_ancestry,
        "n_mandatory_added": len(added_mandatory),
        "mandatory_added": added_mandatory,
        "n_missing_from_csv": len(missing),
        "missing_from_csv": missing,
        "n_final_features": len(kept),
    }
    return kept, report


def load_abcd_outcome_data(
    csv_path: str | Path,
    outcome_column: str,
    feature_list: List[str],
    id_column: str = "Record ID",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load ABCD CSV and return (X, y) for CIA.

    Drops rows with missing values in outcome or any selected feature.
    """
    df = pd.read_csv(csv_path)

    if outcome_column not in df.columns:
        raise ValueError(f"Outcome column not found: {outcome_column}")

    use_cols = [c for c in feature_list if c in df.columns]
    missing = set(feature_list) - set(use_cols)
    if missing:
        raise ValueError(f"Features not in CSV: {missing}")

    subset = df[[outcome_column] + use_cols].dropna()
    X = subset[use_cols]
    y = subset[outcome_column]
    return X, y


def list_outcome_json_files(features_dir: str | Path) -> List[Path]:
    """List per-outcome FIBE feature JSON files (exclude _voting_summary.json)."""
    features_dir = Path(features_dir)
    files = []
    for p in sorted(features_dir.glob("*.json")):
        name = p.name
        if name.startswith("_") or name.startswith("._"):
            continue
        files.append(p)
    return files


def outcome_name_from_json(json_path: Path) -> str:
    """Outcome column name = JSON filename without extension."""
    return json_path.stem


def get_outcome_columns(data_columns: List[str]) -> List[str]:
    """
    Return columns that are cognitive outcomes or alternate score forms.

    Excludes all NIH Toolbox measures and cognition/crystallized composites
    so they are not used as predictors when analysing one outcome.
    """
    excluded = []
    composite_prefixes = (
        "Cognition Fluid Composite",
        "Cognition Total Composite",
        "Crystallized Composite",
    )
    for c in data_columns:
        if c.startswith("NIH Toolbox"):
            excluded.append(c)
            continue
        if any(c.startswith(p) for p in composite_prefixes):
            excluded.append(c)
    return excluded


def prepare_all_cia_features(
    data_columns: List[str],
    outcome_column: str,
    always_include: Optional[List[str]] = None,
    exclude_ancestry: bool = True,
    id_column: str = "Record ID",
) -> Tuple[List[str], dict]:
    """
    Build CIA feature list from all available predictors (not FIBE-selected).

    Excludes: Record ID, all outcome score columns, ancestry proportions.
    Always includes parent education if present.
    """
    if always_include is None:
        always_include = [PARENT_EDUCATION_COLUMN]

    outcome_cols = set(get_outcome_columns(data_columns))
    predictors = []
    removed_ancestry = []

    for col in data_columns:
        if col == id_column or col in outcome_cols:
            continue
        if "Record ID" in col:
            continue
        if exclude_ancestry and is_ancestry_feature(col):
            removed_ancestry.append(col)
            continue
        predictors.append(col)

    added_mandatory = []
    for feat in always_include:
        if feat not in predictors and feat in data_columns:
            predictors.append(feat)
            added_mandatory.append(feat)

    report = {
        "mode": "all_features",
        "outcome_column": outcome_column,
        "n_outcome_columns_excluded": len(outcome_cols),
        "n_ancestry_removed": len(removed_ancestry),
        "ancestry_removed": removed_ancestry,
        "n_mandatory_added": len(added_mandatory),
        "mandatory_added": added_mandatory,
        "n_final_features": len(predictors),
    }
    return predictors, report


def select_top_correlated_features(
    df: pd.DataFrame,
    outcome_column: str,
    candidate_features: List[str],
    top_k: int = 100,
    always_include: Optional[List[str]] = None,
) -> Tuple[List[str], pd.DataFrame, dict]:
    """
    Rank candidates by Pearson correlation p-value with outcome; keep top_k.

    Parent/mandatory features are always included even if outside top_k.
    Returns (selected_features, ranking_table, report).
    """
    if always_include is None:
        always_include = [PARENT_EDUCATION_COLUMN]

    y = df[outcome_column].values.astype(np.float64)
    rows = []
    for feat in candidate_features:
        if feat not in df.columns:
            continue
        x = df[feat].values.astype(np.float64)
        r, p = stats.pearsonr(x, y)
        rows.append({
            "feature": feat,
            "correlation": r,
            "abs_correlation": abs(r),
            "p_value": p,
        })

    ranking = pd.DataFrame(rows).sort_values(
        ["p_value", "abs_correlation"], ascending=[True, False]
    ).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1

    mandatory = [f for f in always_include if f in df.columns]
    top = ranking.head(top_k)["feature"].tolist()
    selected = list(dict.fromkeys(mandatory + top))  # preserve order, dedupe

    report = {
        "top_k": top_k,
        "n_candidates": len(candidate_features),
        "n_ranked": len(ranking),
        "n_selected": len(selected),
        "n_mandatory_included": len(mandatory),
        "mandatory_features": mandatory,
    }
    return selected, ranking, report

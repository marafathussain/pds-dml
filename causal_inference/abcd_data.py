"""Utilities for preparing optional ABCD cohort inputs for DML."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PARENT_EDUCATION_COLUMN = "Parent/caregiver highest education level (demo_prnt_ed_v2)"

ABCD_THREE_OUTCOMES = [
    "Cognition Total Composite Score Age-Corrected Standard Score",
    "NIH Toolbox Flanker Inhibitory Control and Attention Test Ages 8-11 v2.0 Age-Corrected Standard Score",
    "NIH Toolbox Dimensional Change Card Sort Test Ages 8-11 v2.0 Age-Corrected Standard Score",
]

ABCD_OUTCOME_LABELS = {
    ABCD_THREE_OUTCOMES[0]: "Total Composite (IQ proxy)",
    ABCD_THREE_OUTCOMES[1]: "Flanker (EF: inhibition)",
    ABCD_THREE_OUTCOMES[2]: "DCCS (EF: flexibility)",
}

ANCESTRY_KEYWORDS = (
    "proportion of african ancestry",
    "proportion of european ancestry",
    "proportion of east asian ancestry",
    "proportion of american ancestry",
)


def is_ancestry_feature(feature_name: str) -> bool:
    """Return whether a column is an ancestry-proportion feature."""
    name_lower = feature_name.strip().lower()
    return any(keyword in name_lower for keyword in ANCESTRY_KEYWORDS)


def get_outcome_columns(data_columns: List[str]) -> List[str]:
    """Return cognitive outcome columns that must not be used as predictors."""
    prefixes = (
        "Cognition Fluid Composite",
        "Cognition Total Composite",
        "Crystallized Composite",
    )
    return [
        column
        for column in data_columns
        if column.startswith("NIH Toolbox")
        or any(column.startswith(prefix) for prefix in prefixes)
    ]


def prepare_all_features(
    data_columns: List[str],
    outcome_column: str,
    always_include: Optional[List[str]] = None,
    exclude_ancestry: bool = True,
    id_column: str = "Record ID",
) -> Tuple[List[str], dict]:
    """Build the DML predictor pool from all eligible data columns."""
    if always_include is None:
        always_include = [PARENT_EDUCATION_COLUMN]

    outcome_columns = set(get_outcome_columns(data_columns))
    predictors = []
    removed_ancestry = []
    for column in data_columns:
        if column == id_column or "Record ID" in column or column in outcome_columns:
            continue
        if exclude_ancestry and is_ancestry_feature(column):
            removed_ancestry.append(column)
            continue
        predictors.append(column)

    added_mandatory = []
    for feature in always_include:
        if feature not in predictors and feature in data_columns:
            predictors.append(feature)
            added_mandatory.append(feature)

    report = {
        "mode": "all_features",
        "outcome_column": outcome_column,
        "n_outcome_columns_excluded": len(outcome_columns),
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
    top_k: int = 50,
    always_include: Optional[List[str]] = None,
) -> Tuple[List[str], pd.DataFrame, dict]:
    """Rank candidates by Pearson association and retain the top features."""
    if always_include is None:
        always_include = [PARENT_EDUCATION_COLUMN]

    y = df[outcome_column].to_numpy(dtype=np.float64)
    rows = []
    for feature in candidate_features:
        if feature not in df.columns:
            continue
        correlation, p_value = stats.pearsonr(
            df[feature].to_numpy(dtype=np.float64), y
        )
        rows.append(
            {
                "feature": feature,
                "correlation": correlation,
                "abs_correlation": abs(correlation),
                "p_value": p_value,
            }
        )

    ranking = (
        pd.DataFrame(rows)
        .sort_values(["p_value", "abs_correlation"], ascending=[True, False])
        .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1

    mandatory = [feature for feature in always_include if feature in df.columns]
    top = ranking.head(top_k)["feature"].tolist()
    selected = list(dict.fromkeys(mandatory + top))[:top_k]
    report = {
        "top_k": top_k,
        "n_candidates": len(candidate_features),
        "n_ranked": len(ranking),
        "n_selected": len(selected),
        "n_mandatory_included": len(mandatory),
        "mandatory_features": mandatory,
    }
    return selected, ranking, report

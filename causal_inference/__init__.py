"""Double machine learning estimators for high-dimensional feature screening."""

from .chernozhukov_dml import (
    ESTIMAND_DESCRIPTION,
    apply_study_bonferroni,
    estimate_plr_dml,
    estimate_vanilla_dml,
    run_plr_dml_on_candidates,
)
from .dml import DoubleMachineLearning, dml_importance_ranking

__version__ = "1.0.0"

__all__ = [
    "DoubleMachineLearning",
    "dml_importance_ranking",
    "ESTIMAND_DESCRIPTION",
    "estimate_plr_dml",
    "estimate_vanilla_dml",
    "run_plr_dml_on_candidates",
    "apply_study_bonferroni",
]

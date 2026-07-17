"""
Conditional Intervention Analysis (CIA) Framework

A practical approach to causal inference from observational data using
conditional sampling and counterfactual evaluation. This method approximates
interventional effects (do-calculus) by generating biologically plausible
counterfactuals through learned conditional distributions.

Uses consensus of three ML techniques (matching FIBE):
  - Regression: Linear SVR + Gaussian SVR + Random Forest Regressor
  - Classification: Linear SVC + Gaussian SVC + Random Forest Classifier

Reference:
    "We estimated feature-level interventional effects by sampling from the
    conditional distribution P(F_i | F_{-i}), thereby generating biologically
    plausible counterfactuals while preserving the joint feature structure."
"""

from .conditional_intervention import ConditionalInterventionAnalysis
from .utils import generate_synthetic_data, format_results_table
from .outcome_model import (
    get_models,
    train_outcome_models,
    predict_outcome,
    evaluate_outcome_models,
)
from .abcd_data import (
    PARENT_EDUCATION_COLUMN,
    prepare_cia_features,
    load_abcd_outcome_data,
    load_fibe_selected_features,
)
from .baselines import (
    correlation_importance,
    conditional_permutation_importance_ranking,
    linear_dml_importance_ranking,
    permutation_importance_ranking,
)
from .dml import DoubleMachineLearning, dml_importance_ranking
from .hybrid import CIA_DML_Pipeline
from .chernozhukov_dml import (
    ESTIMAND_DESCRIPTION,
    estimate_plr_dml,
    run_plr_dml_on_candidates,
    apply_study_bonferroni,
)

__version__ = "2.3.0"

__all__ = [
    "ConditionalInterventionAnalysis",
    "DoubleMachineLearning",
    "CIA_DML_Pipeline",
    "generate_synthetic_data",
    "format_results_table",
    "get_models",
    "train_outcome_models",
    "predict_outcome",
    "evaluate_outcome_models",
    "PARENT_EDUCATION_COLUMN",
    "prepare_cia_features",
    "load_abcd_outcome_data",
    "load_fibe_selected_features",
    "correlation_importance",
    "conditional_permutation_importance_ranking",
    "linear_dml_importance_ranking",
    "dml_importance_ranking",
    "permutation_importance_ranking",
    "ESTIMAND_DESCRIPTION",
    "estimate_plr_dml",
    "run_plr_dml_on_candidates",
    "apply_study_bonferroni",
]

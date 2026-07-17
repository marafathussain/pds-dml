# CIA-DML / PDS-DML

Code for **Conditional Intervention Analysis (CIA)** and **post-double-selection double machine learning (PDS-DML)** for trustworthy multimodal feature screening in pediatric neuroimaging cohorts.

This repository contains the Python library and examples only (no manuscript or private ABCD data).

## Status

| Component | Status |
|-----------|--------|
| CIA core (`ConditionalInterventionAnalysis`) | ported |
| Baselines (correlation, ANCOVA, permutation, conditional PI, linear DML) | ported |
| Synthetic validation example | included |
| CIA-DML hybrid (`CIA_DML_Pipeline`: screen with CIA, confirm with DML) | implemented |
| ABCD cohort pipeline | not included (add locally under `data/`) |

## Install

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Quick start

```bash
# Interactive demo (synthetic data, regression + classification)
python demo_run.py

# Paper-style synthetic benchmark (CIA vs baselines)
python examples/synthetic_validation.py
python examples/synthetic_validation.py --quick

# Two-stage CIA-DML hybrid
python examples/hybrid_validation.py
python examples/hybrid_validation.py --quick
```

Outputs go to `outputs/` (gitignored).

## Package layout

```
cia-dml/
├── causal_inference/          # Core library
│   ├── conditional_intervention.py
│   ├── outcome_model.py
│   ├── dml.py                 # DoubleMachineLearning (mentor design)
│   ├── hybrid.py              # CIA_DML_Pipeline (two-stage hybrid)
│   ├── baselines.py           # includes linear_dml_importance_ranking
│   ├── utils.py
│   ├── multiple_testing.py
│   └── abcd_data.py           # optional; requires local ABCD CSV + FIBE JSON
├── demo_run.py
├── examples/
│   └── synthetic_validation.py
└── docs/
    └── THEOREM_AND_PROOF.md
```

## CIA-DML hybrid (two-stage pipeline)

```python
from causal_inference import CIA_DML_Pipeline

pipeline = CIA_DML_Pipeline(
    screen_alpha=0.05,       # Stage 1: CIA p-value threshold
    screen_top_k=10,         # Also shortlist top-10 by |CIA effect|
    confirm_alpha=0.05,      # Stage 2: DML confirmation threshold
    nuisance_model="ridge",
    n_bootstrap=200,
    random_state=42,
)
pipeline.fit(X, y)
pipeline.summary()

confirmed = pipeline.get_confirmed_features()  # passed both stages
```

**Stage 1 (CIA):** Screen all features for conditional interventional evidence.
**Stage 2 (DML):** Confirm shortlisted features with debiased partial-linear effects.

A feature is *confirmed* only when both CIA screening and DML confirmation pass.

## DML usage (standalone)

```python
from causal_inference import DoubleMachineLearning

dml = DoubleMachineLearning(
    n_folds=5,
    nuisance_model="ridge",   # 'linear', 'ridge', or 'rf'
    n_bootstrap=200,          # bootstrap CIs; set 0 for analytic SE
    random_state=42,
)

# Default: adjust for all features except i
dml.fit(X, y)
dml.summary()

# Custom adjustment set per feature (e.g., confounders only)
adjustment = {
    "parent_education": ["income", "neighborhood"],
    "cortical_thickness": ["age", "sex"],
}
dml = DoubleMachineLearning(adjustment_spec=adjustment)
dml.fit(X, y)
```

## Next steps

1. Side-by-side benchmarking on ABCD (add local data under `data/`).
2. Compare adjustment modes: all features vs confounder-only subsets in Stage 2.
3. Nonlinear synthetic DGP where CIA screening outperforms linear DML alone.

## Citation

If you use this code, cite the associated workshop paper (details upon acceptance).

## Data

ABCD CSV and feature lists are not included. Place local files under `data/` and pass paths via the example script CLI flags.

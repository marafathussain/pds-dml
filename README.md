# PDS-DML

Code for post-double-selection double machine learning (PDS-DML), a workflow
for trustworthy feature screening in high-dimensional multimodal tables.

The implementation combines sparse adjustment selection, post-Lasso nuisance
models, cross-fitting, influence-function standard errors, global Bonferroni
control, and E-value sensitivity summaries. This repository contains code
only; the manuscript and private ABCD data are not included.

## Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Synthetic validation

Run the six paper ablations:

```bash
python examples/dml_ablation_benchmark.py
python examples/dml_ablation_benchmark.py --quick
```

Outputs are written under `outputs/`, which is gitignored.

## Basic usage

```python
from causal_inference import estimate_plr_dml

result = estimate_plr_dml(
    X_pool=X,
    y=y,
    pool_names=list(X.columns),
    target_name="feature_of_interest",
    n_folds=3,
)
print(result["theta"], result["standard_error"], result["p_value"])
```

For a reusable estimator with configurable nuisance models:

```python
from causal_inference import DoubleMachineLearning

dml = DoubleMachineLearning(
    n_folds=5,
    nuisance_model="ridge",
    n_bootstrap=200,
    random_state=42,
)
dml.fit(X, y)
dml.summary()
```

## ABCD examples

The private ABCD dataset is not distributed. Place a compatible CSV under
`data/` or pass a local path with `--csv`.

```bash
python examples/abcd_chernozhukov_dml.py --three-outcomes --top-k 50 --n-folds 3
python examples/abcd_vanilla_dml.py --three-outcomes --top-k 50 --n-folds 3
```

## Layout

```text
causal_inference/
  chernozhukov_dml.py       PDS-DML and baseline PLR-DML
  dml.py                    reusable DML estimator
  synthetic_dml_ablation.py
  abcd_data.py              optional cohort input preparation
  multiple_testing.py
examples/
  dml_ablation_benchmark.py
  abcd_chernozhukov_dml.py
  abcd_vanilla_dml.py
  generate_paper_figures.py
```

## Citation

Please cite the associated workshop paper; citation details will be added
after acceptance.

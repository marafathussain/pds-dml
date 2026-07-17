"""
Synthetic data generators tuned for DML ablation benchmarks.

Each DGP highlights a specific failure mode of vanilla DML that the
corresponding Chernozhukov improvement addresses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticDMLScenario:
    """Container for one ablation scenario."""

    name: str
    improvement_id: int
    improvement_label: str
    X: pd.DataFrame
    y: pd.Series
    treatment: str
    theta_true: float
    candidate_features: List[str]
    null_features: List[str]
    description: str


def _pack_features(
    arrays: List[np.ndarray],
    names: List[str],
) -> pd.DataFrame:
    return pd.DataFrame(np.column_stack(arrays), columns=names)


def generate_high_dimensional_sparse_dgp(
    n_subjects: int = 280,
    n_noise: int = 140,
    theta_true: float = 2.5,
    random_state: int = 42,
) -> SyntheticDMLScenario:
    """
    Ablation 1: sparse confounders + many irrelevant features.

    Vanilla all-except-i ridge is biased/shrinks theta; post-double-selection recovers truth.
    """
    rng = np.random.default_rng(random_state)
    c0 = rng.normal(0, 1, n_subjects)
    c1 = rng.normal(0, 1, n_subjects)
    c2 = rng.normal(0, 1, n_subjects)

    eta = rng.normal(0, 1, n_subjects)
    treatment = 0.7 * c0 + 0.55 * c1 + 0.45 * c2 + eta

    eps = rng.normal(0, 1.0, n_subjects)
    y = theta_true * treatment + 1.4 * c0 + 1.0 * c1 + 0.8 * c2 + eps

    noise = []
    noise_names = []
    for j in range(n_noise):
        # Mostly irrelevant; a few mild spurious correlates of confounders.
        z = 0.05 * c0 + rng.normal(0, 1, n_subjects)
        noise.append(z)
        noise_names.append(f"Noise_{j + 1}")

    # Near-duplicates inflate vanilla ridge conditioning without adding signal.
    dupes = []
    dupe_names = []
    for j in range(35):
        dupes.append(c0 + rng.normal(0, 0.02, n_subjects))
        dupe_names.append(f"C0_duplicate_{j + 1}")

    X = _pack_features(
        [c0, c1, c2, treatment] + dupes + noise,
        ["Confounder_1", "Confounder_2", "Confounder_3", "Treatment"] + dupe_names + noise_names,
    )
    return SyntheticDMLScenario(
        name="high_dimensional_sparse",
        improvement_id=1,
        improvement_label="Post-double-selection adjustment set",
        X=X,
        y=pd.Series(y, name="Outcome"),
        treatment="Treatment",
        theta_true=theta_true,
        candidate_features=["Treatment"],
        null_features=dupe_names + noise_names,
        description=(
            "PLR with 3 confounders, 25 near-duplicates, and 120 noise features. Vanilla "
            "ridge on all-except-i is regularization-biased; post-double-selection is sparse."
        ),
    )


def generate_sparse_nuisance_dgp(
    n_subjects: int = 500,
    n_confounders: int = 15,
    theta_true: float = 2.0,
    random_state: int = 7,
) -> SyntheticDMLScenario:
    """
    Ablation 2: linear nuisances in mains, plus a large spurious interaction library.

    Ridge penalizes all 120+ adjustment columns; post-Lasso OLS keeps only the active mains.
    """
    rng = np.random.default_rng(random_state)
    mains = [rng.normal(0, 1, n_subjects) for _ in range(n_confounders)]
    main_names = [f"W{i + 1}" for i in range(n_confounders)]

    beta_y = rng.uniform(0.45, 1.0, n_confounders)
    beta_t = rng.uniform(0.3, 0.75, n_confounders)

    eta = rng.normal(0, 1, n_subjects)
    treatment = sum(b * m for b, m in zip(beta_t, mains)) + eta
    eps = rng.normal(0, 1.0, n_subjects)
    y = theta_true * treatment + sum(b * m for b, m in zip(beta_y, mains)) + eps

    spurious_inters = []
    spurious_names = []
    for i in range(n_confounders):
        for j in range(i + 1, n_confounders):
            spurious_inters.append(mains[i] * mains[j])
            spurious_names.append(f"Spur_{i + 1}x{j + 1}")

    X = _pack_features(
        mains + spurious_inters + [treatment],
        main_names + spurious_names + ["Treatment"],
    )
    return SyntheticDMLScenario(
        name="sparse_nuisance",
        improvement_id=2,
        improvement_label="Post-Lasso OLS nuisances",
        X=X,
        y=pd.Series(y, name="Outcome"),
        treatment="Treatment",
        theta_true=theta_true,
        candidate_features=["Treatment"],
        null_features=spurious_names,
        description=(
            f"Nuisances are linear in {n_confounders} mains, but the adjustment library "
            f"includes {len(spurious_names)} spurious interactions. Post-Lasso OLS "
            "selects mains; ridge shrinks across the full library."
        ),
    )


def generate_heteroskedastic_dgp(
    n_subjects: int = 600,
    theta_true: float = 2.2,
    random_state: int = 11,
) -> SyntheticDMLScenario:
    """
    Ablation 3: heteroskedastic errors eps ~ N(0, exp(T)).

    Influence-function SE should achieve better nominal CI coverage than residual SE.
    """
    rng = np.random.default_rng(random_state)
    c0 = rng.normal(0, 1, n_subjects)
    c1 = rng.normal(0, 1, n_subjects)

    eta = rng.normal(0, 1, n_subjects)
    treatment = 0.6 * c0 + 0.5 * c1 + eta

    sigma = np.exp(0.35 * treatment)
    eps = rng.normal(0, 1, n_subjects) * sigma
    y = theta_true * treatment + 1.3 * c0 + 0.9 * c1 + eps

    nulls = []
    null_names = []
    for j in range(25):
        nulls.append(rng.normal(0, 1, n_subjects))
        null_names.append(f"Null_{j + 1}")

    X = _pack_features(
        [c0, c1, treatment] + nulls,
        ["Confounder_A", "Confounder_B", "Treatment"] + null_names,
    )
    return SyntheticDMLScenario(
        name="heteroskedastic",
        improvement_id=3,
        improvement_label="Influence-function standard errors",
        X=X,
        y=pd.Series(y, name="Outcome"),
        treatment="Treatment",
        theta_true=theta_true,
        candidate_features=["Treatment"],
        null_features=null_names,
        description=(
            "Outcome noise scales with treatment magnitude. Residual SE is miscalibrated; "
            "orthogonal IF SE should be closer to nominal 95% coverage."
        ),
    )


def generate_estimand_gap_dgp(
    n_subjects: int = 280,
    n_noise: int = 110,
    theta_true: float = 3.0,
    random_state: int = 19,
) -> SyntheticDMLScenario:
    """
    Ablation 4: vanilla all-except-i targets a biased estimand; sparse PLR W targets truth.
    """
    rng = np.random.default_rng(random_state)
    c0 = rng.normal(0, 1, n_subjects)
    c1 = rng.normal(0, 1, n_subjects)
    c2 = rng.normal(0, 1, n_subjects)

    eta = rng.normal(0, 1, n_subjects)
    treatment = 0.65 * c0 + 0.55 * c1 + 0.4 * c2 + eta

    eps = rng.normal(0, 1.0, n_subjects)
    y = theta_true * treatment + 1.4 * c0 + 1.0 * c1 + 0.75 * c2 + eps

    dupes = []
    dupe_names = []
    for j in range(40):
        dupes.append(c1 + rng.normal(0, 0.02, n_subjects))
        dupe_names.append(f"C1_duplicate_{j + 1}")

    noise = []
    noise_names = []
    for j in range(n_noise):
        z = 0.04 * c2 + rng.normal(0, 1, n_subjects)
        noise.append(z)
        noise_names.append(f"Noise_{j + 1}")

    X = _pack_features(
        [c0, c1, c2, treatment] + dupes + noise,
        ["Confounder_1", "Confounder_2", "Confounder_3", "Treatment"] + dupe_names + noise_names,
    )
    return SyntheticDMLScenario(
        name="estimand_gap",
        improvement_id=4,
        improvement_label="Explicit PLR partial-effect estimand",
        X=X,
        y=pd.Series(y, name="Outcome"),
        treatment="Treatment",
        theta_true=theta_true,
        candidate_features=["Treatment"],
        null_features=dupe_names + noise_names,
        description=(
            "Vanilla all-except-i conflates the estimand with a high-dimensional "
            "regularized residual slope. Improved sparse PLR targets the true partial effect."
        ),
    )


def generate_bad_controls_dgp(
    n_subjects: int = 500,
    n_bad_controls: int = 18,
    theta_true: float = 3.0,
    random_state: int = 19,
) -> SyntheticDMLScenario:
    """
    Ablation 4: bad controls B_j = T + tiny noise.

    Vanilla all-except-i partials out treatment variation; improved sparse W avoids them.
    Demonstrates estimand/identification gap vs explicit PLR partial effect.
    """
    rng = np.random.default_rng(random_state)
    c0 = rng.normal(0, 1, n_subjects)
    c1 = rng.normal(0, 1, n_subjects)

    eta = rng.normal(0, 1, n_subjects)
    treatment = 0.55 * c0 + 0.45 * c1 + eta

    eps = rng.normal(0, 1.0, n_subjects)
    y = theta_true * treatment + 1.1 * c0 + 0.8 * c1 + eps

    bad_controls = []
    bad_names = []
    for j in range(n_bad_controls):
        bc = treatment + rng.normal(0, 0.05, n_subjects)
        bad_controls.append(bc)
        bad_names.append(f"BadControl_{j + 1}")

    nulls = []
    null_names = []
    for j in range(20):
        nulls.append(rng.normal(0, 1, n_subjects))
        null_names.append(f"Null_{j + 1}")

    X = _pack_features(
        [c0, c1, treatment] + bad_controls + nulls,
        ["Confounder_A", "Confounder_B", "Treatment"] + bad_names + null_names,
    )
    return SyntheticDMLScenario(
        name="bad_controls_estimand",
        improvement_id=4,
        improvement_label="Explicit PLR partial-effect estimand",
        X=X,
        y=pd.Series(y, name="Outcome"),
        treatment="Treatment",
        theta_true=theta_true,
        candidate_features=["Treatment"],
        null_features=bad_names + null_names,
        description=(
            "Bad controls are almost deterministic functions of treatment. Vanilla "
            "all-except-i estimates a near-zero biased slope; improved sparse W targets "
            f"the true partial effect ({theta_true})."
        ),
    )


def generate_multiplicity_dgp(
    n_subjects: int = 450,
    n_null_features: int = 35,
    theta_true: float = 2.8,
    random_state: int = 23,
) -> SyntheticDMLScenario:
    """
    Ablation 5: one true treatment + many null features screened jointly.

    Nominal testing yields many false positives; global Bonferroni controls FWER.
    """
    rng = np.random.default_rng(random_state)
    c0 = rng.normal(0, 1, n_subjects)
    c1 = rng.normal(0, 1, n_subjects)

    eta = rng.normal(0, 1, n_subjects)
    treatment = 0.6 * c0 + 0.5 * c1 + eta
    eps = rng.normal(0, 1.0, n_subjects)
    y = theta_true * treatment + 1.2 * c0 + c1 + eps

    nulls = []
    null_names = []
    for j in range(n_null_features):
        nulls.append(rng.normal(0, 1, n_subjects))
        null_names.append(f"Null_{j + 1}")

    X = _pack_features(
        [c0, c1, treatment] + nulls,
        ["Confounder_A", "Confounder_B", "Treatment"] + null_names,
    )
    candidates = ["Treatment"] + null_names
    return SyntheticDMLScenario(
        name="multiplicity",
        improvement_id=5,
        improvement_label="Global Bonferroni multiplicity control",
        X=X,
        y=pd.Series(y, name="Outcome"),
        treatment="Treatment",
        theta_true=theta_true,
        candidate_features=candidates,
        null_features=null_names,
        description=(
            f"Screen {len(candidates)} features with one true causal treatment. "
            "Nominal alpha=0.05 yields false discoveries; Bonferroni suppresses them."
        ),
    )


def generate_sensitivity_dgp(
    n_subjects: int = 650,
    n_null_features: int = 30,
    theta_true: float = 3.5,
    random_state: int = 29,
) -> SyntheticDMLScenario:
    """
    Ablation 6: strong causal treatment vs nulls for E-value discrimination.

    Improved method reports high E-value for true effect and ~1 for nulls.
    """
    rng = np.random.default_rng(random_state)
    c0 = rng.normal(0, 1, n_subjects)
    c1 = rng.normal(0, 1, n_subjects)

    eta = rng.normal(0, 1, n_subjects)
    treatment = 0.5 * c0 + 0.45 * c1 + eta
    eps = rng.normal(0, 0.9, n_subjects)
    y = theta_true * treatment + 1.0 * c0 + 0.7 * c1 + eps

    nulls = []
    null_names = []
    for j in range(n_null_features):
        nulls.append(rng.normal(0, 1, n_subjects))
        null_names.append(f"Null_{j + 1}")

    X = _pack_features(
        [c0, c1, treatment] + nulls,
        ["Confounder_A", "Confounder_B", "Treatment"] + null_names,
    )
    return SyntheticDMLScenario(
        name="sensitivity",
        improvement_id=6,
        improvement_label="Sensitivity analysis (E-value)",
        X=X,
        y=pd.Series(y, name="Outcome"),
        treatment="Treatment",
        theta_true=theta_true,
        candidate_features=["Treatment"] + null_names[:8],
        null_features=null_names,
        description=(
            "Strong treatment effect should yield high E-value; null features should "
            "cluster near E=1. Vanilla DML provides no sensitivity metric."
        ),
    )


def all_ablation_scenarios(random_state: int = 42) -> List[SyntheticDMLScenario]:
    """Return all six ablation scenarios."""
    return [
        generate_high_dimensional_sparse_dgp(random_state=random_state),
        generate_sparse_nuisance_dgp(random_state=random_state + 1),
        generate_heteroskedastic_dgp(random_state=random_state + 2),
        generate_estimand_gap_dgp(random_state=random_state + 3),
        generate_multiplicity_dgp(random_state=random_state + 4),
        generate_sensitivity_dgp(random_state=random_state + 5),
    ]

"""
Utility functions for the Conditional Intervention Analysis framework.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple


def compute_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval for the mean of data.

    Parameters
    ----------
    data : np.ndarray
        Array of observations.
    confidence : float
        Confidence level (e.g., 0.95 for 95% CI).

    Returns
    -------
    ci_low, ci_high : tuple of float
    """
    n = len(data)
    if n < 2:
        return (np.mean(data), np.mean(data))

    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def generate_synthetic_data(
    n_subjects: int = 100,
    n_features: int = 10,
    n_causal: int = 3,
    noise_level: float = 0.5,
    task_type: str = "regression",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Generate synthetic data with known causal structure for testing.

    Creates a dataset where only `n_causal` features truly cause the outcome,
    while the rest are correlated but non-causal.

    Parameters
    ----------
    n_subjects : int
        Number of samples/subjects.
    n_features : int
        Total number of features.
    n_causal : int
        Number of truly causal features (first n_causal features).
    noise_level : float
        Standard deviation of outcome noise (for regression) or
        logistic noise scale (for classification).
    task_type : str
        'regression' for continuous outcome, 'classification' for binary.
    random_state : int
        Random seed.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with named columns.
    y : pd.Series
        Outcome variable (continuous for regression, 0/1 for classification).
    causal_features : list of str
        Names of the truly causal features (ground truth).
    """
    # Initialize random number generator with fixed seed for reproducibility
    rng = np.random.default_rng(random_state)

    # =========================================================================
    # STEP 1: Define causal coefficients (the "true" effect sizes)
    # =========================================================================
    # Each causal feature gets a random coefficient between 1.5 and 3.0
    # These represent how strongly each causal feature influences the outcome
    causal_coefficients = rng.uniform(1.5, 3.0, size=n_causal)
    # Randomly flip signs so some features have positive and some negative effects
    causal_coefficients *= rng.choice([-1, 1], size=n_causal)

    # =========================================================================
    # STEP 2: Generate truly CAUSAL features (independent random variables)
    # =========================================================================
    # These are drawn from a standard normal distribution N(0,1)
    # They are completely independent of each other, no correlations among them
    # IMPORTANT: These are the ONLY features that will appear in the outcome equation
    X_causal = rng.normal(0, 1, size=(n_subjects, n_causal))

    # =========================================================================
    # STEP 3: Generate NON-CAUSAL features (correlated with causal, but NOT in outcome equation)
    # =========================================================================
    # Each non-causal feature is constructed as a linear mix of:
    #   (a) one randomly chosen causal feature (creates correlation with outcome)
    #   (b) independent noise (makes it not perfectly correlated)
    # Formula: NonCausal_j = alpha * Causal_k + (1 - alpha) * noise
    # This makes non-causal features CORRELATED with the outcome (through causal features)
    # but they do NOT directly cause the outcome; they are confounded/spurious
    X_non_causal = np.zeros((n_subjects, n_features - n_causal))
    for i in range(n_features - n_causal):
        # Randomly pick which causal feature this non-causal feature is derived from
        source_idx = rng.integers(0, n_causal)
        # Random correlation strength between 0.3 and 0.7 (moderate correlation)
        correlation_strength = rng.uniform(0.3, 0.7)
        # Mix: part causal feature + part independent noise
        X_non_causal[:, i] = (
            correlation_strength * X_causal[:, source_idx]
            + (1 - correlation_strength) * rng.normal(0, 1, size=n_subjects)
        )

    # =========================================================================
    # STEP 4: Combine all features into one matrix
    # =========================================================================
    # Columns: [Causal_F1, Causal_F2, ..., NonCausal_F1, NonCausal_F2, ...]
    X = np.hstack([X_causal, X_non_causal])

    # Assign human-readable names to each feature
    feature_names = [f"Causal_F{i+1}" for i in range(n_causal)] + [
        f"NonCausal_F{i+1}" for i in range(n_features - n_causal)
    ]
    # Ground truth: only the first n_causal features are truly causal
    causal_feature_names = feature_names[:n_causal]

    # =========================================================================
    # STEP 5: Generate the OUTCOME variable using ONLY causal features
    # =========================================================================
    # THIS IS THE KEY: the outcome depends ONLY on causal features
    # Non-causal features are ABSENT from this equation, guaranteeing they are not causal
    if task_type == "regression":
        # y = beta1*F1 + beta2*F2 + ... + betaN*FN + noise
        # Matrix multiplication: X_causal @ coefficients gives the linear combination
        # Then we add Gaussian noise to simulate measurement uncertainty
        y = X_causal @ causal_coefficients + noise_level * rng.normal(0, 1, size=n_subjects)
    elif task_type == "classification":
        # For classification, we first compute continuous logits (same as regression)
        logits = X_causal @ causal_coefficients + noise_level * rng.normal(0, 1, size=n_subjects)
        # Then pass through sigmoid function to get probabilities between 0 and 1
        probabilities = 1 / (1 + np.exp(-logits))
        # Threshold at 0.5 to create binary class labels (0 or 1)
        y = (probabilities > 0.5).astype(float)
    else:
        raise ValueError(f"task_type must be 'regression' or 'classification', got '{task_type}'.")

    # =========================================================================
    # STEP 6: Package into pandas structures and return
    # =========================================================================
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="Outcome")

    # Return: features, outcome, and the ground-truth list of causal feature names
    return X_df, y_series, causal_feature_names


def format_results_table(results_df: pd.DataFrame, task_type: str = "regression") -> str:
    """
    Format results DataFrame as a readable string table.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from ConditionalInterventionAnalysis.
    task_type : str
        'regression' or 'classification'.

    Returns
    -------
    table : str
        Formatted table string.
    """
    effect_label = "Effect" if task_type == "regression" else "Prob. Effect"
    lines = []
    header = (
        f"{'Rank':<5} {'Feature':<25} {effect_label:>12} {'95% CI':>20} "
        f"{'p-value':>10} {'Significant':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for rank, (_, row) in enumerate(results_df.iterrows(), 1):
        ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        sig_str = "Yes" if row["significant"] else "No"
        lines.append(
            f"{rank:<5} {row['feature']:<25} {row['causal_effect']:>12.4f} "
            f"{ci_str:>20} {row['p_value']:>10.4f} {sig_str:>12}"
        )

    return "\n".join(lines)

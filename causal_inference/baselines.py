"""
Simple baseline methods for comparison with CIA on synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from scipy import stats


def _feature_matrix(X):
    if isinstance(X, pd.DataFrame):
        return list(X.columns), X.values
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    return feature_names, X


def correlation_importance(X, y, method="pearson"):
    """
    Rank features by absolute correlation with outcome.

    Returns DataFrame with columns: feature, score (|correlation|).
    """
    feature_names, X_arr = _feature_matrix(X)

    scores = []
    for j in range(X_arr.shape[1]):
        col = X_arr[:, j]
        if method == "pearson":
            r, _ = stats.pearsonr(col, y)
        else:
            r, _ = stats.spearmanr(col, y)
        scores.append(abs(r))

    df = pd.DataFrame({"feature": feature_names, "score": scores})
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def permutation_importance_ranking(X, y, n_repeats=10, random_state=42):
    """
    Rank features by marginal permutation importance (Random Forest).

    Returns DataFrame with columns: feature, score.
    """
    feature_names, X_arr = _feature_matrix(X)

    model = RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=5)
    model.fit(X_arr, y)
    result = permutation_importance(
        model, X_arr, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    df = pd.DataFrame({
        "feature": feature_names,
        "score": result.importances_mean,
    })
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def conditional_permutation_importance_ranking(
    X,
    y,
    n_repeats=10,
    random_state=42,
):
    """
    Rank features by conditional permutation importance.

    For each feature j, resample values from an estimated conditional model
    P(F_j | F_{-j}) while holding other columns fixed, then measure the drop
    in random-forest R^2 (Molnar-style conditional permutation).
    """
    feature_names, X_arr = _feature_matrix(X)
    n_samples, n_features = X_arr.shape
    rng = np.random.default_rng(random_state)

    model = RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=5)
    model.fit(X_arr, y)
    baseline_r2 = model.score(X_arr, y)

    scores = []
    for j in range(n_features):
        other_idx = [k for k in range(n_features) if k != j]
        cond = LinearRegression()
        cond.fit(X_arr[:, other_idx], X_arr[:, j])
        resid = X_arr[:, j] - cond.predict(X_arr[:, other_idx])
        resid_std = max(float(np.std(resid)), 1e-8)

        drops = []
        for rep in range(n_repeats):
            X_perm = X_arr.copy()
            sampled = cond.predict(X_arr[:, other_idx]) + rng.normal(
                0.0, resid_std, size=n_samples
            )
            X_perm[:, j] = sampled
            perm_r2 = model.score(X_perm, y)
            drops.append(max(baseline_r2 - perm_r2, 0.0))
        scores.append(float(np.mean(drops)))

    df = pd.DataFrame({"feature": feature_names, "score": scores})
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def linear_dml_importance_ranking(
    X,
    y,
    n_folds=5,
    random_state=42,
):
    """
    Rank features by absolute cross-fitted linear double/debiased ML effects.

    Uses pooled residual-on-residual regression across folds (see dml.py).
    """
    from .dml import dml_importance_ranking

    return dml_importance_ranking(
        X,
        y,
        n_folds=n_folds,
        nuisance_model="linear",
        adjustment_spec="all_except_i",
        random_state=random_state,
    )


def ancova_adjusted_ranking(X, y):
    """
    Rank features by absolute multivariate regression coefficient (ANCOVA-style).

    For each feature j, fit Y ~ F_j + F_{-j} and record the partial coefficient of F_j.
    """
    feature_names, X_arr = _feature_matrix(X)

    model = LinearRegression()
    model.fit(X_arr, y)
    coefs = [abs(c) for c in model.coef_]

    df = pd.DataFrame({"feature": feature_names, "score": coefs})
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def evaluate_ranking(recall_features, true_causal, top_k=None):
    """
    Compute precision/recall for a ranked or selected feature set.

    Parameters
    ----------
    recall_features : list
        Identified or top-k feature names.
    true_causal : list
        Ground truth causal feature names.
    top_k : int or None
        If set, use only first top_k from recall_features.
    """
    if top_k is not None:
        identified = set(recall_features[:top_k])
    else:
        identified = set(recall_features)

    true_set = set(true_causal)
    tp = identified & true_set
    fp = identified - true_set
    fn = true_set - identified

    precision = len(tp) / len(identified) if identified else 0.0
    recall = len(tp) / len(true_set) if true_set else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "true_positives": sorted(tp),
        "false_positives": sorted(fp),
        "false_negatives": sorted(fn),
    }

"""
Double Machine Learning (DML) for feature-level causal effect estimation.

Implements cross-fitted partial-linear DML as described in mentor feedback:
1. Split data into K folds.
2. For each fold k, on held-out subjects:
   - deltaMU = Y - E[Y | adjustment_set]
   - deltaFi = F_i - E[F_i | adjustment_set]
3. Pool residuals across folds; slope of deltaMU on deltaFi is theta_i.
4. Optional bootstrap for stability and confidence intervals.

Adjustment sets can be customized per feature (e.g., confounders only)
instead of naively using all features except i.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from typing import Callable, Dict, List, Optional, Union

from .utils import compute_confidence_interval


AdjustmentSpec = Union[
    None,
    str,
    Dict[str, List[str]],
    Callable[[str, List[str]], List[str]],
]


def _feature_matrix(X):
    if isinstance(X, pd.DataFrame):
        return list(X.columns), X.values
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    return feature_names, X


def _get_nuisance_estimator(model_name: str, random_state: Optional[int]):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "ridge":
        return Ridge(alpha=1.0)
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            max_depth=5,
            n_jobs=-1,
        )
    raise ValueError(
        f"Unknown nuisance_model: {model_name}. Choose from 'linear', 'ridge', 'rf'."
    )


def _resolve_adjustment_indices(
    feature_names: List[str],
    target_idx: int,
    adjustment_spec: AdjustmentSpec,
) -> List[int]:
    target_name = feature_names[target_idx]
    all_names = feature_names

    if adjustment_spec is None or adjustment_spec == "all_except_i":
        return [j for j in range(len(feature_names)) if j != target_idx]

    if isinstance(adjustment_spec, dict):
        if target_name not in adjustment_spec:
            raise KeyError(
                f"No adjustment set provided for feature '{target_name}'."
            )
        chosen = set(adjustment_spec[target_name])
        return [j for j, name in enumerate(feature_names) if name in chosen]

    if callable(adjustment_spec):
        chosen = set(adjustment_spec(target_name, all_names))
        return [j for j, name in enumerate(feature_names) if name in chosen]

    raise ValueError(
        "adjustment_spec must be None, 'all_except_i', a dict, or a callable."
    )


def _fit_nuisance_models(X_train, y_train, t_train, adj_idx, nuisance_model, random_state):
    X_adj_train = X_train[:, adj_idx]
    y_model = _get_nuisance_estimator(nuisance_model, random_state)
    t_model = _get_nuisance_estimator(nuisance_model, random_state)
    y_model.fit(X_adj_train, y_train)
    t_model.fit(X_adj_train, t_train)
    return y_model, t_model


def _compute_pooled_theta(
    X: np.ndarray,
    y: np.ndarray,
    target_idx: int,
    adj_idx: List[int],
    n_folds: int,
    nuisance_model: str,
    random_state: Optional[int],
) -> tuple[float, float, float, float, int]:
    """
    Cross-fit nuisances, pool residuals, and estimate theta with SE.

    Returns theta, std_error, t_statistic, p_value, n_residuals.
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    y_res_all: List[float] = []
    t_res_all: List[float] = []

    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        t_train = X_train[:, target_idx]
        t_test = X_test[:, target_idx]

        y_model, t_model = _fit_nuisance_models(
            X_train, y_train, t_train, adj_idx, nuisance_model, random_state
        )

        y_hat = y_model.predict(X_test[:, adj_idx])
        t_hat = t_model.predict(X_test[:, adj_idx])

        y_res_all.extend((y_test - y_hat).tolist())
        t_res_all.extend((t_test - t_hat).tolist())

    y_res = np.asarray(y_res_all, dtype=float)
    t_res = np.asarray(t_res_all, dtype=float)
    denom = float(np.dot(t_res, t_res))

    if denom <= 1e-12 or len(y_res) < 2:
        return 0.0, np.nan, np.nan, np.nan, len(y_res)

    theta = float(np.dot(t_res, y_res) / denom)
    residuals = y_res - theta * t_res
    dof = max(len(y_res) - 1, 1)
    sigma2 = float(np.sum(residuals ** 2) / dof)
    se = float(np.sqrt(sigma2 / denom))
    if se <= 1e-12:
        return theta, np.nan, np.nan, np.nan, len(y_res)

    t_stat = theta / se
    p_value = float(2 * stats.t.sf(abs(t_stat), df=dof))
    return theta, se, t_stat, p_value, len(y_res)


def _bootstrap_theta(
    X: np.ndarray,
    y: np.ndarray,
    target_idx: int,
    adj_idx: List[int],
    n_folds: int,
    nuisance_model: str,
    n_bootstrap: int,
    random_state: Optional[int],
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n_subjects = X.shape[0]
    thetas = []

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_subjects, size=n_subjects, replace=True)
        theta, _, _, _, _ = _compute_pooled_theta(
            X[boot_idx],
            y[boot_idx],
            target_idx,
            adj_idx,
            n_folds,
            nuisance_model,
            random_state,
        )
        thetas.append(theta)

    return np.asarray(thetas, dtype=float)


class DoubleMachineLearning:
    """
    Estimate feature-level partial-linear effects via cross-fitted DML.

    For each feature F_i, treats F_i as treatment and uses an adjustment set
    W (default: all other features in X) to estimate:

        theta_i = Cov(Y - E[Y|W], F_i - E[F_i|W]) / Var(F_i - E[F_i|W])

    Parameters
    ----------
    n_folds : int
        Number of cross-fitting folds (default 5).
    nuisance_model : str
        Learner for E[Y|W] and E[F_i|W]. Options: 'linear', 'ridge', 'rf'.
    adjustment_spec : None, 'all_except_i', dict, or callable
        Controls which features enter W for each target feature.
        - None / 'all_except_i': all columns in X except F_i
        - dict: {feature_name: [adjustment feature names]}
        - callable: fn(target_name, all_names) -> adjustment names
    n_bootstrap : int
        Number of bootstrap replicates for CIs. If 0, uses analytic SE.
    confidence_level : float
        Confidence level for intervals (default 0.95).
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        n_folds: int = 5,
        nuisance_model: str = "ridge",
        adjustment_spec: AdjustmentSpec = "all_except_i",
        n_bootstrap: int = 0,
        confidence_level: float = 0.95,
        random_state: Optional[int] = 42,
    ):
        self.n_folds = n_folds
        self.nuisance_model = nuisance_model
        self.adjustment_spec = adjustment_spec
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.results_: Optional[pd.DataFrame] = None
        self.feature_names_: Optional[List[str]] = None

    def estimate_named_feature(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        target_name: str,
        adjustment_feature_names: Optional[List[str]] = None,
        X_adjustment: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Estimate DML for one named feature.

        If X_adjustment is provided, nuisances use that matrix (mentor's
        "all input features" pool) while treatment is still target_name
        in X.
        """
        if target_name not in feature_names:
            raise KeyError(f"Target '{target_name}' not in feature_names.")

        target_idx = feature_names.index(target_name)
        if X_adjustment is not None:
            if adjustment_feature_names is None:
                raise ValueError(
                    "adjustment_feature_names required when X_adjustment is set."
                )
            if target_name not in adjustment_feature_names:
                raise KeyError(
                    f"Target '{target_name}' not in adjustment_feature_names."
                )
            adj_names = adjustment_feature_names
            adj_idx = [
                j for j, name in enumerate(adj_names) if name != target_name
            ]
            X_nuisance = X_adjustment
            target_idx = adjustment_feature_names.index(target_name)
        else:
            adj_names = feature_names
            X_nuisance = X
            adj_idx = _resolve_adjustment_indices(
                feature_names, target_idx, self.adjustment_spec
            )

        return self._estimate_feature(
            X_nuisance,
            y,
            adj_names,
            target_idx,
            adj_idx_override=adj_idx,
        )

    def _estimate_feature(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        target_idx: int,
        adj_idx_override: Optional[List[int]] = None,
    ) -> dict:
        if adj_idx_override is not None:
            adj_idx = adj_idx_override
        else:
            adj_idx = _resolve_adjustment_indices(
                feature_names, target_idx, self.adjustment_spec
            )
        theta, se, t_stat, p_value, n_res = _compute_pooled_theta(
            X,
            y,
            target_idx,
            adj_idx,
            self.n_folds,
            self.nuisance_model,
            self.random_state,
        )

        if self.n_bootstrap > 0:
            boot_thetas = _bootstrap_theta(
                X,
                y,
                target_idx,
                adj_idx,
                self.n_folds,
                self.nuisance_model,
                self.n_bootstrap,
                self.random_state,
            )
            ci_low, ci_high = compute_confidence_interval(
                boot_thetas, self.confidence_level
            )
            std = float(np.std(boot_thetas, ddof=1)) if len(boot_thetas) > 1 else np.nan
            if np.isnan(p_value):
                p_value = float(
                    2 * np.mean(boot_thetas <= 0.0)
                    if theta >= 0
                    else 2 * np.mean(boot_thetas >= 0.0)
                )
        else:
            if se is not None and not np.isnan(se) and se > 0:
                h = se * stats.t.ppf((1 + self.confidence_level) / 2, max(n_res - 1, 1))
                ci_low, ci_high = theta - h, theta + h
                std = se
            else:
                ci_low, ci_high = theta, theta
                std = np.nan

        return {
            "feature": feature_names[target_idx],
            "causal_effect": theta,
            "std": std,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "t_statistic": t_stat,
            "p_value": p_value,
            "n_residuals": n_res,
            "n_adjustment_features": len(adj_idx),
            "significant": bool(p_value < (1 - self.confidence_level))
            if p_value is not None and not np.isnan(p_value)
            else False,
        }

    def fit(self, X, y, feature_names=None):
        """
        Run DML for every feature in X.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix (n_subjects, n_features).
        y : np.ndarray or pd.Series
            Outcome vector.
        feature_names : list of str, optional
            Feature names. Inferred from DataFrame columns if omitted.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X_arr = X.values
        else:
            X_arr = np.asarray(X, dtype=float)
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(X_arr.shape[1])]

        if isinstance(y, pd.Series):
            y_arr = y.values.astype(float)
        else:
            y_arr = np.asarray(y, dtype=float)

        n_subjects, n_features = X_arr.shape
        self.feature_names_ = feature_names

        print("Running Double Machine Learning...")
        print(f"  Subjects: {n_subjects}, Features: {n_features}")
        print(f"  CV folds: {self.n_folds}, Nuisance model: {self.nuisance_model}")
        if self.n_bootstrap > 0:
            print(f"  Bootstrap replicates: {self.n_bootstrap}")
        print("-" * 60)

        results = []
        for feat_idx in range(n_features):
            print(f"  Feature {feat_idx + 1}/{n_features}: {feature_names[feat_idx]}")
            results.append(self._estimate_feature(X_arr, y_arr, feature_names, feat_idx))

        self.results_ = pd.DataFrame(results)
        self.results_ = self.results_.sort_values("p_value", ascending=True).reset_index(
            drop=True
        )

        print("-" * 60)
        print("DML analysis complete.")
        return self

    def get_results(self) -> pd.DataFrame:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_

    def get_causal_features(self, alpha: float = 0.05) -> pd.DataFrame:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_[self.results_["p_value"] < alpha].reset_index(drop=True)

    def summary(self):
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")

        print("\n" + "=" * 70)
        print("DOUBLE MACHINE LEARNING — CAUSAL EFFECT SUMMARY")
        print("=" * 70)
        print(f"\n{'Feature':<25} {'Effect':>10} {'Std':>8} {'p-value':>10} {'Sig.':>6}")
        print("-" * 70)

        for _, row in self.results_.iterrows():
            p = row["p_value"]
            sig_marker = ""
            if p is not None and not np.isnan(p):
                sig_marker = (
                    "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                )
            std = row["std"]
            std_str = f"{std:>8.4f}" if std is not None and not np.isnan(std) else f"{'n/a':>8}"
            p_str = f"{p:>10.4f}" if p is not None and not np.isnan(p) else f"{'n/a':>10}"
            print(
                f"{row['feature']:<25} {row['causal_effect']:>10.4f} "
                f"{std_str} {p_str} {sig_marker:>6}"
            )

        print("-" * 70)
        n_sig = int((self.results_["p_value"] < 0.05).sum())
        print(f"\nSignificant features (p < 0.05): {n_sig}/{len(self.results_)}")
        print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
        print("=" * 70)


def dml_importance_ranking(
    X,
    y,
    n_folds: int = 5,
    nuisance_model: str = "ridge",
    adjustment_spec: AdjustmentSpec = "all_except_i",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Convenience wrapper: rank features by absolute DML effect.

    Returns DataFrame with columns: feature, score (theta).
    """
    feature_names, X_arr = _feature_matrix(X)
    y_arr = np.asarray(y, dtype=float)
    scores = []

    for j in range(X_arr.shape[1]):
        adj_idx = _resolve_adjustment_indices(feature_names, j, adjustment_spec)
        theta, _, _, _, _ = _compute_pooled_theta(
            X_arr,
            y_arr,
            j,
            adj_idx,
            n_folds,
            nuisance_model,
            random_state,
        )
        scores.append(theta)

    df = pd.DataFrame({"feature": feature_names, "score": scores})
    return df.reindex(df["score"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )

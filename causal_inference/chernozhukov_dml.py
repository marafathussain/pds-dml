"""
Chernozhukov et al. (2018) Partially Linear Regression (PLR) Double Machine Learning.

Improvements over a naive residual-on-residual implementation:
1. Post-double-selection adjustment sets (Belloni et al. 2014)
2. Post-Lasso OLS nuisance learners (cross-validated sparsity)
3. Neyman-orthogonal moment with influence-function standard errors
4. Explicit PLR estimand: theta = partial effect of T on Y given W
5. Global multiplicity helpers (Bonferroni)
6. Sensitivity ingredients (E-value approximation, robustness t-stat)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV, LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from typing import List, Literal, Optional, Sequence, Tuple

from .abcd_data import PARENT_EDUCATION_COLUMN
from .dml import _compute_pooled_theta
from .multiple_testing import global_bonferroni_threshold

AdjustmentMode = Literal["post_double_selection", "all_except_i"]
NuisanceMode = Literal["post_lasso_ols", "ridge"]
SeMode = Literal["influence_function", "residual"]


MANDATORY_ADJUSTMENTS = [
    PARENT_EDUCATION_COLUMN,
    "Age in months at the time of the interview/test/sampling/imaging. of ABCDSocioEconomical_DATA_LABELS_2022-08-23_1557",
    "Sex of subject at birth",
]

ESTIMAND_DESCRIPTION = (
    "theta: change in age-corrected standard-score outcome per one-unit increase "
    "in the treatment feature, holding the post-double-selection adjustment set fixed "
    "(partially linear PLR coefficient under unconfoundedness)."
)


def _post_double_selection_indices(
    X_pool: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    pool_names: List[str],
    target_name: str,
    mandatory: Sequence[str],
    max_lasso_features: int = 40,
    random_state: Optional[int] = 42,
) -> Tuple[List[int], dict]:
    """
    Belloni post-double-selection: union of Lasso supports for Y and T.
    """
    exclude = {target_name}
    candidate_idx = [i for i, n in enumerate(pool_names) if n not in exclude]
    X_c = X_pool[:, candidate_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_c)

    lasso_y = LassoCV(cv=3, n_jobs=-1, random_state=random_state, max_iter=8000)
    lasso_y.fit(X_scaled, y)
    support_y = set(np.where(np.abs(lasso_y.coef_) > 1e-8)[0])

    lasso_t = LassoCV(cv=3, n_jobs=-1, random_state=random_state, max_iter=8000)
    lasso_t.fit(X_scaled, t)
    support_t = set(np.where(np.abs(lasso_t.coef_) > 1e-8)[0])

    selected_local = list(support_y | support_t)
    if len(selected_local) > max_lasso_features:
        combined = np.abs(lasso_y.coef_) + np.abs(lasso_t.coef_)
        top = np.argsort(combined)[::-1][:max_lasso_features]
        selected_local = sorted(top.tolist())

    selected_names = {pool_names[candidate_idx[j]] for j in selected_local}
    for m in mandatory:
        if m in pool_names and m != target_name:
            selected_names.add(m)

    adj_idx = [i for i, n in enumerate(pool_names) if n in selected_names]
    report = {
        "n_pool": len(pool_names) - 1,
        "n_lasso_y": len(support_y),
        "n_lasso_t": len(support_t),
        "n_union": len(selected_names),
        "adjustment_features": sorted(selected_names),
    }
    return adj_idx, report


def _ridge_nuisance(X_train, y_train, X_test, alpha: float = 1.0):
    """Ridge nuisance learner (vanilla DML default)."""
    if X_train.shape[1] == 0:
        mean_y = float(np.mean(y_train))
        return np.full(X_test.shape[0], mean_y)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _post_lasso_ols(X_train, y_train, X_test):
    """Lasso variable selection on train, OLS refit on selected columns."""
    if X_train.shape[1] == 0:
        mean_y = float(np.mean(y_train))
        return np.full(X_test.shape[0], mean_y), []

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    lasso = LassoCV(cv=3, max_iter=5000, n_jobs=-1)
    lasso.fit(X_tr, y_train)
    support = np.where(np.abs(lasso.coef_) > 1e-8)[0].tolist()
    if not support:
        support = [int(np.argmax(np.abs(lasso.coef_)))]

    ols = LinearRegression()
    ols.fit(X_tr[:, support], y_train)
    pred = ols.predict(X_te[:, support])
    return pred, support


def _resolve_adjustment(
    X_pool: np.ndarray,
    y: np.ndarray,
    pool_names: List[str],
    target_name: str,
    adjustment_mode: AdjustmentMode,
    mandatory: Sequence[str],
    random_state: Optional[int],
) -> Tuple[int, List[int], dict]:
    target_idx = pool_names.index(target_name)
    if adjustment_mode == "all_except_i":
        adj_idx = [i for i in range(len(pool_names)) if i != target_idx]
        adj_report = {
            "mode": "all_except_i",
            "n_pool": len(pool_names) - 1,
            "n_union": len(adj_idx),
            "adjustment_features": [pool_names[i] for i in adj_idx],
        }
        return target_idx, adj_idx, adj_report

    adj_idx, adj_report = _post_double_selection_indices(
        X_pool,
        y,
        X_pool[:, target_idx],
        pool_names,
        target_name,
        mandatory,
        random_state=random_state,
    )
    adj_report["mode"] = "post_double_selection"
    return target_idx, adj_idx, adj_report


def _fit_nuisance_predictions(
    X_adj_train: np.ndarray,
    y_train: np.ndarray,
    X_adj_test: np.ndarray,
    nuisance_mode: NuisanceMode,
    ridge_alpha: float = 1.0,
):
    if nuisance_mode == "ridge":
        return _ridge_nuisance(X_adj_train, y_train, X_adj_test, alpha=ridge_alpha)
    pred, _ = _post_lasso_ols(X_adj_train, y_train, X_adj_test)
    return pred


def _compute_se(
    y_res: np.ndarray,
    t_res: np.ndarray,
    theta: float,
    se_mode: SeMode,
) -> Tuple[float, float, float, float, float]:
    n = len(y_res)
    t2_mean = float(np.mean(t_res ** 2))
    if t2_mean <= 1e-12 or n < 3:
        return np.nan, np.nan, np.nan, theta, theta

    if se_mode == "influence_function":
        psi = t_res * (y_res - theta * t_res)
        psi_var = float(np.mean(psi ** 2))
        se = float(np.sqrt(psi_var / (n * t2_mean ** 2)))
    else:
        residuals = y_res - theta * t_res
        dof = max(n - 1, 1)
        sigma2 = float(np.sum(residuals ** 2) / dof)
        se = float(np.sqrt(sigma2 / float(np.dot(t_res, t_res))))

    if se <= 1e-12:
        return np.nan, np.nan, np.nan, theta, theta

    t_stat = theta / se
    p_value = float(2 * stats.t.sf(abs(t_stat), df=max(n - 1, 1)))
    h = se * stats.t.ppf(0.975, df=max(n - 1, 1))
    return se, t_stat, p_value, theta - h, theta + h


def estimate_plr_dml(
    X_pool: np.ndarray,
    y: np.ndarray,
    pool_names: List[str],
    target_name: str,
    n_folds: int = 5,
    mandatory: Optional[Sequence[str]] = None,
    random_state: Optional[int] = 42,
    adjustment_mode: AdjustmentMode = "post_double_selection",
    nuisance_mode: NuisanceMode = "post_lasso_ols",
    se_mode: SeMode = "influence_function",
    ridge_alpha: float = 1.0,
) -> dict:
    """
    Chernozhukov PLR-DML with configurable ablation switches.

    Estimand: theta in Y = theta * T + m(W) + epsilon
    """
    if target_name not in pool_names:
        raise KeyError(f"Target '{target_name}' not in pool_names.")

    if mandatory is None:
        mandatory = [m for m in MANDATORY_ADJUSTMENTS if m in pool_names]

    target_idx, adj_idx, adj_report = _resolve_adjustment(
        X_pool, y, pool_names, target_name, adjustment_mode, mandatory, random_state
    )

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    y_res_all: List[float] = []
    t_res_all: List[float] = []

    for train_idx, test_idx in kfold.split(X_pool):
        X_train, X_test = X_pool[train_idx], X_pool[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        t_train = X_train[:, target_idx]
        t_test = X_test[:, target_idx]

        X_adj_train = X_train[:, adj_idx]
        X_adj_test = X_test[:, adj_idx]

        y_hat = _fit_nuisance_predictions(X_adj_train, y_train, X_adj_test, nuisance_mode, ridge_alpha)
        t_hat = _fit_nuisance_predictions(X_adj_train, t_train, X_adj_test, nuisance_mode, ridge_alpha)

        y_res_all.extend((y_test - y_hat).tolist())
        t_res_all.extend((t_test - t_hat).tolist())

    y_res = np.asarray(y_res_all, dtype=float)
    t_res = np.asarray(t_res_all, dtype=float)
    n = len(y_res)
    t2_mean = float(np.mean(t_res ** 2))

    if t2_mean <= 1e-12 or n < 3:
        return _empty_result(target_name, adj_report, n, adjustment_mode, nuisance_mode, se_mode)

    theta = float(np.dot(t_res, y_res) / np.dot(t_res, t_res))
    se, t_stat, p_value, ci_low, ci_high = _compute_se(y_res, t_res, theta, se_mode)

    e_value = _e_value_approximation(theta, se)
    rv_t = float(abs(t_stat)) if se is not None and not np.isnan(se) else np.nan

    estimand = ESTIMAND_DESCRIPTION
    if adjustment_mode == "all_except_i":
        estimand = (
            "theta under all-except-i adjustment (may differ from sparse PLR partial effect "
            "when irrelevant or bad-control features are included)."
        )

    return {
        "feature": target_name,
        "theta": theta,
        "std_error": se,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "t_statistic": t_stat,
        "p_value": p_value,
        "n_observations": n,
        "n_adjustment_features": len(adj_idx),
        "adjustment_report": adj_report,
        "adjustment_mode": adjustment_mode,
        "nuisance_mode": nuisance_mode,
        "se_mode": se_mode,
        "estimand": estimand,
        "e_value_approx": e_value,
        "robustness_t": rv_t,
        "significant_05": bool(p_value < 0.05) if not np.isnan(p_value) else False,
    }


def nuisance_crossfit_mse(
    X_pool: np.ndarray,
    y: np.ndarray,
    pool_names: List[str],
    target_name: str,
    n_folds: int = 5,
    nuisance_mode: NuisanceMode = "ridge",
    adjustment_mode: AdjustmentMode = "all_except_i",
    ridge_alpha: float = 1.0,
    random_state: Optional[int] = 42,
) -> dict:
    """Cross-fitted nuisance MSE for Y and T (ablation metric for improvement #2)."""
    target_idx, adj_idx, _ = _resolve_adjustment(
        X_pool, y, pool_names, target_name, adjustment_mode, [], random_state
    )
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    y_mse = []
    t_mse = []
    for train_idx, test_idx in kfold.split(X_pool):
        X_train, X_test = X_pool[train_idx], X_pool[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        t_train = X_train[:, target_idx]
        t_test = X_test[:, target_idx]
        X_adj_train = X_train[:, adj_idx]
        X_adj_test = X_test[:, adj_idx]
        y_hat = _fit_nuisance_predictions(X_adj_train, y_train, X_adj_test, nuisance_mode, ridge_alpha)
        t_hat = _fit_nuisance_predictions(X_adj_train, t_train, X_adj_test, nuisance_mode, ridge_alpha)
        y_mse.append(float(np.mean((y_test - y_hat) ** 2)))
        t_mse.append(float(np.mean((t_test - t_hat) ** 2)))
    return {
        "y_nuisance_mse": float(np.mean(y_mse)),
        "t_nuisance_mse": float(np.mean(t_mse)),
        "total_nuisance_mse": float(np.mean(y_mse) + np.mean(t_mse)),
    }


def estimate_vanilla_dml(
    X_pool: np.ndarray,
    y: np.ndarray,
    pool_names: List[str],
    target_name: str,
    n_folds: int = 5,
    random_state: Optional[int] = 42,
) -> dict:
    """Vanilla mentor-style DML: all-except-i ridge nuisances + residual SE."""
    target_idx = pool_names.index(target_name)
    adj_idx = [i for i in range(len(pool_names)) if i != target_idx]
    theta, se, t_stat, p_value, n = _compute_pooled_theta(
        X_pool,
        y,
        target_idx,
        adj_idx,
        n_folds,
        "ridge",
        random_state,
    )
    if se is not None and not np.isnan(se) and se > 0:
        h = se * stats.t.ppf(0.975, df=max(n - 1, 1))
        ci_low, ci_high = theta - h, theta + h
    else:
        ci_low, ci_high = theta, theta

    return {
        "feature": target_name,
        "theta": theta,
        "std_error": se,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "t_statistic": t_stat,
        "p_value": p_value,
        "n_observations": n,
        "n_adjustment_features": len(adj_idx),
        "adjustment_mode": "all_except_i",
        "nuisance_mode": "ridge",
        "se_mode": "residual",
        "estimand": "residual-on-residual slope with all-except-i ridge nuisances",
        "e_value_approx": np.nan,
        "robustness_t": np.nan,
        "significant_05": bool(p_value < 0.05) if not np.isnan(p_value) else False,
    }


def _empty_result(
    target_name: str,
    adj_report: dict,
    n: int,
    adjustment_mode: str = "post_double_selection",
    nuisance_mode: str = "post_lasso_ols",
    se_mode: str = "influence_function",
) -> dict:
    return {
        "feature": target_name,
        "theta": 0.0,
        "std_error": np.nan,
        "ci_lower": 0.0,
        "ci_upper": 0.0,
        "t_statistic": np.nan,
        "p_value": np.nan,
        "n_observations": n,
        "n_adjustment_features": adj_report.get("n_union", 0),
        "adjustment_report": adj_report,
        "adjustment_mode": adjustment_mode,
        "nuisance_mode": nuisance_mode,
        "se_mode": se_mode,
        "estimand": ESTIMAND_DESCRIPTION,
        "e_value_approx": 1.0,
        "robustness_t": np.nan,
        "significant_05": False,
    }


def _e_value_approximation(theta: float, se: float) -> float:
    """
    Rough E-value for sensitivity to unmeasured confounding (VanderWeele-style).
    Uses standardized effect magnitude; interpret as conservative screening metric.
    """
    if se is None or np.isnan(se) or se <= 1e-12:
        return 1.0
    z = abs(theta / se)
    if z < 1.96:
        return 1.0
    return float(min(np.exp(0.5 * min(z**2, 50)), 1e6))


def run_plr_dml_on_candidates(
    X_pool: pd.DataFrame,
    y: pd.Series,
    candidate_features: List[str],
    n_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run Chernozhukov PLR-DML for each candidate treatment feature."""
    pool_names = list(X_pool.columns)
    X_arr = X_pool.values.astype(float)
    y_arr = y.values.astype(float)

    rows = []
    for i, feat in enumerate(candidate_features):
        if feat not in pool_names:
            continue
        if verbose:
            print(f"  [{i + 1}/{len(candidate_features)}] {feat[:70]}...")
        rows.append(
            estimate_plr_dml(
                X_arr, y_arr, pool_names, feat, n_folds=n_folds, random_state=random_state
            )
        )

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("p_value", ascending=True, na_position="last").reset_index(drop=True)
    return df


def apply_study_bonferroni(results_by_outcome: dict[str, pd.DataFrame], alpha: float = 0.05) -> dict:
    """Apply global Bonferroni across all feature-outcome tests."""
    n_tests = sum(len(df) for df in results_by_outcome.values())
    thresh = global_bonferroni_threshold(n_tests, alpha)

    corrected = {}
    for outcome, df in results_by_outcome.items():
        out = df.copy()
        out["p_value_global_bonferroni"] = (out["p_value"] * n_tests).clip(upper=1.0)
        out["significant_global_bonferroni"] = out["p_value"] < thresh
        corrected[outcome] = out

    return {
        "n_tests_total": n_tests,
        "bonferroni_threshold": thresh,
        "alpha": alpha,
        "results": corrected,
    }

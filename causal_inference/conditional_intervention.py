"""
Conditional Intervention Analysis (CIA)

Implements the do-calculus approximation via conditional sampling.
For each feature, estimates the interventional effect on the outcome by:
1. Learning P(F_i | F_{-i}) via regression
2. Sampling plausible counterfactual values of F_i
3. Evaluating the outcome model under counterfactual inputs
4. Aggregating causal effect estimates across subjects and CV folds

Uses the same consensus of three ML techniques as FIBE:
  - Regression: Linear SVR + Gaussian SVR + Random Forest (averaged)
  - Classification: Linear SVC + Gaussian SVC + Random Forest (majority vote)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from typing import Optional

from .outcome_model import get_models, train_outcome_models, predict_outcome
from .utils import compute_confidence_interval


class ConditionalInterventionAnalysis:
    """
    Estimates feature-level interventional (causal) effects on an outcome
    using conditional sampling to generate biologically plausible counterfactuals.

    Uses consensus of three ML techniques (matching FIBE) for both the
    conditional distribution model and the outcome model.

    Parameters
    ----------
    task_type : str
        'regression' or 'classification'. Determines models and metrics.
    n_samples : int
        Number of counterfactual samples per subject per feature (K).
    n_folds : int
        Number of cross-validation folds for stability estimation.
    model_name : str
        Model selection for outcome prediction.
        For regression: 'linearSVR', 'gaussianSVR', 'RegressionForest', 'Ridge',
            'GradientBoosting', 'consensus', or 'consensus_fast'.
        For classification: 'linearSVC', 'gaussianSVC', 'RandomForest',
            'LogisticRegression', 'GradientBoosting', 'consensus', or 'consensus_fast'.
        Default is 'consensus'. Use 'consensus_fast' for large datasets (n > 1000),
        which replaces SVR/SVC with Ridge/LogisticRegression + GradientBoosting.
    conditional_model : str
        Model for learning P(F_i | F_{-i}). Always regression regardless of task_type.
        Options: 'linearSVR', 'gaussianSVR', 'RegressionForest', 'Ridge',
            'GradientBoosting', 'consensus', or 'consensus_fast'.
        Default is 'consensus'. Use 'consensus_fast' for large datasets.
    confidence_level : float
        Confidence level for effect intervals (default 0.95).
    max_subjects : int or None
        Maximum number of test subjects to evaluate per fold. If the test set
        is larger than this, a random subset is used. This dramatically reduces
        runtime for large datasets. E.g., set to 500 for datasets with n > 5000.
        If None (default), all test subjects are used.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        task_type: str = "regression",
        n_samples: int = 50,
        n_folds: int = 5,
        model_name: str = "consensus",
        conditional_model: str = "consensus",
        confidence_level: float = 0.95,
        max_subjects: Optional[int] = None,
        random_state: Optional[int] = 42,
    ):
        if task_type not in ("regression", "classification"):
            raise ValueError(
                f"task_type must be 'regression' or 'classification', got '{task_type}'."
            )
        self.task_type = task_type
        self.n_samples = n_samples
        self.n_folds = n_folds
        self.model_name = model_name
        self.conditional_model = conditional_model
        self.confidence_level = confidence_level
        self.max_subjects = max_subjects
        self.random_state = random_state
        self.results_ = None
        self.feature_names_ = None

    def _learn_conditional(self, X_train, feature_idx):
        """
        Learn P(F_i | F_{-i}) from training data using consensus of 3 regressors.

        The conditional distribution is always a regression problem (predicting
        the continuous/discrete feature value from others), regardless of
        whether the main task is classification or regression.

        Returns the fitted models dict and the residual standard deviation.
        """
        mask = np.ones(X_train.shape[1], dtype=bool)
        mask[feature_idx] = False

        X_other = X_train[:, mask]
        y_target = X_train[:, feature_idx]

        cond_models = get_models("regression", self.conditional_model, self.random_state)
        fitted_cond = {}
        all_residuals = []

        for name, model in cond_models.items():
            model.fit(X_other, y_target)
            fitted_cond[name] = model
            preds = model.predict(X_other)
            all_residuals.append(y_target - preds)

        avg_residuals = np.mean(all_residuals, axis=0)
        sigma = np.std(avg_residuals)

        return fitted_cond, sigma

    def _predict_conditional(self, fitted_cond_models, X_other_single):
        """Consensus prediction for conditional mean (always regression)."""
        preds = []
        for name, model in fitted_cond_models.items():
            preds.append(model.predict(X_other_single.reshape(1, -1))[0])
        return np.mean(preds)

    @staticmethod
    def _auto_detect_feature_types(X, feature_names):
        """
        Automatically detect whether each feature is continuous, categorical, or binary.

        Detection rules:
        1. Check if ALL values in the column are integers (even if stored as float).
           If not, it's 'continuous'.
        2. If all values are integers:
           a. If only 2 unique values exist and they are {0,1} -> 'binary'
           b. If the number of unique values is small (<=10) -> 'categorical'
           c. Otherwise -> 'continuous' (e.g., age stored as integer but with many levels)

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_subjects, n_features).
        feature_names : list of str
            Feature names for reporting.

        Returns
        -------
        types_list : list of str
            Detected type for each feature.
        """
        n_features = X.shape[1]
        types_list = []
        detected_info = []

        for i in range(n_features):
            col = X[:, i]

            # Check 1: Are all values integers?
            all_integer = np.all(col == np.floor(col))

            if not all_integer:
                types_list.append("continuous")
                detected_info.append(f"    {feature_names[i]}: continuous (has non-integer values)")
                continue

            # All values are integers; determine binary vs categorical vs continuous
            unique_values = np.unique(col)
            n_unique = len(unique_values)

            # Check 2: Binary, exactly 2 unique values that are 0 and 1
            if n_unique == 2 and set(unique_values) == {0.0, 1.0}:
                types_list.append("binary")
                detected_info.append(f"    {feature_names[i]}: binary (values: {{0, 1}})")

            # Check 3: Categorical, integer-valued with few unique levels (<=10)
            elif n_unique <= 10:
                types_list.append("categorical")
                vals_str = sorted(int(v) for v in unique_values)
                detected_info.append(
                    f"    {feature_names[i]}: categorical (integer, {n_unique} unique values: {vals_str})"
                )

            # Otherwise: many integer levels (e.g., age 18-90), treat as continuous
            else:
                types_list.append("continuous")
                detected_info.append(
                    f"    {feature_names[i]}: continuous (integer but {n_unique} unique values)"
                )

        print("  Auto-detected feature types:")
        for info in detected_info:
            print(info)

        return types_list

    def _sample_counterfactuals(self, fitted_cond_models, sigma, X_subject, feature_idx, scaler):
        """
        Sample K plausible counterfactual values for feature_idx.

        Handles three feature types:
          - 'continuous': samples from N(mu, sigma) without constraints
          - 'categorical': rounds to nearest integer and clips to [min, max] of
            observed values (e.g., sex: 1 or 2, education: 0-3)
          - 'binary': rounds to 0 or 1

        Since models operate in MinMaxScaled space, discrete constraints are applied
        by: (1) inverse-transforming to original space, (2) rounding/clipping,
        (3) re-scaling back. This ensures sampled values correspond to real
        discrete values in the original feature space.

        Returns array of shape (n_samples,) in scaled space.
        """
        mask = np.ones(len(X_subject), dtype=bool)
        mask[feature_idx] = False

        X_other = X_subject[mask]
        mu = self._predict_conditional(fitted_cond_models, X_other)

        rng = np.random.default_rng(self.random_state)
        counterfactual_values = rng.normal(loc=mu, scale=sigma, size=self.n_samples)

        # Apply discrete constraints based on feature type
        if self.feature_types_ is not None:
            ftype = self.feature_types_[feature_idx]

            if ftype in ("binary", "categorical"):
                # Get original-space scale parameters for this feature
                orig_min = scaler.data_min_[feature_idx]
                orig_max = scaler.data_max_[feature_idx]
                scale_range = orig_max - orig_min

                if scale_range > 0:
                    # Inverse transform: scaled -> original space
                    original_values = counterfactual_values * scale_range + orig_min

                    if ftype == "binary":
                        # Round to 0 or 1 in original space
                        original_values = np.clip(np.round(original_values), 0, 1)
                    elif ftype == "categorical":
                        # Round to integer within observed range
                        f_min = self.feature_ranges_[feature_idx][0]
                        f_max = self.feature_ranges_[feature_idx][1]
                        original_values = np.clip(np.round(original_values), f_min, f_max)

                    # Re-scale back to [0, 1] space
                    counterfactual_values = (original_values - orig_min) / scale_range

            # 'continuous' features: no modification needed

        return counterfactual_values

    def _compute_feature_effect_regression(self, X_test, outcome_models, cond_models, sigma, feature_idx, scaler):
        """
        Compute interventional effect for regression task (vectorized).

        Batches all counterfactual predictions into single model calls for speed.
        Measures normalized delta: (ΔO / ΔF) for each counterfactual.
        """
        n_test = X_test.shape[0]

        # Step 1: Predict original outcomes for all test subjects in one batch
        O_orig_all, _ = predict_outcome(outcome_models, X_test, "regression")

        # Step 2: For each subject, sample counterfactual values and build batch matrix
        all_cf_rows = []
        all_original_values = []
        all_cf_feature_values = []
        subject_indices = []

        for i in range(n_test):
            X_subject = X_test[i]
            original_value = X_subject[feature_idx]

            cf_values = self._sample_counterfactuals(
                cond_models, sigma, X_subject, feature_idx, scaler
            )

            for cf_val in cf_values:
                X_cf = X_subject.copy()
                X_cf[feature_idx] = cf_val
                all_cf_rows.append(X_cf)
                all_original_values.append(original_value)
                all_cf_feature_values.append(cf_val)
                subject_indices.append(i)

        if not all_cf_rows:
            return np.array([])

        # Step 3: Predict all counterfactual outcomes in one batch call
        X_cf_batch = np.array(all_cf_rows)
        O_cf_all, _ = predict_outcome(outcome_models, X_cf_batch, "regression")

        # Step 4: Compute deltas
        all_deltas = []
        for k in range(len(subject_indices)):
            delta_O = O_cf_all[k] - O_orig_all[subject_indices[k]]
            delta_F = all_cf_feature_values[k] - all_original_values[k]

            if abs(delta_F) > 1e-10:
                all_deltas.append(delta_O / delta_F)

        return np.array(all_deltas)

    def _compute_feature_effect_classification(self, X_test, outcome_models, cond_models, sigma, feature_idx, scaler):
        """
        Compute interventional effect for classification task (vectorized).

        Batches all counterfactual predictions into single model calls for speed.
        Uses change in predicted probability (ΔP / ΔF) as the effect measure.
        """
        n_test = X_test.shape[0]

        # Step 1: Predict original probabilities for all test subjects in one batch
        _, P_orig_all = predict_outcome(outcome_models, X_test, "classification")

        # Step 2: For each subject, sample counterfactual values and build batch matrix
        all_cf_rows = []
        all_original_values = []
        all_cf_feature_values = []
        subject_indices = []

        for i in range(n_test):
            X_subject = X_test[i]
            original_value = X_subject[feature_idx]

            cf_values = self._sample_counterfactuals(
                cond_models, sigma, X_subject, feature_idx, scaler
            )

            for cf_val in cf_values:
                X_cf = X_subject.copy()
                X_cf[feature_idx] = cf_val
                all_cf_rows.append(X_cf)
                all_original_values.append(original_value)
                all_cf_feature_values.append(cf_val)
                subject_indices.append(i)

        if not all_cf_rows:
            return np.array([])

        # Step 3: Predict all counterfactual probabilities in one batch call
        X_cf_batch = np.array(all_cf_rows)
        _, P_cf_all = predict_outcome(outcome_models, X_cf_batch, "classification")

        # Step 4: Compute deltas
        all_deltas = []
        for k in range(len(subject_indices)):
            delta_P = P_cf_all[k] - P_orig_all[subject_indices[k]]
            delta_F = all_cf_feature_values[k] - all_original_values[k]

            if abs(delta_F) > 1e-10:
                all_deltas.append(delta_P / delta_F)

        return np.array(all_deltas)

    def fit(self, X, y, feature_names=None, feature_types=None):
        """
        Run the full conditional intervention analysis.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix of shape (n_subjects, n_features).
        y : np.ndarray or pd.Series
            Outcome vector of shape (n_subjects,).
            For regression: continuous values.
            For classification: binary class labels (0/1).
        feature_names : list of str, optional
            Names for the features. If None and X is a DataFrame,
            column names are used.
        feature_types : str, dict, list, or None, optional
            Specifies the type of each feature for proper counterfactual sampling.
            - If None or 'auto': automatically detects types from the data using:
                  * All values are integers? If no → 'continuous'
                  * Only 2 unique values {0, 1}? → 'binary'
                  * Integer with ≤10 unique values? → 'categorical'
                  * Integer with >10 unique values? → 'continuous'
            - If dict: {feature_name_or_index: type_string}
            - If list: [type_string_for_each_feature]
            Valid type strings:
              'continuous'; float-valued, no constraints
              'categorical'; integer-valued within observed min/max range
                              (e.g., education level 0-3, severity grade 1-4)
              'binary'; only 0 or 1 (e.g., sex, presence/absence)

        Returns
        -------
        self : ConditionalInterventionAnalysis
            Fitted instance with results in self.results_.
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = X.astype(np.float64)
        y = y.astype(np.float64)

        n_subjects, n_features = X.shape

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        self.feature_names_ = feature_names

        # Process feature_types into a list indexed by feature position
        if feature_types is None or feature_types == "auto":
            # Auto-detect feature types from data
            self.feature_types_ = self._auto_detect_feature_types(X, feature_names)
        elif isinstance(feature_types, dict):
            # Convert dict (keyed by name or index) to a list
            types_list = ["continuous"] * n_features
            for key, ftype in feature_types.items():
                if isinstance(key, str):
                    idx = feature_names.index(key)
                else:
                    idx = key
                types_list[idx] = ftype
            self.feature_types_ = types_list
        elif isinstance(feature_types, list):
            if len(feature_types) != n_features:
                raise ValueError(
                    f"feature_types list length ({len(feature_types)}) must match "
                    f"number of features ({n_features})."
                )
            self.feature_types_ = feature_types
        else:
            raise ValueError("feature_types must be 'auto', a dict, a list, or None.")

        # Compute observed min/max for categorical/binary features (before scaling)
        self.feature_ranges_ = {}
        for i in range(n_features):
            if self.feature_types_[i] in ("categorical", "binary"):
                self.feature_ranges_[i] = (int(np.min(X[:, i])), int(np.max(X[:, i])))

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        fold_effects = {i: [] for i in range(n_features)}

        model_desc = self.model_name
        if self.model_name == "consensus":
            if self.task_type == "regression":
                model_desc = "consensus (Linear SVR + Gaussian SVR + RF)"
            else:
                model_desc = "consensus (Linear SVC + Gaussian SVC + RF)"
        elif self.model_name == "consensus_fast":
            if self.task_type == "regression":
                model_desc = "consensus_fast (Ridge + Gradient Boosting + RF)"
            else:
                model_desc = "consensus_fast (Logistic Reg + Gradient Boosting + RF)"

        cond_desc = self.conditional_model
        if self.conditional_model == "consensus_fast":
            cond_desc = "consensus_fast (Ridge + Gradient Boosting + RF)"

        print(f"Running Conditional Intervention Analysis...")
        print(f"  Task type: {self.task_type}")
        print(f"  Subjects: {n_subjects}, Features: {n_features}")
        print(f"  CV Folds: {self.n_folds}, Counterfactual samples: {self.n_samples}")
        if self.max_subjects:
            print(f"  Max test subjects per fold: {self.max_subjects}")
        print(f"  Outcome model: {model_desc}")
        print(f"  Conditional model: {cond_desc}")
        print("-" * 60)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Subsample test subjects if max_subjects is set (for large datasets)
            if self.max_subjects is not None and X_test.shape[0] > self.max_subjects:
                rng_sub = np.random.default_rng(self.random_state + fold_idx)
                sub_idx = rng_sub.choice(X_test.shape[0], self.max_subjects, replace=False)
                X_test = X_test[sub_idx]
                n_test_used = self.max_subjects
            else:
                n_test_used = X_test.shape[0]

            print(f"  Fold {fold_idx + 1}/{self.n_folds} "
                  f"(train: {X_train.shape[0]}, test: {n_test_used})...")

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            outcome_fitted = train_outcome_models(
                X_train_scaled, y_train,
                task_type=self.task_type,
                model_name=self.model_name,
                random_state=self.random_state,
            )

            for feat_idx in range(n_features):
                cond_models, sigma = self._learn_conditional(X_train_scaled, feat_idx)

                if sigma < 1e-10:
                    fold_effects[feat_idx].append(0.0)
                    continue

                if self.task_type == "regression":
                    deltas = self._compute_feature_effect_regression(
                        X_test_scaled, outcome_fitted, cond_models, sigma, feat_idx, scaler
                    )
                else:
                    deltas = self._compute_feature_effect_classification(
                        X_test_scaled, outcome_fitted, cond_models, sigma, feat_idx, scaler
                    )

                if len(deltas) > 0:
                    fold_effects[feat_idx].append(np.mean(deltas))
                else:
                    fold_effects[feat_idx].append(0.0)

        results = []
        for feat_idx in range(n_features):
            effects = np.array(fold_effects[feat_idx])
            mean_effect = np.mean(effects)
            std_effect = np.std(effects)
            ci_low, ci_high = compute_confidence_interval(
                effects, self.confidence_level
            )

            t_stat, p_value = stats.ttest_1samp(effects, 0.0)

            results.append({
                "feature": feature_names[feat_idx],
                "causal_effect": mean_effect,
                "std": std_effect,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < (1 - self.confidence_level),
            })

        self.results_ = pd.DataFrame(results)
        self.results_ = self.results_.sort_values(
            "p_value", ascending=True
        ).reset_index(drop=True)

        print("-" * 60)
        print("Analysis complete.")
        return self

    def get_results(self):
        """Return results as a pandas DataFrame."""
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_

    def get_causal_features(self, alpha=0.05):
        """
        Return features with statistically significant causal effects.

        Parameters
        ----------
        alpha : float
            Significance threshold.

        Returns
        -------
        pd.DataFrame
            Subset of results where p_value < alpha.
        """
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_[self.results_["p_value"] < alpha].reset_index(drop=True)

    def summary(self):
        """Print a formatted summary of results."""
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")

        effect_label = "Effect" if self.task_type == "regression" else "Prob. Effect"

        print("\n" + "=" * 70)
        print("CONDITIONAL INTERVENTION ANALYSIS - CAUSAL EFFECT SUMMARY")
        print(f"Task: {self.task_type} | Model: {self.model_name}")
        print("=" * 70)
        print(f"\n{'Feature':<25} {effect_label:>12} {'Std':>8} {'p-value':>10} {'Sig.':>6}")
        print("-" * 70)

        for _, row in self.results_.iterrows():
            sig_marker = "***" if row["p_value"] < 0.001 else (
                "**" if row["p_value"] < 0.01 else (
                    "*" if row["p_value"] < 0.05 else ""
                )
            )
            print(
                f"{row['feature']:<25} {row['causal_effect']:>12.4f} "
                f"{row['std']:>8.4f} {row['p_value']:>10.4f} {sig_marker:>6}"
            )

        print("-" * 70)
        n_sig = (self.results_["p_value"] < 0.05).sum()
        print(f"\nSignificant features (p < 0.05): {n_sig}/{len(self.results_)}")
        print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
        if self.task_type == "classification":
            print("Effect = change in predicted probability per unit change in feature")
        else:
            print("Effect = change in predicted outcome per unit change in feature")
        print("=" * 70)

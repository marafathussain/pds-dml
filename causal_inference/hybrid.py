"""
CIA-DML hybrid: two-stage interventional screening and confirmation.

Stage 1 (CIA): Screen features for conditional interventional evidence.
Stage 2 (DML): Confirm shortlisted features with cross-fitted partial-linear
effects and bootstrap confidence intervals.

A feature is *confirmed* when it passes both stages (CIA screen + DML confirm).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .conditional_intervention import ConditionalInterventionAnalysis
from .dml import AdjustmentSpec, DoubleMachineLearning


class CIA_DML_Pipeline:
    """
    Two-stage CIA-DML hybrid for tabular causal feature screening.

    **Stage 1 — CIA screening:** Run Conditional Intervention Analysis on all
    features. Shortlist candidates by significance (p < alpha) and/or top-k
    |effect|.

    **Stage 2 — DML confirmation:** For each shortlisted feature, estimate a
    debiased partial-linear effect theta via cross-fitted Double Machine
    Learning. A feature is *confirmed* if DML also reaches significance.

    Parameters
    ----------
    screen_alpha : float
        CIA p-value threshold for screening (default 0.05).
    screen_top_k : int or None
        Also shortlist top-k features by |CIA effect|. If None, only alpha
        is used unless no features pass alpha (then top-k fallback applies).
    confirm_alpha : float
        DML p-value threshold for confirmation (default 0.05).
    fallback_top_k : int
        If no features pass CIA screening, shortlist this many by |effect|.
    task_type : str
        'regression' or 'classification' (CIA stage only).
    n_samples, n_folds : int
        Passed to CIA (counterfactual samples, CV folds).
    model_name, conditional_model : str
        CIA outcome and conditional models.
    max_subjects : int or None
        CIA subsampling cap for large datasets.
    nuisance_model : str
        DML nuisance learner ('linear', 'ridge', 'rf').
    adjustment_spec : adjustment set for DML stage.
    n_bootstrap : int
        Bootstrap replicates for DML CIs (0 = analytic SE).
    confidence_level : float
        Confidence level for both stages.
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        screen_alpha: float = 0.05,
        screen_top_k: Optional[int] = None,
        confirm_alpha: float = 0.05,
        fallback_top_k: int = 5,
        task_type: str = "regression",
        n_samples: int = 50,
        n_folds: int = 5,
        model_name: str = "consensus",
        conditional_model: str = "consensus",
        max_subjects: Optional[int] = None,
        nuisance_model: str = "ridge",
        adjustment_spec: AdjustmentSpec = "all_except_i",
        n_bootstrap: int = 200,
        confidence_level: float = 0.95,
        random_state: Optional[int] = 42,
    ):
        self.screen_alpha = screen_alpha
        self.screen_top_k = screen_top_k
        self.confirm_alpha = confirm_alpha
        self.fallback_top_k = fallback_top_k
        self.task_type = task_type
        self.n_samples = n_samples
        self.n_folds = n_folds
        self.model_name = model_name
        self.conditional_model = conditional_model
        self.max_subjects = max_subjects
        self.nuisance_model = nuisance_model
        self.adjustment_spec = adjustment_spec
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state

        self.cia_: Optional[ConditionalInterventionAnalysis] = None
        self.dml_: Optional[DoubleMachineLearning] = None
        self.results_: Optional[pd.DataFrame] = None
        self.screened_features_: Optional[List[str]] = None
        self.confirmed_features_: Optional[List[str]] = None
        self.feature_names_: Optional[List[str]] = None

    def _resolve_shortlist(self, cia_results: pd.DataFrame) -> List[str]:
        by_effect = cia_results.sort_values(
            "causal_effect", key=lambda s: s.abs(), ascending=False
        )
        significant = cia_results.loc[
            cia_results["p_value"] < self.screen_alpha, "feature"
        ].tolist()

        shortlist = set(significant)
        if self.screen_top_k is not None:
            shortlist.update(by_effect["feature"].head(self.screen_top_k).tolist())

        if not shortlist:
            shortlist = set(
                by_effect["feature"].head(self.fallback_top_k).tolist()
            )
            print(
                f"  No CIA features at p < {self.screen_alpha}; "
                f"fallback top-{self.fallback_top_k} by |effect|."
            )

        return sorted(shortlist)

    def fit(
        self,
        X,
        y,
        feature_names=None,
        dml_X=None,
        dml_feature_names=None,
    ):
        """
        Run two-stage CIA screening followed by DML confirmation.

        Parameters
        ----------
        dml_X, dml_feature_names : optional
            If set, DML nuisances adjust for all columns in dml_X (e.g. full
            ABCD predictor pool) while CIA still runs on X.

        Returns
        -------
        self
            Results in self.results_; confirmed features in self.confirmed_features_.
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

        if dml_X is not None:
            if isinstance(dml_X, pd.DataFrame):
                if dml_feature_names is None:
                    dml_feature_names = list(dml_X.columns)
                dml_arr = dml_X.values
            else:
                dml_arr = np.asarray(dml_X, dtype=float)
                if dml_feature_names is None:
                    dml_feature_names = [
                        f"DML_Feature_{i}" for i in range(dml_arr.shape[1])
                    ]
        else:
            dml_arr = None
            dml_feature_names = None

        self.feature_names_ = feature_names
        n_features = len(feature_names)

        print("=" * 70)
        print("  CIA-DML HYBRID PIPELINE")
        print("=" * 70)

        print("\n[Stage 1] CIA screening on all features...")
        self.cia_ = ConditionalInterventionAnalysis(
            task_type=self.task_type,
            n_samples=self.n_samples,
            n_folds=self.n_folds,
            model_name=self.model_name,
            conditional_model=self.conditional_model,
            confidence_level=self.confidence_level,
            max_subjects=self.max_subjects,
            random_state=self.random_state,
        )
        self.cia_.fit(X, y, feature_names=feature_names)
        cia_results = self.cia_.get_results().set_index("feature")

        shortlist = self._resolve_shortlist(cia_results.reset_index())
        self.screened_features_ = shortlist
        print(f"\n  Shortlisted {len(shortlist)}/{n_features} features:")
        for feat in shortlist:
            row = cia_results.loc[feat]
            print(
                f"    {feat}: CIA effect={row['causal_effect']:.4f}, "
                f"p={row['p_value']:.4f}"
            )

        if dml_arr is not None:
            print(
                f"\n[Stage 2] DML confirmation "
                f"(nuisances: {len(dml_feature_names)}-feature full pool)..."
            )
        else:
            print("\n[Stage 2] DML confirmation on shortlisted features...")
        self.dml_ = DoubleMachineLearning(
            n_folds=self.n_folds,
            nuisance_model=self.nuisance_model,
            adjustment_spec=self.adjustment_spec,
            n_bootstrap=self.n_bootstrap,
            confidence_level=self.confidence_level,
            random_state=self.random_state,
        )

        dml_by_feature = {}
        for feat in shortlist:
            print(f"  Confirming: {feat}")
            if dml_arr is not None:
                dml_by_feature[feat] = self.dml_.estimate_named_feature(
                    X_arr,
                    y_arr,
                    feature_names,
                    feat,
                    adjustment_feature_names=dml_feature_names,
                    X_adjustment=dml_arr,
                )
            else:
                feat_idx = feature_names.index(feat)
                dml_by_feature[feat] = self.dml_._estimate_feature(
                    X_arr, y_arr, feature_names, feat_idx
                )

        confirmed = []
        rows = []
        for feat in feature_names:
            cia_row = cia_results.loc[feat]
            screened = feat in shortlist
            dml_row = dml_by_feature.get(feat)

            dml_sig = False
            if dml_row is not None:
                p = dml_row["p_value"]
                dml_sig = p is not None and not np.isnan(p) and p < self.confirm_alpha
                if dml_sig:
                    confirmed.append(feat)

            rows.append({
                "feature": feat,
                "cia_effect": cia_row["causal_effect"],
                "cia_std": cia_row["std"],
                "cia_ci_lower": cia_row["ci_lower"],
                "cia_ci_upper": cia_row["ci_upper"],
                "cia_p_value": cia_row["p_value"],
                "cia_significant": bool(cia_row["p_value"] < self.screen_alpha),
                "screened": screened,
                "dml_theta": dml_row["causal_effect"] if dml_row else np.nan,
                "dml_std": dml_row["std"] if dml_row else np.nan,
                "dml_ci_lower": dml_row["ci_lower"] if dml_row else np.nan,
                "dml_ci_upper": dml_row["ci_upper"] if dml_row else np.nan,
                "dml_p_value": dml_row["p_value"] if dml_row else np.nan,
                "dml_significant": dml_sig,
                "confirmed": screened and dml_sig,
            })

        self.confirmed_features_ = confirmed
        self.results_ = pd.DataFrame(rows)
        self.results_ = self.results_.sort_values(
            ["confirmed", "cia_p_value"],
            ascending=[False, True],
        ).reset_index(drop=True)

        print("-" * 70)
        print(
            f"  Confirmed {len(confirmed)}/{n_features} features "
            f"(CIA screen + DML confirm at p < {self.confirm_alpha})."
        )
        if confirmed:
            print(f"  Confirmed: {confirmed}")
        print("=" * 70)
        return self

    def get_results(self) -> pd.DataFrame:
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")
        return self.results_

    def get_screened_features(self) -> List[str]:
        if self.screened_features_ is None:
            raise RuntimeError("Call .fit() first.")
        return list(self.screened_features_)

    def get_confirmed_features(self) -> List[str]:
        if self.confirmed_features_ is None:
            raise RuntimeError("Call .fit() first.")
        return list(self.confirmed_features_)

    def summary(self):
        if self.results_ is None:
            raise RuntimeError("Call .fit() first.")

        print("\n" + "=" * 78)
        print("CIA-DML HYBRID — SUMMARY")
        print("=" * 78)
        print(
            f"\n{'Feature':<22} {'CIA eff':>8} {'CIA p':>8} "
            f"{'DML eff':>8} {'DML p':>8} {'Stage':>10}"
        )
        print("-" * 78)

        for _, row in self.results_.iterrows():
            if row["confirmed"]:
                stage = "CONFIRMED"
            elif row["screened"]:
                stage = "screened"
            else:
                stage = "—"

            dml_theta = row["dml_theta"]
            dml_p = row["dml_p_value"]
            dml_theta_str = f"{dml_theta:>8.4f}" if not np.isnan(dml_theta) else f"{'—':>8}"
            dml_p_str = f"{dml_p:>8.4f}" if not np.isnan(dml_p) else f"{'—':>8}"

            print(
                f"{row['feature']:<22} {row['cia_effect']:>8.4f} "
                f"{row['cia_p_value']:>8.4f} {dml_theta_str} {dml_p_str} {stage:>10}"
            )

        n_screened = int(self.results_["screened"].sum())
        n_confirmed = int(self.results_["confirmed"].sum())
        print("-" * 78)
        print(f"Screened: {n_screened}  |  Confirmed: {n_confirmed}")
        print("=" * 78)

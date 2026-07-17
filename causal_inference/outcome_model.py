"""
Outcome Model Training and Prediction

Provides the same consensus of three ML techniques used in FIBE:
  - Regression: Linear SVR, Gaussian SVR, Random Forest Regressor
  - Classification: Linear SVC, Gaussian SVC, Random Forest Classifier

Also provides fast/scalable alternatives for large datasets (n > 1000):
  - Regression: Ridge, Gradient Boosting, Random Forest
  - Classification: Logistic Regression, Gradient Boosting, Random Forest

Supports individual models or consensus (averaging for regression,
majority voting for classification).
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, confusion_matrix,
)
from typing import Optional


def get_models(task_type: str, model_name: str = "consensus", random_state: Optional[int] = 42):
    """
    Get model(s) matching FIBE's ML techniques.

    Parameters
    ----------
    task_type : str
        'regression' or 'classification'.
    model_name : str
        Model selection. For regression: 'linearSVR', 'gaussianSVR',
        'RegressionForest', or 'consensus'. For classification:
        'linearSVC', 'gaussianSVC', 'RandomForest', or 'consensus'.
        Default is 'consensus'.
    random_state : int or None
        Random seed.

    Returns
    -------
    models : dict
        Dictionary of {name: model_instance}. For single models,
        dict has one entry. For consensus, dict has three entries.
    """
    if task_type == "regression":
        if model_name == "consensus":
            return {
                "Linear SVR": SVR(kernel="linear", C=1.0, epsilon=0.2),
                "Gaussian SVR": SVR(kernel="rbf", C=1.0, gamma="scale"),
                "Regression Forest": RandomForestRegressor(
                    n_estimators=100, random_state=random_state, max_depth=5
                ),
            }
        elif model_name == "consensus_fast":
            return {
                "Ridge": Ridge(alpha=1.0),
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=100, random_state=random_state, max_depth=5
                ),
                "Regression Forest": RandomForestRegressor(
                    n_estimators=100, random_state=random_state, max_depth=5, n_jobs=-1
                ),
            }
        elif model_name == "linearSVR":
            return {"Linear SVR": SVR(kernel="linear", C=1.0, epsilon=0.2)}
        elif model_name == "gaussianSVR":
            return {"Gaussian SVR": SVR(kernel="rbf", C=1.0, gamma="scale")}
        elif model_name == "RegressionForest":
            return {
                "Regression Forest": RandomForestRegressor(
                    n_estimators=100, random_state=random_state, max_depth=5, n_jobs=-1
                )
            }
        elif model_name == "Ridge":
            return {"Ridge": Ridge(alpha=1.0)}
        elif model_name == "GradientBoosting":
            return {
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=100, random_state=random_state, max_depth=5
                )
            }
        else:
            raise ValueError(
                f"Unknown model_name '{model_name}' for regression. "
                f"Choose from 'linearSVR', 'gaussianSVR', 'RegressionForest', "
                f"'Ridge', 'GradientBoosting', 'consensus', or 'consensus_fast'."
            )

    elif task_type == "classification":
        if model_name == "consensus":
            return {
                "Linear SVC": SVC(kernel="linear", C=1.0, probability=True),
                "Gaussian SVC": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True),
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=random_state, max_depth=5
                ),
            }
        elif model_name == "consensus_fast":
            return {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, random_state=random_state
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=100, random_state=random_state, max_depth=5
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=random_state, max_depth=5, n_jobs=-1
                ),
            }
        elif model_name == "linearSVC":
            return {"Linear SVC": SVC(kernel="linear", C=1.0, probability=True)}
        elif model_name == "gaussianSVC":
            return {"Gaussian SVC": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)}
        elif model_name == "RandomForest":
            return {
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=random_state, max_depth=5, n_jobs=-1
                )
            }
        elif model_name == "LogisticRegression":
            return {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, random_state=random_state
                )
            }
        elif model_name == "GradientBoosting":
            return {
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=100, random_state=random_state, max_depth=5
                )
            }
        else:
            raise ValueError(
                f"Unknown model_name '{model_name}' for classification. "
                f"Choose from 'linearSVC', 'gaussianSVC', 'RandomForest', "
                f"'LogisticRegression', 'GradientBoosting', 'consensus', or 'consensus_fast'."
            )
    else:
        raise ValueError(
            f"Unknown task_type '{task_type}'. Choose 'regression' or 'classification'."
        )


def train_outcome_models(X_train, y_train, task_type="regression",
                         model_name="consensus", random_state=42):
    """
    Train outcome prediction model(s).

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (n_samples, n_features).
    y_train : np.ndarray
        Training outcome vector (n_samples,).
    task_type : str
        'regression' or 'classification'.
    model_name : str
        Model name or 'consensus'.
    random_state : int or None
        Random seed.

    Returns
    -------
    fitted_models : dict
        Dictionary of {name: fitted_model}.
    """
    models = get_models(task_type, model_name, random_state)
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def predict_outcome(fitted_models, X, task_type="regression"):
    """
    Generate consensus outcome predictions.

    For regression: averages predictions across all models.
    For classification: uses majority voting for class, averages probabilities.

    Parameters
    ----------
    fitted_models : dict
        Dictionary of {name: fitted_model}.
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    task_type : str
        'regression' or 'classification'.

    Returns
    -------
    predictions : np.ndarray
        Consensus predictions.
    probabilities : np.ndarray or None
        For classification, averaged class-1 probabilities. None for regression.
    """
    all_preds = []
    all_probs = []

    for name, model in fitted_models.items():
        pred = model.predict(X)
        all_preds.append(pred)

        if task_type == "classification":
            prob = model.predict_proba(X)[:, 1]
            all_probs.append(prob)

    if task_type == "regression":
        consensus_pred = np.mean(all_preds, axis=0)
        return consensus_pred, None
    else:
        if len(all_preds) >= 3:
            consensus_pred = np.array([
                1 if sum(votes) > len(all_preds) / 2 else 0
                for votes in zip(*all_preds)
            ])
        else:
            consensus_pred = all_preds[0]

        consensus_prob = np.mean(all_probs, axis=0) if all_probs else None
        return consensus_pred, consensus_prob


def evaluate_outcome_models(fitted_models, X_test, y_test, task_type="regression", metric=None):
    """
    Evaluate consensus model performance.

    Parameters
    ----------
    fitted_models : dict
        Dictionary of fitted models.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        True outcomes.
    task_type : str
        'regression' or 'classification'.
    metric : str or None
        Metric to use. Defaults: 'MAE' for regression, 'Accuracy' for classification.

    Returns
    -------
    metrics : dict
        Performance metrics.
    """
    predictions, probabilities = predict_outcome(fitted_models, X_test, task_type)

    if task_type == "regression":
        if metric is None:
            metric = "MAE"
        result = {
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R2": r2_score(y_test, predictions),
        }
    else:
        if metric is None:
            metric = "Accuracy"
        acc = accuracy_score(y_test, predictions)
        result = {"Accuracy": acc}

        try:
            cm = confusion_matrix(y_test, predictions)
            tn, fp, fn, tp = cm.ravel()
            epsilon = 1e-7
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            sensitivity = tp / (tp + fn + epsilon)
            specificity = tn / (tn + fp + epsilon)
            result.update({
                "F1-score": f1,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
            })
        except ValueError:
            pass

    return result

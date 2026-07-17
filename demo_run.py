"""
Demo: Conditional Intervention Analysis (CIA)

Demonstrates causal effect estimation for regression and classification.
Also includes a large-dataset demo using consensus_fast and max_subjects.

Usage:
    python demo_run.py                          # regression + classification
    python demo_run.py --task regression        # small-data regression
    python demo_run.py --task classification    # small-data classification
    python demo_run.py --task large             # large-data (consensus_fast)
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from causal_inference import (
    ConditionalInterventionAnalysis,
    generate_synthetic_data,
    format_results_table,
)


def run_regression_demo():
    """Regression demo with FIBE-style consensus (small data)."""

    print("\n" + "=" * 70)
    print("  CIA DEMO, REGRESSION TASK (consensus)")
    print("  Models: Linear SVR + Gaussian SVR + Random Forest")
    print("=" * 70)

    print("\n[Step 1] Generating synthetic regression data...")
    print("  - 80 subjects, 8 features (3 causal, 5 correlated non-causal)")

    X, y, true_causal = generate_synthetic_data(
        n_subjects=80,
        n_features=8,
        n_causal=3,
        noise_level=0.5,
        task_type="regression",
        random_state=42,
    )

    print(f"  Ground truth causal: {true_causal}")
    print(f"  X shape: {X.shape}, y range: [{y.min():.2f}, {y.max():.2f}]")

    print("\n[Step 2] Initializing CIA (consensus)...")
    cia = ConditionalInterventionAnalysis(
        task_type="regression",
        n_samples=30,
        n_folds=5,
        model_name="consensus",
        conditional_model="consensus",
        confidence_level=0.95,
        random_state=42,
    )

    print("\n[Step 3] Running analysis...\n")
    cia.fit(X, y)
    cia.summary()

    print("\n[Step 4] Significant features (p < 0.05):")
    causal_features = cia.get_causal_features(alpha=0.05)
    if len(causal_features) > 0:
        for _, row in causal_features.iterrows():
            print(f"    - {row['feature']}: effect = {row['causal_effect']:.4f}, p = {row['p_value']:.4f}")
    else:
        print("    None found.")

    print("\n[Step 5] Validation against ground truth:")
    _validate(causal_features, true_causal)

    plot_results(
        cia.get_results(),
        "causal_effects_regression.png",
        "Regression",
        true_causal_features=true_causal,
    )

    return cia


def run_classification_demo():
    """Classification demo with FIBE-style consensus (small data)."""

    print("\n" + "=" * 70)
    print("  CIA DEMO, CLASSIFICATION TASK (consensus)")
    print("  Models: Linear SVC + Gaussian SVC + Random Forest")
    print("=" * 70)

    print("\n[Step 1] Generating synthetic classification data...")
    print("  - 100 subjects, 8 features (3 causal, 5 correlated non-causal)")

    X, y, true_causal = generate_synthetic_data(
        n_subjects=100,
        n_features=8,
        n_causal=3,
        noise_level=0.3,
        task_type="classification",
        random_state=42,
    )

    print(f"  Ground truth causal: {true_causal}")
    print(f"  Class distribution: {dict(pd.Series(y).value_counts())}")

    print("\n[Step 2] Initializing CIA (consensus)...")
    cia = ConditionalInterventionAnalysis(
        task_type="classification",
        n_samples=30,
        n_folds=5,
        model_name="consensus",
        conditional_model="consensus",
        confidence_level=0.95,
        random_state=42,
    )

    print("\n[Step 3] Running analysis...\n")
    cia.fit(X, y)
    cia.summary()

    print("\n[Step 4] Significant features (p < 0.05):")
    causal_features = cia.get_causal_features(alpha=0.05)
    if len(causal_features) > 0:
        for _, row in causal_features.iterrows():
            print(f"    - {row['feature']}: prob. effect = {row['causal_effect']:.4f}, p = {row['p_value']:.4f}")
    else:
        print("    None found.")

    print("\n[Step 5] Validation against ground truth:")
    _validate(causal_features, true_causal)

    plot_results(
        cia.get_results(),
        "causal_effects_classification.png",
        "Classification",
        true_causal_features=true_causal,
    )

    return cia


def run_large_dataset_demo():
    """
    Large-dataset demo using consensus_fast and max_subjects.

    Mimics settings suitable for cohorts like ABCD (thousands of subjects,
    dozens of FIBE-selected features). Uses a synthetic dataset with
    500 subjects and 12 features for a runnable example.
    """

    print("\n" + "=" * 70)
    print("  CIA DEMO, LARGE DATASET (consensus_fast)")
    print("  Models: Ridge + Gradient Boosting + Random Forest")
    print("  Options: max_subjects=100, n_samples=15, n_folds=3")
    print("=" * 70)

    print("\n[Step 1] Generating larger synthetic dataset...")
    print("  - 500 subjects, 12 features (3 causal, 9 correlated non-causal)")

    X, y, true_causal = generate_synthetic_data(
        n_subjects=500,
        n_features=12,
        n_causal=3,
        noise_level=0.5,
        task_type="regression",
        random_state=42,
    )

    print(f"  Ground truth causal: {true_causal}")
    print(f"  X shape: {X.shape}")

    print("\n[Step 2] Initializing CIA for large data...")
    print("  Use these settings for real large cohorts (e.g., n=9000, p=47):")
    print("    model_name='consensus_fast'")
    print("    conditional_model='consensus_fast'")
    print("    max_subjects=300, n_samples=20, n_folds=5")

    cia = ConditionalInterventionAnalysis(
        task_type="regression",
        n_samples=15,
        n_folds=3,
        model_name="consensus_fast",
        conditional_model="consensus_fast",
        max_subjects=100,
        confidence_level=0.95,
        random_state=42,
    )

    print("\n[Step 3] Running analysis...\n")
    t0 = time.time()
    cia.fit(X, y)
    elapsed = time.time() - t0
    print(f"\n  Elapsed time: {elapsed:.1f} seconds")

    cia.summary()

    print("\n[Step 4] Significant features (p < 0.05):")
    causal_features = cia.get_causal_features(alpha=0.05)
    if len(causal_features) > 0:
        for _, row in causal_features.iterrows():
            print(f"    - {row['feature']}: effect = {row['causal_effect']:.4f}, p = {row['p_value']:.4f}")
    else:
        print("    None found.")

    print("\n[Step 5] Validation against ground truth:")
    _validate(causal_features, true_causal)

    plot_results(
        cia.get_results(),
        "causal_effects_large.png",
        "Regression (Large Data)",
        true_causal_features=true_causal,
    )

    print("\n[Step 6] Full results table:")
    print(format_results_table(cia.get_results(), task_type="regression"))

    return cia


def _validate(causal_features_df, true_causal):
    """Validate identified features against ground truth."""
    identified = set(causal_features_df["feature"].tolist())
    true_set = set(true_causal)

    true_positives = identified & true_set
    false_positives = identified - true_set
    false_negatives = true_set - identified

    print(f"    True positives:  {true_positives or 'None'}")
    print(f"    False positives: {false_positives or 'None'}")
    print(f"    False negatives: {false_negatives or 'None'}")

    if true_positives:
        precision = len(true_positives) / len(identified) if identified else 0
        recall = len(true_positives) / len(true_set)
        print(f"    Precision: {precision:.2f}, Recall: {recall:.2f}")


def plot_results(results_df, filename, task_label, true_causal_features=None):
    """
    Default CIA visualization: horizontal bar chart with 95% CI.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from cia.get_results().
    filename : str
        Path to save the figure.
    task_label : str
        Title label (e.g., 'Regression').
    true_causal_features : list or None
        If provided (synthetic validation), colors by ground truth + significance.
        If None (real data), colors by p-value only.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    features = results_df["feature"].tolist()
    effects = results_df["causal_effect"].values
    ci_low = results_df["ci_lower"].values
    ci_high = results_df["ci_upper"].values
    p_values = results_df["p_value"].values

    if true_causal_features is not None:
        colors = []
        for i, feat in enumerate(features):
            if feat in true_causal_features:
                colors.append("darkred" if p_values[i] < 0.05 else "salmon")
            else:
                colors.append("steelblue" if p_values[i] < 0.05 else "lightsteelblue")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="darkred", label="True causal (significant)"),
            Patch(facecolor="salmon", label="True causal (not significant)"),
            Patch(facecolor="steelblue", label="Non-causal (significant)"),
            Patch(facecolor="lightsteelblue", label="Non-causal (not significant)"),
        ]
    else:
        colors = []
        for i in range(len(features)):
            if p_values[i] < 0.001:
                colors.append("darkred")
            elif p_values[i] < 0.01:
                colors.append("firebrick")
            elif p_values[i] < 0.05:
                colors.append("darkorange")
            else:
                colors.append("lightgray")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="darkred", label="p < 0.001 (***)"),
            Patch(facecolor="firebrick", label="p < 0.01 (**)"),
            Patch(facecolor="darkorange", label="p < 0.05 (*)"),
            Patch(facecolor="lightgray", label="Not significant"),
        ]

    y_pos = np.arange(len(features))

    ax.barh(y_pos, effects, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    for i in range(len(features)):
        ax.plot(
            [ci_low[i], ci_high[i]], [y_pos[i], y_pos[i]],
            color="black", linewidth=1.5, marker="|", markersize=8
        )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel("Estimated Causal Effect")
    ax.set_title(f"Conditional Intervention Analysis, {task_label}")
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {filename}")


def run_with_custom_data_example():
    """Print copy-paste examples for real data."""
    print("\n" + "=" * 70)
    print("  EXAMPLES: Using CIA with your own data")
    print("=" * 70)
    print("""
    # --- SMALL DATA (n < 500) ---
    from causal_inference import ConditionalInterventionAnalysis
    from demo_run import plot_results
    import pandas as pd

    X = pd.read_csv("features.csv")
    y = pd.read_csv("outcome.csv")["score"]

    cia = ConditionalInterventionAnalysis(
        task_type="regression",
        n_samples=30,
        n_folds=5,
        model_name="consensus",
        conditional_model="consensus",
        random_state=42,
    )
    cia.fit(X, y)                    # feature types auto-detected
    cia.summary()
    plot_results(cia.get_results(), "causal_effects.png", "Regression")

    # --- LARGE DATA (e.g., ABCD: ~9000 subjects, ~47 features) ---
    cia_large = ConditionalInterventionAnalysis(
        task_type="regression",
        n_samples=20,
        n_folds=5,
        model_name="consensus_fast",       # Ridge + GBM + RF
        conditional_model="consensus_fast",
        max_subjects=300,                  # subsample test subjects per fold
        random_state=42,
    )
    cia_large.fit(X, y)
    cia_large.summary()
    plot_results(cia_large.get_results(), "causal_effects_large.png", "Regression")

    # --- AFTER FIBE: analyze only selected features ---
    selected = ["feat1", "feat2", "feat3", ...]
    cia.fit(data[selected], data["outcome"], feature_names=selected)

    # --- CLASSIFICATION ---
    cia_cls = ConditionalInterventionAnalysis(
        task_type="classification",
        model_name="consensus_fast",
        max_subjects=300,
        n_samples=20,
    )
    cia_cls.fit(X, y_binary)
    plot_results(cia_cls.get_results(), "causal_effects_cls.png", "Classification")
    """)


if __name__ == "__main__":
    task_arg = None
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--task" and i + 1 < len(sys.argv):
                task_arg = sys.argv[i + 1]

    if task_arg == "regression":
        run_regression_demo()
    elif task_arg == "classification":
        run_classification_demo()
    elif task_arg == "large":
        run_large_dataset_demo()
    else:
        run_regression_demo()
        run_classification_demo()

    run_with_custom_data_example()

    print("\n" + "=" * 70)
    print("  All demos complete!")
    print("  Tip: run  python demo_run.py --task large  for consensus_fast demo")
    print("=" * 70)

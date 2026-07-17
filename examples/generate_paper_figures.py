"""
Generate manuscript figures for ABCD screening results.

Usage:
    python examples/generate_paper_figures.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
PDS_DIR = ROOT / "outputs" / "abcd_chernozhukov_dml"
BASE_DIR = ROOT / "outputs" / "abcd_vanilla_dml"

OUTCOME_ORDER = [
    "Cognition Total Composite Score Age-Corrected Standard Score",
    "NIH Toolbox Flanker Inhibitory Control and Attention Test Ages 8-11 v2.0 Age-Corrected Standard Score",
    "NIH Toolbox Dimensional Change Card Sort Test Ages 8-11 v2.0 Age-Corrected Standard Score",
]
OUTCOME_LABELS = ["Total composite", "Flanker", "DCCS"]
METHODS = [
    ("Baseline DML", BASE_DIR, "vanilla_dml_results.csv"),
    ("PDS-DML", PDS_DIR, "chernozhukov_dml_results.csv"),
]

COLOR_OTHER = "#B0BEC5"
COLOR_FDR = "#2E7D32"
TOP_K = 10
GLOBAL_N_TESTS = 150


def _save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  saved {path}")


def _load_outcome_df(results_dir: Path, outcome: str, csv_name: str) -> pd.DataFrame:
    folder = results_dir / outcome
    return pd.read_csv(folder / csv_name)


def _prepare_method_frames() -> dict[str, list[pd.DataFrame]]:
    """Load all outcomes for each method and attach study-wide FDR flags."""
    frames: dict[str, list[pd.DataFrame]] = {}
    for label, results_dir, csv_name in METHODS:
        dfs = [_load_outcome_df(results_dir, outcome, csv_name) for outcome in OUTCOME_ORDER]
        combined_p = np.concatenate([df["p_value"].values.astype(float) for df in dfs])
        adj = false_discovery_control(combined_p, method="bh")
        out_dfs = []
        idx = 0
        for df in dfs:
            n = len(df)
            out = df.copy()
            out["significant_global_fdr"] = adj[idx : idx + n] < 0.05
            idx += n
            out_dfs.append(out)
        frames[label] = out_dfs
    return frames


def _short_feature_name(name: str, width: int = 34) -> str:
    name = name.replace("Residential history derived - Area Deprivation Index: ", "ADI: ")
    name = name.replace("Cortical volume in mm^3 of APARC ROI ", "Vol ")
    name = name.replace("Cortical area in mm^2 of APARC ROI ", "Area ")
    name = name.replace("Cortical thickness in mm of APARC ROI ", "Thick ")
    name = name.replace("Volume in mm^3 of ASEG ROI ", "ASEG ")
    name = name.replace("Parent/caregiver highest education level (demo_prnt_ed_v2)", "Parent education")
    return textwrap.shorten(name, width=width, placeholder="...")


def _top_features(df: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    ranked = df.assign(abs_theta=df["theta"].abs()).sort_values("abs_theta", ascending=False)
    return ranked.head(k).sort_values("theta")


def plot_abcd_theta_panel():
    frames = _prepare_method_frames()

    fig, axes = plt.subplots(
        nrows=len(OUTCOME_ORDER),
        ncols=len(METHODS),
        figsize=(11, 9.5),
        sharex=True,
        gridspec_kw={"wspace": 0.12},
    )
    if axes.ndim == 1:
        axes = np.array([axes])

    xmax = 0.0
    for label, _, _ in METHODS:
        for outcome, df in zip(OUTCOME_ORDER, frames[label]):
            xmax = max(xmax, df["theta"].abs().max())
    xmax = min(xmax * 1.15, 5.0)

    for row, (outcome_label, outcome) in enumerate(zip(OUTCOME_LABELS, OUTCOME_ORDER)):
        for col, (method_label, _, _) in enumerate(METHODS):
            ax = axes[row, col]
            top = _top_features(frames[method_label][row], TOP_K)
            y = np.arange(len(top))
            colors = [COLOR_FDR if bool(r["significant_global_fdr"]) else COLOR_OTHER for _, r in top.iterrows()]
            ax.barh(y, top["theta"], color=colors, edgecolor="white", linewidth=0.4)
            ax.set_yticks(y)
            label_width = 28 if col == 1 else 34
            labels = [_short_feature_name(f, width=label_width) for f in top["feature"]]
            if col == 0:
                ax.set_yticklabels(labels, fontsize=7)
            else:
                ax.set_yticklabels([])
                for tick, label in zip(y, labels):
                    ax.text(
                        1.01,
                        tick,
                        label,
                        transform=ax.get_yaxis_transform(),
                        va="center",
                        ha="left",
                        fontsize=7,
                        clip_on=False,
                    )
            ax.axvline(0, color="black", linewidth=0.6)
            ax.set_xlim(-xmax, xmax)
            if row == 0:
                ax.set_title(method_label, fontsize=10)
            if col == 0:
                ax.set_ylabel(outcome_label, fontsize=9)
            if row == len(OUTCOME_ORDER) - 1:
                ax.set_xlabel(r"$\hat{\theta}$ (standard-score units)", fontsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_FDR, label="Global FDR significant"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_OTHER, label="Not FDR significant"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(
        rf"ABCD top-{TOP_K} features by $|\hat{{\theta}}|$ per outcome and method ($n=9{{,}}551$)",
        fontsize=11,
        y=0.995,
    )
    fig.text(
        0.5,
        0.02,
        rf"Bars show signed $\hat{{\theta}}$; ranking within each panel is by $|\hat{{\theta}}|$. "
        f"Global FDR uses Benjamini--Hochberg over {GLOBAL_N_TESTS} tests per method.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.10, top=0.93, left=0.20, right=0.80, hspace=0.35)
    _save(fig, "fig_abcd_theta_panel")
    plt.close(fig)


def main():
    print("Generating figures...")
    plot_abcd_theta_panel()
    print("Done.")


if __name__ == "__main__":
    main()

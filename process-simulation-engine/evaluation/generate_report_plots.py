"""Generate publication-quality bar charts for the LaTeX report."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = REPO_ROOT / "evaluation" / "output" / "batch_evaluation_results.xlsx"
FIGURES_DIR = REPO_ROOT.parent / "latex-report" / "figures"

STRATEGY_MAP = {
    "greedy_random": "R-RMA",
    "greedy_round_robin": "R-RRA",
    "greedy_shortest_queue": "R-SHQ",
    "batch": "1-Batch-1",
    "drl": "DRL",
}
STRATEGY_ORDER = list(STRATEGY_MAP.keys())

BAR_COLOR = "#6a9ec1"       # muted steel blue
BEST_COLOR = "#2b5d8a"      # darker saturated shade


def _report_bar(
    df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    ylabel: str,
    out_path: Path,
) -> None:
    order = [s for s in STRATEGY_ORDER if s in df["strategy"].values]
    df_plot = df.set_index("strategy").loc[order].reset_index()
    labels = [STRATEGY_MAP[s] for s in df_plot["strategy"]]

    means = df_plot[mean_col].values
    stds = df_plot[std_col].values
    best_idx = int(np.nanargmin(means))

    colors = [BAR_COLOR] * len(means)
    colors[best_idx] = BEST_COLOR

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(labels))
    ax.bar(
        x, means, yerr=stds, capsize=3,
        color=colors, edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(bottom=0)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(EXCEL_PATH, sheet_name="per_strategy")

    _report_bar(
        df,
        mean_col="cycle_time_mean_mean",
        std_col="cycle_time_mean_std",
        ylabel="Mean cycle time (days)",
        out_path=FIGURES_DIR / "cycle_time_mean_by_strategy.pdf",
    )
    _report_bar(
        df,
        mean_col="wrf_mean",
        std_col="wrf_std",
        ylabel="Weighted resource fairness (WRF)",
        out_path=FIGURES_DIR / "wrf_by_strategy.pdf",
    )


if __name__ == "__main__":
    main()

"""
Evaluate a single "method" log represented by N runs (CSV or XES) and compare
its metrics against a separate ground-truth log.

Designed to mirror the evaluation metrics/plots used elsewhere in this repo,
but with ground truth as the only comparator (instead of other allocation models).

WRF/Gini are computed via a lifecycle-aware duration extraction that also works
for "complete-only" logs (i.e., when start/resume events are missing).
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pm4py

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

COST_PER_FTE_HOUR_DEFAULT = 50.0

METRIC_KEYS = [
    "cycle_time_mean",
    "cycle_time_median",
    "cycle_time_range_days",
    "cycle_time_p90",
    "makespan_days",
    "throughput_per_day",
    "handovers_mean",
    "wrf",
    "gini",
    "avg_resource_occupation_share",
    "total_busy_hours",
    "cost_per_case",
    "num_resources",
    "events_per_case_mean",
    "total_fte_cost",
    "max_termination_score",
    "mean_termination_score",
    # Productivity/contribution (exported for completeness; not used in the core plots below)
    "productivity_mean",
    "contribution_per_cost_mean",
]

PLOT_SPECS = [
    # Keep plot set aligned with `evaluation/generate_report_plots.py`.
    ("cycle_time_mean", "Mean Cycle Time (days)", "Mean Cycle Time vs Ground Truth", True, "cycle_time_mean_gt_vs_method.pdf"),
    ("wrf", "Weighted resource fairness (WRF)", "WRF vs Ground Truth", True, "wrf_gt_vs_method.pdf"),
]

COLORS = {
    "ground_truth": "#6a9ec1",  # muted steel blue
    "method": "#9b72b0",  # purple-ish
    "best": "#2b5d8a",  # darker saturated shade
}


def _natural_run_id(path: Path) -> int:
    """
    Extract an ordering key from filenames like:
      - simulated_log_1_new.csv -> 1
      - simulated_log (2).csv -> 2
      - run3.xes -> 3
    """
    s = path.name
    nums = re.findall(r"\d+", s)
    if not nums:
        return 0
    return int(nums[-1])


def load_log(path: Path) -> pd.DataFrame:
    """
    Load a XES or CSV log and return a flat DataFrame with standardized column names.
    Required columns:
      - case:concept:name
      - concept:name
      - time:timestamp
    Optional columns:
      - org:resource
      - lifecycle:transition
    """
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        log = pm4py.read_xes(str(path))
        df = pm4py.convert_to_dataframe(log)

    df.columns = [c.lower() for c in df.columns]
    rename_map = {
        "case:concept:name": "case:concept:name",
        "concept:name": "concept:name",
        "time:timestamp": "time:timestamp",
        "org:resource": "org:resource",
        "lifecycle:transition": "lifecycle:transition",
    }
    # Map any case variants back to the canonical pm4py column names we use.
    df = df.rename(columns={c.lower(): canonical for c, canonical in rename_map.items() if c.lower() in df.columns})

    required = {"case:concept:name", "concept:name", "time:timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["time:timestamp"])
    return df


def gini_coefficient(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    v = v[v >= 0]
    n = len(v)
    if n == 0 or v.sum() == 0:
        return np.nan
    ranks = np.arange(1, n + 1)
    return float((2 * (ranks * v).sum()) / (n * v.sum()) - (n + 1) / n)


def wrf_from_busy(df_res: pd.DataFrame) -> float:
    """
    WRF definition used in the repo:
      - work_share = busy_hours / total_busy
      - weights = busy_hours / total_busy
      - WRF = sum_i weights_i * |work_share_i - mean(work_share)|
    """
    if df_res is None or df_res.empty:
        return np.nan
    busy = df_res["busy_hours"].values
    total = float(np.sum(busy))
    if total == 0:
        return np.nan
    shares = df_res["work_share"].values
    weights = busy / total
    return float(np.sum(weights * np.abs(shares - shares.mean())))


def extract_activity_durations(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Extract activity durations in a way that works for:
      1) full lifecycle activities with start+complete
      2) complete-only activities (duration estimated as time since previous event in case)

    Returns df_dur with:
      [case:concept:name, concept:name, (optional org:resource), start_time, complete_time, duration_hours]
    """
    has_lifecycle = "lifecycle:transition" in df.columns
    if not has_lifecycle:
        return pd.DataFrame(), False

    df_sorted = df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    lc = df_sorted["lifecycle:transition"].astype(str).str.lower()
    starts = df_sorted[lc == "start"].copy()
    completes = df_sorted[lc == "complete"].copy()

    if completes.empty:
        return pd.DataFrame(), False

    merge_cols = ["case:concept:name", "concept:name"]
    if "org:resource" in df.columns:
        merge_cols.append("org:resource")

    activities_with_start = set(starts["concept:name"].unique())

    for _df in [starts, completes]:
        _df["_occ"] = _df.groupby(merge_cols).cumcount()

    full_starts = starts[starts["concept:name"].isin(activities_with_start)]
    full_completes = completes[completes["concept:name"].isin(activities_with_start)]

    df_full = pd.DataFrame()
    if not full_starts.empty and not full_completes.empty:
        df_full = pd.merge(
            full_starts.rename(columns={"time:timestamp": "start_time"})[merge_cols + ["_occ", "start_time"]],
            full_completes.rename(columns={"time:timestamp": "complete_time"})[merge_cols + ["_occ", "complete_time"]],
            on=merge_cols + ["_occ"],
            how="inner",
        )
        df_full["duration_hours"] = (df_full["complete_time"] - df_full["start_time"]).dt.total_seconds() / 3600
        df_full = df_full.drop(columns=["_occ"])

    # Complete-only: estimate duration as time since previous event in the case.
    complete_only = completes[~completes["concept:name"].isin(activities_with_start)]
    df_comp_only = pd.DataFrame()
    if not complete_only.empty:
        df_sorted["_prev_time"] = df_sorted.groupby("case:concept:name")["time:timestamp"].shift(1)
        comp_rows = df_sorted.loc[
            (lc == "complete") & (~df_sorted["concept:name"].isin(activities_with_start))
        ].copy()
        comp_rows = comp_rows[comp_rows["_prev_time"].notna()]

        if not comp_rows.empty:
            comp_rows["duration_hours"] = (comp_rows["time:timestamp"] - comp_rows["_prev_time"]).dt.total_seconds() / 3600
            comp_rows = comp_rows.rename(columns={"time:timestamp": "complete_time"})
            comp_rows["start_time"] = comp_rows["_prev_time"]
            out_cols = merge_cols + ["start_time", "complete_time", "duration_hours"]
            df_comp_only = comp_rows[out_cols].copy()

    df_dur = pd.concat([df_full, df_comp_only], ignore_index=True)
    df_dur = df_dur[df_dur["duration_hours"] > 0].reset_index(drop=True)
    return df_dur, not df_dur.empty


def compute_case_metrics(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.sort_values("time:timestamp").groupby("case:concept:name")
    case_start = grp["time:timestamp"].min().rename("case_start")
    case_end = grp["time:timestamp"].max().rename("case_end")
    n_events = grp.size().rename("n_events")

    df_cases = pd.concat([case_start, case_end, n_events], axis=1).reset_index()
    df_cases["cycle_time_days"] = (df_cases["case_end"] - df_cases["case_start"]).dt.total_seconds() / 86_400

    if "org:resource" in df.columns:
        def count_handovers(sub: pd.DataFrame) -> int:
            r = sub.sort_values("time:timestamp")["org:resource"].dropna().astype(str).tolist()
            return sum(1 for a, b in zip(r, r[1:]) if a != b)

        df_cases["handovers"] = grp.apply(count_handovers).values
    else:
        df_cases["handovers"] = np.nan

    return df_cases


def compute_resource_metrics(df_durations: pd.DataFrame, has_durations: bool) -> pd.DataFrame | None:
    if not has_durations or "org:resource" not in df_durations.columns:
        return None

    df_res = (
        df_durations.groupby("org:resource")["duration_hours"]
        .sum()
        .reset_index()
        .rename(columns={"org:resource": "resource", "duration_hours": "busy_hours"})
        .sort_values("busy_hours", ascending=False)
        .reset_index(drop=True)
    )
    total_busy = float(df_res["busy_hours"].sum())
    if total_busy == 0:
        df_res["work_share"] = 0.0
    else:
        df_res["work_share"] = df_res["busy_hours"] / total_busy
    return df_res


def compute_resource_features(
    df_raw: pd.DataFrame,
    df_durations: pd.DataFrame,
    df_cases: pd.DataFrame,
    has_durations: bool,
    cost_per_fte: float,
) -> pd.DataFrame | None:
    if "org:resource" not in df_raw.columns or not has_durations:
        return None

    if "lifecycle:transition" in df_raw.columns:
        df_complete = df_raw[df_raw["lifecycle:transition"].astype(str).str.lower() == "complete"].copy()
    else:
        df_complete = df_raw

    busy = (
        df_durations.groupby("org:resource")["duration_hours"]
        .sum()
        .reset_index()
        .rename(columns={"org:resource": "resource", "duration_hours": "busy_hours"})
    )
    total_busy_hours = float(busy["busy_hours"].sum())
    busy["workload_share"] = (busy["busy_hours"] / total_busy_hours) if total_busy_hours > 0 else 0.0
    busy["fte_cost"] = busy["busy_hours"] * cost_per_fte

    event_counts = (
        df_complete.groupby("org:resource").size().reset_index(name="completed_events_count").rename(columns={"org:resource": "resource"})
    )
    case_counts = (
        df_complete.groupby("org:resource")["case:concept:name"]
        .nunique()
        .reset_index()
        .rename(columns={"org:resource": "resource", "case:concept:name": "distinct_cases_count"})
    )

    rm = busy.merge(event_counts, on="resource", how="left").merge(case_counts, on="resource", how="left")
    rm = rm.fillna({"completed_events_count": 0, "distinct_cases_count": 0})

    rm["productivity"] = np.where(rm["busy_hours"] > 0, rm["completed_events_count"] / rm["busy_hours"], 0.0)
    rm["contribution_per_cost"] = np.where(
        rm["fte_cost"] > 0, rm["distinct_cases_count"] / rm["fte_cost"], 0.0
    )

    # Termination-score helper: compare the cycle time of cases "touched" by a resource to the global mean.
    global_avg_ct = float(df_cases["cycle_time_days"].mean()) if len(df_cases) > 0 else np.nan
    if "org:resource" in df_complete.columns and "case:concept:name" in df_complete.columns and "cycle_time_days" in df_cases.columns:
        resource_cases = (
            df_complete[["org:resource", "case:concept:name"]]
            .drop_duplicates()
            .rename(columns={"org:resource": "resource"})
        )
        case_ct = df_cases[["case:concept:name", "cycle_time_days"]].copy()
        resource_ct = (
            resource_cases.merge(case_ct, on="case:concept:name", how="left")
            .groupby("resource")["cycle_time_days"]
            .mean()
            .reset_index()
            .rename(columns={"cycle_time_days": "avg_cycle_time_of_cases_involved"})
        )
        rm = rm.merge(resource_ct, on="resource", how="left")
        rm["cycle_time_delta"] = rm["avg_cycle_time_of_cases_involved"] - global_avg_ct
    else:
        rm["cycle_time_delta"] = np.nan

    return rm


def compute_metrics_for_log(log_path: Path, cost_per_fte_hour: float) -> dict:
    df_raw = load_log(log_path)
    df_cases = compute_case_metrics(df_raw)

    num_cases = int(df_cases["case:concept:name"].nunique())
    num_events = len(df_raw)
    events_per_case_mean = float(df_cases["n_events"].mean()) if len(df_cases) > 0 else np.nan

    # Cycle time (case end - case start)
    cycle_times = df_cases["cycle_time_days"].dropna()
    cycle_time_mean = float(cycle_times.mean()) if len(cycle_times) else np.nan
    cycle_time_median = float(cycle_times.median()) if len(cycle_times) else np.nan
    cycle_time_range_days = float(cycle_times.max() - cycle_times.min()) if len(cycle_times) else np.nan
    cycle_time_p90 = float(np.percentile(cycle_times.values, 90)) if len(cycle_times) else np.nan

    # Throughput
    if len(df_cases) > 0:
        span_days = float((df_cases["case_end"].max() - df_cases["case_start"].min()).total_seconds() / 86_400)
        makespan_days = span_days
        throughput_per_day = float(num_cases / span_days) if span_days > 0 else np.nan
    else:
        throughput_per_day = np.nan
        makespan_days = np.nan

    # Handovers
    handovers_mean = float(df_cases["handovers"].mean()) if "handovers" in df_cases.columns and df_cases["handovers"].notna().any() else np.nan

    # Resource-centric metrics via extracted busy time.
    df_dur, has_dur = extract_activity_durations(df_raw)
    df_res = compute_resource_metrics(df_dur, has_dur)

    if df_res is None or df_res.empty:
        total_busy_hours = np.nan
        total_fte_cost = np.nan
        cost_per_case = np.nan
        gini = np.nan
        wrf = np.nan
        avg_resource_occupation_share = np.nan
        num_resources = np.nan
        productivity_mean = np.nan
        contribution_per_cost_mean = np.nan
        max_termination_score = np.nan
        mean_termination_score = np.nan
    else:
        total_busy_hours = float(df_res["busy_hours"].sum())
        total_fte_cost = total_busy_hours * cost_per_fte_hour
        cost_per_case = float(total_fte_cost / num_cases) if num_cases > 0 else np.nan
        gini = gini_coefficient(df_res["busy_hours"].values)
        wrf = wrf_from_busy(df_res)
        avg_resource_occupation_share = float(df_res["work_share"].mean()) if "work_share" in df_res.columns else np.nan
        num_resources = int(len(df_res))

        rm_features = compute_resource_features(df_raw, df_dur, df_cases, has_dur, cost_per_fte_hour)
        if rm_features is None or rm_features.empty:
            productivity_mean = np.nan
            contribution_per_cost_mean = np.nan
            max_termination_score = np.nan
            mean_termination_score = np.nan
        else:
            productivity_mean = float(rm_features["productivity"].mean())
            contribution_per_cost_mean = float(rm_features["contribution_per_cost"].mean())

            def percentile_normalize(series: pd.Series, p_low: float = 5, p_high: float = 95) -> pd.Series:
                vals = series.dropna()
                if len(vals) == 0:
                    return pd.Series(np.nan, index=series.index)
                lo = float(np.percentile(vals.values, p_low))
                hi = float(np.percentile(vals.values, p_high))
                clipped = series.clip(lower=lo, upper=hi)
                if (hi - lo) == 0:
                    return pd.Series(0.5, index=series.index)
                return (clipped - lo) / (hi - lo)

            # Termination-score weights (as in the repo notebooks).
            W_COST, W_LOW_PRODUCTIVITY, W_LOW_VALUE = 0.25, 0.25, 0.20
            W_LOW_CONTRIBUTION, W_CRITICALITY, W_CYCLE_IMPACT = 0.15, 0.10, 0.05

            rm_features = rm_features.copy()
            rm_features["norm_cost"] = percentile_normalize(rm_features["fte_cost"]) if "fte_cost" in rm_features.columns else np.nan
            rm_features["norm_productivity"] = percentile_normalize(rm_features["productivity"]) if "productivity" in rm_features.columns else np.nan
            rm_features["norm_value"] = percentile_normalize(rm_features["contribution_per_cost"]) if "contribution_per_cost" in rm_features.columns else np.nan
            rm_features["norm_contribution"] = percentile_normalize(rm_features["distinct_cases_count"]) if "distinct_cases_count" in rm_features.columns else np.nan
            rm_features["norm_workload_share"] = percentile_normalize(rm_features["workload_share"]) if "workload_share" in rm_features.columns else np.nan
            rm_features["norm_cycle_delta"] = percentile_normalize(rm_features["cycle_time_delta"]) if "cycle_time_delta" in rm_features.columns else np.nan

            rm_features["termination_score"] = (
                W_COST * rm_features["norm_cost"]
                + W_LOW_PRODUCTIVITY * (1 - rm_features["norm_productivity"])
                + W_LOW_VALUE * (1 - rm_features["norm_value"])
                + W_LOW_CONTRIBUTION * (1 - rm_features["norm_contribution"])
                + W_CRITICALITY * (1 - rm_features["norm_workload_share"])
                + W_CYCLE_IMPACT * rm_features["norm_cycle_delta"]
            )

            max_termination_score = float(rm_features["termination_score"].max())
            mean_termination_score = float(rm_features["termination_score"].mean())

    return {
        "num_cases": num_cases,
        "num_events": num_events,
        "cycle_time_mean": cycle_time_mean,
        "cycle_time_median": cycle_time_median,
        "cycle_time_range_days": cycle_time_range_days,
        "cycle_time_p90": cycle_time_p90,
        "makespan_days": makespan_days,
        "throughput_per_day": throughput_per_day,
        "handovers_mean": handovers_mean,
        "events_per_case_mean": events_per_case_mean,
        "avg_resource_occupation_share": avg_resource_occupation_share,
        "total_busy_hours": total_busy_hours,
        "total_fte_cost": total_fte_cost,
        "cost_per_case": cost_per_case,
        "num_resources": num_resources,
        "gini": gini,
        "wrf": wrf,
        "productivity_mean": productivity_mean,
        "contribution_per_cost_mean": contribution_per_cost_mean,
        "max_termination_score": max_termination_score,
        "mean_termination_score": mean_termination_score,
    }


def build_comparison_table(method_runs_df: pd.DataFrame, gt_metrics: dict, method_label: str) -> pd.DataFrame:
    """
    Build a 2-row table suitable for plotting:
      - ground_truth: mean = gt value, std = 0
      - method: mean = average across runs, std = sample std across runs
    """
    rows = []
    gt_row = {"label": "Ground truth"}
    for k in METRIC_KEYS:
        gt_row[f"{k}_mean"] = gt_metrics.get(k, np.nan)
        gt_row[f"{k}_std"] = 0.0
    rows.append(gt_row)

    method_row = {"label": method_label}
    for k in METRIC_KEYS:
        vals = method_runs_df[k].dropna()
        method_row[f"{k}_mean"] = float(vals.mean()) if len(vals) else np.nan
        method_row[f"{k}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
    rows.append(method_row)

    return pd.DataFrame(rows)


def plot_two_bars(
    df_comp: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    lower_is_better: bool,
) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    labels = df_comp["label"].tolist()
    means = df_comp[mean_col].to_numpy(dtype=float)
    stds = df_comp[std_col].to_numpy(dtype=float)

    # Determine best among the two bars (ignoring NaNs).
    valid_mask = ~np.isnan(means)
    if valid_mask.any():
        valid_means = means[valid_mask]
        best_valid_idx = int(np.nanargmin(valid_means)) if lower_is_better else int(np.nanargmax(valid_means))
        best_idx = int(np.flatnonzero(valid_mask)[best_valid_idx])
    else:
        best_idx = None

    colors = [COLORS["ground_truth"], COLORS["method"]]
    if best_idx is not None:
        colors[best_idx] = COLORS["best"]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=3,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _fmt_mean_std(mean: float, std: float, decimals: int) -> str:
    if np.isnan(mean):
        return f"nan $\\pm$ nan"
    std = 0.0 if np.isnan(std) else float(std)
    m = f"{mean:,.{decimals}f}"
    s = f"{std:,.{decimals}f}"

    # Remove the ± / plus-minus symbol; keep std in parentheses.
    return f"{m} ({s})"


def _fmt_mean_latex(mean: float, decimals: int) -> str:
    """Format mean for the *left* column (no deviation)."""
    if np.isnan(mean):
        return "nan"
    return f"{mean:,.{decimals}f}"


def _fmt_std_latex(std: float, decimals: int) -> str:
    if np.isnan(std):
        std = 0.0
    return f"{float(std):,.{decimals}f}"


def _fmt_mean_pm_latex(mean: float, std: float, decimals: int) -> str:
    """Format mean + deviation for the *right* column."""
    if np.isnan(mean):
        return "nan"
    return f"{_fmt_mean_latex(mean, decimals)} $\\pm$ {_fmt_std_latex(std, decimals)}"


def _latex_escape(s: str) -> str:
    # Minimal escaping for LaTeX table content.
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def generate_latex_gt_vs_method_table(
    df_comp: pd.DataFrame,
    method_label: str,
    out_tex_path: Path,
    caption: str,
    selected_rows: list[tuple[str, str, bool, int]],
    label: str = "tab:pmsp-gt-vs-method",
) -> None:
    """
    selected_rows:
      (pretty_metric_name, metric_key_in_df_comp, lower_is_better, decimals)
    """
    gt_row = df_comp[df_comp["label"] == "Ground truth"]
    m_row = df_comp[df_comp["label"] == method_label]
    if gt_row.empty or m_row.empty:
        raise ValueError("Could not find both 'Ground truth' and method_label rows in df_comp.")
    gt_row = gt_row.iloc[0]
    m_row = m_row.iloc[0]

    decimals_map = {metric_key: dec for (_name, metric_key, _lower, dec) in selected_rows}

    header = (
        r"\textbf{Metric} & "
        r"\textbf{Ground truth} & "
        f"\\textbf{{{_latex_escape(method_label)}}} \\\\\n"
    )
    body_lines: list[str] = []

    for pretty_name, metric_key, lower_is_better, dec in selected_rows:
        gt_mean = float(gt_row[f"{metric_key}_mean"])
        m_mean = float(m_row[f"{metric_key}_mean"])
        gt_std = float(gt_row[f"{metric_key}_std"])
        m_std = float(m_row[f"{metric_key}_std"])

        gt_cell = _fmt_mean_latex(gt_mean, decimals=dec)
        m_cell = _fmt_mean_pm_latex(m_mean, std=m_std, decimals=dec)

        if np.isnan(gt_mean) and not np.isnan(m_mean):
            m_cell = f"\\textbf{{{m_cell}}}"
        elif np.isnan(m_mean) and not np.isnan(gt_mean):
            gt_cell = f"\\textbf{{{gt_cell}}}"
        elif not (np.isnan(gt_mean) and np.isnan(m_mean)):
            best_is_gt = (gt_mean <= m_mean) if lower_is_better else (gt_mean >= m_mean)
            if best_is_gt:
                gt_cell = f"\\textbf{{{gt_cell}}}"
            else:
                m_cell = f"\\textbf{{{m_cell}}}"
        # If both are NaN: no bolding.

        body_lines.append(f"{pretty_name} & {gt_cell} & {m_cell} \\\\")

    latex = (
        "\\begin{table*}[htbp]\n"
        "\\scriptsize\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{lrr}\n"
        "\\toprule\n"
        + header
        + "\n"
        "\\midrule\n"
        + "\n".join(body_lines)
        + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )

    out_tex_path.parent.mkdir(parents=True, exist_ok=True)
    out_tex_path.write_text(latex, encoding="utf-8")
    print(f"Latex table exported: {out_tex_path}")


def _fmt_mean_std_png(mean: float, std: float, decimals: int) -> str:
    if np.isnan(mean):
        return "nan"
    if np.isnan(std):
        std = 0.0
    m = f"{mean:.{decimals}f}"
    s = f"{std:.{decimals}f}"
    return f"{m} +- {s}"


def _fmt_mean_png(mean: float, decimals: int) -> str:
    if np.isnan(mean):
        return "nan"
    return f"{mean:.{decimals}f}"


def generate_png_gt_vs_method_table(
    df_comp: pd.DataFrame,
    method_label: str,
    out_png_path: Path,
    selected_rows: list[tuple[str, str, bool, int]],
) -> None:
    """
    Render a PNG table (matplotlib) for the same metrics selection as the LaTeX export.
    Best value among GT vs Method is bolded.
    """
    import matplotlib.pyplot as _plt

    gt_row = df_comp[df_comp["label"] == "Ground truth"]
    m_row = df_comp[df_comp["label"] == method_label]
    if gt_row.empty or m_row.empty:
        raise ValueError("Could not find both 'Ground truth' and method_label rows in df_comp.")
    gt_row = gt_row.iloc[0]
    m_row = m_row.iloc[0]

    columns = ["Metric", "Ground truth", "Method (5 runs)"]
    cell_text: list[list[str]] = []
    best_flags: list[tuple[bool, bool]] = []  # (best_gt, best_method)

    for pretty_name, metric_key, lower_is_better, dec in selected_rows:
        gt_mean = float(gt_row[f"{metric_key}_mean"])
        gt_std = float(gt_row[f"{metric_key}_std"])
        m_mean = float(m_row[f"{metric_key}_mean"])
        m_std = float(m_row[f"{metric_key}_std"])

        gt_s = _fmt_mean_png(gt_mean, decimals=dec)
        m_s = _fmt_mean_std_png(m_mean, std=m_std, decimals=dec)

        if np.isnan(gt_mean) and not np.isnan(m_mean):
            best_flags.append((False, True))
        elif np.isnan(m_mean) and not np.isnan(gt_mean):
            best_flags.append((True, False))
        elif np.isnan(gt_mean) and np.isnan(m_mean):
            best_flags.append((False, False))
        else:
            best_is_gt = (gt_mean <= m_mean) if lower_is_better else (gt_mean >= m_mean)
            best_flags.append((best_is_gt, not best_is_gt))

        cell_text.append([pretty_name, gt_s, m_s])

    fig, ax = _plt.subplots(figsize=(10.5, 4.9))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )

    table.auto_set_font_size(False)
    base_font = 11
    header_font = 12

    # Style cells.
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold", fontsize=header_font)
        else:
            cell.set_text_props(fontsize=base_font)

    # Bold best among the two value columns.
    # For matplotlib tables: row indices in get_celld are 0..n for header+body.
    for i, (best_gt, best_method) in enumerate(best_flags, start=1):
        if best_gt:
            table[(i, 1)].get_text().set_weight("bold")
        if best_method:
            table[(i, 2)].get_text().set_weight("bold")

    table.scale(1.0, 1.45)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_path, dpi=300, bbox_inches="tight")
    _plt.close(fig)
    print(f"PNG table exported: {out_png_path}")


def export_diff_stats_latex(
    df_runs: pd.DataFrame,
    gt_metrics: dict,
    out_tex_path: Path,
) -> None:
    """
    Export a booktabs-style table with mean/std of (GT - Method) differences per run.
    This yields a comparable summary for optimization.
    """
    diff_spec = [
        ("Avg Cycle Time (days)", "cycle_time_mean"),
        ("Median Cycle Time (days)", "cycle_time_median"),
        ("Range Cycle Time (days)", "cycle_time_range_days"),
        ("P90 Cycle Time (days)", "cycle_time_p90"),
        ("Makespan (days)", "makespan_days"),
        ("Throughput (cases/day)", "throughput_per_day"),
        ("Avg Resource Occupation (share)", "avg_resource_occupation_share"),
        ("Weighted Resource Fairness", "wrf"),
        ("Workload Gini Coefficient", "gini"),
    ]

    def safe_float(x: object) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    diffs_rows: list[tuple[str, float, float]] = []
    for pretty, key in diff_spec:
        gt_val = safe_float(gt_metrics.get(key, float("nan")))
        # Keep table layout stable: always emit the metrics from `diff_spec`.
        # If a metric column is missing in `df_runs`, write `nan` instead of skipping the row.
        if key not in df_runs.columns:
            diffs_rows.append((pretty, float("nan"), float("nan")))
            continue
        vals = pd.to_numeric(df_runs[key], errors="coerce").astype(float)
        d = gt_val - vals
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) == 0:
            mean, std = float("nan"), float("nan")
        elif len(d) == 1:
            mean, std = float(d.mean()), 0.0
        else:
            mean, std = float(d.mean()), float(d.std(ddof=1))
        diffs_rows.append((pretty, mean, std))

    # Keep the LaTeX layout identical to the snippet you provided.
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Mean and standard deviation of metric differences across paired runs (baseline $-$ resource termination).}"
    )
    lines.append(r"\label{tab:termination-comparison}")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{Mean} & \textbf{Std.} \\")
    lines.append(r"\midrule")
    for pretty, mean, std in diffs_rows:
        if np.isnan(mean) or np.isnan(std):
            lines.append(f"{pretty} & nan & nan \\\\")
        else:
            lines.append(f"{pretty} & {mean:.4f} & {std:.4f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_tex_path.parent.mkdir(parents=True, exist_ok=True)
    out_tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Diff stats table exported: {out_tex_path}")


def discover_method_runs(runs_dir: Path, runs_glob: str, expected_runs: int | None) -> list[Path]:
    paths = sorted(runs_dir.glob(runs_glob), key=_natural_run_id)
    if expected_runs is not None:
        paths = paths[:expected_runs]
    if not paths:
        raise FileNotFoundError(f"No method run logs found with glob='{runs_glob}' in {runs_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare N method runs against ground truth (metrics + plots).")
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parent, help="Folder containing the N method run logs.")
    parser.add_argument("--runs-glob", type=str, default="simulated_log*.csv", help="Glob to discover method run logs.")
    parser.add_argument("--expected-runs", type=int, default=5, help="How many runs to aggregate (first N by natural sort).")
    parser.add_argument("--ground-truth", type=Path, required=False, help="Path to the ground truth log (CSV or XES).")
    parser.add_argument("--method-label", type=str, default="Method (5 runs)", help="Label used in the plots.")
    parser.add_argument("--cost-per-fte-hour", type=float, default=COST_PER_FTE_HOUR_DEFAULT, help="FTE cost rate (€/occupied resource-hour).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: <repo_root>/evaluation/output/gt_vs_method).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = args.out_dir or (repo_root / "evaluation" / "output" / "gt_vs_method")

    # Default ground truth discovery (repo local).
    if args.ground_truth is None:
        # Prefer the smaller evaluation-local ground truth when available.
        gt_candidates = [
            repo_root / "evaluation" / "ground_truth_log.csv",
            repo_root / "evaluation" / "ground_truth_log.xes",
            repo_root / "simulation" / "output" / "ground_truth_log.csv",
            repo_root / "simulation" / "output" / "ground_truth_log.xes",
        ]
        gt_path = None
        for c in gt_candidates:
            if c.exists():
                gt_path = c
                break
        if gt_path is None:
            raise FileNotFoundError(
                "Ground truth log not found. Provide --ground-truth or ensure one of these exists:\n"
                + "\n".join(f"  {c}" for c in gt_candidates)
            )
    else:
        gt_path = args.ground_truth

    run_paths = discover_method_runs(args.runs_dir, args.runs_glob, args.expected_runs)

    # Compute ground truth once.
    print(f"Ground truth: {gt_path}")
    gt_metrics = compute_metrics_for_log(gt_path, args.cost_per_fte_hour)

    # Compute method runs.
    runs = []
    for idx, p in enumerate(run_paths, 1):
        run_id = _natural_run_id(p)
        print(f"[{idx}/{len(run_paths)}] Method run #{run_id}: {p}")
        m = compute_metrics_for_log(p, args.cost_per_fte_hour)
        m["run"] = run_id
        m["path"] = str(p)
        runs.append(m)

    df_runs = pd.DataFrame(runs)

    # Order columns for readability.
    per_run_cols = ["run", "path"] + [k for k in METRIC_KEYS if k in df_runs.columns]
    df_runs = df_runs[per_run_cols]

    df_comp = build_comparison_table(df_runs, gt_metrics, args.method_label)

    # Export Excel.
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / "gt_vs_method_metrics.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_runs.to_excel(writer, sheet_name="per_run", index=False)
        df_comp.to_excel(writer, sheet_name="comparison", index=False)
        pd.DataFrame([gt_metrics]).to_excel(writer, sheet_name="ground_truth", index=False)
        method_summary = df_comp[df_comp["label"] == args.method_label].copy()
        method_summary.to_excel(writer, sheet_name="method_aggregate", index=False)

    print(f"Excel exported: {xlsx_path}")

    # Plots (keep them inside the workspace so the script runs reliably everywhere).
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, title, lower_is_better, filename in PLOT_SPECS:
        out_path = plots_dir / filename
        plot_two_bars(df_comp, metric, ylabel, title, out_path, lower_is_better)

    print(f"Plots saved in: {plots_dir}")

    # LaTeX table (similar layout to your screenshot/table style).
    selected_rows = [
        ("Avg Cycle Time (days)", "cycle_time_mean", True, 2),
        ("Median Cycle Time (days)", "cycle_time_median", True, 2),
        ("Range Cycle Time (days)", "cycle_time_range_days", True, 2),
        ("P90 Cycle Time (days)", "cycle_time_p90", True, 2),
        ("Makespan (days)", "makespan_days", True, 2),
        ("Throughput (cases/day)", "throughput_per_day", False, 2),
        ("Avg Resource Occupation (share)", "avg_resource_occupation_share", False, 3),
        ("Weighted Resource Fairness (WRF)", "wrf", True, 3),
        ("Workload Gini Coefficient", "gini", True, 3),
    ]
    tex_out = out_dir / "pmsp_gt_vs_method_table.tex"
    caption = (
        "PMSP metrics comparison between ground truth and the proposed method "
        f"({args.expected_runs} runs; mean $\\pm$ std). Best value per metric is shown in bold."
    )
    generate_latex_gt_vs_method_table(
        df_comp,
        args.method_label,
        tex_out,
        caption,
        selected_rows,
        label="tab:pmsp-gt-vs-method",
    )

    # Additional diff-statistics table (GT - Method) in booktabs layout.
    diff_tex_out = out_dir / "pmsp_gt_vs_method_diff_stats_table.tex"
    export_diff_stats_latex(df_runs, gt_metrics, diff_tex_out)

    png_out = out_dir / "pmsp_gt_vs_method_table.png"
    generate_png_gt_vs_method_table(df_comp, args.method_label, png_out, selected_rows)


if __name__ == "__main__":
    main()


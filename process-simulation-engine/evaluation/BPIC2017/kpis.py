"""
BPIC17: was die Auswertung (evaluation/evaluation.py) nur für diesen Datensatz berechnet (DATASET_KPIS):
- Entscheidungspunkt: bewilligt (A_Pending) je Aktion; decision_kpis auch für next_event_prediction/train.py
- Fallattribute: Anteil Limit raise
- Mitarbeiter-Effekte: bewilligt je erstem Anrufer (W_Call incomplete files)
- Angebotsattribute (an O_Create Offer) und ihr Diagramm
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

Events = List[Tuple[str, str]]  # (activity, lifecycle)

OFFER_ATTRIBUTES = ["CreditScore", "OfferedAmount", "FirstWithdrawalAmount", "NumberOfTerms", "MonthlyCost", "Selected", "Accepted"]
OFFER_ACTIVITY, OUTCOME_ACTIVITY, CALL_WORK_ITEM = "O_Create Offer", "A_Pending", "W_Call incomplete files"
_OPENING = ("start", "resume")
MIN_CALLS = 50


def decision_kpis(traces: List[Events], decision_activity: str = "A_Validating", action_activity: str = "A_Incomplete",
                  outcome_activity: str = OUTCOME_ACTIVITY) -> Dict[str, float]:
    """Shares at the first decision point: how often it is reached, how often the action is taken, and the outcome per action."""
    reached, called, outcome = [], [], []
    for trace in traces:
        activities = [activity for activity, lifecycle in trace if lifecycle == "complete"]
        outcome.append(outcome_activity in activities)
        if decision_activity not in activities:
            reached.append(False)
            continue
        reached.append(True)
        first_round = activities[activities.index(decision_activity) + 1:]
        if decision_activity in first_round:
            first_round = first_round[: first_round.index(decision_activity)]
        called.append(action_activity in first_round)
    reached, called, outcome = np.array(reached, dtype=bool), np.array(called, dtype=bool), np.array(outcome, dtype=bool)
    with_decision = outcome[reached]
    return {
        "erreicht Entscheidungspunkt (%)": 100.0 * reached.mean(),
        "Call am ersten Entscheidungspunkt (%)": 100.0 * called.mean() if len(called) else float("nan"),
        "bewilligt, alle Fälle (%)": 100.0 * outcome.mean(),
        "bewilligt ohne Call (%)": 100.0 * with_decision[~called].mean() if (~called).any() else float("nan"),
        "bewilligt mit Call (%)": 100.0 * with_decision[called].mean() if called.any() else float("nan"),
    }


def prepare(df: pd.DataFrame, log) -> None:
    """Zum Log (evaluation.Log): je Fall bewilligt, je O_Create Offer die Angebotsattribute (log.extra["offers"])."""
    if log.attributes is None:
        return
    approved = (df["concept:name"] == OUTCOME_ACTIVITY).groupby(df["case:concept:name"]).any()
    log.attributes["bewilligt"] = approved.reindex(log.attributes.index).fillna(False).to_numpy()
    if not all(column in df.columns for column in OFFER_ATTRIBUTES):
        return
    rows = df[(df["concept:name"] == OFFER_ACTIVITY) & (df["lifecycle:transition"].str.lower() == "complete")]
    offers = rows[["case:concept:name"] + OFFER_ATTRIBUTES].copy()
    for column in OFFER_ATTRIBUTES[:5]:
        offers[column] = pd.to_numeric(offers[column], errors="coerce")
    for column in ("Selected", "Accepted"):
        offers[column] = offers[column].astype(str).str.lower().isin(["true", "1", "1.0"])
    offers["Angebot Nr."] = offers.groupby("case:concept:name").cumcount() + 1
    log.extra["offers"] = offers.join(log.attributes, on="case:concept:name")


def _callers(log) -> pd.Series:
    """Share of approved cases per employee who first worked on W_Call incomplete files in the case."""
    first = {}
    for case, (events, resources) in enumerate(zip(log.events, log.resources)):
        for (activity, lifecycle), resource in zip(events, resources):
            if activity == CALL_WORK_ITEM and lifecycle in _OPENING:
                first[case] = (resource, any(a == OUTCOME_ACTIVITY for a, _ in events))
                break
    frame = pd.DataFrame(first.values(), columns=["resource", "approved"])
    counts = frame["resource"].value_counts()
    return frame[frame["resource"].isin(counts[counts >= MIN_CALLS].index)].groupby("resource")["approved"].mean()


def kpis(real, simulated) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Die Zeilen der Kennzahlentabelle (Bereich, Kennzahl) -> (echt, simuliert), die es nur für BPIC17 gibt."""
    rows: Dict[Tuple[str, str], Tuple[float, float]] = {}
    decision_real, decision_simulated = decision_kpis(real.events), decision_kpis(simulated.events)
    for name in ("bewilligt, alle Fälle (%)", "bewilligt ohne Call (%)", "bewilligt mit Call (%)"):
        rows[("Entscheidungspunkt", name)] = (decision_real[name], decision_simulated[name])
    if real.resources is not None and simulated.resources is not None:
        area = "Mitarbeiter-Effekte"
        rates_real, rates_simulated = _callers(real), _callers(simulated)
        rows[(area, "Bewilligt je erstem Anrufer: Streuung (Std)")] = (float(rates_real.std()), float(rates_simulated.std()))
        common = rates_real.index.intersection(rates_simulated.index)
        rows[(area, "Bewilligt je erstem Anrufer: Korrelation mit echt")] = (
            1.0, float(np.corrcoef(rates_real[common], rates_simulated[common])[0, 1]) if len(common) > 2 else float("nan"))
    if real.attributes is not None and simulated.attributes is not None and "ApplicationType" in real.attributes:
        rows[("Fallattribute", "ApplicationType: Anteil Limit raise (%)")] = tuple(
            float((frame["ApplicationType"] == "Limit raise").mean() * 100) for frame in (real.attributes, simulated.attributes))
    o_r, o_s = real.extra.get("offers"), simulated.extra.get("offers")
    if o_r is None or o_s is None:
        return rows
    from evaluation.evaluation import _ks
    area = "Angebotsattribute"
    rows[(area, "Angebote")] = (len(o_r), len(o_s))
    rows[(area, "Angebote pro Fall mit Angebot")] = tuple(float(frame.groupby("case:concept:name").size().mean()) for frame in (o_r, o_s))

    def varies(frame: pd.DataFrame) -> float:
        # Share of the cases with several offers whose offers differ in OfferedAmount
        several = frame.groupby("case:concept:name")["OfferedAmount"].agg(["size", "nunique"])
        several = several[several["size"] > 1]
        return float((several["nunique"] > 1).mean() * 100) if len(several) else float("nan")

    rows[(area, "Fälle mit mehreren Angeboten: Angebote verschieden (%)")] = (varies(o_r), varies(o_s))
    rows[(area, "CreditScore: Anteil 0 (%)")] = tuple(float((frame["CreditScore"] == 0).mean() * 100) for frame in (o_r, o_s))
    rows[(area, "CreditScore > 0: Median")] = tuple(float(frame.loc[frame["CreditScore"] > 0, "CreditScore"].median()) for frame in (o_r, o_s))
    rows[(area, "FirstWithdrawalAmount: Anteil 0 (%)")] = tuple(float((frame["FirstWithdrawalAmount"] == 0).mean() * 100) for frame in (o_r, o_s))
    for column in OFFER_ATTRIBUTES[:5]:
        rows[(area, f"{column}: Median")] = tuple(float(frame[column].median()) for frame in (o_r, o_s))
        rows[(area, f"{column}: KS-Abstand")] = (0.0, _ks(o_r[column], o_s[column]))
    # Do the offer attributes fit together and to the case?
    for name, frame_function in (
        ("OfferedAmount / RequestedAmount: Median", lambda f: (f["OfferedAmount"] / f["RequestedAmount"].where(f["RequestedAmount"] > 0)).median()),
        ("OfferedAmount = RequestedAmount (%)", lambda f: (f["OfferedAmount"] == f["RequestedAmount"]).mean() * 100),
        ("MonthlyCost × NumberOfTerms / OfferedAmount: Median", lambda f: (f["MonthlyCost"] * f["NumberOfTerms"] / f["OfferedAmount"].where(f["OfferedAmount"] > 0)).median()),
    ):
        rows[(area, name)] = (float(frame_function(o_r)), float(frame_function(o_s)))
    rows[(area, "Selected: Anteil (%)")] = tuple(float(frame["Selected"].mean() * 100) for frame in (o_r, o_s))
    rows[(area, "Accepted: Anteil (%)")] = tuple(float(frame["Accepted"].mean() * 100) for frame in (o_r, o_s))
    # Accepted belongs to the offer that the customer later accepted: in the real log it goes with the outcome
    for value, label in ((True, "Accepted"), (False, "nicht Accepted")):
        rows[(area, f"Fall bewilligt, wenn {label} (%)")] = tuple(
            float(frame.loc[frame["Accepted"] == value, "bewilligt"].mean() * 100) for frame in (o_r, o_s))
    return rows


def plots(real, simulated, save) -> None:
    """Das Diagramm der Angebotsattribute: je Attribut die Verteilung, dazu bewilligt je Accepted."""
    if real.extra.get("offers") is None or simulated.extra.get("offers") is None:
        return
    import matplotlib.pyplot as plt

    from evaluation.evaluation import COLORS, INK, REAL, SIMULATED, _axes, cdf
    logs = ((REAL, real), (SIMULATED, simulated))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (column, log_x) in zip(axes.flat, (("CreditScore", False), ("OfferedAmount", True), ("FirstWithdrawalAmount", False),
                                               ("NumberOfTerms", False), ("MonthlyCost", True))):
        cdf(ax, [(name, log.extra["offers"][column]) for name, log in logs], column, "Wert" + (" (log)" if log_x else ""), log_x=log_x)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower right")
    ax = axes[1, 2]
    positions = np.arange(2)
    for offset, (name, log) in zip((-0.2, 0.2), logs):
        offers = log.extra["offers"]
        values = [offers.loc[offers["Accepted"] == value, "bewilligt"].mean() * 100 for value in (True, False)]
        bars = ax.bar(positions + offset, values, width=0.36, color=COLORS[name], label=name)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, color=INK, padding=2)
    ax.set_xticks(positions, ["Accepted", "nicht Accepted"])
    ax.set_ylim(0, 100)
    _axes(ax, "Fall bewilligt (A_Pending), je Angebot", ylabel="%")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "10_angebotsattribute.png")

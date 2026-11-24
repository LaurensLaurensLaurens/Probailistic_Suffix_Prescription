"""Prospective binary offer outcomes and multiplicity-adjusted AIPW inference.

The target is a completed future event, not the retrospective Accepted attribute
or A_Accepted (the bank's application assessment). Previously reached targets are
excluded using the prefix before treatment. Outcomes are defined per application:
at least one offer reaches the target before the observed trace ends.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from helpers.datasets import ACTIVITY, CASE, LIFECYCLE

SEMANTICS_URL = "https://promforum.win.tue.nl/discussion/749/meaning-of-columns-selected-and-accepted"
ACTIVITY_URL = "https://promforum.win.tue.nl/discussion/comment/2126/"
ENDPOINTS = {
    "accepted": {"event": "O_Accepted", "label": "Erfolgreiche Angebotsannahme"},
    "returned": {"event": "O_Returned", "label": "Rücksendung des unterschriebenen Angebots"},
}


def prospective_outcome(data, log, event, groups=None):
    """Retain decisions strictly before first target; map future target to 0/1.

    Event positions preserve ordering even for equal timestamps. Absence means
    not observed by trace end; it does not imply eventual rejection. No decision
    is removed merely because its observed subsequent outcome is zero.
    """
    if data.decision_pos is None:
        raise ValueError("A prospective outcome requires decision positions")
    completed = log[LIFECYCLE].eq("complete") if LIFECYCLE in log else True
    first = log.loc[log[ACTIVITY].eq(event) & completed].groupby(CASE)["_pos"].min()
    first = first.reindex(data.X.index)
    before = first.isna() | (first > data.decision_pos.reindex(data.X.index))
    keep = before.to_numpy(dtype=bool)
    index = data.X.index[keep]
    result = replace(data, X=data.X.loc[index].copy(), T=data.T.loc[index].copy(),
                     Y=first.loc[index].notna().astype(float),
                     decision_pos=data.decision_pos.loc[index].copy(),
                     pool=data.pool.loc[index].copy() if data.pool is not None else None)
    return result, np.asarray(groups)[keep] if groups is not None else None


def contrast_pvalues(details, randomized):
    """Two-sided t tests with the same independent units as confidence intervals."""
    result = details.copy()
    if not len(result):
        return result
    p = []
    for _, row in result.iterrows():
        mean, se = row["Effekt (AIPW)"], row["Std.-Fehler"]
        units = row["RCT-Läufe im Vergleich"] if randomized else row["Vergleichsfälle"]
        if not np.isfinite(mean) or not np.isfinite(se) or not np.isfinite(units) or units < 2:
            value = np.nan
        elif se == 0:
            value = 1.0 if mean == 0 else 0.0
        else:
            value = float(2 * student_t.sf(abs(mean / se), units - 1))
        p.append(value)
    result["p-Wert"] = p
    return result


def holm(pvalues):
    """Holm adjustment over all finite contrasts in one source/endpoint family."""
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(values[finite], kind="stable")]
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate((len(order) - np.arange(len(order))) * values[order]))
    return adjusted


def annotate_results(summary, details):
    summary, details = summary.copy(), details.copy()
    if len(details):
        details["p Holm"] = holm(details["p-Wert"])
        details["Hinweis nach Holm (5 %)"] = details["p Holm"].le(.05) & details["Effekt (AIPW)"].notna()
        for col in ("Effekt (AIPW)", "KI unten", "KI oben", "Std.-Fehler"):
            details[f"{col} (Prozentpunkte)"] = 100 * details[col]
    for i, row in summary.iterrows():
        part = details[details["Interventionspunkt"].eq(row["Interventionspunkt"])] if len(details) else details
        valid = part[part["p Holm"].notna()] if len(part) else part
        significant = valid[valid["p Holm"].le(.05)] if len(valid) else valid
        summary.loc[i, "Schätzbare Kontraste"] = len(valid)
        summary.loc[i, "Kontraste mit Hinweis nach Holm"] = len(significant)
        summary.loc[i, "Einfluss"] = ("Hinweis nach Holm-Korrektur" if len(significant) else
                                      "kein gesicherter Hinweis" if len(valid) else "nicht schätzbar")
        raw_comparison = str(row["Vergleich"]).split(" (zugewiesene Frist")[0]
        if len(part):
            keys = part["Option"].astype(str) + " vs. " + part["Referenz"].astype(str)
            main = part[keys.eq(raw_comparison)]
            if len(main) == 1:
                summary.loc[i, "p-Wert des gezeigten Vergleichs"] = main.iloc[0]["p-Wert"]
                summary.loc[i, "p Holm des gezeigten Vergleichs"] = main.iloc[0]["p Holm"]
        for col in ("Effekt (AIPW)", "KI unten", "KI oben"):
            summary.loc[i, f"{col} (Prozentpunkte)"] = 100 * row[col] if pd.notna(row[col]) else np.nan
    return summary, details

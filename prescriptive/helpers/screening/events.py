"""
Entscheidungs-Event und Aktion je Fall, aus dem Log (sortiert nach Fall und Zeit, mit Spalte _pos = Position im Fall).
"""
import numpy as np
import pandas as pd
from interventions import option_token

from ..datasets import ACTIVITY, CASE, LIFECYCLE, RESOURCE, TIMESTAMP


def _matches(activities, exact=None, contains=None):
    activities = activities.astype(str)
    if contains:
        return np.logical_or.reduce([activities.str.contains(text, regex=False) for text in contains])
    return activities.isin(exact if isinstance(exact, (list, tuple)) else [exact]).to_numpy()


# Ist das Event die Aktion? action: genaue Namen, action_contains: Teile des Namens; nur Events mit action_lifecycle
# (Standard complete, wie enriched_log.enrich)
def action_mask(log, test):
    mask = _matches(log[ACTIVITY], test.get("action"), test.get("action_contains"))
    if LIFECYCLE in log:
        mask = mask & (log[LIFECYCLE] == (test.get("action_lifecycle") or "complete")).to_numpy()
    return mask


# Je Fall das erste Event von decision (genau) bzw. decision_contains (Teil des Namens), mit lifecycle (Standard complete);
# eine Zeile je Fall (Index: Fall)
def decision_events(log, test):
    match = _matches(log[ACTIVITY], test.get("decision"), test.get("decision_contains"))
    if LIFECYCLE in log:
        match = match & (log[LIFECYCLE] == test.get("lifecycle", "complete")).to_numpy()
    return log[match].groupby(CASE).head(1).set_index(CASE)


# Events bis einschließlich Entscheidungs-Event
def prefix_until(log, decision_pos):
    return log[log["_pos"] <= log[CASE].map(decision_pos)]


# Aktion laut Regel (rule): Entscheidungs-Position je Fall, Aktion ja/nein, Stunden bis zur ersten Aktion, je
# Aktions-Event ein Systemnutzer?, Prefix
#   follows_before_next_decision  Aktion vor dem nächsten Entscheidungs-Event
#   follows_later                 Aktion irgendwann danach
#   first_activity                der Fall beginnt mit der Aktion (entschieden zu Fallbeginn, leerer Prefix)
#   next_event                    direkt nach dem Entscheidungs-Event eine der beiden Optionen (action); Aktion: die zweite
def actions(log, test, spec):
    is_action = action_mask(log, test)
    system_users = spec["system_users"]
    if test["rule"] == "first_activity":
        start = log["_pos"].to_numpy() == 0
        cases = log.loc[start, CASE].to_numpy()
        treated = pd.Series(is_action[start], index=cases)
        acts = log[start & is_action]
        lag = pd.Series(0.0, index=treated.index[treated])
        return pd.Series(0, index=cases), treated, lag, acts[RESOURCE].astype(str).str.contains(system_users), log.iloc[0:0]
    complete = (log[LIFECYCLE] == "complete") if LIFECYCLE in log else pd.Series(True, index=log.index)
    if test["rule"] == "next_event":
        # Entscheidungspunkt: das erste Entscheidungs-Event, nach dem direkt eine der beiden Optionen kommt; Aktion: Option 1
        options = [option_token(option) for option in test["action"]]
        tokens = log[ACTIVITY].astype(str) + "|" + (log[LIFECYCLE].astype(str) if LIFECYCLE in log else "complete")
        following = tokens.reindex(log.sort_values([CASE, "_pos"]).index).groupby(log[CASE]).shift(-1).reindex(log.index).astype(str)
        decided = (log[LIFECYCLE] == test.get("lifecycle", "complete")) if LIFECYCLE in log else complete
        at = (log[ACTIVITY].astype(str) == test["decision"]) & decided & following.isin(options)
        decision_pos = log.loc[at].groupby(CASE)["_pos"].min()
        dpos = log[CASE].map(decision_pos)
        acts = log[(log["_pos"] == dpos + 1) & (tokens == options[1])]
        treated = acts.groupby(CASE).size().reindex(decision_pos.index, fill_value=0) > 0
        decided_at = log[log["_pos"] == dpos].set_index(CASE)[TIMESTAMP]
        lag = (acts.set_index(CASE)[TIMESTAMP] - decided_at.reindex(acts[CASE])).dt.total_seconds() / 3600
        system = acts[RESOURCE].astype(str).str.contains(system_users) if RESOURCE in acts else pd.Series(dtype=bool)
        return decision_pos, treated, lag, system, log[log["_pos"] <= dpos]
    positions = log.loc[(log[ACTIVITY].astype(str) == test["decision"]) & complete].groupby(CASE)["_pos"]
    decision_pos = positions.min()
    dpos = log[CASE].map(decision_pos)
    upper = pd.Series(np.inf, index=log.index)
    if test["rule"] == "follows_before_next_decision":
        second = positions.nth(1)
        upper = log[CASE].map(pd.Series(second.to_numpy(), index=log.loc[second.index, CASE].to_numpy())).fillna(np.inf)
    acts = log[(log["_pos"] > dpos) & (log["_pos"] < upper) & is_action]
    treated = acts.groupby(CASE).size().reindex(decision_pos.index, fill_value=0) > 0
    decided_at = log[log["_pos"] == dpos].set_index(CASE)[TIMESTAMP]
    first = acts.groupby(CASE)[TIMESTAMP].min()
    lag = (first - decided_at.reindex(first.index)).dt.total_seconds() / 3600
    system = acts[RESOURCE].astype(str).str.contains(system_users) if RESOURCE in acts else pd.Series(dtype=bool)
    return decision_pos, treated, lag, system, log[log["_pos"] <= dpos]

"""
Merkmale X am Entscheidungspunkt: nur was zu diesem Zeitpunkt bekannt ist (Spalten aus dem Block screening des
Datensatzes: case_attributes, numeric_attributes, event_numeric).
"""
import re

import pandas as pd

from ..datasets import ACTIVITY, CASE, TIMESTAMP

MAX_ACTIVITY_COUNTS = 50  # Zählmerkmale für die häufigsten Tokens (Aktivität mit Lifecycle)


# Event als Token aus Aktivität und Lifecycle, wie die Spalte event des angereicherten Logs (enriched_log.event_token):
# "W_Validate application:suspend", complete-Events nur die Aktivität
def _events(frame):
    from ..datasets import LIFECYCLE, add_paths
    add_paths()
    from prescriptive.enriched_log import event_token
    return pd.Series(event_token(frame[ACTIVITY], frame[LIFECYCLE] if LIFECYCLE in frame else None), index=frame.index)


# Fallattribute, letzte bekannte Event-Attribute, Zahl der Events je Token (Aktivität mit Lifecycle), vorheriges Token,
# Zahl der Events, Stunden seit Fallbeginn, Stunde, Wochentag, Monat. Mit leerem Prefix (Entscheidung zu Fallbeginn) nur
# die Fallattribute und der Kalender des Fallbeginns.
# Wird aufgerufen von candidate in candidates
def features(log, prefix, decision_pos, spec):
    cases = log.groupby(CASE)
    X = pd.DataFrame(index=decision_pos.index)
    first = cases.first().reindex(X.index)
    for column in spec.get("case_attributes", []):
        X[column] = first[column].astype(str).astype("category")
    for column in spec.get("numeric_attributes", []):
        X[column] = pd.to_numeric(first[column], errors="coerce")
    start = cases[TIMESTAMP].min().reindex(X.index)
    now = start
    if len(prefix):
        at = prefix[CASE].map(decision_pos)
        now = prefix[prefix["_pos"] == at].set_index(CASE)[TIMESTAMP].reindex(X.index)
        for column in spec.get("event_numeric", []):
            X["last:" + column] = pd.to_numeric(prefix[column], errors="coerce").groupby(prefix[CASE]).last().reindex(X.index)
        top = _events(log).value_counts().index[:MAX_ACTIVITY_COUNTS]
        events = _events(prefix)
        part = prefix[events.isin(top)]
        counts = part.groupby([CASE, events[part.index]]).size().unstack(fill_value=0).reindex(X.index, fill_value=0)
        # XGBoost erlaubt [, ] und < nicht in Merkmalsnamen
        counts.columns = ["n:" + re.sub(r"[\[\]<]", "_", str(name)) for name in counts.columns]
        X = X.join(counts)
        previous = pd.Series(events[prefix["_pos"] == at - 1].to_numpy(), index=prefix.loc[prefix["_pos"] == at - 1, CASE].to_numpy()).reindex(X.index)
        X["previous_activity"] = previous.fillna("<start>").astype("category")
        X["events"] = prefix.groupby(CASE).size().reindex(X.index).astype(float)
        X["elapsed_h"] = (now - start).dt.total_seconds() / 3600
    X["hour"] = now.dt.hour.astype(float)
    X["weekday"] = now.dt.weekday.astype(float)
    X["month"] = (now.dt.year * 12 + now.dt.month).astype(float)
    return X

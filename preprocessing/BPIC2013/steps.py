"""BPIC13 (IT-Incidents, Volvo IT): die Maßnahmen des Preprocessings nur für diesen Datensatz (STEPS)."""
import numpy as np
import pandas as pd

from prescriptive.helpers.datasets import ACTIVITY, CASE, LIFECYCLE

BPIC13_CASE_ATTRIBUTES = {"case:impact": "impact", "case:product": "product", "case:org_line": "organization involved"}


def bpic13_support_level(groups):
    """Support-Ebene je Event aus dem Namen der Gruppe: "3rd" bzw. "2nd" im Namen, sonst 1st."""
    groups = groups.astype(str)
    return pd.Series(np.where(groups.str.contains("3rd"), "3rd", np.where(groups.str.contains("2nd"), "2nd", "1st")), index=groups.index)


def bpic13_activity_with_level(log):
    """Aktivität = Status/Unterstatus [Support-Ebene], z. B. "Accepted/In Progress [1st]"; alle Events werden complete."""
    log[ACTIVITY] = log[ACTIVITY].astype(str) + "/" + log[LIFECYCLE].astype(str) + " [" + bpic13_support_level(log["org:group"]) + "]"
    log[LIFECYCLE] = "complete"
    return log


def bpic13_case_attributes(log):
    """Fallattribute case:impact, case:product, case:org_line: der erste vorhandene Wert des Event-Attributs im Fall."""
    cases = log.groupby(CASE, sort=False)
    for column, source in BPIC13_CASE_ATTRIBUTES.items():
        log[column] = cases[source].transform("first").astype(str)
    return log


STEPS = [bpic13_activity_with_level, bpic13_case_attributes]

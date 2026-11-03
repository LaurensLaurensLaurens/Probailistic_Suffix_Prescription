"""BPIC15 (Bauanträge, fünf niederländische Gemeinden): das Laden der fünf Logs, die Maßnahmen des Preprocessings nur für diesen Datensatz (STEPS) und die Phasenebene für das Prozessmodell."""
import os

import pandas as pd

from prescriptive.helpers.datasets import ACTIVITY, CASE, RESOURCE, TIMESTAMP

MUNICIPALITIES = range(1, 6)
TIME_FORMAT = "%m/%d/%y %H:%M:%S"  # im Export steht der Monat zuerst
MIN_VARIANT_CASES = 25  # Mindestzahl Fälle einer Variante im Prozessmodell (wie Teinemaa et al., BPIC 2015)


def load_bpic15(folder):
    """
    Die fünf Logs (BPIC15_<n>_sorted.csv, Disco-Export) als ein Roh-Log mit den Spalten wie im XES. Fall-IDs je Gemeinde
    eindeutig (zwei kämen sonst doppelt vor), Gemeinde in case:municipality. Aktivität ist der Code (concept:name, z. B.
    01_HOOFD_010); die Spalte Activity ist derselbe Code mit angehängtem Lifecycle und fällt weg.
    """
    parts = []
    for number in MUNICIPALITIES:
        part = pd.read_csv(os.path.join(folder, f"BPIC15_{number}_sorted.csv"), dtype=str)
        part[CASE] = f"M{number}_" + part["Case ID"]
        part["case:municipality"] = f"M{number}"
        parts.append(part)
    log = pd.concat(parts, ignore_index=True).drop(columns=["Case ID", "Activity"])
    log = log.rename(columns={"Resource": RESOURCE, "Complete Timestamp": TIMESTAMP})
    log = log.rename(columns=lambda column: column.replace("(case) ", "case:"))
    log[TIMESTAMP] = pd.to_datetime(log[TIMESTAMP], format=TIME_FORMAT)
    return log


def bpic15_numeric_leges(log):
    """case:SUMleges (Gebühren) als Zahl."""
    log["case:SUMleges"] = pd.to_numeric(log["case:SUMleges"], errors="coerce")
    return log


def bpic15_phases(log):
    """
    Das Log auf Phasenebene für das Prozessmodell (Teinemaa et al., BPIC 2015, Fig. 23), kein Schritt des Preprocessings: nur
    die 35 Phasen-Aktivitäten (activityNameEN "phase ...", "phased application" ist keine) unter ihrem Namen. Gleiche
    Zeitstempel (Batches) nach Aktivitätscode geordnet, dieselbe Phase direkt hintereinander einmal (eine Phase hat teils
    mehrere Codes), nur Varianten mit mindestens MIN_VARIANT_CASES Fällen.
    """
    phases = log[log["activityNameEN"].str.startswith("phase ")].sort_values([CASE, TIMESTAMP, ACTIVITY], kind="stable")
    phases = phases.assign(**{ACTIVITY: phases["activityNameEN"]})[[CASE, ACTIVITY, TIMESTAMP]]
    phases = phases[(phases[CASE] != phases[CASE].shift()) | (phases[ACTIVITY] != phases[ACTIVITY].shift())]
    variants = phases.groupby(CASE)[ACTIVITY].agg(tuple)
    frequent = variants[variants.map(variants.value_counts()) >= MIN_VARIANT_CASES].index
    return phases[phases[CASE].isin(frequent)]


STEPS = [bpic15_numeric_leges]

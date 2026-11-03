"""BPIC19 (Einkauf): die Maßnahmen des Preprocessings nur für diesen Datensatz (STEPS)."""
import pandas as pd

from prescriptive.helpers.datasets import CASE, TIMESTAMP

BPIC19_MIN_START = pd.Timestamp("2017-01-01")
# Das Log endet faktisch am 17.01.2019. Mit Clear Invoice (ohne Consignment) je Startmonat: Januar bis Juli 2018 91 bis 93 %,
# August 89 %, September 76 %, Oktober 53 %, November 29 %, Dezember 5 %: spätere Fälle sind abgeschnitten, nicht zu Ende
BPIC19_MAX_START = pd.Timestamp("2018-08-01")


def bpic19_drop_early_cases(log):
    """Fälle, die vor 2017 beginnen, entfernen (falsche Zeitstempel ab 1948; der Log umfasst Bestellungen ab 2018)."""
    start = log.groupby(CASE)[TIMESTAMP].transform("min")
    return log[start >= BPIC19_MIN_START].reset_index(drop=True)


def bpic19_drop_late_cases(log):
    """Fälle, die nach Juli 2018 beginnen, entfernen (das Log endet im Januar 2019, sie wären noch offen, nicht zu Ende)."""
    start = log.groupby(CASE)[TIMESTAMP].transform("min")
    return log[start < BPIC19_MAX_START].reset_index(drop=True)


def bpic19_numeric_net_worth(log):
    """Cumulative net worth (EUR) als Zahl."""
    log["Cumulative net worth (EUR)"] = pd.to_numeric(log["Cumulative net worth (EUR)"], errors="coerce")
    return log


STEPS = [bpic19_drop_early_cases, bpic19_drop_late_cases, bpic19_numeric_net_worth]

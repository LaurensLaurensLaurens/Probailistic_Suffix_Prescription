"""
Ankünfte der simulierten Fälle je Datensatz. dataset_arrivals ist der Einstieg für alle Simulatoren (die Engine
simulation/case_engine.py und simulation/dataset_simulator.py); ARRIVALS (unten) legt je Datensatz fest, wie die
Ankünfte entstehen.

Hier steht, was für mehrere Datensätze gilt; was nur ein Datensatz nutzt, liegt in dessen Ordner:
    real_starts    (BPIC13, BPIC19) hier: Startzeit und Fall echter Fälle zusammen
    resample_days  (BPIC17) BPIC2017/resample_days.py: generate_arrivals, echte Tage neu gezogen
"""
import numpy as np
import pandas as pd

from BPIC2017.resample_days import generate_arrivals  # Ordner "Instance Spawn Rate" liegt im Pfad, wie für dieses Modul


# ------------------------------------------------------------------ je Datensatz

# Wie der Simulator die Ankünfte eines Datensatzes bildet:
#   resample_days  echte Tage neu gezogen (BPIC2017/resample_days.py, generate_arrivals), aus period (None: der ganze
#                  Log). Für die Engine mit Mitarbeitern, in der die Ankunftsrate die Auslastung bestimmt.
#   real_starts    Startzeit und Fall echter Fälle zusammen (real_starts). Für Fälle ohne Konkurrenz um Mitarbeiter,
#                  deren Verlauf vom Startdatum abhängt.
ARRIVALS = {
    "bpic17": {"method": "resample_days", "period": None},
    # Ein Ausschnitt: 96 % der Fälle im April und Mai 2012, ältere nur, wenn noch offen, ab dem 6. Mai keine mehr.
    # Fälle vom Anfang Mai sind kurz, aus dem April lang; neu gezogene Tage verschöben diese Mischung (17 statt 9
    # Events je Fall), und mit unbeschränkten Tagen fielen 20.000 Ankünfte auf fünf Jahre (im Median 2 je Tag)
    "bpic13": {"method": "real_starts"},
    # Fast nur 2018 (2017 und 2019 je ein paar hundert Positionen am Rand); Fälle unabhängig wie bei BPIC13
    "bpic19": {"method": "real_starts"},
    # Die Werkstatt: Fälle konkurrieren um die Mitarbeiter, die Ankunftsrate bestimmt die Auslastung (wie BPIC17)
    "repair": {"method": "resample_days", "period": None},
}


def arrival_spec(dataset: str) -> dict:
    if dataset not in ARRIVALS:
        raise ValueError(f"Keine Ankünfte für '{dataset}' festgelegt, möglich: {', '.join(ARRIVALS)}")
    return ARRIVALS[dataset]


def real_starts(df: pd.DataFrame, num_cases: int, seed: int = 42):
    """
    Startzeiten echter Fälle, mit Zurücklegen gezogen, und je neuer Ankunft der echte Fall, von dem sie stammt (seine
    Fallattribute gehören dazu). Für Logs, deren Fälle ohne Konkurrenz um Mitarbeiter laufen und deren Verlauf vom
    Startdatum abhängt: dann zählt die Mischung der Startdaten, nicht die Ankunftsrate.
    """
    starts = pd.to_datetime(df.groupby("case:concept:name")["time:timestamp"].min())
    if starts.dt.tz is not None:
        starts = starts.dt.tz_convert("UTC").dt.tz_localize(None)
    picks = np.random.default_rng(seed).integers(len(starts), size=num_cases)
    return [ts.floor("us").to_pydatetime() for ts in starts.iloc[picks]], list(starts.index[picks])


def dataset_arrivals(dataset: str, df: pd.DataFrame, num_cases: int, seed: int = 42, start_date=None):
    """
    Ankünfte für den Datensatz nach ARRIVALS; resample_days ab start_date (Standard: der erste Tag des Zeitraums bzw.
    des Logs). Liefert die Ankunftszeiten und je Ankunft den echten Fall, dessen Attribute sie bekommt (bei
    resample_days None: die Attribute kommen vom Simulator).
    """
    spec = arrival_spec(dataset)
    if spec["method"] == "real_starts":
        return real_starts(df, num_cases, seed)
    period = spec.get("period")
    if start_date is None:
        start_date = pd.Timestamp(period[0]) if period else pd.to_datetime(df["time:timestamp"]).min().floor("D")
        start_date = (start_date.tz_convert("UTC").tz_localize(None) if start_date.tzinfo else start_date).to_pydatetime()
    return generate_arrivals(df, num_cases, start_date, seed, period=period), None

"""
Datensatz wählen und laden: Pfade, Katalog (prescriptive/catalog.yaml) und Roh-Log.

    from helpers.datasets import load_dataset
    log, info, catalog = load_dataset("bpic13")
"""
import os
import sys

import pandas as pd

# XES-Spalten, gemeinsam für alle helpers
CASE, ACTIVITY, TIMESTAMP, LIFECYCLE, RESOURCE = "case:concept:name", "concept:name", "time:timestamp", "lifecycle:transition", "org:resource"

DATASETS = ("bpic17", "bpic13", "bpic15", "bpic19", "repair")

# prescriptive/ und daneben process-simulation-engine/ (Simulator, models/); eventlog/ und preprocessing/ im Ordner darüber
HELPERS = os.path.dirname(os.path.abspath(__file__))
PRESCRIPTIVE = os.path.dirname(HELPERS)
PROJECT = os.path.dirname(PRESCRIPTIVE)
ROOT = os.path.join(PROJECT, "process-simulation-engine")
CACHE = os.path.join(PRESCRIPTIVE, "data", "cache")


# ------------------------------------------------------------------ Umgebung

# Importpfade: ROOT für die Pakete des Simulators, prescriptive/ für "from helpers... import" und der Ordner darüber für
# "import prescriptive", auch nach dem Wechsel des Arbeitsordners. ROOT steht vorn: prescriptive/ hat Ordner mit Namen wie
# Pakete des Simulators (evaluation), die sonst dessen Pakete verdeckten.
def add_paths():
    # TensorFlow (Next-Event-Modell) ohne Hinweise zu CUDA und CPU-Befehlen im Notebook
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    for path in (os.path.dirname(PRESCRIPTIVE), PRESCRIPTIVE, ROOT):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


# Arbeitsordner und Importpfade auf process-simulation-engine, Profil der Pipeline in prescriptive/ setzen (prescriptive/profiles.py)
def setup(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unbekannter Datensatz '{dataset}', möglich: {', '.join(DATASETS)}")
    os.chdir(ROOT)
    add_paths()
    profile, current = dataset, os.environ.get("PRESCRIPTIVE_PROFILE", "bpic17")
    # Die Schätzer in prescriptive/estimators/ legen ihre Spalten beim Import fest; ein anderes Profil braucht einen neuen
    # Kernel. Event-Log und RCT-Daten (enriched_log.enrich, helpers/rct.py) bekommen den Datensatz je Aufruf mit.
    loaded = [name for name in ("psp", "xgb_estimator", "lstm_estimator") if f"prescriptive.estimators.{name}" in sys.modules]
    if loaded and current != profile:
        print(f"Hinweis: die Schätzer ({', '.join(loaded)}) sind schon mit Profil {current} geladen. Für {dataset} den Kernel neu starten.")
    os.environ["PRESCRIPTIVE_PROFILE"] = profile


# ------------------------------------------------------------------ Katalog

def _catalog():
    add_paths()
    import yaml
    import interventions
    with open(interventions.CANDIDATES_FILE) as f:
        return yaml.safe_load(f)


# Eintrag des Datensatzes: event_log, prozess, outcome (type, beschreibung), screening, nicht_beeinflussbar, interventionen
def dataset_info(dataset):
    return _catalog()[dataset]


# Interventionen des Datensatzes als Tabelle (full: alle Felder, auch test für das Screening)
def intervention_table(dataset, full=False):
    import interventions
    table = pd.DataFrame(interventions.candidates(dataset))
    for column in ("intervention", "test"):
        if column not in table:
            table[column] = None
    # Zur Anzeige: der Name, die Art, die Intervention im Simulator und die Infofelder, soweit der Katalog sie hat
    shown = [column for column in ("name", "beschreibung", "art", "variable_typ", "simulator", "intervention", "daten", "hinweis") if column in table]
    return table if full else table[shown]


# ------------------------------------------------------------------ Roh-Log

# Unverändert aus der XES-Datei, bei BPIC15 aus den fünf CSV-Dateien (Preprocessing: preprocessing/), zwischengespeichert in prescriptive/data/cache/
def load_raw_log(dataset, path):
    cache = os.path.join(CACHE, f"{dataset}_raw.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    import contextlib
    import io

    import pm4py
    if dataset == "bpic15":
        from preprocessing.BPIC2015.steps import load_bpic15
        log = load_bpic15(os.path.join(PROJECT, path))
    else:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            log = pm4py.convert_to_dataframe(pm4py.read_xes(os.path.join(PROJECT, path)))
    os.makedirs(CACHE, exist_ok=True)
    log.to_pickle(cache)
    return log


def describe(log):
    return (f"Fälle {log[CASE].nunique():,}, Events {len(log):,}, Aktivitäten {log[ACTIVITY].nunique()}, "
            f"Zeitraum {log[TIMESTAMP].min():%Y-%m-%d} bis {log[TIMESTAMP].max():%Y-%m-%d}")


# Für das Notebook: Umgebung setzen, Roh-Log, Katalog-Eintrag und Tabelle der Interventionen
def load_dataset(dataset):
    setup(dataset)
    info = dataset_info(dataset)
    log = load_raw_log(dataset, info["event_log"])
    print(f"{dataset}: {info.get('prozess', dataset)}\nOutcome: {info['outcome'].get('beschreibung', info['outcome']['type'])}\nLog: {info['event_log']}\n{describe(log)}")
    return log, info, intervention_table(dataset)


# ------------------------------------------------------------------ mehrere Datensätze im Notebook

def per_dataset(datasets, function, show=True, heading=True):
    """
    function(datensatz) für jeden der gewählten Datensätze; im Notebook je Datensatz eine Überschrift (heading) und das
    Ergebnis (show: True, False oder eine Funktion, die aus dem Ergebnis macht, was angezeigt wird). Gibt die Ergebnisse
    je Datensatz zurück: {datensatz: ergebnis}.
    """
    try:
        from IPython.display import Markdown, display
    except ImportError:
        Markdown, display = str, print
    results = {}
    for dataset in datasets:
        if heading:
            display(Markdown(f"### {dataset}: {dataset_info(dataset).get('prozess', dataset)}"))
        results[dataset] = function(dataset)
        shown = show(results[dataset]) if callable(show) else (results[dataset] if show else None)
        if shown is not None:
            display(shown)
    return results

"""
Der Simulator je Datensatz: das Next-Event-Modell (next_event_prediction/event_time_model.py) in einer von zwei Engines, die
Wahlpunkte aus dem Katalog (interventions.choice_points) und die Zuteilungsstrategie aus lauf.zuteilung. Mit Arbeitsschritten
(work_item_prefix, BPIC17) simulation/case_engine_lifecycle_aware.py: alle Fälle konkurrieren um die Mitarbeiter (Dienstplan,
Berechtigungen). Mit performers (BPIC19, nur complete-Events) simulation/case_engine_lifecycle_unaware.py: jeder Fall läuft für
sich, das Modell zieht auch, wer ein Event ausführt, und die Policy kann Aktivität und Person ersetzen. BPIC13 hat noch keine
Engine: sein Modell zieht keine Mitarbeiter.

    simulator = Simulator("bpic17", log)        # log: vorverarbeitet; Modell vorhanden: geladen, sonst trainiert (retrain=True: immer)
    events, choices = simulator.run(2000, seed=0)                          # ohne Policy: wie die Bank
    events, choices = simulator.run(2000, policy=RCTPolicy(seed=1), seed=0)                             # RCT-Daten
    events, choices = simulator.run(4000, policy=XGBPolicy(), points=["channel"], window=(700, 2700), seed=3)

run: Ohne Policy zieht das Modell jedes Event wie die Bank und die Zuteilungsstrategie teilt die Mitarbeiter zu. Mit Policy
fragt die Engine sie an den Wahlpunkten (points: nur diese, Standard alle) für die Fälle mit Ankunftsnummer in window
(Standard alle). events: das Event-Log (Spalten wie der echte Log), choices: jede Wahl der Policy (case_engine._ask).
Ankünfte und Fallattribute kommen aus dem Log (Instance Spawn Rate/case_arrivals.py, case_attributes/), Dienstplan und
Berechtigungen ebenso (resources/).
"""
import os
import time

import numpy as np
import pandas as pd

import interventions
from next_event_prediction.datasets import model_spec

CASE = "case:concept:name"
ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING = {"width": 128, "depth": 2, "dropout": 0.1, "epochs": 80, "patience": 8, "seed": 42, "threads": 8}
DATASETS = ("bpic17", "bpic13", "bpic19", "repair")  # Datensätze mit Einträgen in NEXT_EVENT, CASE_ATTRIBUTES, ARRIVALS und RESOURCES
CHOICE_COLUMNS = ["case", "point", "kind", "choice", "index", "options", "time"]
WAIT_COLUMNS = ["case", "activity", "lifecycle", "seconds", "reason"]


class Simulator:
    def __init__(self, dataset, log, intervention=None, retrain=False, **training):
        """
        log: vorverarbeitet (prescriptive/helpers/preprocessing). intervention: Control-Flow-Intervention (Standard
        standard_intervention im Katalog). training: Einstellungen für das Training des Modells (TRAINING).
        """
        if dataset not in DATASETS:
            raise ValueError(f"Simulator: bisher nur {', '.join(DATASETS)}, nicht {dataset}")
        self.dataset, self.log = dataset, log
        self.intervention = interventions.get(intervention or interventions.catalog()[dataset]["standard_intervention"])
        self.points = interventions.choice_points(dataset, self.intervention)
        self.model_folder = model_folder(dataset)
        if retrain or not is_trained(self.model_folder):
            print(f"Trainiere den Simulator für {dataset} ...", flush=True)
            train_model(log, dataset, self.model_folder, **training)
        self.waiting_hours = None  # Wartezeit auf Mitarbeiter aller Fälle im letzten Lauf
        self.waits = None  # jede Wartezeit auf einen Mitarbeiter im letzten Lauf (WAIT_COLUMNS)

    def __repr__(self):
        return (f"Simulator({self.dataset}, Wahlpunkte {', '.join(point.name for point in self.points)}, Zuteilung "
                f"{interventions.run_allocation(self.dataset)}, Modell {os.path.relpath(self.model_folder, ENGINE_ROOT)})")

    def run(self, n_cases, policy=None, points=None, window=None, seed=0, observer=None, observe_hours=None):
        """
        n_cases Fälle gemeinsam simulieren. policy: mit choose(point, case, options, time, engine), None: wie die Bank;
        points: Namen der Wahlpunkte, an denen die Policy gefragt wird (Standard alle); window: (erste, letzte + 1)
        Ankunftsnummer der Fälle, für die sie gefragt wird. observer: wird alle observe_hours mit (Zeit, Engine) aufgerufen
        (z. B. ein Lagebild), ändert den Verlauf nicht. Liefert events und choices (DataFrames).
        """
        if model_spec(self.dataset)["work_item_prefix"]:
            from simulation.case_engine_lifecycle_aware import build_engine
        else:
            from simulation.case_engine_lifecycle_unaware import build_engine
        chosen = self.points
        if points is not None:
            unknown = set(points) - {point.name for point in self.points}
            if unknown:
                raise ValueError(f"Unbekannte Wahlpunkte {', '.join(sorted(unknown))}, möglich: {', '.join(point.name for point in self.points)}")
            chosen = [point for point in self.points if point.name in points]
        engine = build_engine(self.log, n_cases, seed, self.model_folder, allocation=interventions.run_allocation(self.dataset), dataset=self.dataset,
                              load=interventions.arrival_load(self.dataset))
        engine.set_policy(policy, chosen if policy is not None else [], window)
        if observer is not None:
            engine.observer, engine.observe_seconds = observer, observe_hours * 3600
        events = pd.DataFrame(engine.run(n_cases))
        self.waiting_hours = sum(engine.waiting_seconds.values()) / 3600
        self.waits = pd.DataFrame(engine.waits, columns=WAIT_COLUMNS)
        self.forced_assignments = engine.forced_assignments  # Aufgaben, die nach MAX_WAIT jemanden außer Dienst bekamen
        return events, pd.DataFrame(engine.choices, columns=CHOICE_COLUMNS)


# ------------------------------------------------------------------ Simulationsmodell

# Fälle mit höchstens so vielen Events, wie das Modell abbilden kann (max_events, next_event_prediction/datasets.py)
def modelable_cases(log, dataset, verbose=False):
    from next_event_prediction.datasets import model_spec
    longest = model_spec(dataset)["max_events"] - 1
    sizes = log.groupby(CASE).size()
    keep = set(sizes.index[sizes <= longest])
    if verbose and len(keep) < len(sizes):
        print(f"Für den Simulator weggelassen: {len(sizes) - len(keep)} Fälle mit mehr als {longest} Events")
    return keep


# Traces des Next-Event-Modells aus dem vorverarbeiteten Log, ohne zu lange Fälle
# Aufgerufen von train_model
def training_traces(log, dataset, verbose=False):
    from next_event_prediction.data_prep import load_traces
    return load_traces(log[log[CASE].isin(modelable_cases(log, dataset, verbose))], dataset)



# Next-Event-Modell trainieren wie next_event_prediction/train.py --model lstm_time --all-data: alle Fälle, ein
# zufälliges 5 % beendet nur das Training. Die Eingänge legt event_time_model.py aus dem Datensatz fest
# (next_event_prediction/datasets.py): Mitarbeiter bei Datensätzen mit Arbeitsschritten, die aktuellen dynamischen Attribute
# Aufgerufen von Simulator (Modell fehlt oder retrain)
def train_model(log, dataset, folder, **params):
    import keras
    import tensorflow as tf

    from next_event_prediction.data_prep import Encoder
    from next_event_prediction.event_time_model import EventTimeModel
    params = {**TRAINING, **params}
    try:
        tf.config.threading.set_intra_op_parallelism_threads(params["threads"])
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass  # TensorFlow lief in diesem Prozess schon
    keras.utils.set_random_seed(params["seed"])
    # Vorverarbeitung der training traces. Siehe in training_traces
    traces = training_traces(log, dataset, verbose=True)
    # Wir wollen, dass nicht die letzten 5% der traces alleine der Validierung dienen, sondern dass sie gemischt werden
    order = np.random.default_rng(params["seed"]).permutation(len(traces))
    validation = [traces[i] for i in order[: len(traces) // 20]]
    train = [traces[i] for i in order[len(traces) // 20:]]
    encoder = Encoder(traces, dataset, month_traces=train + validation)
    spec = model_spec(dataset)
    model = EventTimeModel(encoder, spec["max_events"], width=params["width"], depth=params["depth"], dropout=params["dropout"])
    print(f"Eingänge des Simulators: Fallattribute" + (", Mitarbeiter (bisherige im Prefix, nächster für die Zeit)" if model.use_resources else "")
          + (", aktuelle dynamische Attribute" if model.use_offers else ""), flush=True)
    started = time.time()
    history = model.fit(train, validation, epochs=params["epochs"], patience=params["patience"])
    model.save(folder)
    print(f"Simulator trainiert: {len(history.history['loss'])} Epochen in {(time.time() - started) / 60:.1f} min, gespeichert in {folder}")


# Modellordner des Datensatzes: model_folder in next_event_prediction/datasets.py, sonst models/next_event/<datensatz>_lstm_time
def model_folder(dataset):
    return os.path.join(ENGINE_ROOT, model_spec(dataset).get("model_folder") or os.path.join("models", "next_event", f"{dataset}_lstm_time"))


def is_trained(folder):
    return os.path.exists(os.path.join(folder, "config.json"))

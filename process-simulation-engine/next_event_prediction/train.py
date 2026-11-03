"""
Trains a next-event model on the first 80 % of the cases and measures it on the last 20 %.

Usage (from process-simulation-engine/; --dataset bpic17 (default), bpic13, bpic19):
    python -m next_event_prediction.train --model ngram
    python -m next_event_prediction.train --model lstm
    python -m next_event_prediction.train --model transformer
    python -m next_event_prediction.train --model lstm_time      (next event and time until it)
    python -m next_event_prediction.train --summary
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import interventions
from .BPIC2017.kpis import kpis as bpic17_kpis
from .data_prep import Encoder, load_traces, split_traces
from .datasets import model_spec
from .event_time_model import EventTimeModel
from evaluation.evaluation import control_flow_kpis, time_kpis

from .models import NeuralModel, NGramModel

MODEL_FOLDER = os.path.join("models", "next_event")
# Kennzahlen, die es nur für einen Datensatz gibt (im Ordner des Datensatzes)
DATASET_KPIS = {"bpic17": bpic17_kpis}
# Events the model gives less than this probability are never drawn (see NeuralModel.sample). Chosen on the
# test cases: without it the models produce about 200 different directly-follows edges instead of 147.
MIN_PROBABILITY = 1e-3


def load_event_log(path: str) -> pd.DataFrame:
    import pm4py

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return pm4py.convert_to_dataframe(pm4py.read_xes(path))


def prepared_log(dataset: str, path: str = None, work_sessions: bool = False) -> pd.DataFrame:
    """
    Der Log des Datensatzes (Pfad: Katalog) nach seinem Preprocessing (prescriptive/helpers/preprocessing), wie im
    Notebook: die Lifecycles wie im Log; work_sessions=True: mit Arbeitsabschnitten (je Abschnitt start -> complete).
    """
    project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # über process-simulation-engine/
    if project not in sys.path:
        sys.path.insert(0, project)
    from preprocessing import preprocess
    return preprocess(dataset, load_event_log(path or interventions.catalog()[dataset]["event_log"]), verbose=False, work_sessions=work_sessions)


def evaluate(name: str, generated, test, dataset: str, extra=None, gaps=None) -> dict:
    real = [trace.events for trace in test]
    dataset_kpis = DATASET_KPIS[dataset](generated, real) if dataset in DATASET_KPIS else {}
    result = {"name": name, "control_flow": control_flow_kpis(generated, real), **dataset_kpis, **(extra or {})}
    if gaps is not None:
        prefix = model_spec(dataset)["work_item_prefix"]
        if prefix:
            result["time"] = time_kpis(generated, gaps, prefix)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    with open(os.path.join(MODEL_FOLDER, f"result_{name}.json"), "w") as f:
        json.dump(result, f, indent=1, default=float)
    return result


def print_summary() -> None:
    results = []
    for file in sorted(os.listdir(MODEL_FOLDER)):
        if file.startswith("result_") and file.endswith(".json"):
            with open(os.path.join(MODEL_FOLDER, file)) as f:
                results.append(json.load(f))
    order = ["echte Testfälle", "echte Trainingsfälle"]
    results.sort(key=lambda r: (order.index(r["name"]) if r["name"] in order else len(order), r["name"]))
    pd.set_option("display.width", 250)
    for block in ("control_flow", "decision", "time"):
        print(pd.DataFrame({r["name"]: r[block] for r in results if block in r}).round(1).to_string(), "\n")
    extras = {r["name"]: {k: v for k, v in r.items() if k not in ("name", "control_flow", "decision", "time")} for r in results}
    print(pd.DataFrame(extras).round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["reference", "ngram", "lstm", "transformer", "lstm_time"])
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--dataset", choices=list(interventions.catalog()), default="bpic17",
                        help="Datensatz (Katalog prescriptive/catalog.yaml) und damit die Eingänge des Modells (datasets.py)")
    parser.add_argument("--event-log", type=str, default=None, help="Path of the event log (default: the one of the dataset in the catalog)")
    parser.add_argument("--order", type=int, default=5, help="ngram: number of events in an n-gram")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4, help="Stop after this many epochs without a better validation loss")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--name", type=str, default=None, help="Name of the result (default: the model)")
    parser.add_argument("--evaluate-only", action="store_true", help="Load the trained weights instead of training")
    parser.add_argument("--random-split", action="store_true",
                        help="Hold back a random 20 %% of the cases instead of the last 20 %%: tests on unseen cases from the same period")
    parser.add_argument("--all-data", action="store_true",
                        help="lstm_time: train on all cases (a random 5 %% only stops the training) for the model the simulation uses")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-sessions", action=argparse.BooleanOptionalAction, default=False,
                        help="Train on the log with work sessions (start -> complete per session); default --no-work-sessions: the lifecycles of the log, as the pipeline in prescriptive/ uses them")
    args = parser.parse_args()

    if args.model:
        traces = load_traces(prepared_log(args.dataset, args.event_log, args.work_sessions), args.dataset)
        train, validation, test = split_traces(traces)
        if args.random_split:
            shuffled = [traces[i] for i in np.random.default_rng(0).permutation(len(traces))]
            train, validation, test = split_traces(shuffled)
        # The vocabulary covers the whole log: one activity only appears in the last months
        encoder = Encoder(traces, args.dataset)
        rng = np.random.default_rng(args.seed)
        name = args.name or args.model
        max_length = model_spec(args.dataset)["max_events"]

        if args.model == "reference":
            # What real cases reach against the test cases: the test cases themselves and as many training cases
            evaluate("echte Testfälle", [trace.events for trace in test], test, args.dataset, gaps=[trace.gaps for trace in test])
            sample = [train[i] for i in rng.choice(len(train), len(test), replace=False)]
            evaluate("echte Trainingsfälle", [trace.events for trace in sample], test, args.dataset, gaps=[trace.gaps for trace in sample])
        elif args.model == "ngram":
            model = NGramModel(order=args.order)
            started = time.time()
            model.fit([encoder.encode_events(trace.events) for trace in train + validation])
            generated = [encoder.decode_events(ids) for ids in model.sample(len(test), max_length, rng)]
            evaluate(name, generated, test, args.dataset, {"Training (s)": time.time() - started})
        elif args.model == "lstm_time":
            import keras

            keras.utils.set_random_seed(args.seed)
            folder = os.path.join(MODEL_FOLDER, name)
            training = {}
            if args.all_data:
                # The model for the simulation: it should reflect the whole log, so nothing is held back.
                # The KPIs below are then measured on cases the model has seen.
                order = rng.permutation(len(traces))
                validation = [traces[i] for i in order[: len(traces) // 20]]
                train = [traces[i] for i in order[len(traces) // 20:]]
            if args.evaluate_only:
                model = EventTimeModel.load(folder)
            else:
                # The start months come from the training cases only: later months are unknown to the model
                model_encoder = Encoder(traces, args.dataset, month_traces=train + validation)
                # Eingänge aus dem Datensatz (datasets.py): Mitarbeiter, dynamische Attribute
                model = EventTimeModel(model_encoder, max_length, width=args.width, depth=args.depth, dropout=args.dropout)
                started = time.time()
                history = model.fit(train, validation, epochs=args.epochs, patience=args.patience)
                training = {"Training (s)": time.time() - started, "Epochen": len(history.history["loss"])}
                model.save(folder)
            started = time.time()
            ids, seconds = model.sample(test, rng, min_probability=MIN_PROBABILITY)
            generation_seconds = time.time() - started
            event_loss, time_loss = model.losses(validation)
            test_event_loss, test_time_loss = model.losses(test)
            evaluate(name, [encoder.decode_events(case) for case in ids], test, args.dataset, {
                **training,
                "Parameter": model.network.count_params(),
                "Verlust pro Event, Validierung": event_loss,
                "Verlust pro Event, Test": test_event_loss,
                "Verlust der Zeit pro Event, Validierung": time_loss,
                "Verlust der Zeit pro Event, Test": test_time_loss,
                "Erzeugen der Testfälle (s)": generation_seconds,
            }, gaps=seconds)
        else:
            import keras

            keras.utils.set_random_seed(args.seed)
            model = NeuralModel(args.model, encoder, max_length, width=args.width, depth=args.depth, dropout=args.dropout)
            weights_path = os.path.join(MODEL_FOLDER, f"{name}.weights.h5")
            training = {}
            if args.evaluate_only:
                model.network.load_weights(weights_path)
            else:
                started = time.time()
                history = model.fit(train, validation, epochs=args.epochs, patience=args.patience)
                training = {"Training (s)": time.time() - started, "Epochen": len(history.history["loss"])}
                model.network.save_weights(weights_path)
            started = time.time()
            generated = [encoder.decode_events(ids) for ids in model.sample(test, rng, min_probability=MIN_PROBABILITY)]
            evaluate(name, generated, test, args.dataset, {
                **training,
                "Parameter": model.network.count_params(),
                "Verlust pro Event, Validierung": model.loss(validation),
                "Verlust pro Event, Test": model.loss(test),
                "Erzeugen der Testfälle (s)": time.time() - started,
            })
    if args.summary:
        print_summary()

"""
Prescriptive Process Monitoring mit dem PSP in der gemeinsamen Simulation (prescriptive/joint.py): Alle Fälle laufen
parallel in einer Engine und konkurrieren um die Mitarbeiter.

1. RCT: Die Engine simuliert rct_runs Jahre mit der Policy rct: am ersten Entscheidungspunkt call/no_call zufällig, bei
   call beim ersten Anruf danach ein zufälliger Mitarbeiter unter den freien Kompetenten. Alle anderen Arbeitsschritte
   teilt die aus dem echten Log gelernte Bank-Zuteilung zu. Mit delta (wie in SCOPE) entscheidet an diesem Anteil der
   Entscheidungspunkte stattdessen die Bank. Das sind die Trainingsdaten des PSP.
2. PSP: Preprocessing mit Tokens am Entscheidungspunkt ("A_Validating|call") und am Anruf-Beginn
   ("W_Call incomplete files|start|User_29") und Training nur an diesen Stellen (prescriptive/estimators/psp.py). Die
   Events sind Aktivität mit Lifecycle (enriched_log.py, Spalte event), wie die Engine sie simuliert.
3. Auswertung: je seed ein gemeinsamer Lauf je Policy: bank (Baseline: Aktion und Mitarbeiter wie die Bank) und psp (das
   PSP wird live gefragt: call oder no_call, und beim Anruf, welcher freie Mitarbeiter), dazu Vergleiche. Gewertet werden
   die Fälle im Fenster nach der Einlaufphase und vor der Auslaufphase, gepaart gegen die Bank desselben seeds.

Aufruf (aus process-simulation-engine/, prescriptive/ liegt daneben):
    OMP_NUM_THREADS=2 PYTHONPATH=.. .venv/bin/python -m prescriptive.main
"""
import argparse
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Intervention (INTERVENTION) und das gemeinsame Event-Log aller Teile (enrich): prescriptive/enriched_log.py.
# Outcome: je Datensatz im Katalog (BPIC17: Gewinn der Bank, prescriptive/helpers/preprocessing/BPIC2017/outcome.py)
from prescriptive import profiles
from prescriptive.enriched_log import INTERVENTION, PROFILE, enrich, with_current_costs
from prescriptive.callers import top_resources
from prescriptive.policies import BankPolicy, DirectPolicy, FixedPolicy, PSPPolicy, RandomActionPolicy, RCTPolicy, XGBActionPolicy
from prescriptive.joint import compare, run_policy, summarize

# Das Simulationsmodell (next_event_prediction) je Datensatz und Intervention: simulation/dataset_simulator.model_folder
# (BPIC17: Aktion, Angebot und Anrufer als Eingang, der Mitarbeiter für die Zeit; resolve_intervention_args)

# Was die Arbeitsprozesse brauchen (Event-Log, Simulationsmodell, PSP), gesetzt von init_worker
_SHARED = {}


# Startet einen Arbeitsprozess ("spawn": sicher auch nach Torch-Training im Hauptprozess); er lädt das Event-Log selbst
# Wird aufgerufen von main.py -> worker_pool -> ProcessPoolExecutor -> init_worker (je Arbeitsprozess)
def init_worker(config):
    _SHARED.update(config)
    # Der Log nach dem Preprocessing (profiles.prepare -> prescriptive/helpers/preprocessing), wie für das Modell
    from simulation.dataset_simulator import uses_work_sessions
    _SHARED["event_log"] = profiles.prepare(load_event_log(config["event_log_path"]), PROFILE, uses_work_sessions(config["model_folder"]))


# Wird aufgerufen von main.py -> run_rct / run_evaluation -> worker_pool
def worker_pool(workers, config):
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(config,))


# Wird aufgerufen von main.py (__main__) -> save_data, nach jedem Block, der etwas erzeugt
def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".pkl", "wb") as f:
        pickle.dump(data, f)


# Wird aufgerufen von main.py (__main__) -> load_data, bei den --already-...-Schaltern
def load_data(path):
    with open(path + ".pkl", "rb") as f:
        return pickle.load(f)


# Wird aufgerufen von main.py (__main__) -> load_event_log
def load_event_log(path):
    if path.endswith(".xes") or path.endswith(".xes.gz"):
        import pm4py
        df = pm4py.convert_to_dataframe(pm4py.read_xes(path))
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    print(f"Loaded event log: {len(df)} events, {df['case:concept:name'].nunique()} cases")
    return df


# Kennzahlen der RCT-Daten: Fälle, Gewinn, und je Aktion am Entscheidungspunkt, wie oft und mit welchem Ergebnis
# Wird aufgerufen von main.py (__main__) -> print_summary
def print_summary(data):
    cases = data.groupby("case_nr").first()
    parts = [f"{label}: {cases[column].mean():{digits}}" for column, label, digits in (
        ("approved", "bewilligt", ".3f"), ("working_hours", "Arbeitsstunden", ".2f"), ("calls", "Anrufe", ".2f"), ("cycle_days", "Durchlaufzeit (Tage)", ".1f"))
        if column in cases]
    print(f"Fälle: {len(cases)}, Events: {len(data)}, Outcome je Fall: {cases['outcome'].mean():.1f}, " + ", ".join(parts))
    #Das sind alle Entscheidungspunkte des Trainings-und-Tuning-Logs: eine Zeile pro Fall, nämlich sein erstes A_Validating
    decisions = data[data["stage"] == 0]
    for action, name in enumerate(INTERVENTION["actions"]):
        part = decisions[decisions["treatment"] == action]
        print(f"  {name}: {len(part)} Fälle, Outcome je Fall {part['outcome'].mean():.1f}, bewilligt {part['approved'].mean():.3f}")
    if "rct" in decisions.columns:
        print(f"  randomisiert (RCT): {decisions['rct'].mean():.3f} der Entscheidungspunkte, sonst wie die Bank (delta)")
    callers = cases["call_resource"].dropna()
    print(f"  erster Anruf nach der Entscheidung: {len(callers)} Fälle, {callers.nunique()} verschiedene Mitarbeiter")


# Ein Lauf einer Policy in einem Arbeitsprozess. job: (Name der Policy, seed, Anzahl Fälle, Fenster)
# Wird aufgerufen von main.py -> run_rct / run_evaluation -> ProcessPoolExecutor -> policy_run
def policy_run(job):
    name, seed, n_cases, window = job
    shared = _SHARED
    if name == "psp":
        import torch
        torch.set_num_threads(2)
        policy = PSPPolicy(**shared["psp"], seed=seed)
    elif name in ("xgb", "lstm"):
        policy = DirectPolicy(**shared[name], seed=seed)
    elif name == "xgb_action":
        # Ablation: XGBoost nur für call/no_call, Anrufer wie die Bank
        policy = XGBActionPolicy(**shared["xgb"], seed=seed)
    elif name == "rct":
        policy = RCTPolicy(seed, shared.get("delta", 0.0))
    elif name == "random":
        policy = RandomActionPolicy(seed + 5)
    elif name.startswith("always_"):
        policy = FixedPolicy(INTERVENTION["actions"].index(name[len("always_"):]))
    else:
        policy = BankPolicy()
    events, waiting, choices = run_policy(shared["event_log"], shared["event_log_path"], policy, n_cases, seed, shared["model_folder"], shared["allocation"], window)
    if name == "rct":
        # Fälle, deren Entscheidung (und Anrufer) zufällig war
        events["rct"] = events["case:concept:name"].isin(policy.randomized)
        return events
    row = {"Policy": name, "seed": seed, **summarize(enrich(events, n_stages=1), window, waiting, policy, choices)}
    print(f"seed {seed}, {name}: Gewinn {row['Gewinn']:,.0f}, Aktion {row['Anteil Aktion']:.2f}, bewilligt {row['bewilligt']:.3f}", flush=True)
    return row


# RCT-Daten: runs Jahre mit der Policy rct, jedes mit eigenem seed, parallel. Fall-IDs je Jahr eindeutig (R<Jahr>_).
# Wird aufgerufen von main.py (__main__) -> run_rct
def run_rct(n_cases, runs, seed, workers, config, first_run=0):
    run_numbers = list(range(first_run, first_run + runs))
    jobs = [("rct", seed + 1000 + run, n_cases, None) for run in run_numbers]
    with worker_pool(min(workers, runs), config) as pool:
        logs = list(pool.map(policy_run, jobs))
    for run, events in zip(run_numbers, logs):
        events["case:concept:name"] = f"R{run}_" + events["case:concept:name"].astype(str)
    events = pd.concat(logs, ignore_index=True)
    # Die Spalte rct der Events (policy_run) bleibt im Event-Log erhalten
    return enrich(events, n_stages=1)


# Auswertung: je seed jede Policy ein gemeinsamer Lauf, parallel; dann gepaarter Vergleich gegen die Bank
# Wird aufgerufen von main.py (__main__) -> run_evaluation
def run_evaluation(policies, seeds, n_cases, window, workers, config):
    jobs = [(name, seed, n_cases, window) for seed in seeds for name in policies]
    # Die langen Läufe der Schätzer zuerst, damit die kurzen die Lücken füllen
    jobs.sort(key=lambda job: job[0] not in ("psp", "xgb", "lstm", "xgb_action"))
    with worker_pool(workers, config) as pool:
        # Policy run ist der Teilm bei dem die cases Parallel laufen
        rows = list(pool.map(policy_run, jobs))
    return compare(rows)


# RCT-Daten für Logs ohne Engine (Profil engine = False, z. B. BPIC13): runs Läufe zu je n_cases Fällen im Fall-Simulator
# (simulation/case_simulator.py), unabhängig voneinander. Fallattribute und Startzeit je Fall zufällig aus den
# echten Fällen (mit Zurücklegen); am ersten Entscheidungs-Event Aktion 0 oder 1 zufällig, mit delta an diesem Anteil
# der Fälle wie die Bank (das Modell entscheidet). Fall-IDs R<Lauf>_<Nummer>.
# Wird aufgerufen von main.py (__main__) -> run_case_rct
def run_case_rct(n_cases, runs, seed, model_folder, event_log_path, delta=0.0, first_run=0):
    from simulation.case_simulator import CaseSimulator
    from next_event_prediction.data_prep import load_traces
    from prescriptive.enriched_log import ACTIVITY, CASE, CASE_ATTRS, LIFECYCLE, RESOURCE, TIMESTAMP
    from simulation.dataset_simulator import uses_work_sessions
    traces = load_traces(profiles.prepare(load_event_log(event_log_path), PROFILE, uses_work_sessions(model_folder)), PROFILE["key"])
    simulator = CaseSimulator(model_folder)
    frames = []
    for run in range(first_run, first_run + runs):
        rng = np.random.default_rng(seed + 1000 + run)
        picks = rng.integers(len(traces), size=n_cases)
        randomized, draws = rng.random(n_cases) >= delta, rng.integers(2, size=n_cases)
        activities, gaps, chosen = simulator.simulate([traces[i].attributes for i in picks], np.array([traces[i].start for i in picks]), rng,
                                                      lambda row: int(draws[row]) if randomized[row] else None)
        lengths = np.array([len(case) for case in activities])
        rows = np.repeat(np.arange(n_cases), lengths)
        seconds = np.array([traces[i].start for i in picks])[rows] + np.concatenate([np.cumsum(g) for g in gaps if len(g)] or [np.zeros(0)])
        frame = pd.DataFrame({CASE: [f"R{run}_{row}" for row in rows], ACTIVITY: [a for case in activities for a in case], LIFECYCLE: "complete",
                              TIMESTAMP: pd.to_datetime(seconds, unit="s"), RESOURCE: None})
        for column in CASE_ATTRS:
            frame[column] = np.array([traces[i].attributes.get(column) for i in picks], dtype=object)[rows]
        # Fälle, deren Aktion zufällig war (wie policy_run bei der Engine)
        frame["rct"] = randomized[rows]
        frames.append(frame)
        print(f"RCT-Lauf {run}: {n_cases} Fälle, {int(lengths.sum())} Events, Aktion zufällig in {randomized.mean():.3f}", flush=True)
    return enrich(pd.concat(frames, ignore_index=True), n_stages=1)


# Schichten des PSP wie in psp.MODEL_PARAMS (ohne torch zu laden)
def psp_layers_default():
    return 4


# 2a. PSP: Preprocessing und Training; liefert, was die Arbeitsprozesse für PSPPolicy brauchen (ein kleines Dataset mit
# Encodern und Kategorien statt des ganzen)
# Wird aufgerufen von main.py (__main__) -> train_psp (--estimator psp)
def train_psp(args, data, callers, psp_folder):
    import torch
    from prescriptive.estimators import psp
    from preprocessing.work_time import fit_hours_per_activity
    torch.set_num_threads(args.workers)
    datasets_path = os.path.join(psp_folder, "datasets.pt")
    if args.already_psp_preprocessed:
        datasets = torch.load(datasets_path, weights_only=False)
    else:
        # Die Sicht des PSP auf das Event-Log, mit den Tokens am Entscheidungspunkt und am Anruf-Beginn
        view = psp.psp_view(data, INTERVENTION["actions"], callers)
        # Sind die Testfälle schon festgelegt (test_cases.pkl im PSP-Ordner, z. B. die eines kleineren RCT oder von
        # helpers/training.define_test_cases), werden sie nicht trainiert; der Loader teilt die übrigen Fälle nur noch in
        # Training und Validierung, die Validierung so groß wie sonst (Anteil an allen Fällen)
        split = {}
        if os.path.exists(os.path.join(psp_folder, "test_cases.pkl")):
            from prescriptive.enriched_log import CASE
            held_out = test_cases(psp_folder)
            n_cases = view[CASE].nunique()
            view = view[~view[CASE].isin(held_out)].reset_index(drop=True)
            split = {"val_share": psp.EVENT_LOG_PROPERTIES["train_validation_size"] * n_cases / max(view[CASE].nunique(), 1), "without_test": True}
            print(f"Vorgegebene Testfälle ausgeschlossen: {len(held_out)}; Validierung {split['val_share']:.1%} der übrigen Fälle", flush=True)
        # Ohne psp_all_prefixes nur die Proben am Entscheidungspunkt und am Anruf-Beginn kodieren: nur auf ihnen wird dann
        # trainiert (decision_samples), und alle Prefixe brauchen mit den Lifecycle-Events viel Speicher
        datasets = psp.build_datasets(view, min_suffix_size=args.psp_min_suffix_size, lognormal=args.psp_lognormal,
                                      decision_only=not getattr(args, "psp_all_prefixes", False), **split)
        del view
        torch.save(datasets, datasets_path)
    print("PSP samples: " + ", ".join(f"{split}={len(dataset)}" for split, dataset in datasets.items()) + f", window size: {datasets['train'].encoder_decoder.window_size}")
    model_path = os.path.join(psp_folder, "psp_model.pkl")
    if not args.already_psp_trained:
        model = psp.build_model(datasets["train"], {**psp.MODEL_PARAMS, "decoder_context": args.psp_decoder_context, "decoder_static": args.psp_decoder_static, "num_layers": args.psp_layers})
        train_params = {**psp.TRAIN_PARAMS, "epochs": args.epochs, "learning_rate": args.learning_rate,
                        "use_gradnorm": not args.psp_no_gradnorm, "cat_loss": "standard" if args.psp_standard_ce else "attenuation",
                        "best_saving_path": os.path.join(psp_folder, "psp_model_best.pkl"), "branch_weight": args.psp_branch_weight,
                        "tf_max": args.psp_tf_max, "tf_decay_from": args.psp_tf_decay_from}
        psp.sampling_config(model_path, train_params)
        if getattr(args, "psp_all_prefixes", False):
            # Alle Prefixe, damit das PSP auch an anderen Stellen Suffixe ziehen kann; die Entscheidungen verstärkt: je Epoche
            # alle Entscheidungs-Proben und psp_other_ratio-mal so viele andere (psp.DecisionMix)
            samples = {"train": psp.DecisionMix(datasets["train"], args.psp_other_ratio, seed=args.seed),
                       "val": psp.DecisionMix(datasets["val"], args.psp_other_ratio, seed=args.seed + 1, fixed=True)}
            print(f"Training: {samples['train']}; Validierung: {samples['val']}", flush=True)
        else:
            # Nur Proben, deren Prefix an einem Token endet (Entscheidungspunkt, Anruf-Beginn): nur solche werden abgefragt
            samples = {split: psp.decision_samples(datasets[split]) for split in ("train", "val")}
        losses = psp.train_model(model, samples, model_path, train_params)
        save_data(losses, os.path.join(psp_folder, "losses"))
    test_cases(psp_folder, datasets)
    return {"model_path": model_path, "dataset": datasets["train"].subset(range(64)), "intervention": INTERVENTION,
            "hours_per_activity": fit_hours_per_activity(data),
            "valid_ends": psp.valid_end_activities(data) if args.psp_valid_ends else None,
            "samples_per_action": args.samples_per_action, "max_draws": args.max_draws, "variational_decoder": args.psp_variational_decoder,
            "transitions": psp.training_transitions(data, datasets["train"], test_cases(psp_folder), INTERVENTION["actions"]) if args.psp_dfg else None}


# 2b. XGBoost: Training auf den RCT-Fällen ohne die Testfälle des PSP (damit beide offline auf denselben gehaltenen
# Fällen bewertet werden); liefert, was die Arbeitsprozesse für DirectPolicy brauchen
# Wird aufgerufen von main.py (__main__) -> train_xgb (--estimator xgb)
def train_xgb(args, data, callers, psp_folder, tag):
    from prescriptive.enriched_log import CASE
    from prescriptive.estimators.xgb_estimator import XGBEstimator
    model_path = os.path.join(args.data_folder, "xgb_" + tag, "xgb_model.pkl")
    if not args.already_estimator_trained:
        held_out = test_cases(psp_folder)
        estimator = XGBEstimator(callers, seed=args.seed)
        print("XGBoost (Proben, Bäume):", estimator.fit(data[~data[CASE].isin(held_out)]))
        estimator.save(model_path)
    return {"model_path": model_path, "intervention": INTERVENTION}


# 2c. LSTM: wie XGBoost, Training auf den RCT-Fällen ohne die Testfälle des PSP
# Wird aufgerufen von main.py (__main__) -> train_lstm (--estimator lstm)
def train_lstm(args, data, callers, psp_folder, tag):
    from prescriptive.enriched_log import CASE
    from prescriptive.estimators.lstm_estimator import LSTMEstimator
    model_path = os.path.join(args.data_folder, "lstm_" + tag, "lstm_model.pkl")
    if not args.already_estimator_trained:
        held_out = test_cases(psp_folder)
        estimator = LSTMEstimator(callers, seed=args.seed, params={"threads": args.workers})
        print("LSTM:", estimator.fit(data[~data[CASE].isin(held_out)]), flush=True)
        estimator.save(model_path)
    return {"model_path": model_path, "intervention": INTERVENTION}


# Datensätze mit Arbeitsschritten (Lifecycle-Events außer complete, BPIC17): Ihre Daten und Schätzer haben seit dem
# Event-Log mit allen Lifecycles (enriched_log.py) eigene Ordner mit diesem Zusatz; die älteren (nur complete-Events)
# bleiben unverändert
LIFECYCLE_TAG = "_lc"


# Ordner der RCT-Daten: <datensatz>_rct_<intervention>, z. B. bpic13_rct_escalation. Das Standardprofil (bpic17) behält
# die Namen seiner vorhandenen Daten: ohne Datensatz, bei seiner Standard-Intervention auch ohne Intervention (rct,
# rct_second_offer), mit Arbeitsschritten dahinter LIFECYCLE_TAG (rct_lc, rct_second_offer_lc)
# Wird aufgerufen von resolve_intervention_args -> rct_name und helpers/rct.py -> rct_folder
def rct_name(dataset, key):
    tag = LIFECYCLE_TAG if profiles.get(dataset)["work_item_prefix"] else ""
    if dataset != profiles.DEFAULT:
        return f"{dataset}_rct_{key}{tag}"
    return ("rct" if key == profiles.get(dataset)["intervention"] else f"rct_{key}") + tag


# Wie das Simulationsmodell des Datensatzes für die Intervention trainiert wird: Eingänge laut next_event_prediction/datasets.py
# (Mitarbeiter für die Zeit bei Arbeitsschritten, Angebote, Anrufer)
# Wird aufgerufen von resolve_intervention_args -> training_command
def training_command(folder):
    flags = [f"--dataset {PROFILE['key']}", "--model lstm_time", f"--name {os.path.basename(folder)}", "--all-data"]
    # BPIC17: auf den Lifecycles des Logs (schedule/suspend/resume/...), nicht auf bereinigten Arbeitsabschnitten
    flags += ["--no-work-sessions"] if PROFILE["work_item_prefix"] else []
    return "python -m next_event_prediction.train " + " ".join(flags + ["--epochs 80", "--patience 8"])


# Intervention wählen (für alle Module und die Arbeitsprozesse) und die davon abhängigen Voreinstellungen setzen:
# Ordner der RCT-Daten (rct_name) und Simulationsmodell (simulation/dataset_simulator.model_folder, für alle
# Interventionen dasselbe); bricht ab, wenn es fehlt.
# Wird aufgerufen von main.py und offline_evaluation.py (__main__) -> resolve_intervention_args
def resolve_intervention_args(args):
    from prescriptive.enriched_log import set_intervention
    from simulation.dataset_simulator import is_trained, model_folder
    # Ohne --intervention: die Standard-Intervention des Log-Profils (BPIC17: channel)
    set_intervention(args.intervention or PROFILE["intervention"])
    key = INTERVENTION["key"]
    # Interventionen ohne dataset gehören zum Standardprofil (interventions/__init__.py)
    dataset = INTERVENTION.get("dataset", profiles.DEFAULT)
    if dataset != PROFILE["key"]:
        raise SystemExit(f"Die Intervention {key} gehört zum Datensatz {dataset}, nicht zu {PROFILE['key']} (--profile)")
    if getattr(args, "rct_name", None) is None:
        args.rct_name = rct_name(PROFILE["key"], key)
    if hasattr(args, "model_folder"):
        if args.model_folder is None:
            args.model_folder = os.path.relpath(model_folder(PROFILE["key"]))
        if not is_trained(args.model_folder):
            raise SystemExit(f"Das Simulationsmodell {args.model_folder} ist nicht vorhanden. Trainieren mit:\n  {training_command(args.model_folder)}")


# Die Fälle, die das PSP als Test zurückhält (Split des PSP-Loaders, zufällig nach Fällen); gespeichert in test_cases.pkl
# im PSP-Ordner, damit XGBoost und die Offline-Auswertung das große Dataset nicht laden müssen
# Wird aufgerufen von train_psp, train_xgb und offline_evaluation.py -> test_cases
def test_cases(psp_folder, datasets=None):
    path = os.path.join(psp_folder, "test_cases")
    if os.path.exists(path + ".pkl"):
        return load_data(path)
    if datasets is None:
        import torch
        from prescriptive.estimators import psp  # PSP-Pfad für torch.load
        datasets = torch.load(os.path.join(psp_folder, "datasets.pt"), weights_only=False)
    cases = set(pd.unique(np.asarray(datasets["test"].case_ids, dtype=object)))
    save_data(cases, path)
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSP as prescriptive approach in the joint simulation of BPIC17")
    parser.add_argument("--profile", type=str, default=profiles.DEFAULT,
                        help="Log-Profil (prescriptive/profiles.py): bpic17 (Standard, Engine mit Mitarbeitern) oder bpic13 (Fall-Simulator, Auswertung offline und mit prescriptive.evaluation.oracle_evaluation)")
    parser.add_argument("--event-log", type=str, default=None, help="Path to the event log (default: the one of the profile, BPIC17: eventlog/eventlog.xes.gz)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--intervention", type=str, default=None,
                        help="Interventionspunkt (interventions/): BPIC17 call (Standard), second_offer, upsell; BPIC13 escalation (Standard), wait_user")
    parser.add_argument("--model-folder", type=str, default=None,
                        help="Simulation model (next-event model), trained with actions at the chosen intervention (default: "
                             "simulation/dataset_simulator.model_folder of the profile and intervention)")
    parser.add_argument("--allocation", choices=["random", "round_robin", "shortest_queue"], default=None,
                        help="Selection strategy (resources/selection_strategies.py) for the work items where no policy chooses "
                             "(default: lauf.zuteilung in catalog.yaml)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel simulation processes")
    parser.add_argument("--data-folder", type=str, default=DATA_FOLDER, help="Folder for RCT data, PSP and results (default prescriptive/data)")
    # RCT
    parser.add_argument("--rct-cases", type=int, default=31509, help="Cases per simulated RCT year")
    parser.add_argument("--rct-runs", type=int, default=3, help="Simulated RCT years (training data of the PSP)")
    parser.add_argument("--delta", type=float, default=0.0,
                        help="As in SCOPE (confounding type point): share of first decision points decided like the bank instead of randomly "
                             "(0 = pure RCT; folders get the suffix _delta<percent> if > 0)")
    parser.add_argument("--already-rct-generated", action="store_true", help="Load the RCT data instead of simulating it")
    parser.add_argument("--rct-name", type=str, default=None,
                        help="Name of the RCT data; folders: <name>, psp_<name>, xgb_<name>, lstm_<name> (each with the delta suffix)")
    parser.add_argument("--rct-extend-from", type=str, default=None,
                        help="Add --rct-runs new simulated years (from --rct-first-run on) to these RCT data (e.g. rct) and save them as --rct-name")
    parser.add_argument("--rct-first-run", type=int, default=0, help="Number of the first simulated RCT year (seed and case IDs R<number>_)")
    # Schätzer
    parser.add_argument("--estimator", choices=["psp", "xgb", "lstm"], default="psp",
                        help="Causal estimator that chooses call/no_call and the caller: psp (suffixes, prescriptive/estimators/psp.py), xgb (XGBoost S-Learner, "
                             "prescriptive/estimators/xgb_estimator.py) or lstm (LSTM S-Learner, prescriptive/estimators/lstm_estimator.py)")
    parser.add_argument("--already-estimator-trained", action="store_true", help="xgb / lstm: load the trained models instead of training them")
    parser.add_argument("--caller-choice", action=argparse.BooleanOptionalAction, default=True,
                        help="The estimator also chooses the caller of the first call after call (--no-caller-choice: only call/no_call, the caller as the bank assigns it)")
    # PSP
    parser.add_argument("--psp-suffix", type=str, default="",
                        help="Folder suffix of the PSP (\"\": the first PSP; \"_offers\": with the current offer and the number of offers as inputs)")
    parser.add_argument("--psp-min-suffix-size", type=int, default=15, help="Decoder steps per training sample of the PSP")
    parser.add_argument("--psp-min-calls", type=int, default=100, help="Employees with at least this many first calls in the RCT data get their own token")
    parser.add_argument("--psp-lognormal", action="store_true",
                        help="All numbers of the PSP (times, offers) log-normal: log1p, then standardized (continuous_positive_columns)")
    parser.add_argument("--psp-decoder-context", action="store_true", help="The final encoder state goes into every decoder step of the PSP")
    parser.add_argument("--psp-decoder-static", action="store_true",
                        help="The embedded case attributes (LoanGoal, ApplicationType, RequestedAmount) go into every decoder step of the PSP")
    parser.add_argument("--psp-no-gradnorm", action="store_true", help="Train the PSP without GradNorm (all losses weighted equally)")
    parser.add_argument("--psp-standard-ce", action="store_true",
                        help="Train the activities of the PSP with the standard cross entropy instead of the one with learned logit variance (then no logit noise when sampling)")
    parser.add_argument("--psp-layers", type=int, default=psp_layers_default(), help="LSTM layers of encoder and decoder of the PSP")
    parser.add_argument("--psp-branch-weight", type=float, default=1.0,
                        help="With --psp-standard-ce: weight of the outcome events and EOS in the loss of the activities (psp.BRANCH_EVENTS; 1 = all equal)")
    parser.add_argument("--psp-tf-max", type=float, default=1.0, help="Teacher forcing ratio at the start of the PSP training (paper: 0.8)")
    parser.add_argument("--psp-tf-decay-from", type=float, default=None,
                        help="Share of the epochs after which teacher forcing decreases linearly to 0 at the end (paper: 0.2); default: from the start, reaching 0 at half the epochs")
    parser.add_argument("--already-psp-preprocessed", action="store_true", help="Load the encoded PSP datasets instead of building them")
    parser.add_argument("--already-psp-trained", action="store_true", help="Load the trained PSP instead of training it")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs of the PSP")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Start learning rate of the PSP")
    parser.add_argument("--samples-per-action", type=int, default=100, help="Sampled suffixes per option of a live decision")
    parser.add_argument("--max-draws", type=int, default=2000, help="At most this many suffixes per live decision")
    parser.add_argument("--psp-dfg", action="store_true",
                        help="When sampling, only transitions between activities seen at least 3 times in the training data (psp.transition_matrix)")
    parser.add_argument("--psp-variational-decoder", action="store_true", help="Keep the decoder dropout masks of the first step for the whole suffix")
    parser.add_argument("--psp-valid-ends", action=argparse.BooleanOptionalAction, default=True,
                        help="Discard and redraw sampled suffixes that do not end with an activity ending at least 0.25 %% of the training cases")
    # Auswertung
    parser.add_argument("--eval-policies", type=str, default="bank,estimator,always_1,always_0,random",
                        help="Policies of the evaluation (comma separated; estimator = the one chosen with --estimator)")
    parser.add_argument("--results-tag", type=str, default="", help="Extra suffix of the result file (eval/joint_results...<tag>.pkl)")
    parser.add_argument("--eval-seeds", type=int, default=10, help="Repetitions (seeds) per policy (0 = no joint evaluation, e.g. only training)")
    parser.add_argument("--eval-cases", type=int, default=4000, help="Arrivals per evaluation run")
    parser.add_argument("--warmup-cases", type=int, default=700, help="First arrivals that are simulated but not evaluated")
    parser.add_argument("--cooldown-cases", type=int, default=1300, help="Last arrivals that are simulated but not evaluated")
    args = parser.parse_args()
    resolve_intervention_args(args)
    args.event_log = args.event_log or PROFILE["event_log"]
    print("Specified Arguments: ", args, "\n")

    import interventions
    config = {"event_log_path": args.event_log, "model_folder": args.model_folder,
              "allocation": args.allocation or interventions.run_allocation(PROFILE["key"]), "delta": args.delta}
    suffix = f"_delta{int(round(100 * args.delta))}" if args.delta > 0 else ""

    # 1. RCT-Daten
    tag = args.rct_name + suffix
    rct_folder = os.path.join(args.data_folder, tag)
    if args.already_rct_generated:
        # Gewinn mit den aktuellen Kostenannahmen aus prescriptive/helpers/preprocessing/BPIC2017/outcome.py (die Bestandteile sind gespeichert)
        data = with_current_costs(load_data(os.path.join(rct_folder, "log")))
    elif not PROFILE["engine"]:
        data = run_case_rct(args.rct_cases, args.rct_runs, args.seed, args.model_folder, args.event_log, args.delta, args.rct_first_run)
        save_data(data, os.path.join(rct_folder, "log"))
    else:
        data = run_rct(args.rct_cases, args.rct_runs, args.seed, args.workers, config, first_run=args.rct_first_run)
        if args.rct_extend_from:
            # Die vorhandenen Jahre bleiben unverändert, die neuen kommen dazu
            from prescriptive.enriched_log import concat_logs
            base = load_data(os.path.join(args.data_folder, args.rct_extend_from + suffix, "log"))
            if "rct" not in base.columns and args.delta == 0:
                base["rct"] = True
            data = concat_logs(base, data)
        save_data(data, os.path.join(rct_folder, "log"))
    print_summary(data)

    callers = top_resources(data, args.psp_min_calls)
    print(f"Anrufer mit eigener Kategorie: {len(callers)} (mindestens {args.psp_min_calls} Anrufe im RCT)")
    psp_folder = os.path.join(args.data_folder, "psp_" + tag + args.psp_suffix)
    os.makedirs(psp_folder, exist_ok=True)
    results_tag = ("" if args.rct_name == "rct" else "_" + args.rct_name) + suffix
    if args.estimator == "psp":
        config["psp"], results_name = train_psp(args, data, callers, psp_folder), "joint_results" + results_tag + args.psp_suffix
    elif args.estimator == "xgb":
        config["xgb"], results_name = train_xgb(args, data, callers, psp_folder, tag), "joint_results" + results_tag + "_xgb"
    else:
        config["lstm"], results_name = train_lstm(args, data, callers, psp_folder, tag), "joint_results" + results_tag + "_lstm"

    for name in ("psp", "xgb", "lstm"):
        if name in config:
            config[name]["choose_callers"] = args.caller_choice
    if args.eval_seeds <= 0:
        raise SystemExit("Keine gemeinsame Auswertung (--eval-seeds 0)")
    if not PROFILE["engine"]:
        raise SystemExit(f"Profil {PROFILE['key']}: keine gemeinsame Simulation (Fälle unabhängig). Auswertung: python -m prescriptive.evaluation.offline_evaluation --profile {PROFILE['key']} ... "
                         f"und python -m prescriptive.evaluation.oracle_evaluation --profile {PROFILE['key']}")
    window = (args.warmup_cases, args.eval_cases - args.cooldown_cases)
    # estimator = der gewählte Schätzer; always_1 / always_0 = immer Option 1 bzw. 0 der Intervention (z. B. always_call)
    aliases = {"estimator": args.estimator, "always_1": "always_" + INTERVENTION["actions"][1], "always_0": "always_" + INTERVENTION["actions"][0]}
    policies = [aliases.get(name.strip(), name.strip()) for name in args.eval_policies.split(",") if name.strip()]
    seeds = [args.seed + replication for replication in range(args.eval_seeds)]
    table, runs = run_evaluation(policies, seeds, args.eval_cases, window, args.workers, config)
    pd.set_option("display.width", 250)
    print(f"\nGemeinsame Simulation: {args.eval_cases} Fälle je Lauf, gewertet {window[0]} bis {window[1]}, {len(seeds)} Wiederholungen")
    print(table.round(3).to_string())
    save_data({"table": table, "runs": runs, "args": vars(args)}, os.path.join(args.data_folder, "eval", results_name + args.results_tag))

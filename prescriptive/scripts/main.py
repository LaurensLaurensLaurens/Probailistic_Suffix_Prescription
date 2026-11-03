"""
prescriptive_guide.ipynb als Skript für die Kommandozeile (wie SCOPE_WORKSPACE/scripts/main.py): dieselben Schritte in
derselben Reihenfolge, die Parameter der Zellen als Optionen (Standardwerte wie im Notebook). Ein eigener Prozess je
Lauf: was ein Schritt nicht mehr braucht, wird freigegeben, und nach dem Lauf ist der ganze Speicher wieder frei.
Alle Ergebnisse kommen nach --out (Standard prescriptive/scripts/results/), je Schritt in einen festen Ordner; ein neuer
Lauf überschreibt die Dateien der Schritte, die er ausführt:
    daten/                                    Katalog der Interventionen, RCT-Daten, Training, zufällige Fälle
    simulator/                                Simulator gegen echt (check), KPIs mit Diagrammen in kpis_diagramme/ (kpis)
    causal_estimator_eval/<Modell>/           wie gut das Modell (XGBoost, LSTM) das Outcome je Wahlpunkt trifft, mit Kalibrierung und Streuung (models)
    prescription/<Ansatz>/                    Online-Auswertung gegen die Bank (online)
    prescription/screening/                   control_flow/, variables/, details/ (screening)
    lagebild/                                 Lagebild des PSP (lagebild)
Je Tabelle eine CSV und ein Bild, dazu Diagramme und bericht.html mit allem dieses Laufs (helpers/report.py); die
Tabellen auch auf die Konsole. Daten und Modelle (RCT-Daten, Schätzer, Zwischenspeicher) bleiben in prescriptive/data.

Schritte (--steps, Standard alle, immer in dieser Reihenfolge):
    screening   Beispielfall eines Kandidaten, Screening der Control-Flow- und der Variablen-Kandidaten
    check       Simulator gegen den echten Log (wie die Bank)
    kpis        Evaluation des Simulators (KPIs; Tabellen und Diagramme in <out>/<datensatz>_simulator_kpis/)
    cases       zufälliger echter Fall und zufälliger Fall der Bank-Policy
    rct         RCT-Daten (vorhanden: geladen)
    train       Training der Modelle (helpers/training.py, gespeichert in prescriptive/models/; lädt oder erzeugt dazu die RCT-Daten)
    models      Kennzahlen je Modell und Wahlpunkt (helpers/training.py evaluate_estimators): Vorhersage des Outcomes auf den
                Testfällen des RCT und dem echten Log, am Control-Flow-Wahlpunkt auf den randomisierten Testfällen auch der Effekt
    online      Online-Auswertung auf dem Simulator: der Simulator gibt die Umgebung, die Policy (bank, random, xgb, lstm, psp,
                psp_lookahead, psp_erm) wählt live an den Wahlpunkten (alle oder --online-points), gegen die Bank
    lagebild    Lagebild des PSP auf dem Simulator

Optionen je Datensatz als datensatz=wert, z. B. --outcome bpic17=gewinn --eval-point "bpic17=Mitarbeiter: O_Accepted".

Aufruf mit der Umgebung des Simulators (wie der Notebook-Kernel), im Hintergrund und unabhängig von der Sitzung:
    cd process-simulation-engine
    setsid nohup .venv/bin/python -u ../prescriptive/scripts/main.py --steps train > ../prescriptive/data/train.log 2>&1 &
    .venv/bin/python -u ../prescriptive/scripts/main.py --steps rct train models --delta 0.5 --rct-cases 10000
"""
import argparse
import gc
import os
import sys
import time

# Diagramme nur als Dateien (kein Fenster)
os.environ.setdefault("MPLBACKEND", "Agg")

PRESCRIPTIVE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PRESCRIPTIVE)

import pandas as pd

from helpers.datasets import DATASETS, add_paths, dataset_info
from helpers.report import calibration_chart, check_chart, portfolio_chart, scatter_chart, screening_chart, uplift_chart

# Ordnernamen je Art des Schätzers bzw. Ansatzes; Vergleichs-Policies, die in jedem Ordner eines Ansatzes mitstehen
APPROACHES = {"xgb": "XGBoost", "lstm": "LSTM", "psp": "PSP", "psp_lookahead": "PSP_Vorausschau", "psp_erm": "PSP_ERM"}
ONLINE_BASELINES = ("bank", "random")


# Modell (Ordnername, z. B. xgb_rct_lc_delta50_30k) -> Ordner prescription/<Ansatz>; mehrere Modelle derselben Art
# bekommen je einen eigenen Ordner <Ansatz>_<Modell>
def approach_folders(models):
    kinds = [str(model).split("_")[0] for model in models]
    return {model: "prescription/" + (APPROACHES.get(kind, kind) if kinds.count(kind) == 1 else f"{APPROACHES.get(kind, kind)}_{model}")
            for model, kind in zip(models, kinds)}


# Die Interventionspunkte der Mitarbeiterwahl dieses Laufs: --eval-point, sonst lauf.mitarbeiterwahl im Katalog, sonst
# der erste Mitarbeiter-Kandidat mit Modellen
def choice_points(args, dataset, points):
    if args.eval_point.get(dataset):
        return [args.eval_point[dataset]]
    chosen = (dataset_info(dataset).get("lauf") or {}).get("mitarbeiterwahl") or []
    unknown = [point for point in chosen if point not in points.index]
    if unknown:
        raise ValueError(f"lauf.mitarbeiterwahl im Katalog: unbekannte Interventionspunkte {', '.join(unknown)}; möglich: {', '.join(points.index)}")
    return list(chosen) or [(list(points.index[points["Modelle"] > 0]) + list(points.index))[0]]


# Ordner eines Ansatzes an einem Interventionspunkt: prescription/<Ansatz>, bei mehreren Punkten je Punkt ein Unterordner
def point_folder(folder, point, points):
    from helpers.report import slug
    return folder if len(points) == 1 else f"{folder}/{slug(point)}"


# Spalten der Screening-Tabellen, die im Bild fehlen (in der CSV stehen sie)
SCREENING_HIDDEN = ("Ergebnis", "Optionen")

STEPS = ("screening", "check", "kpis", "cases", "rct", "train", "models", "online", "lagebild")


# ------------------------------------------------------------------ Optionen

def per_dataset_values(pairs):
    """["bpic17=wert", ...] -> {"bpic17": "wert"}; der Wert darf Leerzeichen und weitere = enthalten."""
    values = {}
    for pair in pairs or []:
        dataset, sep, value = pair.partition("=")
        if not sep or dataset not in DATASETS:
            raise argparse.ArgumentTypeError(f"'{pair}': erwartet datensatz=wert mit datensatz aus {', '.join(DATASETS)}")
        values[dataset] = value
    return values


def optional_int(text):
    return None if text.lower() in ("none", "") else int(text)


# Einstellungen des PSP-Trainings aus --psp-epochs, --psp-train-cases, --psp-decoder-steps, --psp-learning-rate und --psp-point-weight
# (ohne Angabe: wie im Original)
def psp_params(args):
    return {"train_cases": args.psp_train_cases, **({"epochs": args.psp_epochs} if args.psp_epochs else {}),
            **({"suffix_size": args.psp_decoder_steps} if args.psp_decoder_steps else {}),
            **({"learning_rate": args.psp_learning_rate} if args.psp_learning_rate else {}),
            **({"point_weight": args.psp_point_weight} if args.psp_point_weight else {})}


def parse():
    parser = argparse.ArgumentParser(description="prescriptive_guide.ipynb als Skript", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["bpic17"], help="Datensätze (Zelle 1)")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS), help="Schritte (siehe oben)")
    parser.add_argument("--out", default=None, help="Ordner für die Ergebnisse (Standard prescriptive/scripts/results)")

    group = parser.add_argument_group("Preprocessing")
    group.add_argument("--outcome", nargs="*", default=[], metavar="DATENSATZ=OUTCOME",
                       help="je Datensatz eine der outcome_optionen des Katalogs statt des Standards (z. B. bpic17=gewinn); gilt für die "
                            "ganze Pipeline, XGBoost und LSTM in eigenen Ordnern mit _<Option>")

    group = parser.add_argument_group("Screening")
    group.add_argument("--intervention", nargs="*", default=["bpic17=Mitarbeiter: O_Accepted"], metavar="DATENSATZ=KANDIDAT",
                       help="Kandidat des Beispielfalls; fehlt er: die Standard-Intervention")
    group.add_argument("--example-seed", type=optional_int, default=None, help="None: jedes Mal ein anderer Fall")
    group.add_argument("--pool", default="kompetent", help='Pool der Mitarbeiter-Kandidaten: "kompetent", "im Dienst", "frei" oder "none"')
    group.add_argument("--min-overlap", type=float, default=0.5)
    group.add_argument("--min-cases-resources", type=int, default=1, help="eigene Schätzung für jeden Mitarbeiter mit so vielen Fällen; übrige zusammen als other")

    group = parser.add_argument_group("Simulator")
    group.add_argument("--n-cases", type=int, default=10000, help="simulierte Fälle je Lauf (Check, Evaluation, Bank-Fall)")
    group.add_argument("--seed", type=int, default=0, help="Zufall der Simulation")
    group.add_argument("--hourly-rate", type=float, default=50.0, help="€ je Arbeitsstunde")
    group.add_argument("--daily-cost", type=float, default=10.0, help="€ je Tag Durchlaufzeit")
    group.add_argument("--sim-intervention", nargs="*", default=[], metavar="DATENSATZ=INTERVENTION",
                       help="je Datensatz eine andere Intervention als die Standard-Intervention, z. B. bpic13=wait_user")
    group.add_argument("--retrain-simulator", action="store_true", help="das Next-Event-Modell neu trainieren")
    group.add_argument("--sim-width", type=int, default=128)
    group.add_argument("--sim-depth", type=int, default=2)
    group.add_argument("--sim-dropout", type=float, default=0.1)
    group.add_argument("--sim-epochs", type=int, default=80)
    group.add_argument("--sim-patience", type=int, default=8)
    group.add_argument("--sim-seed", type=int, default=42)
    group.add_argument("--sim-threads", type=int, default=8)

    group = parser.add_argument_group("Zufällige Fälle")
    group.add_argument("--case-seed", type=optional_int, default=None, help="None: jedes Mal ein anderer Fall")
    group.add_argument("--case-ids", nargs="*", default=[], metavar="DATENSATZ=FALL", help="bestimmter echter Fall")
    group.add_argument("--bank-case-seed", type=optional_int, default=None)
    group.add_argument("--bank-case-ids", nargs="*", default=[], metavar="DATENSATZ=FALL", help="bestimmter simulierter Fall")

    group = parser.add_argument_group("RCT-Daten")
    group.add_argument("--rct-cases", type=optional_int, default=None, help="Fälle je Lauf; None: so viele wie der echte Log")
    group.add_argument("--rct-runs", type=int, default=3, help="Läufe (simulierte Jahre)")
    group.add_argument("--rct-seed", type=int, default=42, help="Lauf r: RCT_SEED + 1000 + r")
    group.add_argument("--delta", type=float, default=0.0, help="Anteil der ersten Entscheidungspunkte wie die Bank statt zufällig")
    group.add_argument("--workers", type=int, default=8, help="parallele Simulationsprozesse; auch Threads des PSP-Trainings")
    group.add_argument("--regenerate-rct", action="store_true", help="neu simulieren und vorhandene RCT-Daten überschreiben")

    group = parser.add_argument_group("Training")
    group.add_argument("--estimators", nargs="+", choices=("xgb", "lstm", "psp"), default=["xgb", "lstm", "psp"], help="Modelle (estimators/, gespeichert in models/)")
    group.add_argument("--psp-epochs", type=int, default=None, help="Trainingsepochen des PSP (Standard wie im Original: 100)")
    group.add_argument("--psp-train-cases", type=optional_int, default=1000, help="Trainingsfälle des PSP (None: alle)")
    group.add_argument("--psp-decoder-steps", type=int, default=None, help="Schritte des PSP-Decoders im Training (Standard wie im Original: 5)")
    group.add_argument("--psp-learning-rate", type=float, default=None, help="Lernrate des PSP (Standard wie im Original: 1e-5)")
    group.add_argument("--psp-point-weight", type=int, default=None,
                       help="PSP: Proben, deren Prefix am Options-Event eines Wahlpunkts endet, so oft im Training (Standard wie im Original: 1)")
    group.add_argument("--psp-suffix", default="", help="Lagebild: Zusatz des PSP-Ordners")
    group.add_argument("--train-seed", type=int, default=42, help="Zufall des Trainings")
    group.add_argument("--retrain-estimators", action="store_true", help="neu trainieren, auch wenn ein Modell auf denselben Daten vorhanden ist")
    group.add_argument("--train-cases", type=optional_int, default=None, help="nur so viele zufällig gezogene Fälle der RCT-Daten (mit --train-seed)")

    group = parser.add_argument_group("Interventionspunkte")
    group.add_argument("--eval-point", nargs="*", default=[], metavar="DATENSATZ=PUNKT",
                       help="Interventionspunkt statt lauf.mitarbeiterwahl im Katalog (prescriptive/catalog.yaml)")

    group = parser.add_argument_group("Kennzahlen je Modell (models)")
    group.add_argument("--kpi-real-log", action=argparse.BooleanOptionalAction, default=True, help="auch auf dem echten Log bewerten")

    group = parser.add_argument_group("Online-Auswertung")
    group.add_argument("--online-policies", nargs="+", choices=("bank", "random", "xgb", "lstm", "psp", "psp_lookahead", "psp_erm"), default=["bank", "random", "xgb", "lstm"],
                       help="Policies (evaluation/online_evaluation.py); die Bank immer als Vergleich")
    group.add_argument("--online-points", nargs="*", default=[], metavar="DATENSATZ=WAHLPUNKT",
                       help='nur an diesen Wahlpunkten wählt die Policy, mehrfach möglich, z. B. bpic17=channel; Standard: alle des Simulators')
    group.add_argument("--online-cases", type=int, default=4000, help="Ankünfte je Lauf")
    group.add_argument("--online-warmup", type=int, default=700, help="erste Ankünfte, die mitlaufen, aber nicht gewertet werden")
    group.add_argument("--online-cooldown", type=int, default=1300, help="letzte Ankünfte, die mitlaufen, aber nicht gewertet werden")
    group.add_argument("--online-seeds", type=int, default=5, help="Wiederholungen je Policy (seed, seed + 1, ...)")
    group.add_argument("--online-workers", type=int, default=8)
    group.add_argument("--erm-gamma", type=float, default=None,
                       help="psp_erm: Risikoaversion γ in 1/Tag (Standard policies/psp_erm_policy.py GAMMA)")
    group.add_argument("--xgb-model", default=None, help="xgb: dieses XGBoost-Modell statt models/xgb_model_<datensatz><outcome>.pkl")

    group = parser.add_argument_group("Lagebild")
    group.add_argument("--overview-model", nargs="*", default=[], metavar="DATENSATZ=ORDNER",
                       help="PSP-Modell je Datensatz; Standard: das PSP aus dem Training (psp_<rct-Ordner>[_<train-cases>]<psp-suffix>)")
    group.add_argument("--overview-cases", type=int, default=2000)
    group.add_argument("--overview-every-hours", type=float, default=24)
    group.add_argument("--overview-samples", type=int, default=20)
    group.add_argument("--overview-workers", type=int, default=8)
    group.add_argument("--overview-seed", type=int, default=None, help="Standard: --seed")

    args = parser.parse_args()
    for name in ("outcome", "intervention", "sim_intervention", "case_ids", "bank_case_ids", "eval_point", "overview_model"):
        setattr(args, name, per_dataset_values(getattr(args, name)))
    # Das gewählte Outcome gilt in der ganzen Pipeline (Profil, angereicherte Logs, RCT-Daten, Training, Auswertung) und in
    # den Arbeitsprozessen; vor dem ersten Import von enriched_log (prescriptive/profiles.py set_outcome)
    add_paths()
    from prescriptive import profiles
    for dataset in args.datasets:
        profiles.set_outcome(dataset, args.outcome.get(dataset))
    args.pool = None if args.pool.lower() == "none" else args.pool
    # --online-points: je Datensatz eine Liste (bpic17=channel "bpic17=Mitarbeiter: W_Call incomplete files")
    online_points = {}
    for pair in args.online_points:
        dataset, _, point = pair.partition("=")
        if dataset not in DATASETS or not point:
            raise argparse.ArgumentTypeError(f"'{pair}': erwartet datensatz=wahlpunkt mit datensatz aus {', '.join(DATASETS)}")
        online_points.setdefault(dataset, []).append(point)
    args.online_points = online_points
    args.overview_seed = args.seed if args.overview_seed is None else args.overview_seed
    args.costs = {"hourly_rate": args.hourly_rate, "daily_cost": args.daily_cost}
    args.out = os.path.abspath(args.out or os.path.join(PRESCRIPTIVE, "scripts", "results"))
    return args


# ------------------------------------------------------------------ Ausgabe

# Der Bericht des Laufs (helpers/report.py), angelegt in main
REPORT = None


# Statt display im Notebook: Überschrift und Tabelle auf die Konsole, die Tabelle als CSV und Bild nach out und in den Bericht
# (sub: Unterordner von out; hide: Spalten, die im Bild und im Bericht fehlen)
def report(args, dataset, name, table, title=None, save=True, sub=None, hide=(), transpose=None):
    print(f"\n==== {dataset}: {title or name}", flush=True)
    if table is None:
        return
    if isinstance(table, pd.Series):
        table = table.to_frame()
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 250, "display.max_colwidth", 80):
        print(table.to_string(), flush=True)
    if save:
        REPORT.table(dataset, name, table, title, sub=sub, hide=hide, transpose=transpose)


# Ein Diagramm (helpers/report.py) nach out und in den Bericht
def chart(dataset, name, figure, title, sub=None):
    REPORT.figure(dataset, name, figure, title, sub=sub)


def heading(text):
    print(f"\n######## {text} ({time.strftime('%H:%M:%S')})", flush=True)
    if REPORT is not None:
        REPORT.heading(text)


# ------------------------------------------------------------------ Lauf

def main():
    global REPORT
    from helpers.report import Report
    args = parse()
    steps = set(args.steps)
    from prescriptive import profiles
    from helpers.report import set_outcome
    # Beschriftungen nach dem Outcome des (ersten) Datensatzes, z. B. −Durchlaufzeit in Tagen statt Gewinn in €
    outcome = profiles.get(args.datasets[0])
    set_outcome(outcome["outcome"])
    REPORT = Report(args.out, f"Lauf vom {time.strftime('%Y-%m-%d %H:%M')}: {', '.join(s for s in STEPS if s in steps)}; Outcome: "
                              f"{outcome['outcome_option'] or 'Standard'} ({outcome['outcome']})")
    print("Optionen:", vars(args), flush=True)
    add_paths()
    if len(args.datasets) > 1 and steps & {"train", "online", "lagebild"}:
        print("Hinweis: die Schätzer legen ihr Profil beim Import fest; mehrere Datensätze mit Training oder Auswertung besser je "
              "Datensatz in einem eigenen Aufruf.", flush=True)

    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    from helpers.example_case import example_case
    from helpers.process_model import process_model

    # ---- Datensätze laden und Preprocessing (Zellen 1 und 2); das Outcome gleich mit den gewählten Kosten
    # GEPRÜFT
    heading("Datensätze und Preprocessing")
    loaded, logs = {}, {}
    for dataset in args.datasets:
        raw, info, catalog = load_dataset(dataset)
        report(args, dataset, "interventionen", catalog, f"{info.get('prozess', dataset)}: Katalog der Interventionen", sub="daten")
        loaded[dataset] = raw
        logs[dataset] = preprocess(dataset, raw, costs=args.costs, outcome=args.outcome.get(dataset))

    # ---- Prozessmodell und Beispielfall je Datensatz, falls noch nicht vorhanden (process_models/, example_cases/)
    heading("Prozessmodelle und Beispielfälle")
    for dataset in args.datasets:
        process_model(dataset, logs[dataset])
        example_case(dataset, logs[dataset])

    # ---- Screening (Zellen 3 bis 5)
    # GEPRÜFT
    if "screening" in steps:
        from helpers.screening import candidate_example, screen_control_flow, screen_variables
        heading("Screening")
        for dataset in args.datasets:
            # Hier wird ein Beispielkandidat genommen, wenn dieser nicht vorhanden ist, dann ein Standard kandidat
            example = candidate_example(dataset, logs[dataset], args.intervention.get(dataset), seed=args.example_seed, raw_log=loaded[dataset])
            report(args, dataset, "beispiel_angaben", example["summary"], "Beispielfall des Kandidaten", sub="prescription/screening/details")
            report(args, dataset, "beispiel_merkmale", example["features"], "Merkmale X (wie sie in das Propensity-Modell gehen)", sub="prescription/screening/details")
            if example.get("original") is not None:
                report(args, dataset, "beispiel_original", example["original"], "Original (Roh-Log)", sub="prescription/screening/details")
            report(args, dataset, "beispiel_events", example["events"], "Nach dem Preprocessing (Teil: Prefix = bekannt am Entscheidungspunkt)", sub="prescription/screening/details")

            # Hier das Variablen und Control flow screnening
            # Je Art ein Unterordner mit Diagramm und Tabelle (ohne Ergebnis und Optionen; die CSV hat alle Spalten)
            control_flow = screen_control_flow(dataset, logs[dataset])
            chart(dataset, "screening_control_flow_diagramm", screening_chart(control_flow, "Screening: Control-Flow-Kandidaten"), "Control-Flow-Kandidaten",
                  sub="prescription/screening/control_flow")
            report(args, dataset, "screening_control_flow", control_flow.round(3), "Control-Flow-Kandidaten", sub="prescription/screening/control_flow", hide=SCREENING_HIDDEN, transpose=False)
            variables = screen_variables(dataset, logs[dataset], pool=args.pool, min_cases=args.min_cases_resources, min_overlap=args.min_overlap)
            chart(dataset, "screening_variablen_diagramm", screening_chart(variables, "Screening: Variablen-Kandidaten"), "Variablen-Kandidaten",
                  sub="prescription/screening/variables")
            report(args, dataset, "screening_variablen", variables.round(3), "Variablen-Kandidaten", sub="prescription/screening/variables", hide=SCREENING_HIDDEN, transpose=False)
            for candidate, details in variables.attrs.get("details", {}).items():
                report(args, dataset, "screening_" + candidate, details.round(4), f"{candidate}: Mitarbeiter (Fälle und Propensity)", sub="prescription/screening/details")

    # ---- Simulator aufstellen (Zelle 6); der Log mit den Lifecycles, auf denen sein Modell trainiert ist
    simulators, simulator_logs = {}, {}
    if steps & {"check", "kpis", "cases", "rct", "train", "models", "online", "lagebild"}:
        from simulation.dataset_simulator import Simulator
        heading("Simulator")
        training = {"width": args.sim_width, "depth": args.sim_depth, "dropout": args.sim_dropout, "epochs": args.sim_epochs,
                    "patience": args.sim_patience, "seed": args.sim_seed, "threads": args.sim_threads}
        for dataset in args.datasets:
            simulator_logs[dataset] = logs[dataset]
            simulators[dataset] = Simulator(dataset, logs[dataset], args.sim_intervention.get(dataset), retrain=args.retrain_simulator, **training)
            print(simulators[dataset], flush=True)
    # Die Roh-Logs braucht danach niemand mehr
    del loaded
    gc.collect()

    # ---- Simulator gegen echt, wie die Bank (Zelle 6)
    if "check" in steps:
        from helpers.simulator_check import check_simulator
        heading("Simulator gegen echt")
        for dataset in args.datasets:
            check = check_simulator(simulators[dataset], simulator_logs[dataset], n_cases=args.n_cases, seed=args.seed, costs=args.costs)
            chart(dataset, "simulator_check_diagramm", check_chart(check), "Simulator gegen echt: Abweichung je Kennzahl", sub="simulator")
            report(args, dataset, "simulator_check", check, "Simuliert gegen echt: wie die Bank", sub="simulator")

    # ---- Evaluation des Simulators: KPIs (Zelle 7)
    if "kpis" in steps:
        from helpers.simulator_check import evaluate_simulator
        heading("Evaluation des Simulators")
        for dataset in args.datasets:
            folder = os.path.join(args.out, "simulator", "kpis_diagramme")
            table = evaluate_simulator(simulators[dataset], simulator_logs[dataset], n_cases=args.n_cases, seed=args.seed, costs=args.costs,
                                       folder=folder, show=False)
            report(args, dataset, "simulator_kpis", table.round(2), f"KPIs (Tabellen und Diagramme in {folder})", sub="simulator")
            REPORT.folder(dataset, folder, "Diagramme der Simulator-KPIs")
    simulator_logs = None
    gc.collect()

    # ---- Zufälliger echter Fall und Fall der Bank-Policy (Zellen 8 und 9)
    if "cases" in steps:
        from helpers.cases import random_case, simulated_log
        heading("Zufällige Fälle")
        for dataset in args.datasets:
            intervention = simulators[dataset].intervention
            case = random_case(dataset, logs[dataset], intervention, seed=args.case_seed, case_id=args.case_ids.get(dataset), costs=args.costs)
            report(args, dataset, "fall_echt_angaben", case["summary"], "Zufälliger Fall aus dem echten Log", sub="daten")
            report(args, dataset, "fall_echt_events", case["events"].round(2), "Events", sub="daten")
            bank_log = simulated_log(simulators[dataset], args.n_cases, "bank", args.seed, costs=args.costs)
            case = random_case(dataset, bank_log, intervention, seed=args.bank_case_seed, case_id=args.bank_case_ids.get(dataset), costs=args.costs)
            report(args, dataset, "fall_bank_angaben", case["summary"], "Zufälliger Fall der Bank-Policy (simuliert)", sub="daten")
            report(args, dataset, "fall_bank_events", case["events"].round(2), "Events", sub="daten")
            del bank_log, case
    # Den echten Log brauchen danach nur noch die Kennzahlen der Modelle
    if not steps & {"models"}:
        logs = None
        gc.collect()

    # ---- RCT-Daten (Zelle 10), auch für das Training und die Kennzahlen der Schätzer
    rct = {}
    if steps & {"rct", "train", "models"}:
        from helpers.rct import rct_summary, simulate_rct
        from helpers.training import train_estimators
        heading("RCT-Daten" + (" und Training" if "train" in steps else ""))
        for dataset in args.datasets:
            data = simulate_rct(simulators[dataset], n_cases=args.rct_cases, runs=args.rct_runs, seed=args.rct_seed, delta=args.delta,
                                workers=args.workers, costs=args.costs, regenerate=args.regenerate_rct)
            report(args, dataset, "rct", rct_summary(data, simulators[dataset].intervention), "RCT-Daten", sub="daten")
            # ---- Training der Schätzer (Zelle 11); je Datensatz direkt nach seinen RCT-Daten, die danach freigegeben werden
            if "train" in steps:
                table = train_estimators(simulators[dataset], data, estimators=args.estimators, delta=args.delta, seed=args.train_seed,
                                         retrain=args.retrain_estimators, cases=args.train_cases, params={"psp": psp_params(args)})
                report(args, dataset, "training", table, "Training der Schätzer", sub="daten")
            if "models" in steps:
                rct[dataset] = data
            del data
            gc.collect()

    # ---- Kennzahlen je Schätzer: wie gut er vorhersagt, auf dem simulierten Log (Testfälle des RCT) und dem echten Log
    if "models" in steps:
        from helpers.training import evaluate_estimators
        heading("Kennzahlen je Schätzer")
        for dataset in args.datasets:
            table, values = evaluate_estimators(simulators[dataset], rct[dataset], logs[dataset] if args.kpi_real_log else None, estimators=args.estimators)
            # Je Modell ein Ordner: Kennzahlen je Wahlpunkt (Spalten: Daten), Kalibrierung und Streuung
            for name in dict.fromkeys(estimator for _, estimator in table.columns):
                folder = "causal_estimator_eval/" + APPROACHES[name]
                part = table.xs(name, axis=1, level=1).dropna(how="all")
                report(args, dataset, "kennzahlen", part.round(3), f"{APPROACHES[name]}: Kennzahlen", sub=folder)
                predicted = {source: by_point for (source, estimator), by_point in values.items() if estimator == name}
                if predicted:
                    chart(dataset, "kalibrierung", calibration_chart(predicted, f"{APPROACHES[name]}: Kalibrierung"), f"{APPROACHES[name]}: Kalibrierung", sub=folder)
                    chart(dataset, "streuung", scatter_chart(predicted, f"{APPROACHES[name]}: vorhergesagt gegen tatsächlich"), f"{APPROACHES[name]}: Streuung", sub=folder)
    logs = rct = None
    gc.collect()

    # ---- Online-Auswertung auf dem Simulator: die Policy wählt live an den Wahlpunkten, gegen die Bank
    if "online" in steps:
        from helpers.report import slug
        from prescriptive.evaluation.online_evaluation import evaluate_online
        heading("Online-Auswertung")
        for dataset in args.datasets:
            points = args.online_points.get(dataset)
            table, runs = evaluate_online(simulators[dataset], policies=args.online_policies, points=points, n_cases=args.online_cases,
                                          warmup=args.online_warmup, cooldown=args.online_cooldown, seeds=args.online_seeds, seed=args.seed,
                                          workers=args.online_workers, costs=args.costs,
                                          policy_params={**({"psp_erm": {"gamma": args.erm_gamma}} if args.erm_gamma else {}),
                                                         **({"xgb": {"model": args.xgb_model}} if args.xgb_model else {})})
            # Je Ansatz ein Ordner: seine Policy mit Bank und Zufall; mit --online-points ein Unterordner je Auswahl
            baselines = [name for name in ONLINE_BASELINES if name in table.index]
            for model in [name for name in table.index if name not in baselines]:
                folder = "prescription/" + APPROACHES.get(model, model) + ("/" + slug("_".join(points)) if points else "")
                part, part_runs = table.loc[baselines + [model]], runs[runs["Policy"].isin(baselines + [model])]
                chart(dataset, "online_diagramm", uplift_chart(part, part_runs), f"{APPROACHES.get(model, model)}: online, Uplift gegen die Bank", sub=folder)
                report(args, dataset, "online", part.round(3), f"{APPROACHES.get(model, model)}: Online-Auswertung, Mittel je Policy", sub=folder)
                report(args, dataset, "online_laeufe", part_runs.round(3), f"{APPROACHES.get(model, model)}: Online-Auswertung je Lauf", sub=folder)

    # ---- Lagebild des PSP auf dem Simulator (Zelle 13)
    if "lagebild" in steps:
        from helpers.overview import simulate_overview
        from helpers.training import model_tag
        heading("Lagebild")
        for dataset in args.datasets:
            model = args.overview_model.get(dataset) or "psp_" + model_tag(simulators[dataset], args.delta, args.train_cases) + args.psp_suffix
            print(f"{dataset}: PSP-Modell {model}", flush=True)
            result = simulate_overview(simulators[dataset], model, n_cases=args.overview_cases, every_hours=args.overview_every_hours,
                                       samples=args.overview_samples, workers=args.overview_workers, seed=args.overview_seed, costs=args.costs)
            report(args, dataset, "lagebild", result["overview"].summary.round(1), "Lagebild je Beobachtung", sub="lagebild")
            report(args, dataset, "lagebild_guete", result["quality"]["cases"].round(3), "Güte der Prognosen je Phase", sub="lagebild")
            report(args, dataset, "lagebild_abschluesse", result["quality"]["portfolio"].round(1), "Abschlüsse: erwartet gegen tatsächlich", sub="lagebild")
            chart(dataset, "lagebild_diagramm", portfolio_chart(result["quality"]["portfolio"]), "Lagebild: Abschlüsse erwartet gegen tatsächlich", sub="lagebild")

    heading(f"Fertig; Ergebnisse in {args.out}, Übersicht: bericht.html")


if __name__ == "__main__":
    main()

"""
Offline-Bewertung eines Schätzers (PSP oder XGBoost, --estimator) auf gehaltenen RCT-Fällen, in Minuten statt der Stunden einer gemeinsamen Simulation.

Im RCT ist die Aktion am Entscheidungspunkt zufällig (Wahrscheinlichkeit 0,5 je Aktion). Der Wert einer Policy lässt sich
deshalb aus den Fällen schätzen, deren zufällige Aktion mit der Wahl der Policy übereinstimmt (Inverse Propensity
Weighting, selbstnormiert: Mittel des Gewinns dieser Fälle). Gehalten sind die Fälle im Test-Teil des PSP-Splits
(zufällig nach Fällen, im Training nie gesehen); mit delta > 0 nur die randomisierten.

Je Fall schätzt das PSP am Entscheidungspunkt wie in der Simulation den Gewinn von call und no_call (encode_live_prefix,
sample_per_action). Bewertet werden PSP, immer call, immer no_call und zufällig, dazu gepaarte Unterschiede mit
Bootstrap-Konfidenzintervall. Diagnose: je Aktion, was das PSP für Gewinn, Bewilligung, Anrufe, Arbeitsstunden und
Durchlaufzeit vorhersagt, gegen das, was in den Fällen mit dieser Aktion eintrat.

Grenzen: Der Anrufer war im RCT zufällig und wird hier nicht bewertet; Konkurrenz um Mitarbeiter wirkt nur, wie sie im
RCT-Lauf war. Für das endgültige Ergebnis bleibt die gemeinsame Simulation (prescriptive/main.py).

Aufruf (aus process-simulation-engine/):
    OMP_NUM_THREADS=2 python -m prescriptive.evaluation.offline_evaluation [--delta 0] [--cases 0] [--workers 8]
"""
import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

_SHARED = {}


# Wird aufgerufen von ProcessPoolExecutor -> init_worker (je Arbeitsprozess)
def init_worker(config):
    _SHARED.update(config)
    if config["estimator"] in ("xgb", "lstm"):
        import pickle
        with open(config["model_path"], "rb") as f:
            _SHARED["direct"] = pickle.load(f)
        if config["estimator"] == "xgb":
            _SHARED["direct"].decision_model.set_params(n_jobs=1)
        else:
            import torch
            torch.set_num_threads(1)
        return
    import torch

    from prescriptive.estimators import psp
    torch.set_num_threads(1)
    # Mit eos_mask darf EOS nur direkt nach einer gültigen Endaktivität gezogen werden (BatchSampler, end_activities)
    _SHARED["sampler"] = psp.BatchSampler(psp.load_evaluator(config["model_path"], config["dataset"]),
                                          end_activities=config["valid_ends"] if config.get("eos_mask") else None,
                                          transitions=config.get("transitions"), variational_decoder=config.get("variational_decoder", False))


# Schätzungen des PSP für einen Fall am Entscheidungspunkt und was tatsächlich geschah
# Wird aufgerufen von main -> pool.map -> evaluate_case (je Fall, in einem Arbeitsprozess)
def evaluate_case(case):
    import torch

    from prescriptive.estimators import psp
    from prescriptive.enriched_log import CASE, INTERVENTION
    shared = _SHARED
    case = case.reset_index(drop=True)
    decision, last = case[case["stage"] == 0].iloc[0], case.iloc[-1]
    row = {"case_id": last[CASE], "action": int(decision["treatment"]), "outcome": float(last["outcome"]), "approved": float(last["approved"]),
           # Bestandteile, die das Outcome des Datensatzes nicht hat (BPIC13: keine Anrufe), sind 0
           "calls": float(last.get("calls", 0.0)), "hours": float(last["working_hours"]),
           # Outcome negative_days_until (BPIC13): Lösungsdauer statt Durchlaufzeit, wie das PSP sie schätzt
           "days": float(last["resolution_days"] if "resolution_days" in last.index else last["cycle_days"])}
    if shared["estimator"] in ("xgb", "lstm"):
        # XGBoost und LSTM schätzen nur den Gewinn je Aktion, keine Bestandteile
        return {**row, **{f"q{action}": value for action, value in enumerate(shared["direct"].decision_values(case, range(len(INTERVENTION["actions"]))))}}
    prefix = psp.encode_live_prefix(case, shared["dataset"], "decision", INTERVENTION["actions"])
    torch.manual_seed(shared["seed"] + int(case["case_nr"].iloc[0]))
    result = psp.sample_per_action(shared["sampler"], prefix, INTERVENTION, shared["samples_per_action"], shared["max_draws"],
                                   len(INTERVENTION["actions"]), shared["hours_per_activity"], action_input=True, valid_ends=shared["valid_ends"],
                                   common_random_numbers=shared["common_random_numbers"])
    row.update(prefix["outcome_inputs"])
    for action in range(len(INTERVENTION["actions"])):
        row[f"q{action}"] = result["q"][action]
        row[f"discarded{action}"], row[f"length{action}"] = result["discarded"][action], result["suffix_len"][action]
        for name, value in result["parts_by_variant"].get((action, None), {}).items():
            row[f"{name}{action}"] = value
    return row


# Selbstnormierter IPW-Wert je Policy und gepaarte Unterschiede, mit Bootstrap über die Fälle
# Wird aufgerufen von main -> policy_values
def policy_values(rows, n_boot=1000, seed=0, name="psp"):
    rows = rows.dropna(subset=["q0", "q1"])
    choices = {name: (rows["q1"] > rows["q0"]).astype(int).to_numpy(), "always_call": np.ones(len(rows), int), "always_no_call": np.zeros(len(rows), int)}
    action, outcome = rows["action"].to_numpy(), rows["outcome"].to_numpy()

    def value(choice, index):
        match = action[index] == choice[index]
        return outcome[index][match].mean()

    def random_value(index):
        return outcome[index].mean()

    everything = np.arange(len(rows))
    estimates = {policy: value(choice, everything) for policy, choice in choices.items()}
    estimates["random"] = random_value(everything)
    rng = np.random.default_rng(seed)
    boot = {name: [] for name in estimates}
    for _ in range(n_boot):
        index = rng.integers(len(rows), size=len(rows))
        for policy, choice in choices.items():
            boot[policy].append(value(choice, index))
        boot["random"].append(random_value(index))
    boot = pd.DataFrame(boot)
    table = pd.DataFrame({"Gewinn je Fall": estimates, "95-%-KI unten": boot.quantile(0.025), "95-%-KI oben": boot.quantile(0.975)})
    for other in ("always_call", "random"):
        difference = boot[name] - boot[other]
        table.loc[f"{name} − {other}", ["Gewinn je Fall", "95-%-KI unten", "95-%-KI oben"]] = [estimates[name] - estimates[other], difference.quantile(0.025), difference.quantile(0.975)]
    table["call"] = pd.Series({policy: choice.mean() for policy, choice in choices.items()} | {"random": 0.5})
    return table


# Je Aktion: Vorhersage des PSP gegen das, was in den Fällen mit dieser Aktion tatsächlich eintrat (randomisiert, also vergleichbar)
# Wird aufgerufen von main -> calibration
def calibration(rows):
    table = {}
    from prescriptive.enriched_log import INTERVENTION
    for action, name in enumerate(INTERVENTION["actions"]):
        part = rows[rows["action"] == action]
        for measure, label in (("q", "Gewinn"), ("approved", "bewilligt"), ("calls", "Anrufe"), ("hours", "Arbeitsstunden"), ("days", "Durchlaufzeit (Tage)")):
            predicted = part[f"{measure}{action}"].mean() if f"{measure}{action}" in part else np.nan
            actual = part["outcome" if measure == "q" else measure].mean()
            table[(name, label)] = {"Schätzer": predicted, "tatsächlich": actual, "Abweichung (%)": (predicted - actual) / abs(actual) * 100 if actual else np.nan}
    return pd.DataFrame(table).T


if __name__ == "__main__":
    import torch

    from prescriptive import profiles
    from prescriptive.estimators import psp
    from prescriptive.enriched_log import CASE, with_current_costs
    from prescriptive.main import DATA_FOLDER, load_data, resolve_intervention_args, save_data, test_cases
    from preprocessing.work_time import fit_hours_per_activity
    parser = argparse.ArgumentParser(description="Offline evaluation of the PSP on held-out RCT cases (inverse propensity weighting)")
    parser.add_argument("--data-folder", type=str, default=DATA_FOLDER)
    parser.add_argument("--delta", type=float, default=0.0, help="Which run (as in prescriptive/main.py)")
    parser.add_argument("--profile", type=str, default=profiles.DEFAULT, help="Log-Profil (as in prescriptive/main.py)")
    parser.add_argument("--intervention", type=str, default=None, help="Interventionspunkt (as in prescriptive/main.py; default: the one of the profile)")
    parser.add_argument("--rct-name", type=str, default=None, help="Which RCT data and models (as in prescriptive/main.py; default from --intervention)")
    parser.add_argument("--estimator", choices=["psp", "xgb", "lstm"], default="psp", help="Causal estimator (as in prescriptive/main.py)")
    parser.add_argument("--psp-suffix", type=str, default="", help="Folder suffix of the PSP (as in prescriptive/main.py); its test cases are the held-out cases")
    parser.add_argument("--cases", type=int, default=0, help="At most this many held-out decision cases (0 = all)")
    parser.add_argument("--samples-per-action", type=int, default=100)
    parser.add_argument("--max-draws", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--common-random-numbers", action=argparse.BooleanOptionalAction, default=True,
                        help="Draw all options of a case with the same random numbers (BatchSampler.sample_common)")
    parser.add_argument("--eos-mask", action="store_true", help="PSP: EOS only directly after an activity that ends at least 0.25 %% of the cases (instead of discarding such suffixes)")
    parser.add_argument("--dfg", action="store_true", help="PSP: when sampling, only transitions between activities seen at least 3 times in the training data (psp.transition_matrix)")
    parser.add_argument("--variational-decoder", action="store_true", help="PSP: keep the decoder dropout masks of the first step for the whole suffix (as in training)")
    parser.add_argument("--tag", type=str, default="", help="Extra suffix of the result file")
    args = parser.parse_args()
    resolve_intervention_args(args)
    suffix = f"_delta{int(round(100 * args.delta))}" if args.delta > 0 else ""
    tag = args.rct_name + suffix
    psp_folder = os.path.join(args.data_folder, "psp_" + tag + args.psp_suffix)

    data = with_current_costs(load_data(os.path.join(args.data_folder, tag, "log")))
    held_out = test_cases(psp_folder)
    decisions = data[(data["stage"] == 0) & data[CASE].isin(held_out)]
    if "rct" in data.columns:
        decisions = decisions[decisions["rct"]]
    case_ids = decisions[CASE].to_numpy()
    if args.cases:
        case_ids = np.random.default_rng(args.seed).choice(case_ids, min(args.cases, len(case_ids)), replace=False)
    print(f"Gehaltene randomisierte Entscheidungsfälle: {len(case_ids)} (von {len(held_out)} gehaltenen Fällen)", flush=True)
    cases = [group for _, group in data[data[CASE].isin(set(case_ids))].groupby(CASE, sort=False)]
    config = {"estimator": args.estimator, "model_path": os.path.join(psp_folder, "psp_model.pkl"), "dataset": None,
              "hours_per_activity": fit_hours_per_activity(data), "valid_ends": psp.valid_end_activities(data),
              "samples_per_action": args.samples_per_action, "max_draws": args.max_draws, "seed": args.seed, "common_random_numbers": args.common_random_numbers,
              "eos_mask": args.eos_mask}
    if args.estimator == "psp":
        config["dataset"] = torch.load(os.path.join(psp_folder, "datasets.pt"), weights_only=False)["train"].subset(range(64))
        if args.dfg:
            config["transitions"] = psp.training_transitions(data, config["dataset"], held_out)
            print(f"Erlaubte Übergänge: {int(config['transitions'].sum())} von {config['transitions'].numel()}", flush=True)
        config["variational_decoder"] = args.variational_decoder
    else:
        config["model_path"] = os.path.join(args.data_folder, f"{args.estimator}_" + tag, f"{args.estimator}_model.pkl")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(config,)) as pool:
        rows = pd.DataFrame(list(pool.map(evaluate_case, cases, chunksize=16)))

    pd.set_option("display.width", 220)
    values, calib = policy_values(rows, name=args.estimator), calibration(rows)
    print("\nWert je Policy (selbstnormiertes IPW, Gewinn je Fall in €):")
    print(values.round(1).to_string())
    print("\nVorhersage des Schätzers gegen tatsächlich, je Aktion:")
    print(calib.round(3).to_string())
    save_data({"rows": rows, "values": values, "calibration": calib, "args": vars(args)}, os.path.join(args.data_folder, "eval", "offline" + ("" if args.rct_name == "rct" else "_" + args.rct_name) + suffix
                                                                + (args.psp_suffix if args.estimator == "psp" else "_" + args.estimator) + args.tag))

"""
DR-Learner (AIPW als Schätzer, in SCOPE "AIPWE") für call/no_call, mit dem PSP als Outcome-Modell; zum Vergleich mit
XGBoost als Outcome-Modell. Offline bewertet mit AIPW (prescriptive/aipw.py) auf den gehaltenen RCT-Fällen.

1. Outcome-Modell μ_a(x) (geschätzter Gewinn bei Aktion a) auf Fällen, die es nie gesehen hat: Entscheidungsfälle der
   Jahre R3 bis R8 der dreifachen RCT-Daten (rct3x). Das erste PSP (psp_rct) und XGBoost (xgb_rct) wurden nur auf
   R0 bis R2 trainiert; ohne diese Trennung wären ihre Schätzungen dort zu gut und die Korrektur verzerrt.
2. Pseudo-Outcome je Fall (Effekt des Anrufs, doppelt robust; e = 0,5 im RCT):
       Γ = μ_1(x) − μ_0(x) + T/e · (Y − μ_1(x)) − (1 − T)/(1 − e) · (Y − μ_0(x))
3. Effekt-Modell τ(x): XGBoost auf die Merkmale am Entscheidungspunkt (wie xgb_estimator) mit Ziel Γ.
4. Entscheidung: call, wenn τ(x) > 0.

Aufruf (aus process-simulation-engine/): python -m prescriptive.estimators.dr_learner [--cases 40000] [--workers 8]
"""
import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from prescriptive.enriched_log import CASE, CASE_ATTRS, EVENT, with_current_costs
from prescriptive.main import DATA_FOLDER, load_data, save_data, test_cases
from prescriptive.evaluation import offline_evaluation

PROPENSITY = 0.5


# Merkmale am Entscheidungspunkt (ohne Aktion) für die Fälle cases, wie bei XGBoost
# Wird aufgerufen von __main__ -> decision_features
def decision_features(log, cases, activities, goals, types):
    from prescriptive.estimators.xgb_estimator import training_frames
    (X, _), _ = training_frames(log[log[CASE].isin(set(cases))], activities, goals, types, [])
    return X.loc[cases].drop(columns="action")  # das Effekt-Modell bekommt die beobachtete Aktion nicht


# PSP-Schätzungen μ_0, μ_1 je Fall, mit derselben Abfrage wie die Offline-Auswertung (offline.evaluate_case)
# Wird aufgerufen von __main__ -> psp_outcomes (zwischengespeichert in eval/dr_psp_nuisance.pkl)
def psp_outcomes(log, cases, workers, seed, psp_suffix="", dfg=False):
    import torch

    from prescriptive import psp
    from preprocessing.work_time import fit_hours_per_activity
    psp_folder = os.path.join(DATA_FOLDER, "psp_rct" + psp_suffix)
    base = with_current_costs(load_data(os.path.join(DATA_FOLDER, "rct", "log")))
    config = {"estimator": "psp", "model_path": os.path.join(psp_folder, "psp_model.pkl"),
              "dataset": torch.load(os.path.join(psp_folder, "datasets.pt"), weights_only=False)["train"].subset(range(64)),
              # wie für das PSP in der Offline-Auswertung: Stunden je Aktivität und gültige Enden aus seinen Trainingsdaten
              "hours_per_activity": fit_hours_per_activity(base), "valid_ends": psp.valid_end_activities(base),
              "samples_per_action": 100, "max_draws": 2000, "seed": seed, "common_random_numbers": True, "eos_mask": False}
    if dfg:
        config["transitions"] = psp.training_transitions(base, config["dataset"], test_cases(psp_folder))
    groups = [group for _, group in log[log[CASE].isin(set(cases))].groupby(CASE, sort=False)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=offline_evaluation.init_worker, initargs=(config,)) as pool:
        rows = pd.DataFrame(list(pool.map(offline_evaluation.evaluate_case, groups, chunksize=16)))
    return rows.set_index("case_id")


def pseudo_outcomes(mu0, mu1, action, outcome, e=PROPENSITY):
    return mu1 - mu0 + action / e * (outcome - mu1) - (1 - action) / (1 - e) * (outcome - mu0)


# Effekt-Modell τ(x) auf Γ, mit frühem Stoppen auf 10 % der Fälle
def fit_effect(X, gamma, seed):
    from xgboost import XGBRegressor

    from prescriptive.estimators.xgb_estimator import XGB_PARAMS
    validation = np.random.default_rng(seed).random(len(X)) < 0.1
    model = XGBRegressor(**{**XGB_PARAMS, "min_child_weight": 50}, random_state=seed)
    model.fit(X[~validation], gamma[~validation], eval_set=[(X[validation], gamma[validation])], verbose=False)
    return model


if __name__ == "__main__":
    from prescriptive.aipw import RUNS, contributions, cross_fitted_outcomes, interval
    from prescriptive.estimators.xgb_estimator import XGBEstimator
    parser = argparse.ArgumentParser(description="DR-Learner (AIPW) with PSP or XGBoost as outcome model, evaluated offline with AIPW")
    parser.add_argument("--cases", type=int, default=40000, help="Decision cases of the years R3 to R8 for the DR-Learner")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--psp-suffix", type=str, default="", help="Which PSP (trained on rct, i.e. R0 to R2) is the outcome model")
    parser.add_argument("--dfg", action="store_true", help="PSP: sample only transitions seen in its training data")
    args = parser.parse_args()
    name = args.psp_suffix + ("_dfg" if args.dfg else "")

    log = with_current_costs(load_data(os.path.join(DATA_FOLDER, "rct3x", "log")))
    held_out = test_cases(os.path.join(DATA_FOLDER, "psp_rct"))
    decisions = log[log["stage"] == 0].set_index(CASE)
    new_years = decisions[~decisions.index.str.match(r"R[012]_") & ~decisions.index.isin(held_out)]
    train_cases = sorted(np.random.default_rng(args.seed).choice(new_years.index.to_numpy(), min(args.cases, len(new_years)), replace=False))
    print(f"DR-Training: {len(train_cases)} Entscheidungsfälle aus R3 bis R8", flush=True)

    nuisance_path = os.path.join(DATA_FOLDER, "eval", "dr_psp_nuisance" + name)
    if os.path.exists(nuisance_path + ".pkl") and len(load_data(nuisance_path)) >= len(train_cases):
        psp_rows = load_data(nuisance_path)
    else:
        psp_rows = psp_outcomes(log, train_cases, args.workers, args.seed, args.psp_suffix, args.dfg)
        save_data(psp_rows, nuisance_path)
    train_cases = [c for c in train_cases if c in psp_rows.index and np.isfinite(psp_rows.loc[c, ["q0", "q1"]]).all()]
    print(f"PSP-Schätzungen: {len(train_cases)} Fälle", flush=True)

    activities = sorted(log[EVENT].dropna().astype(str).unique())
    goals = sorted(log[CASE_ATTRS[0]].dropna().astype(str).unique())
    types = sorted(log[CASE_ATTRS[1]].dropna().astype(str).unique())
    X_train = decision_features(log, train_cases, activities, goals, types)
    action = decisions.loc[train_cases, "treatment"].to_numpy().astype(int)
    outcome = decisions.loc[train_cases, "outcome"].to_numpy()

    xgb = XGBEstimator.load(os.path.join(DATA_FOLDER, "xgb_rct", "xgb_model.pkl"))
    Xa = decision_features(log, train_cases, xgb.activities, xgb.goals, xgb.types)
    mu_xgb = [xgb.decision_model.predict(Xa.assign(action=a)[xgb.columns[0]]) for a in (0, 1)]
    nuisances = {"PSP": (psp_rows.loc[train_cases, "q0"].to_numpy(), psp_rows.loc[train_cases, "q1"].to_numpy()), "XGBoost": tuple(mu_xgb)}

    # Bewertung auf den gehaltenen Fällen, gemeinsam mit den bisherigen Schätzern (AIPW)
    rows = {name: load_data(os.path.join(DATA_FOLDER, "eval", file))["rows"].dropna(subset=["q0", "q1"]).set_index("case_id")
            for name, file in RUNS.items() if os.path.exists(os.path.join(DATA_FOLDER, "eval", file + ".pkl"))}
    test = sorted(set.intersection(*[set(r.index) for r in rows.values()]))
    base_log = with_current_costs(load_data(os.path.join(DATA_FOLDER, "rct", "log")))
    test_action = rows["PSP erstes (100 Suffixe)"].loc[test, "action"].to_numpy().astype(int)
    test_outcome = rows["PSP erstes (100 Suffixe)"].loc[test, "outcome"].to_numpy()
    mu_eval = cross_fitted_outcomes(base_log, test)
    X_test = decision_features(base_log, test, activities, goals, types)
    choices = {"immer call": np.ones(len(test), int)}
    choices.update({name: (r.loc[test, "q1"] > r.loc[test, "q0"]).astype(int).to_numpy() for name, r in rows.items()})
    summary = {}
    for name, (mu0, mu1) in nuisances.items():
        gamma = pseudo_outcomes(mu0, mu1, action, outcome)
        naive = mu1 - mu0
        model = fit_effect(X_train, gamma, args.seed)
        tau = model.predict(X_test)
        choices[f"DR-Learner, Outcome-Modell {name}"] = (tau > 0).astype(int)
        summary[name] = {"mittlerer Effekt laut Outcome-Modell": naive.mean(), "mittleres Γ": gamma.mean(),
                         "mittlerer Effekt RCT (call − no_call)": outcome[action == 1].mean() - outcome[action == 0].mean(), "Bäume": model.best_iteration}
    print(pd.DataFrame(summary).round(1).to_string(), flush=True)
    contrib = {name: contributions(choice, test_action, test_outcome, mu_eval) for name, choice in choices.items()}
    table = []
    for name, c in contrib.items():
        row = {"Policy": name, "call": choices[name].mean()}
        row["Gewinn je Fall"], row["KI unten"], row["KI oben"] = interval(c)
        for ref in ("immer call", "PSP erstes (100 Suffixe)", "XGBoost"):
            if ref != name:
                d = interval(c - contrib[ref])
                row[f"gegen {ref}"] = f"{d[0]:+.1f} [{d[1]:+.0f}; {d[2]:+.0f}]"
        table.append(row)
    table = pd.DataFrame(table).set_index("Policy")
    pd.set_option("display.width", 300)
    print(table.round(1).to_string())
    save_data({"table": table, "summary": summary, "train_cases": train_cases}, os.path.join(DATA_FOLDER, "eval", "dr_learner" + name))

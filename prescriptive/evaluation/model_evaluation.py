"""
Offline-Auswertung aller Modelle auf dem echten Event-Log an einem wählbaren Interventionspunkt, derzeit nur die Wahl
des Mitarbeiters (Kandidaten "Mitarbeiter: <Tätigkeit>" des Katalogs, Art resource).

Der Wert einer Policy wird mit AIPW aus dem Log geschätzt, wie im Screening (helpers.screening):
- Fälle, Merkmale X, Wahl der Bank T, Outcome Y und Pool je Fall: der Mitarbeiter-Kandidat des Screenings
  (build_candidate; Pool POOL: kompetent, im Dienst, frei)
- Optionen: die Mitarbeiter mit eigener Option wie im Screening (mindestens min_cases Fälle), alle übrigen zusammen
  other. Gewählt wird nur unter
  den eigenen Optionen im Pool (other_choosable=False), für other gibt es kaum vergleichbare Fälle im Log. Propensity
  und Outcome-Modell kennen other trotzdem, damit e und μ die Wahl der Bank unter allen Mitarbeitern beschreiben.
- Ausgewertet werden die Fälle mit Wahl (mindestens zwei wählbare Mitarbeiter im Pool)
- Beitrag eines Falls zum Wert einer Policy π: ψ[π(x)] = μ_π(x) + 1{T = π(x)} (Y − μ_π(x)) / e_π(x)
  (helpers.screening.causal.aipw, e mit Cross-Fitting geschätzt, je Fall auf den Pool normiert)

Policies: Bank (das beobachtete Outcome), Zufall (Mittel über die wählbaren Mitarbeiter), der DR-Learner des Screenings
(auf dem echten Log selbst geschätzt, mit Cross-Fitting: was ein Modell erreicht, das die echten Daten kennt) und die
trainierten Modelle (PSP, XGBoost, LSTM), wenn der Interventionspunkt im Katalog zu einer Intervention des Simulators
gehört (Feld intervention) und deren Arbeitsschritt (action_work_item) ist; BPIC17: "Mitarbeiter: W_Call incomplete
files" zu call. Die Modelle sind auf den RCT-Daten dieser Intervention trainiert (find_models) und wählen den Mitarbeiter
dort, wo sie es im RCT taten: am ersten Beginn des Arbeitsschritts nach call am ersten Entscheidungspunkt. Gewertet
werden dann nur diese Fälle, für alle Policies dieselben.

Ältere Modelle (vor den Lifecycle-Events, Vokabular nur mit complete-Events) bekommen den Fall nur mit complete-Events,
so wie sie trainiert sind. Das PSP fragt in Arbeitsprozessen ab, geladen wie in evaluation/offline_evaluation.py
(init_worker). Die Werte je Modell liegen in data/eval/models_<datensatz>/ und werden beim nächsten Lauf mit denselben
Einstellungen und derselben Modelldatei wiederverwendet (recompute=True: neu).

    from prescriptive.evaluation.model_evaluation import evaluate_models, resource_points
    resource_points("bpic17")                                   # wählbare Interventionspunkte und ihre Modelle
    table = evaluate_models("bpic17", log, "Mitarbeiter: W_Call incomplete files", max_cases=2000)
"""
import glob
import hashlib
import json
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from helpers.datasets import CASE, LIFECYCLE, PRESCRIPTIVE, TIMESTAMP, add_paths, intervention_table
from helpers.screening.candidates import build_candidate, prepared_log
from helpers.screening.causal import aipw, interval, nuisances, option_scores
from helpers.screening.evaluation import MIN_GROUP, OTHER, employee_propensities, grouped, pool_mask, within_pool

DATA_FOLDER = os.path.join(PRESCRIPTIVE, "data")
KINDS = ("psp", "xgb", "lstm")
BASELINES = ["Bank", "Zufall", "DR-Learner (echter Log)"]
# Wie aus dem PSP gezogen wird (evaluation/offline_evaluation.py --eos-mask, --variational-decoder)
PSP_SAMPLING = {"eos_mask": False, "variational_decoder": False}


# Die wählbaren Interventionspunkte (Mitarbeiter-Kandidaten des Katalogs), die Intervention des Simulators, zu der sie
# gehören, und deren gespeicherte Modelle
def resource_points(dataset, data_folder=DATA_FOLDER):
    rows = []
    for _, row in intervention_table(dataset, full=True).iterrows():
        test = row["test"] if isinstance(row["test"], dict) else {}
        if test.get("kind") != "resource":
            continue
        intervention = linked_intervention(dataset, row["name"])
        models = find_models(dataset, intervention, data_folder) if intervention else {}
        rows.append({"Interventionspunkt": row["name"], "Entscheidung": test.get("decision"), "Intervention": intervention["key"] if intervention else "",
                     "Modelle": len(models), "Namen": ", ".join(models)})
    return pd.DataFrame(rows).set_index("Interventionspunkt")


# Die Intervention des Simulators zum Interventionspunkt: Feld intervention im Katalog, ihr Arbeitsschritt ist die
# Tätigkeit des Kandidaten (sonst None: es gibt keine Modelle, nur die Policies auf dem echten Log)
def linked_intervention(dataset, point):
    add_paths()
    import interventions as intervention_catalog
    table = intervention_table(dataset, full=True).set_index("name")
    if point not in table.index:
        raise ValueError(f"Unbekannter Interventionspunkt '{point}' für {dataset}, möglich: {', '.join(resource_points(dataset).index)}")
    key, test = table.loc[point, "intervention"], table.loc[point, "test"]
    if not isinstance(key, str) or not key:
        return None
    intervention = intervention_catalog.get(key)
    return intervention if intervention.get("action_work_item") == test.get("decision") else None


# Alle gespeicherten Modelle der Intervention: Ordner <art>_<rct-Name>... mit <art>_model.pkl (main.py: psp_<tag><psp-suffix>,
# xgb_<tag>, lstm_<tag>; tag = rct-Name mit _delta<Prozent>). Gesucht wird auch ohne die Lifecycle-Endung (LIFECYCLE_TAG),
# so werden die älteren Modelle auf complete-Events mit gefunden. XGBoost und LSTM nur zum gewählten Outcome
# (profiles.outcome_tag: Ordner mit _<Option> bzw. ohne eine Option); das PSP gilt für jedes Outcome.
def find_models(dataset, intervention, data_folder=DATA_FOLDER):
    add_paths()
    import interventions as catalog
    from prescriptive import profiles
    from prescriptive.main import LIFECYCLE_TAG, rct_name
    name = rct_name(dataset, intervention["key"])
    base = name[:-len(LIFECYCLE_TAG)] if name.endswith(LIFECYCLE_TAG) else name
    chosen = profiles.outcome_tag(dataset)
    options = ["_" + option for option in (catalog.catalog()[dataset].get("outcome_optionen") or {})]
    models = {}
    for kind in KINDS:
        for path in sorted(glob.glob(os.path.join(data_folder, f"{kind}_{base}*", f"{kind}_model.pkl"))):
            folder = os.path.basename(os.path.dirname(path))
            if kind != "psp" and (not folder.endswith(chosen) if chosen else any(folder.endswith(option) for option in options)):
                continue
            models[folder] = (kind, path)
    return models


# Auswahl unter den gefundenen Modellen (find_models): eine Art ("psp", "xgb" bzw. "xgboost", "lstm"; Groß- und
# Kleinschreibung egal) wählt alle Modelle dieser Art, ein Ordnername genau dieses; auch eine Liste aus beidem
ALIASES = {"psp": "psp", "xgb": "xgb", "xgboost": "xgb", "lstm": "lstm"}


def select_models(found, models):
    selected = {}
    for name in [models] if isinstance(models, str) else models:
        kind = ALIASES.get(str(name).lower())
        if kind:
            selected.update({model: entry for model, entry in found.items() if entry[0] == kind})
        elif name in found:
            selected[name] = found[name]
        else:
            raise ValueError(f"Unbekanntes Modell '{name}': eine Art ({', '.join(ALIASES)}) oder ein Ordner aus {', '.join(found) or 'keine'}")
    return selected


def evaluate_models(dataset, log, point, models=None, pool="frei", min_cases=MIN_GROUP, other_choosable=False,
                    max_cases=None, samples_per_action=100, max_draws=2000, psp_sampling=None, workers=8, folds=5, seed=0, costs=None,
                    recompute=False, data_folder=DATA_FOLDER):
    """
    Tabelle je Policy (siehe oben). log: der vorverarbeitete echte Log (helpers.preprocessing, Outcome mit den Kosten
    costs); point: Interventionspunkt (resource_points); models: Art oder Ordnername oder Liste davon (select_models;
    None: alle); max_cases: höchstens
    so viele Fälle mit Wahl, zufällig und für alle Policies dieselben (das PSP zieht je Fall und Mitarbeiter
    samples_per_action Suffixe). table.attrs: "choices" (je Fall und Policy der gewählte Mitarbeiter), "info" (Umfang,
    Optionen), "skipped" (Modelle ohne Mitarbeiterwahl und warum).
    """
    add_paths()
    from prescriptive import profiles
    intervention = linked_intervention(dataset, point)
    print(f"{point}: Fälle, Pools ({pool}) und Optionen aus dem echten Log ...", flush=True)
    data = _case_data(dataset, log, point, intervention, profiles.get(dataset), pool, min_cases, other_choosable, folds, seed, costs)
    cases = data["cases"]
    if max_cases and len(cases) > max_cases:
        cases = np.sort(np.random.default_rng(seed).choice(cases, max_cases, replace=False))
    print(f"Ausgewertet: {len(cases):,} Fälle mit Wahl" + (f" (von {len(data['cases']):,})" if len(cases) < len(data["cases"]) else ""), flush=True)

    found = find_models(dataset, intervention, data_folder) if intervention else {}
    if not intervention:
        print("Keine Intervention des Simulators mit diesem Arbeitsschritt: keine trainierten Modelle, nur die Policies auf dem echten Log", flush=True)
    if models is not None:
        found = select_models(found, models)
    print(f"Modelle: {', '.join(found) or 'keine'}", flush=True)
    psp_sampling = {**PSP_SAMPLING, **(psp_sampling or {})}
    settings = {"point": point, "cases": [str(case) for case in cases], "pool": pool, "min_cases": min_cases,
                "other_choosable": other_choosable, "samples_per_action": samples_per_action, "max_draws": max_draws, "psp_sampling": psp_sampling,
                "seed": seed, "costs": costs}
    folder = os.path.join(data_folder, "eval", f"models_{dataset}")
    values, skipped = {}, {}
    for name, (kind, path) in found.items():
        # Mit dem Fingerabdruck der Modelldatei: ein neu trainiertes Modell unter demselben Namen rechnet neu
        model_settings = {**settings, "model": _fingerprint(path)}
        cached = None if recompute else _load_cached(folder, name, model_settings)
        if cached is not None:
            values[name] = cached
            print(f"{name}: gespeichert, geladen", flush=True)
            continue
        print(f"{name} ...", flush=True)
        try:
            values[name] = _model_values(kind, path, data, cases, intervention, samples_per_action, max_draws, psp_sampling, workers, seed)
        except _NoResourceChoice as reason:
            skipped[name] = str(reason)
            print(f"  übersprungen: {reason}", flush=True)
            continue
        _save_cached(folder, name, model_settings, values[name])

    table, choices = _policy_table(data, cases, values, seed)
    table.attrs.update(choices=choices, info={**data["info"], "ausgewertet": len(cases)}, skipped=skipped)
    return table


# ------------------------------------------------------------------ Fälle, Optionen und AIPW

def _case_data(dataset, log, point, intervention, profile, pool, min_cases, other_choosable, folds, seed, costs):
    # Der Mitarbeiter-Kandidat des Screenings: X, die Wahl der Bank T, Outcome Y und der Pool je Fall
    candidate = build_candidate(dataset, log, point, seed=seed, pool=pool)
    if candidate.skip or candidate.pool is None:
        raise ValueError(f"{point}: nicht auswertbar ({candidate.skip or 'kein Pool'})")
    table = employee_propensities(candidate, folds, seed)
    own = list(table.index[table["Fälle"] >= min_cases])
    T_all, pools, frequent, rare, members = grouped(candidate, min_cases, own)
    counts = T_all.value_counts()
    options = frequent + ([OTHER] if counts.get(OTHER, 0) >= min_cases else [])
    keep = T_all.isin(options).to_numpy()
    X, Y = candidate.X[keep], candidate.Y[keep].to_numpy()
    codes = T_all[keep].map({o: i for i, o in enumerate(options)}).to_numpy(dtype=int)
    allowed = pool_mask(pools[keep], options)
    e, mu = nuisances(X, codes, Y, len(options), folds, seed)
    e = within_pool(e, allowed)
    psi = aipw(codes, Y, e, mu)
    dr = option_scores(X, psi, folds, seed)

    # Wählbar: die eigenen Optionen im Pool (mit other_choosable auch alle übrigen Mitarbeiter des Pools)
    index = X.index
    code_of = {o: i for i, o in enumerate(options)}
    choosable = [sorted(p for p in people if p in code_of or (other_choosable and OTHER in code_of)) for people in candidate.pool.reindex(index)]
    with_choice = np.array([len(people) >= 2 for people in choosable])
    info = {"Fälle (Screening)": len(candidate.X), "Fälle mit Wahl": int(with_choice.sum()), "eigene Optionen": len(frequent),
            "other": f"{rare} Mitarbeiter, wählbar: {other_choosable}"}
    data = {"position": {case: i for i, case in enumerate(index)}, "choosable": dict(zip(index, choosable)), "code_of": code_of,
            "Y": Y, "e": e, "psi": psi, "dr": dr, "size": allowed.sum(axis=1), "bank": candidate.T.reindex(index), "info": info}

    if intervention is None:
        data["cases"] = index[with_choice].to_numpy()
        return data
    # Mit Modellen: nur die Fälle, an denen sie im RCT wählten (call am ersten Entscheidungspunkt, danach der erste Beginn
    # des Arbeitsschritts, call_start), zur selben Zeit und mit demselben Mitarbeiter wie die Wahl im Screening
    from prescriptive.enriched_log import before_call_start, enrich, naive_utc
    enriched = enrich(log, n_stages=1, profile=profile, intervention=intervention, costs=costs)
    decisions = enriched[enriched["stage"] == 0].drop_duplicates(CASE).set_index(CASE)
    decisions = decisions[(decisions["treatment"] == 1) & decisions["call_start"].notna()]
    order = prepared_log(log)
    decided_at = order[order["_pos"] == order[CASE].map(candidate.decision_pos)].set_index(CASE)[TIMESTAMP]
    same = (index.isin(decisions.index)
            & (naive_utc(decided_at.reindex(index)).to_numpy() == naive_utc(decisions["call_start"].reindex(index)).to_numpy())
            & (decisions["call_resource"].reindex(index).astype(object).to_numpy() == candidate.T.reindex(index).astype(object).to_numpy()))
    cases = index[same & with_choice]
    rows = enriched[enriched[CASE].isin(set(cases))]
    rows = rows[before_call_start(rows, intervention)]
    info.update({f"davon nach {intervention['actions'][1]} wie im RCT": int((same & with_choice).sum())})
    data.update(cases=cases.to_numpy(), case_logs={case: part for case, part in rows.groupby(CASE, sort=False)},
                times=decisions["call_start"], enriched=enriched)
    return data


# Wahl je Policy, Beiträge und Tabelle
def _policy_table(data, cases, values, seed):
    psi, e, size, Y, code_of = data["psi"], data["e"], data["size"], data["Y"], data["code_of"]
    rows = np.array([data["position"][case] for case in cases])
    choosable = [data["choosable"][case] for case in cases]
    bank = data["bank"].reindex(cases).astype(object).to_numpy()

    def code(person):
        return code_of.get(person, code_of.get(OTHER))

    rng = np.random.default_rng(seed)
    choices = {"Bank": bank,
               "Zufall": np.array([people[rng.integers(len(people))] for people in choosable], dtype=object),
               "DR-Learner (echter Log)": np.array([max(people, key=lambda p, r=r: data["dr"][r, code(p)]) for r, people in zip(rows, choosable)], dtype=object)}
    estimated = {}
    for name, per_case in values.items():
        picked, missing = [], 0
        for case, people, chosen_by_bank in zip(cases, choosable, bank):
            scores = per_case.get(case, {})
            known = {p: scores[p] for p in people if p in scores and np.isfinite(scores[p])}
            # Ohne Schätzung entscheidet die Bank
            picked.append(max(known, key=known.get) if known else chosen_by_bank)
            missing += not known
        choices[name] = np.array(picked, dtype=object)
        estimated[name] = 1 - missing / max(len(cases), 1)

    table = []
    for name, chosen in choices.items():
        if name == "Bank":
            contribution = Y[rows]
        elif name == "Zufall":
            # Erwartungswert der zufälligen Wahl: Mittel über die wählbaren Mitarbeiter
            contribution = np.array([np.mean([psi[r, code(p)] for p in people]) for r, people in zip(rows, choosable)])
        else:
            contribution = psi[rows, [code(p) for p in chosen]]
        gain, versus = interval(contribution), interval(contribution - Y[rows])
        # Positivität der Wahl: die Bank wählt die gewählte Option in solchen Fällen mit mehr als 0,1 / Poolgröße
        supported = e[rows, [code(p) for p in chosen]] > 0.1 / size[rows]
        table.append({"Policy": name, "Gewinn je Fall": gain[0], "KI unten": gain[1], "KI oben": gain[2],
                      "ggü. Bank": versus[0], "Bank KI unten": versus[1], "Bank KI oben": versus[2],
                      "wie die Bank": float(np.mean(chosen == bank)), "Wahl gestützt": float(np.mean(supported)),
                      "mit Schätzung": estimated.get(name, 1.0), "verschiedene Gewählte": len(set(chosen))})
    table = pd.DataFrame(table).set_index("Policy")
    table = pd.concat([table.loc[BASELINES], table.drop(index=BASELINES).sort_values("ggü. Bank", ascending=False)])
    return table, pd.DataFrame(choices, index=cases)


# ------------------------------------------------------------------ Werte der Modelle je Fall und Mitarbeiter

class _NoResourceChoice(Exception):
    pass


# Ältere Modelle kennen nur complete-Events: ihr Vokabular hat keine Lifecycle-Tokens (enriched_log.event_token: "<Aktivität>:<lifecycle>")
def _complete_only(tokens):
    return not any(":" in str(token) for token in tokens)


def _model_log(case_log, complete_only):
    return case_log[case_log[LIFECYCLE] == "complete"] if complete_only else case_log


# {Fall: {Mitarbeiter: geschätzter Gewinn}} für die Fälle cases
def _model_values(kind, path, data, cases, intervention, samples_per_action, max_draws, psp_sampling, workers, seed):
    if kind == "psp":
        return _psp_values(path, data, cases, intervention, samples_per_action, max_draws, psp_sampling, workers, seed)
    with open(path, "rb") as f:
        estimator = pickle.load(f)
    if kind == "xgb":
        if estimator.caller_model is None:
            raise _NoResourceChoice("kein Modell für den Mitarbeiter (RCT ohne Mitarbeiterwahl)")
        tokens = estimator.activities
    else:
        if "caller" not in estimator.nets:
            raise _NoResourceChoice("kein Netz für den Mitarbeiter (RCT ohne Mitarbeiterwahl)")
        tokens = estimator.vocab
    complete_only = _complete_only(tokens)
    values = {}
    for i, case in enumerate(cases):
        case_log = _model_log(data["case_logs"][case], complete_only)
        values[case] = estimator.caller_values(case_log, data["times"][case], data["choosable"][case])
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:,} von {len(cases):,} Fällen", flush=True)
    return values


def _psp_values(path, data, cases, intervention, samples_per_action, max_draws, psp_sampling, workers, seed):
    from prescriptive.enriched_log import set_intervention
    from prescriptive.estimators import psp
    from prescriptive.evaluation.offline_evaluation import init_worker
    from preprocessing.work_time import fit_hours_per_activity
    # Die Arbeitsprozesse übernehmen die Intervention über die Umgebungsvariable (enriched_log.INTERVENTION_ENV)
    set_intervention(intervention["key"])
    print("  lade das Dataset des Modells (datasets.pt) ...", flush=True)
    dataset = psp.load_small_dataset(os.path.join(os.path.dirname(path), "datasets.pt"))
    if not psp.token_callers(dataset):
        raise _NoResourceChoice("keine Mitarbeiter-Tokens im Vokabular")
    complete_only = _complete_only(dataset.all_categories[0][0][2])
    # Arbeitsstunden je Token und gültige Enden wie im Training, aber aus dem echten Log im Format des Modells
    enriched = _model_log(data["enriched"], complete_only)
    config = {"estimator": "psp", "model_path": path, "dataset": dataset, "valid_ends": psp.valid_end_activities(enriched),
              "hours_per_activity": fit_hours_per_activity(enriched), "intervention": intervention,
              "samples_per_action": samples_per_action, "max_draws": max_draws, "seed": seed, **psp_sampling}
    jobs = [(case, _model_log(data["case_logs"][case], complete_only), data["times"][case], data["choosable"][case]) for case in cases]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(config,)) as pool:
        return dict(pool.map(_psp_case, jobs, chunksize=8))


# Je Fall die Suffixe je wählbarem Mitarbeiter (Mitarbeiter ohne eigenes Token teilen sich "other")
# Wird aufgerufen von _psp_values -> pool.map -> _psp_case (je Fall, im Arbeitsprozess nach offline_evaluation.init_worker)
def _psp_case(job):
    import torch
    from prescriptive.estimators import psp
    from prescriptive.evaluation.offline_evaluation import _SHARED as shared
    case, case_log, time, people = job
    intervention = shared["intervention"]
    try:
        prefix = psp.encode_live_prefix(case_log, shared["dataset"], "caller", intervention["actions"], time=time, candidates=people)
        torch.manual_seed(shared["seed"] + int(prefix["case_nr"]))
        result = psp.sample_per_action(shared["sampler"], prefix, intervention, shared["samples_per_action"], shared["max_draws"],
                                       len(intervention["actions"]), shared["hours_per_activity"], action_input=True, valid_ends=shared["valid_ends"])
    except Exception as error:  # ein Fall, den das Modell nicht kodieren kann: dort entscheidet die Bank
        print(f"  {case}: {type(error).__name__}: {error}", flush=True)
        return case, {}
    known = set(psp.token_callers(shared["dataset"]))
    values = result["q_by_variant"]
    return case, {p: values.get((1, p if p in known else psp.OTHER), np.nan) for p in people}


# ------------------------------------------------------------------ Zwischenspeicher je Modell

def _key(settings):
    return hashlib.sha1(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:16]


# Inhalt der Modelldatei (<art>_model.pkl) als Teil des Schlüssels
def _fingerprint(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:16]


def _load_cached(folder, name, settings):
    path = os.path.join(folder, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return saved["values"] if saved.get("key") == _key(settings) else None


def _save_cached(folder, name, settings, values):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{name}.pkl"), "wb") as f:
        pickle.dump({"key": _key(settings), "values": values}, f)

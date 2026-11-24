"""Compare frozen OE and D on identical captured online decision prefixes."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()
import numpy as np
import pandas as pd
import torch
from prescriptive.estimators.psp_model import PSPModel
from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
from prescriptive.policies.ev_policy_event_costs.psp_policy import EventCostPSPModel

BASE = ROOT / "prescriptive/scripts/results_100k/d_sampling_diagnosis_20261115"
OUT = ROOT / "prescriptive/scripts/results_100k/oe_vs_d_sampling_20261115"
REF = ROOT / "prescriptive/policies/ev_policy_event_costs/results/psp_d_ue_200_faelle_500_ziehungen_seed0_20261114"
MODELS = ROOT / "prescriptive/models"
OE_PATH = MODELS / "psp_model_durchlaufzeit_real_alle_s60_token_oe_ue.pkl"
D_PATH = MODELS / "psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder_ue.pkl"
os.environ["PSP_CATALOG"] = str(REF / "catalog.yaml")
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib"))
warnings.filterwarnings("ignore", category=FutureWarning)
ACT, LC, TS = "concept:name", "lifecycle:transition", "time:timestamp"
EDGES = np.array([1, 600, 14400, 86400, 259200, 604800, 1209600, 2592000])
ENDS = {"A_Pending", "A_Denied", "A_Cancelled"}


def read(path):
    return json.loads(Path(path).read_text())


def write(name, data):
    path = OUT / name
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model, reference = PSPModel.load(OE_PATH), PSPModel.load(D_PATH)
    model.set_threads(1)
    assert model.gap_tokens() and not model.gap_classes() and model.option_in_encoder()
    assert reference.gap_classes() and not reference.gap_tokens() and reference.option_in_encoder()
    assert model.test_cases == reference.test_cases and model.points == reference.points
    pairs = set(map(tuple, model.transition_rules["pairs"]))
    assert pairs == set(map(tuple, reference.transition_rules["pairs"]))
    assert set(model.transition_rules["ends"]) == set(reference.transition_rules["ends"])
    differences = {k: {"D": reference.params.get(k), "OE": model.params.get(k)}
                   for k in set(model.params)|set(reference.params) if model.params.get(k)!=reference.params.get(k)}
    write("manifest.json", {"oe_model": str(OE_PATH), "d_model": str(D_PATH), "parameter_differences": differences,
        "same_held_out_cases_choice_points_and_transition_rules": True, "samples_per_option": 500,
        "repeats_per_random_prefix": 3, "random_prefixes": 12, "extra_case750_repeats": 5,
        "original_inputs_sha256": {str(p): sha(p) for p in [OE_PATH, D_PATH, BASE / "prefixes.pkl",
            BASE / "selected_decisions.json", BASE / "psp_draws.csv", BASE / "simulator_branches.csv",
            REF / "event_costs.yaml", ROOT / "prescriptive/estimators/psp_model.py"]}})
    del reference
    costs = EventCosts(REF / "event_costs.yaml")
    adapter = EventCostPSPModel(model, costs)
    prefixes = pickle.loads((BASE / "prefixes.pkl").read_bytes())
    targets = set(read(BASE / "selected_decisions.json"))
    # Descriptive support check against the complete observed log; not used to draw.
    raw = pd.read_pickle(ROOT / "prescriptive/data/cache/bpic17_raw.pkl")
    raw = raw[["case:concept:name", ACT, LC, TS]].copy().sort_values(["case:concept:name", TS], kind="stable")
    gaps = pd.to_datetime(raw[TS]).groupby(raw["case:concept:name"], observed=True).diff().dt.total_seconds().fillna(0)
    tokens = raw[ACT].astype(str).str.strip()+"|"+raw[LC].astype(str)
    seen = set(zip(tokens, np.searchsorted(EDGES, gaps.to_numpy(), side="right")))
    del raw, gaps, tokens
    estimates, details, intervals = [], [], []
    captured = {}
    original = model.option_suffixes
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["drawn"], captured["elapsed"] = result
        return result
    model.option_suffixes = capture
    checked_pairs = 0
    started = time.time()
    for record in prefixes:
        rid = record["id"]
        if rid not in targets and rid != 102:
            continue
        point = model.points[record["point"]]
        observed = costs.state(record["events"])
        sunk = observed.summary(record["now"])
        cap = 1+max(model.encoder_decoder.window_size-model.encoder_decoder.min_suffix_size-len(record["events"]), 1)
        for repeat in range(5 if rid == 102 else 3):
            torch.manual_seed(200000+rid*100+repeat)
            scores = -adapter.outcome_samples(point, record["events"], record["options"], record["now"], 500)
            drawn, elapsed = captured["drawn"], captured["elapsed"]
            same_first = np.array_equal([s[0][2] for s in drawn[0]], [s[0][2] for s in drawn[1]])
            paired_se = (scores[0]-scores[1]).std(ddof=1)/np.sqrt(500)
            for oi, (option, suffixes) in enumerate(zip(record["options"], drawn)):
                estimates.append({"id": rid, "case": record["case"], "point": record["point"], "option": option,
                    "repeat": repeat, "random_sample": rid in targets, "mean_cost": scores[oi].mean(),
                    "median_cost": np.median(scores[oi]), "sd_cost": scores[oi].std(ddof=1), "paired_se": paired_se,
                    "same_first_gaps_for_both_options": bool(same_first)})
                if repeat == 0:
                    with (OUT / f"suffixes_{rid:03d}_{oi}.pkl").open("wb") as stream:
                        pickle.dump(suffixes, stream)
                for k, suffix in enumerate(suffixes):
                    state = observed.copy()
                    at = max(pd.Timestamp(record["events"][0][TS]).timestamp()+elapsed, record["now"], state.last)
                    previous = record["events"][-1][ACT]+"|"+record["events"][-1][LC]
                    unseen = 0
                    max_work = 0.
                    for step, (token, _, gap) in enumerate(suffix):
                        assert (previous, token) in pairs, (rid, previous, token)
                        checked_pairs += 1
                        previous = token
                        a, _, lc = token.rpartition("|")
                        at += gap
                        unseen += (token, int(np.searchsorted(EDGES, gap, side="right"))) not in seen
                        kind = None
                        if a in state.opened:
                            if lc in ("start", "resume"):
                                kind = "reopen"
                            elif lc in ("suspend", "complete", "ate_abort", "withdraw"):
                                kind = "close"
                            if kind:
                                hours = (at-state.opened[a])/3600
                                max_work = max(max_work, hours)
                                if repeat == 0:
                                    intervals.append({"id": rid, "option": oi, "draw": k, "random_sample": rid in targets,
                                        "step": step, "activity": a, "kind": kind, "hours": hours,
                                        "cost": hours*costs.tariff(a)["eur_per_work_hour"]})
                        state.push(a, lc, at)
                    for a, began in state.opened.items():
                        hours = (at-began)/3600
                        max_work = max(max_work, hours)
                        if repeat == 0:
                            intervals.append({"id": rid, "option": oi, "draw": k, "random_sample": rid in targets,
                                "step": len(suffix), "activity": a, "kind": "unclosed", "hours": hours,
                                "cost": hours*costs.tariff(a)["eur_per_work_hour"]})
                    result = state.summary(at)
                    assert np.isclose(max(result["total_eur"]-sunk["total_eur"], 0), scores[oi, k])
                    row = {"id": rid, "case": record["case"], "point": record["point"], "option": option,
                        "repeat": repeat, "draw": k, "random_sample": rid in targets, "events": len(suffix),
                        "days": sum(s[2] for s in suffix)/86400, "first_gap_seconds": suffix[0][2],
                        "missing_business_end": not any(e[ACT] in ENDS for e in record["events"])
                            and not any(s[0].split("|")[0] in ENDS for s in suffix),
                        "open_at_end": len(state.opened), "at_cap": len(suffix)>=cap,
                        "new_reopens": state.reopened_segments-observed.reopened_segments,
                        "new_orphans": state.orphan_closures-observed.orphan_closures,
                        "unseen_event_classes": unseen, "max_work_hours": max_work}
                    for key in ("total_eur", "base_eur", "lifecycle_eur", "time_eur", "active_hours"):
                        row[key] = result[key]-sunk[key]
                    details.append(row)
            pd.DataFrame(estimates).to_csv(OUT / "estimates.csv", index=False)
            pd.DataFrame(details).to_csv(OUT / "draws.csv", index=False)
            pd.DataFrame(intervals).to_csv(OUT / "work_intervals.csv", index=False)
            write("status.json", {"status": "sampling", "last_id": rid, "last_repeat": repeat,
                "draws_completed": len(details), "checked_transitions": checked_pairs, "elapsed_seconds": time.time()-started})
            print("OE_SAMPLES", rid, repeat, "elapsed", round(time.time()-started), flush=True)
    write("sampling_validation.json", {"status": "passed", "random_suffixes": 36000, "extra_suffixes": 5000,
        "costs_match_outcome_adapter": True, "checked_transitions": checked_pairs, "unobserved_transitions": 0})
    summarize()


def summarize():
    oe_all = pd.read_csv(OUT / "draws.csv")
    oe = oe_all[oe_all.random_sample]
    d = pd.read_csv(BASE / "psp_draws.csv")
    assert len(d) == len(oe) == 36000
    branches = pd.read_csv(BASE / "simulator_branches.csv")
    comparison = []
    for name, frame in (("D", d), ("OE", oe)):
        comparison.append({"model": name, "remaining_cost_eur": frame.total_eur.mean(), "median_cost_eur": frame.total_eur.median(),
            "active_hours": frame.active_hours.mean(), "remaining_days": frame.days.mean(), "events": frame.events.mean(),
            "missing_business_end": frame.missing_business_end.mean(), "open_session_at_end": frame.open_at_end.gt(0).mean(),
            "reopens": frame.new_reopens.mean(), "at_cap": frame.at_cap.mean()})
    table = pd.DataFrame(comparison).set_index("model")
    table.to_csv(OUT / "comparison.csv")
    means = oe.groupby(["id", "point", "option"]).agg(oe_cost=("total_eur", "mean"), oe_days=("days", "mean"),
                          oe_first_gap=("first_gap_seconds", "mean"))
    means = means.join(d.groupby(["id", "point", "option"]).agg(d_cost=("total_eur", "mean"), d_days=("days", "mean"),
                          d_first_gap=("first_gap_seconds", "mean")))
    means = means.join(branches.groupby(["id", "point", "option"]).agg(sim_cost=("total_eur", "mean"),
                          sim_days=("remaining_days", "mean")))
    means.to_csv(OUT / "by_option.csv")
    noisy = []
    for name, frame in (("D", d), ("OE", oe)):
        for objective in ("total_eur", "days"):
            rows = []
            for (rid, repeat), group in frame.groupby(["id", "repeat"]):
                wide = group.pivot(index="draw", columns="option", values=objective)
                delta = wide.iloc[:, 0]-wide.iloc[:, 1]
                se = delta.std(ddof=1)/np.sqrt(len(delta))
                rows.append({"id": rid, "repeat": repeat, "diff": delta.mean(), "se": se})
            noise = pd.DataFrame(rows)
            noisy.append({"model": name, "objective": objective, "median_paired_se": noise.se.median(),
                "repeat_choice_flip_rate": noise.groupby("id")["diff"].agg(lambda x: x.min()<0<x.max()).mean()})
    pd.DataFrame(noisy).to_csv(OUT / "uncertainty.csv", index=False)
    estimates = pd.read_csv(OUT / "estimates.csv")
    prefix = {r["id"]: r for r in pickle.loads((BASE / "prefixes.pkl").read_bytes())}
    lines = ["# Zieht OE besser als D?", "", "Verglichen wurden die vorhandenen Modelle auf denselben zwölf "
        "Entscheidungszuständen des früheren D-Laufs: je drei verschiedene Fälle pro Wahlpunkt, "
        "dreimal 500 Suffixe je Option, also je Modell 36.000 Suffixe. "
        "Gleiche Tarife, Testfälle, Wahlpunkte und Übergangssperre. "
        "Zusätzlich: Fall 750, fünfmal 500 Suffixe je Option. "
        "Der Simulatorvergleich verwendet die bereits geprüften 72 Fortsetzungen "
        "(eine erzwungene Option, anschließend Bankverhalten). Kein neuer Online-Benchmark.", "",
        "| Kennzahl | D | OE |", "|---|---:|---:|"]
    labels = {"remaining_cost_eur": "Verbleibende Kosten (€)", "active_hours": "Aktive Arbeitsstunden",
        "remaining_days": "Verbleibende Tage", "events": "Suffix-Events", "missing_business_end": "Ohne geschäftlichen Endmarker (%)",
        "open_session_at_end": "Offene Arbeitssitzung am Ende (%)", "reopens": "Erneute Beginne bei offener Sitzung je Suffix",
        "at_cap": "An der Längengrenze (%)"}
    for col, label in labels.items():
        scale = 100 if col in ("missing_business_end", "open_session_at_end", "at_cap") else 1
        lines.append(f"| {label} | {table.loc['D', col]*scale:.3f} | {table.loc['OE', col]*scale:.3f} |")
    lines += ["", f"Simulator-Fortsetzungen derselben Zustände: {branches.total_eur.mean():.2f} € "
        f"und {branches.remaining_days.mean():.2f} Tage im Mittel. Nur drei Fortsetzungen pro Option; "
        "für individuelle Optionsrangfolgen ist diese Stichprobe klein.", "",
        f"OE: identische erste Optionsabstände in "
        f"{int(estimates[estimates.random_sample].groupby(['id','repeat']).same_first_gaps_for_both_options.first().sum())} "
        "von 36 Ziehungsaufrufen. Bei D waren es 36 von 36.", "",
        f"OE: {oe.unseen_event_classes.sum()/oe.events.sum()*100:.4f} % der gezogenen Events haben eine "
        "Kombination aus Event und aus dem tatsächlichen Abstand abgeleiteter Zeitklasse, die im gesamten "
        "echten Log nicht vorkommt. Bei D: 8,06 % in der ersten Wiederholung. "
        "OE zieht ein gemeinsames Token für Event und Klasse; Randwert-Rundungen können beim "
        "nachträglichen Ableiten der Klasse aus Sekunden Unterschiede erzeugen.", "", "## Fall 750", ""]
    special = oe_all[oe_all.id.eq(102)]
    for option, group in special.groupby("option"):
        lines.append(f"- {option}: {group.total_eur.mean():.2f} €; erster Abstand im Mittel "
                     f"{group.first_gap_seconds.mean()/3600:.3f} Stunden.")
    wide = special.groupby(["repeat", "option"]).total_eur.mean().unstack()
    starts = int((wide['W_Complete application|start']<wide['A_Accepted']).sum())
    lines += ["", f"OE wählt in {starts} von fünf Wiederholungen den erneuten Start. "
        "D wählte ihn in einer von fünf neuen Wiederholungen und im ursprünglichen 500er-Lauf. "
        "Die fehlende Sitzungsprüfung des Simulators wird durch den Modellwechsel nicht behoben.", "",
        "## Modellunterschiede", "", json.dumps(read(OUT / 'manifest.json')['parameter_differences'], indent=2, ensure_ascii=False), "",
        "Die Wiederholungen verwenden identische Start-Seeds je Zustand. Wegen unterschiedlicher "
        "Modellausgaben sind einzelne Suffixe zwischen D und OE keine gepaarten identischen Zufallspfade. "
        "Gemeinsame Zufallszahlen zwischen den Optionen werden innerhalb jedes Modells verwendet.", "",
        "Artefakte: comparison.csv, by_option.csv, draws.csv, estimates.csv, work_intervals.csv, uncertainty.csv, "
        "sampling_validation.json und manifest.json. Die Originalmodelle und laufenden Trainings wurden nicht verändert.", ""]
    (OUT / "auswertung.md").write_text("\n".join(lines))
    write("status.json", {"status": "complete", "oe_suffixes": len(oe_all), "main_comparison_suffixes_per_model": 36000})
    print(table.to_string(), flush=True)


if __name__ == "__main__":
    if "--summarize" in sys.argv:
        summarize()
    else:
        evaluate()

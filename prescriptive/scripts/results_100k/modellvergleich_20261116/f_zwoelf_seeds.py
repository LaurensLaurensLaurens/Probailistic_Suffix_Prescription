"""Aufbau F auf 12 Seeds erweitern (2026-11-17, Wunsch des Nutzers): genauere gepaarte Vergleiche (SE ≈ 0,2 statt 0,4 Tage).
Hintergrund: die zufällige Mitarbeiterzuteilung zieht aus einem globalen Strom; sobald eine Wahl abweicht, laufen die Läufe auseinander
(befunde.md 17.11. 18:35), je Seed schwankt der Vergleich um ~1 Tag. Seeds 0-2 gibt es schon (gleicher Aufbau, werden in
f_zwoelf_auswerten.py wiederverwendet); hier laufen die fehlenden Seeds.

Gleiches Gerüst wie cf_im_korrigierten_simulator.py / c_first_only_im_korrigierten_simulator.py: je Seed 200 bewertete Fälle (700 Warmup,
1.300 Cooldown), Katalog d_oe_zustand_20261115/catalog.yaml (zufällige Zuteilung), Ziel Durchlaufzeit. Arme:
  bank           keine Policy (Referenz; für Seeds 0-2 liegt sie in d_oe_zustand_20261115/bank_reference_seed*)
  regel          feste AIPW-Regel (policies/rule_policy.py; der Causal Forest wählt in F fallweise identisch)
  c_fest100      PSP (c), 100 Ziehungen fest, first_only = angebot_nachlegen
  c_fest300      PSP (c), 300 Ziehungen fest, first_only = angebot_nachlegen
  zztf1_fest100  PSP ZZ-TF1 (fertiges Modell, 15 Epochen), 100 Ziehungen fest, first_only = angebot_nachlegen
  tl_fest300     T-Learner-PSP (Aktion/Referenz, warm vom ZZ-TF1), 300 Ziehungen fest, first_only = angebot_nachlegen
  tl_fest500     wie tl_fest300 mit 500 Ziehungen (18.11.: die Optionen des T-Learners sind kaum gekoppelt, ρ ≈ 0,1)
  tlbnll_fest300 T-Learner-PSP warm vom β-NLL-Modell (ZZU-TF1 + β-NLL 0,5; 18.11.), 300 Ziehungen fest, first_only
  tlks_fest300   T-Learner mit Zeitkopf je Event (Versatz + Steigung, nur Kopf nachtrainiert; 18.11.), 300 Ziehungen fest, first_only
  tlko_fest300   wie tlks_fest300, nur Versatz je Event
  tlkb_fest300   wie tlks_fest300, Zeitköpfe mit β-NLL β = 1 nachtrainiert (19.11.)
  tlcrn_fest300  wie tl_fest300, Zufallszahlen je Ereignis statt je Schritt (params keyed_noise; 19.11.)
  tlsv_fest300   wie tlcrn_fest300, beide Arme mit Suffix-Verlust nachtrainiert (sequence_loss; 19.11.)
  tlsvk_fest300  wie tlsv_fest300, zusätzlich Kalibrierung je (Wahlpunkt, Option) im Suffix-Verlust (19.11. abends)
  tlsvp_fest300  wie tlsvk_fest300, Kalibrierung gepaart (dieselben Prefixe, zwei Ziehungshälften; 20.11.)
  tlsvh_fest300  wie tlsvk_fest300, Kalibrierung mit disjunkten Prefix-Hälften gegen das Gruppenmittel (20.11.)
  olsv_fest300   ein PSP je Wahlpunkt und Option (8 Netze, warm vom ZZ-TF1) mit Suffix-Verlust, ohne Kalibrierung, keyed_noise (22.11.)
  plsv_fest300   S-Learner je Wahlpunkt (4 Netze, beide Optionen, warm vom ZZ-TF1) mit Suffix-Verlust, ohne Kalibrierung (22.11.)
  olsvcf_fest300 wie olsv_fest300, aber bei nicht unterscheidbaren Optionen (|Δ| < 1,96 · SE) entscheidet der Causal Forest (psp_cf_policy.py; 23.11.)
Ergebnis je Arm und Seed: f_zwoelf_seeds/<arm>_seed<s>/ (case_costs.csv, choices.csv, events.pkl, result.json), je Aufruf runs_<zeit>.csv.
Quellen-Hashes werden am Anfang und Ende festgehalten (Abweichungen nur gemeldet, kein Abbruch).
Aufruf im Ordner process-simulation-engine:
  .venv/bin/python -u <dieses Skript> --arms c_fest300,c_fest100,regel,bank --seeds 3-11 --workers 6
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import warnings
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
MODELS = {"c": ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl",
          "zztf1": ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1.pkl",
          "tl": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_zztf1.pkl",  # T-Learner-Hülle (t_learner_policy.py)
          "zzrct": ROOT/"prescriptive/models/psp_model_durchlaufzeit_rct_zaehler_zeit_tf1.pkl",  # ZZ-TF1 auf Simulator-RCT-Daten (18.11.)
          "tlrct": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_rct.pkl",  # T-Learner auf Simulator-RCT-Daten
          "tlbnll": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_bnll05.pkl",  # T-Learner warm vom β-NLL-Modell (18.11.)
          "tlks": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_kopf_steigung.pkl",  # Zeitkopf je Event, Steigung (18.11.)
          "tlko": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_kopf_versatz.pkl",  # Zeitkopf je Event, nur Versatz
          "tlkb": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_kopf_steigung_bnll1.pkl",  # Zeitkopf je Event, β-NLL 1 (19.11.)
          "tlcrn": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_zztf1_crn.pkl",  # T-Learner, keyed_noise (19.11.)
          "tlsv": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_sv_crn.pkl",  # T-Learner + Suffix-Verlust, keyed_noise (19.11.)
          "tlsvk": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_svk_crn.pkl",  # + Kalibrierung je Gruppe, keyed_noise
          "tlsvp": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_svp_crn.pkl",  # + gepaarte Kalibrierung, keyed_noise
          "tlsvh": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_svh_crn.pkl",  # + Kalibrierung Hälften, keyed_noise
          "olsv": ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_sv_crn.pkl",  # Netz je (Punkt, Option), Suffix-Verlust
          "plsv": ROOT/"prescriptive/models/psp_model_durchlaufzeit_punkt_learner_sv_crn.pkl",  # S-Learner je Wahlpunkt, Suffix-Verlust
          "olsvcf": ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_sv_crn.pkl"}  # olsv + Causal Forest bei Unsicherheit (23.11.)
FOREST = ROOT/"prescriptive/models/cate_model_cf_durchlaufzeit_real_alle.pkl"  # Causal Forest (12.11., echter Log) für die Arme <modell>cf
OUT = Path(__file__).resolve().parent/"f_zwoelf_seeds"
FIRST_ONLY = ("angebot_nachlegen",)
COST = {"tl_fest500": 110, "tlbnll_fest300": 40, "tlks_fest300": 40, "tlko_fest300": 40, "tlkb_fest300": 40, "tlcrn_fest300": 45, "tlsv_fest300": 45, "tlsvk_fest300": 45, "tlsvp_fest300": 45, "tlsvh_fest300": 45, "olsv_fest300": 45, "plsv_fest300": 45, "olsvcf_fest300": 47, "tlrct_fest300": 40, "zzrct_fest100": 15, "tl_fest300": 40, "c_fest300": 35, "zztf1_fest100": 15, "c_fest100": 12, "regel": 3, "bank": 3}  # Minuten je Job (Reihenfolge: teuerste zuerst)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import pandas as pd  # noqa: E402

SCORED = 200


def now():
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+"\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def simulator():
    import interventions
    interventions.CANDIDATES_FILE = str(SOURCE/"catalog.yaml")
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    raw, _, _ = load_dataset("bpic17")
    return Simulator("bpic17", preprocess("bpic17", raw))


def score(events):
    from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
    numbers = events["case:concept:name"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    events = events.loc[numbers.ge(700) & numbers.lt(700+SCORED)]
    bills = EventCosts(SOURCE/"event_costs.yaml").case_costs(events)
    times = events.groupby("case:concept:name")["time:timestamp"]
    bills["cycle_days"] = (times.max()-times.min()).dt.total_seconds()/86400
    assert len(bills) == SCORED
    return events, bills


def bank_check(seed=0):
    """Derselbe Simulator wie am 15.11.? Die Bank von Seed 0 muss die gespeicherte Referenz je Fall reproduzieren."""
    events, _ = simulator().run(700+SCORED+1300, policy=None, window=(700, 700+SCORED), seed=seed)
    _, bills = score(events)
    reference = pd.read_csv(SOURCE/f"bank_reference_seed{seed}/case_costs.csv", index_col=0)
    difference = float((bills.cycle_days-reference.cycle_days.reindex(bills.index)).abs().max())
    return {"seed": seed, "max_abs_case_difference_days": difference, "reproduced": bool(difference < 1e-6)}


def make_policy(arm, seed):
    if arm == "bank":
        return None
    if arm == "regel":
        from prescriptive.policies.rule_policy import RulePolicy
        return RulePolicy()
    from prescriptive.policies.ev_policy.psp_policy import PSPPolicy
    model, draws = arm.split("_fest")
    if model.endswith("cf"):  # PSP + Causal Forest bei nicht unterscheidbaren Optionen (23.11.)
        from prescriptive.policies.ev_policy.psp_cf_policy import PSPCausalForestPolicy
        policy = PSPCausalForestPolicy(str(MODELS[model]), str(FOREST), seed=seed, samples=int(draws), first_only=FIRST_ONLY)
    else:
        policy = PSPPolicy(model_path=str(MODELS[model]), seed=seed, samples=int(draws), first_only=FIRST_ONLY)
    assert policy.first_only == set(FIRST_ONLY) and "sampling_control" not in policy.model.params
    assert policy.model._outcome_kind() == "negative_cycle_days" and policy.model.value_source() == "suffix"
    return policy


def job(task):
    import torch
    torch.set_num_threads(1)
    arm, seed = task
    policy = make_policy(arm, seed)
    folder = OUT/f"{arm}_seed{seed}"
    folder.mkdir(exist_ok=False)
    started = time.time()
    progress = {"status": "running", "arm": arm, "seed": seed, "started_at": now(), "decisions": 0}
    write(folder/"progress.json", progress)
    if policy is not None:
        original = policy.choose

        def choose(point, case, options, at, engine):
            selected = original(point, case, options, at, engine)
            progress["decisions"] += 1
            if progress["decisions"] == 1 or progress["decisions"] % 10 == 0:
                progress.update(updated_at=now(), elapsed_seconds=time.time()-started)
                write(folder/"progress.json", progress)
            return selected
        policy.choose = choose
    events, choices = simulator().run(700+SCORED+1300, policy=policy, window=(700, 700+SCORED), seed=seed)
    events, bills = score(events)
    result = {"arm": arm, "seed": seed, "cases": len(bills), "cycle_days": float(bills.cycle_days.mean()),
              "labor_eur": float(bills.time_eur.mean()), "cost_eur": float(bills.total_eur.mean()),
              "decisions": progress["decisions"], "sampling_fallbacks": getattr(policy, "sampling_fallbacks", 0),
              "seconds": time.time()-started}
    bills.to_csv(folder/"case_costs.csv")
    choices.to_csv(folder/"choices.csv", index=False)
    if getattr(policy, "entscheidungen", None):  # psp_cf: je Entscheidung Δ, SE, unsicher, PSP-/Forest-Wahl
        pd.DataFrame(policy.entscheidungen).to_csv(folder/"entscheidungen.csv", index=False)
        e = pd.DataFrame(policy.entscheidungen)
        result.update(unsicher=int(e.unsicher.sum()), forest_entscheidet=int(e.forest.notna().sum()), forest_anders=int((e.forest.notna() & (e.forest != e.psp)).sum()))
    events.to_pickle(folder/"events.pkl")
    write(folder/"result.json", result)
    write(folder/"progress.json", {**progress, "status": "complete", "updated_at": now()})
    print("COMPLETE", json.dumps(result), flush=True)
    return result


def seeds_from(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a)+1))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    arms, seeds = args.arms.split(","), seeds_from(args.seeds)
    assert set(arms) <= set(COST), arms
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(OUT/"matplotlib")
    stamp = time.strftime("%Y%m%d_%H%M")
    files = [Path(__file__), SOURCE/"catalog.yaml", SOURCE/"event_costs.yaml", ROOT/"prescriptive/policies/ev_policy/psp_policy.py",
             ROOT/"prescriptive/policies/rule_policy.py", ROOT/"prescriptive/estimators/psp_model.py",
             ROOT/"process-simulation-engine/simulation/dataset_simulator.py",
             ROOT/"process-simulation-engine/simulation/case_engine_lifecycle_aware.py",
             ROOT/"process-simulation-engine/interventions/work_state.py",
             ROOT/"process-simulation-engine/resources/selection_strategies.py"]
    files += [MODELS[a.split("_fest")[0]] for a in arms if "_fest" in a]
    files += [FOREST, ROOT/"prescriptive/policies/ev_policy/psp_cf_policy.py"] if any(a.split("_fest")[0].endswith("cf") for a in arms) else []
    hashes = {str(path): sha(path) for path in files}
    check = bank_check(0)
    print("BANK CHECK", json.dumps(check), flush=True)
    write(OUT/f"manifest_{stamp}.json", {"prepared_at": now(), "arms": arms, "seeds": seeds, "scored_cases": SCORED, "warmup": 700,
                                          "cooldown": 1300, "allocation": "random", "first_only": list(FIRST_ONLY),
                                          "models": {k: str(v) for k, v in MODELS.items()}, "bank_check": check, "source_sha256": hashes})
    if not check["reproduced"]:
        raise RuntimeError("Die Bankreferenz vom 15.11. wird nicht reproduziert; kein gepaarter Vergleich möglich")
    tasks = sorted([(arm, seed) for arm in arms for seed in seeds], key=lambda t: -COST[t[0]])
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(job, task) for task in tasks]
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:  # ein Fehler soll die anderen Jobs nicht stoppen
                print("FEHLER", repr(error), flush=True)
            if rows:  # je Aufruf eine eigene Datei (zwei Aufrufe können sich überlappen); die Auswertung liest die result.json
                pd.DataFrame(rows).sort_values(["arm", "seed"]).to_csv(OUT/f"runs_{stamp}.csv", index=False)
    changed = [path for path, expected in hashes.items() if sha(path) != expected]
    write(OUT/f"manifest_{stamp}_ende.json", {"finished_at": now(), "changed_sources": changed})
    print("ALLES FERTIG", now(), "geänderte Quellen:", changed or "keine", flush=True)


if __name__ == "__main__":
    main()

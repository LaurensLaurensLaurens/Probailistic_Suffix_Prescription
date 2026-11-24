"""Online-Auswertung Conformance im Simulator mit künstlichen Effekten v3 (2026-11-20, Nutzer: „okay mach das“ – prüfen, ob das PSP
Conformance-Abweichungen verhindern kann; Zielgröße separat, eigene Policy policies/konformanz_policy.py).

Gerüst wie Aufbau F (modellvergleich_20261116/f_zwoelf_seeds.py): je Seed 700 Einlauf-, 200 bewertete und 1.300 Auslauffälle, Katalog
d_oe_zustand_20261115 (zufällige Zuteilung), die Policy entscheidet nur für die bewerteten Fälle. Dazu die Effekte
kuenstliche_effekte/effekte_v3.yaml in jedem Lauf. Arme:
  bank        keine Policy
  regel       feste Regel auf Durchlaufzeit (policies/rule_policy.py)
  orakel      kennt die Teilgruppen der Effekte und wählt je Teilgruppe die Option mit weniger Abweichungen laut wahrheit_effekte_v3.csv
              (andere − Regel-Option < 0 → andere Option, sonst Regel-Option; nachlegen nur 1. Besuch, sonst Bank)
  konformanz  KonformanzPolicy mit dem T-Learner auf dem v3-Log (models/psp_model_durchlaufzeit_t_learner_k3.pkl), 100 Ziehungen,
              first_only angebot_nachlegen
  konformanz_x4  wie konformanz, Arme auf dem 4-fachen v3-Log (128.000 Fälle; t_learner_k3x4.pkl; 2026-11-21)
  konformanz_ol  wie konformanz_x4, aber ein Netz je Wahlpunkt und Option (Option-Learner, option_learner_k3x4.pkl; 2026-11-22)
  konformanz_ol300  Kontrolle 2026-11-23: dieselben Netze wie konformanz_ol, aber keyed_noise und 300 Ziehungen
  konformanz_sim  2026-11-23: Option-Learner mit Basis nur auf dem Simulator (ZZ-TF1 auf log_effekte_v3), keyed_noise, 300 Ziehungen
  konformanz_ol_ro  2026-11-23: wie konformanz_ol300, dazu Rollout: an späteren Wahlpunkten in den gezogenen Suffixen entscheidet der
                  Conformance-Forest (policies/ev_policy/rollout.py, models/cate_model_cf_konformanz_simv3.pkl)
  konformanz_sd  2026-11-24: wie konformanz_ol300, aber die 8 Netze mit statischen Merkmalen in jedem Decoder-Schritt (psp_model
                  static_in_decoder, warm von den k3x4ol-Netzen; data/bpic17_psp_statik_decoder.sh), keyed_noise, 300 Ziehungen
  cf_konformanz  2026-11-24 (Nutzer: „teste parallel mal, wie gut der Causal Forest bei der Konformitätsaufgabe performt“): nur der
                  Conformance-Forest (models/cate_model_cf_konformanz_simv3.pkl, Outcome −Abweichungen, log_effekte_v3 1-fach) entscheidet
                  (policies/ev_policy/cate_policy.py, Schwelle 0), first_only angebot_nachlegen; kein PSP
  cf_rueckwaerts  2026-11-24: je Punkt ein Forest aus der Rückwärtsinduktion (rueckwaerts_cf_20261124/rueckwaerts.py: Stufenpunkt im
                  Simulator zufällig, spätere Punkte ihr Forest), policies/ev_policy/rueckwaerts_forest.PunktForestPolicy
Zielgrößen je bewertetem Fall: Conformance-Abweichungen gegen BPIC17_ToBe_angepasst (Alignment, conformance_probe.py), konform
ja/nein, Durchlaufzeit. Ausgabe online_konformanz/<arm>_seed<s>/ (events.pkl, choices.csv, faelle.csv, result.json, ggf.
entscheidungen.csv) und online_konformanz.md (Mittel je Arm, gepaarte Differenzen über Seeds).
Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Arme, Komma] [Seeds a-b] [Prozesse]
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
OUT = HERE/os.environ.get("KONFORMANZ_OUT", "online_konformanz")
EFFEKTE = "kuenstliche_effekte/effekte_v3.yaml"
WAHRHEIT = HERE/"wahrheit_effekte_v3.csv"
MODELL = ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_k3.pkl"
# Arme mit KonformanzPolicy: Name → Hülle (2026-11-21: konformanz_x4 = gleiche Arme auf dem 4-fachen v3-Log)
KONFORMANZ_MODELLE = {"konformanz": MODELL, "konformanz_x4": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_k3x4.pkl",
                      "konformanz_x3": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_k3x3.pkl",
                      "konformanz_x2": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_k3x2.pkl",
                      "konformanz_ol": ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_k3x4_schlank.pkl"}  # 2026-11-22; ab 11-23 die schlanke Kopie (ohne mitgespeicherten Log, huelle_schlank.py: gleiche Entscheidungen; Original 10 GB je Prozess → OOM)
KONFORMANZ_MODELLE.update({  # 2026-11-23: Kontrolle (keyed_noise + 300) und Basis nur auf dem Simulator
    "konformanz_ol300": ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_k3x4_schlank_crn.pkl",
    "konformanz_sim": ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_simv3_schlank_crn.pkl"})
KONFORMANZ_MODELLE["konformanz_ol_ro"] = KONFORMANZ_MODELLE["konformanz_ol300"]
KONFORMANZ_MODELLE["konformanz_sd"] = ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_k3x4sd_schlank_crn.pkl"  # 2026-11-24
ZIEHUNGEN = {"konformanz_ol300": 300, "konformanz_sim": 300, "konformanz_ol_ro": 300, "konformanz_sd": 300}  # sonst 100
ROLLOUT = {"konformanz_ol_ro": ROOT/"prescriptive/models/cate_model_cf_konformanz_simv3.pkl"}  # Arm -> Forest für spätere Wahlpunkte
FOREST_ARME = {"cf_konformanz": ROOT/"prescriptive/models/cate_model_cf_konformanz_simv3.pkl"}  # 2026-11-24: nur der Forest entscheidet
# 2026-11-24: Rückwärtsinduktion (rueckwaerts_cf_20261124/rueckwaerts.py) – je Punkt ein eigener Forest, trainiert mit den späteren Forests
RUECKWAERTS_ARME = {"cf_rueckwaerts": {p: ROOT/f"prescriptive/models/cate_model_cf_konformanz_rw_{k}.pkl" for p, k in (
    ("angebot_nachlegen", "nachlegen"), ("annehmen_nach_vervollstaendigung", "nachv"), ("annehmen_bei_konzept", "konzept"),
    ("handle_leads", "leads"))}}
ARME = (sys.argv[1] if len(sys.argv) > 1 else "bank,regel,orakel,konformanz").split(",")
_s = sys.argv[2] if len(sys.argv) > 2 else "0-11"
SEEDS = list(range(int(_s.split("-")[0]), int(_s.split("-")[1]) + 1)) if "-" in _s else [int(x) for x in _s.split(",")]
PROZESSE = int(sys.argv[3]) if len(sys.argv) > 3 else 12
SCORED, WARM, COOL = 200, 700, 1300
FIRST_ONLY = ("angebot_nachlegen",)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def simulator():
    import interventions
    interventions.CANDIDATES_FILE = str(SOURCE/"catalog.yaml")
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    raw, _, _ = load_dataset("bpic17")
    return Simulator("bpic17", preprocess("bpic17", raw, verbose=False))


class Orakel:
    """Kennt die Teilgruppen der Effektregeln (wahrheit.gruppe) und die wahre Wirkung auf die Conformance je Teilgruppe."""

    # wie wahrheit.PUNKTE / wahrheit.gruppe (nicht importiert: wahrheit.py liest beim Import seine Kommandozeile)
    PUNKTE = {"annehmen_bei_konzept": ("A_Accepted", "W_Complete application|start", False),
              "annehmen_nach_vervollstaendigung": ("A_Accepted", "W_Complete application|resume", False),
              "angebot_nachlegen": ("O_Sent (mail and online)", "O_Create Offer", True),
              "handle_leads": ("W_Handle leads|withdraw", "W_Handle leads|start", False)}

    def gruppe(self, zeile, punkt):
        import kuenstliche_effekte as ke
        regeln = [r for regeln in ((self.effekte.get("punkte") or {}).get(punkt) or {}).values() for r in regeln]
        texte = sorted({", ".join(f"{k}={v}" for k, v in r.items() if k in ke.BEDINGUNGEN) for r in regeln
                        if ke.trifft_zu(r, zeile, zeile["dauer_tage"], 1)})
        return " | ".join(texte) if texte else "keine"

    def __init__(self):
        import kuenstliche_effekte as ke
        self.effekte = ke.laden(EFFEKTE)
        tabelle = pd.read_csv(WAHRHEIT)
        self.delta = {(r.Punkt, r.Gruppe): r["n Abweichungen andere−Regel"] for _, r in tabelle.iterrows()}

    def choose(self, point, case, options, time, engine):
        if point.name not in self.PUNKTE or len(options) < 2:
            return None
        regel, andere, erster = self.PUNKTE[point.name]
        if erster and sum(1 for e in case.prefix if e.get("concept:name") == point.activity
                          and e.get("lifecycle:transition", "complete") == point.lifecycle) > 1:
            return None
        from simulation.case_engine_lifecycle_aware import to_seconds
        dauer = (time - to_seconds(case.prefix[0]["time:timestamp"])) / 86400 if case.prefix else 0.0
        zeile = {"case:RequestedAmount": case.attributes.get("case:RequestedAmount"), "case:LoanGoal": case.attributes.get("case:LoanGoal"),
                 "case:ApplicationType": case.attributes.get("case:ApplicationType"), "dauer_tage": dauer}
        delta = self.delta.get((point.name, self.gruppe(zeile, point.name)))
        if delta is None or not np.isfinite(delta):
            return None
        wahl = andere if delta < 0 else regel
        return wahl if wahl in options else None


def make_policy(arm, seed):
    if arm == "bank":
        return None
    if arm == "regel":
        from prescriptive.policies.rule_policy import RulePolicy
        return RulePolicy()
    if arm == "orakel":
        return Orakel()
    if arm in FOREST_ARME:
        from prescriptive.policies.ev_policy.cate_policy import CATEPolicy
        return CATEPolicy(str(FOREST_ARME[arm]), seed=seed, threshold=0.0, first_only=FIRST_ONLY)
    if arm in RUECKWAERTS_ARME:
        from prescriptive.policies.ev_policy.rueckwaerts_forest import PunktForestPolicy
        return PunktForestPolicy(RUECKWAERTS_ARME[arm], seed=seed, first_only=FIRST_ONLY)
    if arm in KONFORMANZ_MODELLE:
        from prescriptive.policies.konformanz_policy import KonformanzPolicy
        policy = KonformanzPolicy(str(KONFORMANZ_MODELLE[arm]), seed=seed, samples=ZIEHUNGEN.get(arm, 100), first_only=FIRST_ONLY)
        if arm in ROLLOUT:
            from prescriptive.policies.ev_policy.rollout import ForestRollout
            ForestRollout.einschalten(policy.model, str(ROLLOUT[arm]), first_only=FIRST_ONLY)
        return policy
    raise ValueError(arm)


def bewerten(events):
    import conformance_probe as cp
    nummer = events["case:concept:name"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    events = events.loc[nummer.ge(WARM) & nummer.lt(WARM + SCORED)]
    zeiten = events.groupby("case:concept:name")["time:timestamp"]
    faelle = pd.DataFrame({"dlz_tage": (zeiten.max() - zeiten.min()).dt.total_seconds() / 86400})
    erg = cp.align(cp.projektion(events))
    faelle["abweichungen"] = pd.Series({c: cost // 10000 for c, _, cost, _ in erg})
    faelle["konform"] = faelle.abweichungen == 0
    assert len(faelle) == SCORED, len(faelle)
    return events, faelle


def job(task):
    import torch
    torch.set_num_threads(1)
    arm, seed = task
    import kuenstliche_effekte as ke
    ke.aktivieren(ke.laden(EFFEKTE))
    ordner = OUT/f"{arm}_seed{seed}"
    if (ordner/"result.json").exists():
        return json.loads((ordner/"result.json").read_text())
    ordner.mkdir(parents=True, exist_ok=True)
    start = time.time()
    policy = make_policy(arm, seed)
    events, choices = simulator().run(WARM + SCORED + COOL, policy=policy, window=(WARM, WARM + SCORED), seed=seed)
    events, faelle = bewerten(events)
    result = {"arm": arm, "seed": seed, "faelle": len(faelle), "abweichungen": float(faelle.abweichungen.mean()),
              "konform": float(faelle.konform.mean()), "dlz_tage": float(faelle.dlz_tage.mean()), "sekunden": time.time() - start,
              "fallbacks": getattr(policy, "sampling_fallbacks", 0)}
    events.to_pickle(ordner/"events.pkl")
    choices.to_csv(ordner/"choices.csv", index=False)
    faelle.to_csv(ordner/"faelle.csv")
    if getattr(policy, "entscheidungen", None):
        pd.DataFrame(policy.entscheidungen).to_csv(ordner/"entscheidungen.csv", index=False)
    pd.DataFrame(ke.LETZTE_ENGINE.effect_log).to_csv(ordner/"effekte.csv", index=False)
    (ordner/"result.json").write_text(json.dumps(result, indent=2))
    print("COMPLETE", json.dumps(result), flush=True)
    return result


def auswerten():
    zeilen = [json.loads(p.read_text()) for p in OUT.glob("*_seed*/result.json")]
    if not zeilen:
        return "keine Ergebnisse"
    d = pd.DataFrame(zeilen)
    mittel = d.groupby("arm").agg(seeds=("seed", "nunique"), abweichungen=("abweichungen", "mean"), konform=("konform", "mean"),
                                  dlz_tage=("dlz_tage", "mean"), stunden=("sekunden", lambda s: s.mean() / 3600)).round(3)
    breit = d.pivot(index="seed", columns="arm")
    paare = []
    for a, b in (("bank", "konformanz"), ("regel", "konformanz"), ("orakel", "konformanz"), ("bank", "orakel"), ("bank", "regel"),
                 *[(a, f"konformanz_x{n}") for n in (4, 3, 2) for a in ("bank", "regel", "orakel", "konformanz")],
                 *[(a, "konformanz_ol") for a in ("bank", "regel", "orakel", "konformanz_x4")],
                 *[(a, "konformanz_ol300") for a in ("bank", "regel", "orakel", "konformanz_ol")],
                 *[(a, "konformanz_sim") for a in ("bank", "regel", "orakel", "konformanz_ol", "konformanz_ol300")],
                 *[(a, "konformanz_ol_ro") for a in ("bank", "regel", "orakel", "konformanz_ol300")],
                 *[(a, "konformanz_sd") for a in ("bank", "regel", "orakel", "konformanz_ol300")],
                 *[(a, "cf_konformanz") for a in ("bank", "regel", "orakel", "konformanz_ol300")],
                 *[(a, "cf_rueckwaerts") for a in ("bank", "regel", "orakel", "konformanz_ol300", "cf_konformanz")]):
        if a in d.arm.values and b in d.arm.values:
            for groesse in ("abweichungen", "konform", "dlz_tage"):
                diff = (breit[groesse][a] - breit[groesse][b]).dropna()
                if len(diff) > 1:
                    paare.append({"vergleich": f"{a} − {b}", "größe": groesse, "seeds": len(diff), "mittel": diff.mean(),
                                  "se": diff.std(ddof=1) / np.sqrt(len(diff))})
    paare = pd.DataFrame(paare).round(3)
    text = (f"# Online Conformance, Simulator mit Effekten v3 (Aufbau F, {SCORED} bewertete Fälle je Seed)\n\n"
            f"Abweichungen je Fall gegen BPIC17_ToBe_angepasst; konform = Anteil Fälle ohne Abweichung.\n\n```\n{mittel.to_string()}\n```\n\n"
            f"Gepaart je Seed (erste minus zweite; bei abweichungen positiv = zweite besser):\n\n```\n{paare.to_string(index=False)}\n```\n")
    (HERE/"online_konformanz.md").write_text(text)
    return text


def main():
    OUT.mkdir(exist_ok=True)
    tasks = [(arm, seed) for arm in ARME for seed in SEEDS]
    # die teuren Arme zuerst, damit die Prozesse gleichmäßig ausgelastet sind
    tasks.sort(key=lambda t: not t[0].startswith("konformanz"))
    with ProcessPoolExecutor(PROZESSE, mp_context=multiprocessing.get_context("spawn")) as pool:
        for _ in pool.map(job, tasks):
            pass
    print(auswerten())
    print("FERTIG", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "auswerten":
        print(auswerten())
    else:
        main()

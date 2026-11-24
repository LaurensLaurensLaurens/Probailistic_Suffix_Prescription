"""Prüfungen für den Rollout mit Causal Forest (2026-11-23).
  referenz   vor der Codeänderung: Ziehungen der Hülle k3x4ol (schlank, keyed_noise) an 8 Entscheidungszuständen (je Punkt 2, erste Besuche
             im Bank-Lauf online_konformanz/bank_seed0) mit fester Saat speichern → referenz.pkl
  vergleich  nach der Änderung, ohne Rollout: dieselben Ziehungen müssen bitgenau gleich sein
  rollout    mit Rollout (Forest models/cate_model_cf_konformanz_simv3.pkl): nach jedem Auslöser eines Wahlpunkts im gezogenen Suffix muss das
             nächste Event die Wahl des Forests sein (nachlegen nur beim 1. Besuch); Zählung der Eingriffe
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> referenz|vergleich|rollout
"""
import os
from pathlib import Path
import pickle
import sys
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
K = ROOT/"prescriptive/scripts/results_100k/kuenstliche_effekte_20261120"
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
HUELLE = ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_k3x4_schlank_crn.pkl"
FOREST = ROOT/"prescriptive/models/cate_model_cf_konformanz_simv3.pkl"
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import pandas as pd  # noqa: E402

CASE, ACT, LC, TIME = "case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"


def zustaende(je_punkt=2):
    import interventions
    from interventions import option_token
    interventions.CANDIDATES_FILE = str(SOURCE/"catalog.yaml")
    punkte = {p.name: p for p in interventions.choice_points("bpic17")}
    e = pd.read_pickle(K/"online_konformanz/bank_seed0/events.pkl").sort_values([CASE, TIME], kind="stable")
    out, zahl = [], {n: 0 for n in punkte}
    for case, t in e.groupby(CASE, sort=False):
        t = t.reset_index(drop=True)
        toks = [f"{a}|{l}" for a, l in zip(t[ACT], t[LC])]
        for name, p in punkte.items():
            if zahl[name] >= je_punkt:
                continue
            ausloeser = f"{p.activity}|{p.lifecycle}"
            if ausloeser in toks[:-1]:
                i = toks.index(ausloeser)
                out.append((p, t.iloc[:i + 1].to_dict("records"), list(p.options), pd.Timestamp(t.loc[i, TIME]).timestamp()))
                zahl[name] += 1
        if all(v >= je_punkt for v in zahl.values()):
            break
    return out


def ziehen(model, n=20):
    import torch
    erg = []
    for j, (p, prefix, options, jetzt) in enumerate(zustaende()):
        torch.manual_seed(123 + j)
        drawn, elapsed = model.option_suffixes(model.points[p.name], prefix, options, jetzt, n)
        erg.append((p.name, drawn, elapsed))
    return erg


def main():
    from prescriptive.estimators.psp_model import PSPModel
    modus = sys.argv[1]
    model = PSPModel.load(str(HUELLE))
    model.set_threads(4)
    if modus == "referenz":
        erg = ziehen(model)
        (HERE/"referenz.pkl").write_bytes(pickle.dumps(erg))
        print(f"Referenz gespeichert: {len(erg)} Zustände, {sum(len(s) for _, d, _ in erg for o in d for s in o)} Suffixe")
    elif modus == "vergleich":
        ref = pickle.loads((HERE/"referenz.pkl").read_bytes())
        erg = ziehen(model)
        gleich = ref == erg
        print("GLEICH (bitgenau)" if gleich else "VERSCHIEDEN")
        sys.exit(0 if gleich else 1)
    elif modus == "rollout":
        from prescriptive.policies.ev_policy.rollout import ForestRollout
        ForestRollout.einschalten(model, str(FOREST), first_only=("angebot_nachlegen",))
        erg = ziehen(model)
        ro = model.rollout
        print(f"Forest-Eingriffe: {ro.eingriffe} (je Punkt {dict(ro.je_punkt)}), Optionen {dict(ro.wahlen)}")
        # Prüfung: nach jedem Auslöser (1. Besuch bei first_only) steht als nächstes eine Option des Punkts
        ausloeser = {f"{p.activity}|{p.lifecycle}": p for p in model.points.values()}
        fehler = 0
        for name, drawn, _ in erg:
            for suffixes in drawn:
                for suffix in suffixes:
                    toks = [t for t, _, _ in suffix]
                    for i, tk in enumerate(toks[:-1]):
                        if tk in ausloeser and ausloeser[tk].name not in ro.first_only:  # erste Besuche bei first_only: Prefix nötig, hier nicht geprüft
                            from interventions import option_token
                            if toks[i + 1] not in {option_token(o) for o in ausloeser[tk].options}:
                                fehler += 1
        print(f"Auslöser ohne Option danach: {fehler}")
        print("OK" if fehler == 0 and ro.eingriffe > 0 else "FEHLER")


if __name__ == "__main__":
    main()

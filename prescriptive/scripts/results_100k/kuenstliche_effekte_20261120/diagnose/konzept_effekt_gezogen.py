"""Hypothese 1 vom 23.11. 23:00 (Nutzer: „ja“): Geht der Betrag im Decoder verloren? Zieht das aktuelle Modell k3x4ol (schlank, keyed_noise)
an konzept-Zuständen die beiden eingebauten konzept-Effekte je Betragsgruppe?
  - Netz „vervollständigen“ (W_Complete application|start): bei < 11.000 € in 30 % der Fälle A_Complete doppelt (Effekt v3), bei ≥ 11.000 € nicht
  - Netz „annehmen“ (A_Accepted): bei ≥ 11.000 € in 40 % W_Validate application ausgelassen, bei < 11.000 € nicht
Zustände: erste Besuche von annehmen_bei_konzept in den Bank-Läufen online_konformanz/bank_seed* (frischer Simulator v3, nicht im Training),
je Betragsgruppe bis zu N_ZUSTAENDE, je Zustand und Option N_ZIEHUNGEN gezogene Suffixe. Vergleich: dieselben Raten im Trainings-Log von
k3x4ol (log_effekte_v3_x4, Fortsetzung des Falls mit der Option der Bank) und im Bank-Lauf selbst (tatsächlich simuliert).
Gezählt über den ganzen Fall (Prefix + Option + Suffix): A_Complete ≥ 2-mal; kein W_Validate application nach dem Wahlpunkt.
Ausgabe konzept_effekt_gezogen.md / .csv hier. Aufruf im Ordner process-simulation-engine:
  .venv/bin/python -u <dieses Skript> [Zustände je Gruppe] [Ziehungen]
"""
import os
from pathlib import Path
import sys
import time
import warnings

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parents[3]
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
HUELLE = Path(os.environ.get("DIAG_HUELLE", ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_k3x4_schlank_crn.pkl"))  # 2026-11-24: DIAG_HUELLE / DIAG_NAME für andere Hüllen (static_in_decoder)
NAME = os.environ.get("DIAG_NAME", "konzept_effekt_gezogen")
LOG = PARENT/"log_effekte_v3_x4/log_raw.pkl"
N_ZUSTAENDE = int(sys.argv[1]) if len(sys.argv) > 1 else 120
N_ZIEHUNGEN = int(sys.argv[2]) if len(sys.argv) > 2 else 30
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CASE, ACT, LC, TIME = "case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"
PUNKT = "annehmen_bei_konzept"
ANNEHMEN, VERVOLLST = "A_Accepted", "W_Complete application|start"
GRENZE = 11000


def merkmale(toks_nach):
    """toks_nach: Tokens ab dem Wahlpunkt (Option + Rest). A_Complete im ganzen Fall kommt erst nach konzept vor."""
    a_complete = sum(1 for t in toks_nach if t.split("|")[0] == "A_Complete")
    validate = any(t.split("|")[0] == "W_Validate application" for t in toks_nach)
    return a_complete >= 2, not validate


def zustaende(punkt):
    from interventions import option_token
    ausloeser = f"{punkt.activity}|{punkt.lifecycle}"
    opt = {option_token(o): o for o in punkt.options}
    out, echt = {"< 11k": [], "≥ 11k": []}, []
    for ordner in sorted((PARENT/"online_konformanz").glob("bank_seed*")):
        e = pd.read_pickle(ordner/"events.pkl").sort_values([CASE, TIME], kind="stable")
        for case, t in e.groupby(CASE, sort=False):
            t = t.reset_index(drop=True)
            toks = [f"{a}|{l}" for a, l in zip(t[ACT], t[LC])]
            if ausloeser not in toks[:-1]:
                continue
            i = toks.index(ausloeser)
            gruppe = "< 11k" if float(t["case:RequestedAmount"].iloc[0]) < GRENZE else "≥ 11k"
            if toks[i + 1] in opt:
                echt.append({"gruppe": gruppe, "option": opt[toks[i + 1]], **dict(zip(("a_complete_doppelt", "validate_fehlt"), merkmale(toks[i + 1:])))})
            if len(out[gruppe]) < N_ZUSTAENDE:
                out[gruppe].append((str(case), t.iloc[:i + 1].to_dict("records"), pd.Timestamp(t.loc[i, TIME]).timestamp()))
    return out, pd.DataFrame(echt)


def log_raten(punkt):
    from interventions import option_token
    ausloeser = f"{punkt.activity}|{punkt.lifecycle}"
    opt = {option_token(o): o for o in punkt.options}
    raw = pd.read_pickle(LOG).sort_values([CASE, TIME], kind="stable")
    raw["tok"] = raw[ACT] + "|" + raw[LC].fillna("complete")
    z = []
    for _, t in raw.groupby(CASE, sort=False):
        toks = t.tok.tolist()
        if ausloeser not in toks[:-1]:
            continue
        i = toks.index(ausloeser)
        if toks[i + 1] not in opt:
            continue
        gruppe = "< 11k" if float(t["case:RequestedAmount"].iloc[0]) < GRENZE else "≥ 11k"
        z.append({"gruppe": gruppe, "option": opt[toks[i + 1]], **dict(zip(("a_complete_doppelt", "validate_fehlt"), merkmale(toks[i + 1:])))})
    return pd.DataFrame(z)


def main():
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    start = time.time()
    model = PSPModel.load(str(HUELLE))
    model.set_threads(8)
    punkt = model.points[PUNKT]
    options = [o for o in punkt.options if o in (ANNEHMEN, VERVOLLST)]
    zust, echt_bank = zustaende(punkt)
    print({g: len(v) for g, v in zust.items()}, "Zustände;", f"{time.time() - start:.0f} s", flush=True)
    zeilen = []
    j = 0
    for gruppe, liste in zust.items():
        for case, prefix, jetzt in liste:
            torch.manual_seed(1000 + j)
            j += 1
            drawn, _ = model.option_suffixes(punkt, prefix, options, jetzt, N_ZIEHUNGEN)
            for option, suffixes in zip(options, drawn):
                from interventions import option_token
                for s in suffixes:
                    toks = [str(t) for t, _, _ in s]
                    if not toks or toks[0] != option_token(option):
                        toks = [option_token(option)] + toks
                    d, v = merkmale(toks)
                    zeilen.append({"gruppe": gruppe, "option": option, "case": case, "a_complete_doppelt": d, "validate_fehlt": v})
        print(f"{gruppe} fertig ({time.time() - start:.0f} s)", flush=True)
    gez = pd.DataFrame(zeilen)
    log = log_raten(punkt)
    tab = []
    for quelle, d in (("gezogen", gez), ("Trainings-Log x4 (Bank)", log), ("Bank-Lauf simuliert", echt_bank)):
        g = d.groupby(["option", "gruppe"]).agg(n=("a_complete_doppelt", "size"), a_complete_doppelt=("a_complete_doppelt", "mean"),
                                                validate_fehlt=("validate_fehlt", "mean")).reset_index()
        g.insert(0, "quelle", quelle)
        tab.append(g)
    tab = pd.concat(tab).round(3)
    tab.to_csv(HERE/f"{NAME}.csv", index=False)
    text = (f"# Konzept-Effekte in gezogenen Suffixen ({HUELLE.name}) je Betragsgruppe ({time.strftime('%Y-%m-%d %H:%M')})\n\n"
            f"Effekt v3: vervollständigen & < 11k → 30 % A_Complete doppelt; annehmen & ≥ 11k → 40 % W_Validate application ausgelassen.\n"
            f"{sum(len(v) for v in zust.values())} Zustände (erste Besuche, Bank-Läufe online_konformanz/bank_seed*), {N_ZIEHUNGEN} Ziehungen je Option. "
            f"Anteile je Fall ab dem Wahlpunkt.\n\n```\n{tab.to_string(index=False)}\n```\n\nLaufzeit {time.time() - start:.0f} s\n")
    (HERE/f"{NAME}.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()

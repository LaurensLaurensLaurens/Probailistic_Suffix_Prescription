"""Tor an konzept nicht bestanden (2026-11-24 01:16): Lernt das Netz die Effekt-Übergänge überhaupt, wenn man ihm den wahren Verlauf gibt?
Teacher Forcing beim Ziehen: an ersten Besuchen von annehmen_bei_konzept in den Bank-Läufen (online_konformanz/bank_seed*, Simulator v3)
wird die wahre Fortsetzung nach der Option bis einschließlich zum ersten A_Complete erzwungen (draw_cases first= für Schritt 0, force=
danach; Zeiten wie sonst aus dem Decoder). Danach frei gezogen.
  - Netz „vervollständigen“ (W_Complete application|start): P(nächstes Event = A_Complete) aus den mittleren Logits (Softmax, ohne
    Logit-Rauschen, gemittelt über die Ziehungen = Dropout-Masken) und gezogener Anteil; Wahrheit: < 11k 30 %, ≥ 11k 0 %.
  - Netz „annehmen“ (A_Accepted): Anteil der Ziehungen ohne W_Validate application im freien Rest; Wahrheit über den Bank-Lauf.
Hüllen: k3x4ol und die Zwischenhülle k3x4sd_konzept (static_in_decoder). Keine Hooks (Modul-Attribute werden für den Lauf umgehängt).
Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Fälle je Gruppe] [Ziehungen]
"""
from pathlib import Path
import os
import sys
import time
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
K = HERE.parent/"kuenstliche_effekte_20261120"
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
HUELLEN = {"k3x4ol": ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_k3x4_schlank_crn.pkl",
           "k3x4sd": ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_k3x4sd_konzept_schlank_crn.pkl"}
if os.environ.get("UL_HUELLEN"):  # 2026-11-24 Lernbarkeitstest: "name=pfad;name=pfad" (die erste Hülle liefert die Zustände)
    HUELLEN = {n: Path(x) for n, x in (e.split("=", 1) for e in os.environ["UL_HUELLEN"].split(";") if e)}
NAME = os.environ.get("UL_NAME", "uebergang_lehrer")
N_FAELLE = int(sys.argv[1]) if len(sys.argv) > 1 else 120
N_ZIEHUNGEN = int(sys.argv[2]) if len(sys.argv) > 2 else 10
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
VERVOLLST, ANNEHMEN = "W_Complete application|start", "A_Accepted|complete"
AC = "A_Complete|complete"
GRENZE = 11000


def faelle():
    """Je Option (Bank-Wahl) und Gruppe bis zu N_FAELLE: (Fall, Prefix-Records, jetzt, wahre Fortsetzung bis zum 1. A_Complete, Rest)."""
    out = {}
    for ordner in sorted((K/"online_konformanz").glob("bank_seed*")):
        e = pd.read_pickle(ordner/"events.pkl").sort_values([CASE, TIME], kind="stable")
        for case, t in e.groupby(CASE, sort=False):
            t = t.reset_index(drop=True)
            toks = [f"{a}|{l}" for a, l in zip(t[ACT], t[LC])]
            if AUSLOESER not in toks[:-1]:
                continue
            i = toks.index(AUSLOESER)
            option = toks[i + 1]
            if option not in (VERVOLLST, ANNEHMEN) or AC not in toks[i + 2:]:
                continue
            j = toks.index(AC, i + 2)
            gruppe = "< 11k" if float(t["case:RequestedAmount"].iloc[0]) < GRENZE else "≥ 11k"
            liste = out.setdefault((option, gruppe), [])
            if len(liste) < N_FAELLE:
                liste.append((f"{ordner.name}/{case}", t.iloc[:i + 1].to_dict("records"), pd.Timestamp(t.loc[i, TIME]).timestamp(),
                              toks[i + 2:j + 1], toks[j + 1:]))
    return out


class Erzwingen:
    """rollout-Ersatz für _options_in_encoder: kontext() liefert die force-Funktion (Zeile k gehört zu Fall k // n)."""
    def __init__(self):
        self.fortsetzung = None

    def kontext(self, cases, token, reached, n):
        cont = self.fortsetzung

        def force(labels, active, suffixes):
            return {int(k): cont[len(suffixes[k])] for k in active if len(suffixes[k]) < len(cont)}
        return force


def main():
    import torch
    from interventions import option_token
    from prescriptive.estimators import psp_model as pm
    from prescriptive.estimators.psp_model import PSPModel
    global AUSLOESER
    start = time.time()
    zeilen = []
    for name, pfad in HUELLEN.items():
        huelle = PSPModel.load(str(pfad))
        huelle.set_threads(8)
        punkt = huelle.points[PUNKT]
        AUSLOESER = f"{punkt.activity}|{punkt.lifecycle}"
        if name == next(iter(HUELLEN)):
            alle = faelle()
            print({k: len(v) for k, v in alle.items()}, f"{time.time() - start:.0f} s", flush=True)
        for (option, gruppe), liste in sorted(alle.items()):
            opt = next(o for o in punkt.options if option_token(o) == option)
            modell = huelle.netze[(PUNKT, option)] if getattr(huelle, "netze", None) else huelle  # einzelnes PSP (z. B. Simulator-Basis)
            net = modell.net
            labels = {str(v): int(k) for k, v in modell._evaluator().inverted_suffix_categories[0].items()}
            if AC not in labels:
                raise RuntimeError(f"{AC} nicht im Vokabular")
            aufnahme, ziehen_orig, inferenz_orig = [], modell.draw_cases, net.inference
            ro = Erzwingen()

            def ziehen(encoded, n, *a, force=None, **kw):
                if force is not None:  # zweiter Aufruf in _options_in_encoder: der Rest nach der Option
                    kw["first"] = [ro.fortsetzung[0]] * len(encoded)
                    aufnahme.clear()
                return ziehen_orig(encoded, n, *a, force=force, **kw)

            def inferenz(*a, **kw):
                out = inferenz_orig(*a, **kw)
                aufnahme.append(out[0][0][0][pm.ACTIVITY + "_mean"].detach().reshape(-1, out[0][0][0][pm.ACTIVITY + "_mean"].shape[-1]))
                return out
            modell.draw_cases, net.inference, modell.rollout = ziehen, inferenz, ro
            fehler = 0
            try:
                for j, (case, prefix, jetzt, cont, rest) in enumerate(liste):
                    if any(t not in labels for t in cont):
                        fehler += 1
                        continue
                    ro.fortsetzung = cont
                    torch.manual_seed(9000 + j)
                    try:
                        with torch.no_grad():
                            drawn, _ = huelle.option_suffixes(punkt, prefix, [opt], jetzt, N_ZIEHUNGEN)
                    except (RuntimeError, ValueError) as err:
                        fehler += 1
                        print(f"{name} {option} {gruppe} {case}: {err}", flush=True)
                        continue
                    s = len(cont)
                    p = torch.softmax(aufnahme[s].float(), dim=1)[:, labels[AC]].numpy() if s < len(aufnahme) else np.full(N_ZIEHUNGEN, np.nan)
                    for i, suffix in enumerate(drawn[0]):
                        toks = [str(t) for t, _, _ in suffix][1:]  # ohne Options-Event
                        if toks[:s] != cont:
                            raise RuntimeError(f"Erzwingen fehlgeschlagen: {toks[:s]} statt {cont}")
                        frei = toks[s:]
                        zeilen.append({"huelle": name, "option": option, "gruppe": gruppe, "case": case,
                                       "p_ac_naechstes": float(p[i]), "gezogen_ac_naechstes": bool(frei[:1] == [AC]),
                                       "validate_fehlt_rest": not any(t.startswith("W_Validate application") for t in frei),
                                       "wahr_ac_naechstes": bool(rest[:1] == [AC]),
                                       "wahr_validate_fehlt_rest": not any(t.startswith("W_Validate application") for t in rest)})
            finally:
                del modell.draw_cases, net.inference, modell.rollout
            print(f"{name} {option} {gruppe}: {len(liste)} Fälle, {fehler} übersprungen ({time.time() - start:.0f} s)", flush=True)
    d = pd.DataFrame(zeilen)
    d.to_csv(HERE/f"{NAME}_zeilen.csv", index=False)
    je_fall = d.groupby(["huelle", "option", "gruppe", "case"]).mean(numeric_only=True).reset_index()
    tab = je_fall.groupby(["option", "gruppe", "huelle"]).agg(
        Faelle=("case", "size"), P_AC_naechstes=("p_ac_naechstes", "mean"), gezogen_AC_naechstes=("gezogen_ac_naechstes", "mean"),
        wahr_AC_naechstes=("wahr_ac_naechstes", "mean"), validate_fehlt_gezogen=("validate_fehlt_rest", "mean"),
        validate_fehlt_wahr=("wahr_validate_fehlt_rest", "mean")).round(3).reset_index()
    tab.to_csv(HERE/f"{NAME}.csv", index=False)
    text = (f"# Teacher Forcing bis zum ersten A_Complete an konzept – {', '.join(HUELLEN)} ({time.strftime('%Y-%m-%d %H:%M')})\n\n"
            f"Bank-Läufe (Simulator v3), erste Besuche, Bank-Option; wahre Fortsetzung nach der Option bis einschließlich zum ersten "
            f"A_Complete erzwungen, {N_ZIEHUNGEN} Ziehungen je Fall. P_AC_naechstes: Softmax der mittleren Logits für A_Complete als "
            f"nächstes Event (ohne Logit-Rauschen, Mittel über Dropout-Masken); gezogen: mit Rauschen und Übergangsregeln; wahr: Bank-Lauf. "
            f"validate_fehlt: kein W_Validate application im freien Rest. Effekt v3: vervollständigen & < 11k → 30 % A_Complete doppelt "
            f"(+1 ms, also direkt danach); annehmen & ≥ 11k → 40 % W_Validate ausgelassen.\n\n```\n{tab.to_string(index=False)}\n```\n\n"
            f"Laufzeit {time.time() - start:.0f} s\n")
    (HERE/f"{NAME}.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()

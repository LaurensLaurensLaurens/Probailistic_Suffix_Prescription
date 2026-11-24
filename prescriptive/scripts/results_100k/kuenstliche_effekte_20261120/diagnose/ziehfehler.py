"""Diagnose 2026-11-21 (Nutzer: „Ich dachte, beim Ziehen macht das PSP durch TF wenige Fehler“): woher kommen die Conformance-Abweichungen
der gezogenen Suffixe? An frischen Zuständen (Simulator v3, Bank) je erstem Besuch eines Wahlpunkts: 20 gezogene Suffixe je Option (k3)
und der tatsächlich simulierte Verlauf des Falls (mit der Option, die die Bank nahm). Je Spur: Abweichungen gegen BPIC17_ToBe_angepasst
nach Art (Logzug / Modellzug je Aktivität), ob ein im v3-Log nie beobachteter Übergang (Aktivität|Lifecycle-Paar) vorkommt, ob der Suffix
mit EOS endet, Länge. Ausgabe ziehfehler.md / ziehfehler.csv hier.
Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Prozesse] [Fälle je Prozess] [Ziehungen]
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import warnings

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parents[3]
PROZESSE = int(sys.argv[1]) if len(sys.argv) > 1 else 3
FAELLE = int(sys.argv[2]) if len(sys.argv) > 2 else 40
ZIEHUNGEN = int(sys.argv[3]) if len(sys.argv) > 3 else 20
MODELL = ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_k3.pkl"
LOG = PARENT/"log_effekte_v3/log_raw.pkl"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PARENT))
import online_konformanz as ok  # noqa: E402
import pandas as pd  # noqa: E402


def zuege(spur):
    """Alignment-Züge (Logzug / Modellzug je Aktivität) der Spur gegen das angepasste Modell."""
    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
    from pm4py.objects.log.obj import Event, Trace
    from prescriptive.policies import konformanz_policy as kp
    net, im, fm = kp._netz(kp.MODELL)
    ergebnis = alignments.apply_trace(Trace([Event({"concept:name": a}) for a in spur]), net, im, fm)
    out = []
    for a, b in ergebnis["alignment"]:
        if a == ">>" and b is not None:
            out.append(f"fehlt {b}")
        elif b == ">>" and a is not None:
            out.append(f"unerlaubt {a}")
    return out


class Aufzeichner:
    def __init__(self):
        self.zustaende, self.gesehen = [], set()

    def choose(self, point, case, options, time, engine):
        if point.name in ok.Orakel.PUNKTE and len(options) == 2 and (case.case_id, point.name) not in self.gesehen:
            self.gesehen.add((case.case_id, point.name))
            self.zustaende.append({"point": point, "case": case.case_id, "prefix": [dict(e) for e in case.prefix], "time": time,
                                   "options": list(options)})
        return None


def job(k):
    import torch
    torch.set_num_threads(1)
    import kuenstliche_effekte as ke
    from interventions import option_token
    from prescriptive.estimators.psp_model import PSPModel
    from prescriptive.policies import konformanz_policy as kp
    ke.aktivieren(ke.laden(ok.EFFEKTE))
    auf = Aufzeichner()
    events, _ = ok.simulator().run(400 + FAELLE, policy=auf, window=(400, 400 + FAELLE), seed=500 + k)
    log = pd.read_pickle(LOG).sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    tok = log["concept:name"] + "|" + log["lifecycle:transition"]
    nach = tok.groupby(log["case:concept:name"]).shift(-1)
    paare = set(zip(tok, nach.fillna("EOS")))
    events = events.sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    model = PSPModel.load(str(MODELL))
    model.set_threads(1)
    torch.manual_seed(500 + k)
    zeilen = []
    for z in auf.zustaende:
        point, prefix = z["point"], z["prefix"]
        vorne = kp.projektion_events(prefix)
        letzter = f"{prefix[-1]['concept:name']}|{prefix[-1]['lifecycle:transition']}"
        drawn, _ = model.option_suffixes(point, prefix, z["options"], z["time"], ZIEHUNGEN)
        for option, suffixes in zip(z["options"], drawn):
            o = option_token(option)
            for suffix in suffixes:
                labels = [s[0] for s in suffix]
                ende = len(labels) < 100  # ohne EOS-Ausgabe: Ende vor der Schrittgrenze (grob)
                labels = kp.mit_option(o, labels)  # Suffix beginnt schon mit der Option (2026-11-21)
                folge = [letzter] + [x for x in labels if "|" in str(x)] + ["EOS"]  # draw_cases gibt EOS nicht aus: Ende = EOS
                ungueltig = sum((a, b) not in paare for a, b in zip(folge, folge[1:]))
                spur = vorne + kp.projektion_tokens(labels)
                moves = zuege(spur)
                zeilen.append({"quelle": "gezogen", "punkt": point.name, "option": option, "case": z["case"], "abweichungen": len(moves),
                               "zuege": "; ".join(moves), "ungueltige_uebergaenge": ungueltig, "mit_eos": ende, "laenge": len(labels)})
        # tatsächlicher Verlauf des Falls (mit der Option der Bank)
        fall = events[events["case:concept:name"] == z["case"]]
        n = len(prefix)
        if len(fall) > n:
            genommen = f"{fall.iloc[n]['concept:name']}|{fall.iloc[n]['lifecycle:transition']}"
            option = next((x for x in z["options"] if option_token(x) == genommen), None)
            if option is not None:
                moves = zuege(kp.projektion_events(fall.to_dict("records")))
                folge = list(fall["concept:name"] + "|" + fall["lifecycle:transition"])[n - 1:] + ["EOS"]
                zeilen.append({"quelle": "echt (Simulator)", "punkt": point.name, "option": option, "case": z["case"],
                               "abweichungen": len(moves), "zuege": "; ".join(moves),
                               "ungueltige_uebergaenge": sum((a, b) not in paare for a, b in zip(folge, folge[1:])), "mit_eos": True,
                               "laenge": len(fall) - n})
    return pd.DataFrame(zeilen)


def main():
    with ProcessPoolExecutor(PROZESSE, mp_context=multiprocessing.get_context("spawn")) as pool:
        d = pd.concat(list(pool.map(job, range(PROZESSE))), ignore_index=True)
    d.to_csv(HERE/"ziehfehler.csv", index=False)
    tab = d.groupby(["punkt", "option", "quelle"]).agg(spuren=("case", "size"), abweichungen=("abweichungen", "mean"),
                                                       mit_ungueltigem_uebergang=("ungueltige_uebergaenge", lambda s: (s > 0).mean()),
                                                       ohne_eos=("mit_eos", lambda s: 1 - s.mean()), laenge=("laenge", "mean")).round(3)
    teile = [f"# Woher die Abweichungen der gezogenen Suffixe kommen (k3, Diagnose 2026-11-21)\n\n{ZIEHUNGEN} Ziehungen je Option, "
             f"{PROZESSE} × {FAELLE} Fälle; „echt“ = tatsächlich simulierter Verlauf mit der Option der Bank.\n\n```\n{tab.to_string()}\n```\n"]
    for quelle, teil in d.groupby("quelle"):
        c = Counter(m for s in teil.zuege.dropna() for m in s.split("; ") if m)
        n = len(teil)
        teile.append(f"\n## Häufigste Züge, {quelle} (je Spur)\n\n" + "\n".join(f"- {k}: {v / n:.3f}" for k, v in c.most_common(15)))
        ungueltig = teil[teil.ungueltige_uebergaenge > 0]
        teile.append(f"\nAbweichungen je Spur mit / ohne ungültigen Übergang: "
                     f"{ungueltig.abweichungen.mean():.2f} / {teil[teil.ungueltige_uebergaenge == 0].abweichungen.mean():.2f} "
                     f"(Anteil Spuren mit ungültigem Übergang {len(ungueltig) / max(1, n):.1%})")
    text = "\n".join(teile) + "\n"
    (HERE/"ziehfehler.md").write_text(text)
    print(text)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()

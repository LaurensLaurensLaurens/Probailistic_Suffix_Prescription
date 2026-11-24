"""Wie realistisch zieht olsv die Suffixe online? (2026-11-23, Nutzer: „Wie realistisch zieht olsv die Suffixe online?“)

Online-Umgebung = Simulator in Aufbau F. Zustände: erste Besuche der 4 Wahlpunkte in den bewerteten Fällen (700–899) der Bank-Läufe
(Seeds 0–11: f_zwoelf_seeds/bank_seed<s> bzw. d_oe_zustand_20261115/bank_reference_seed<s>), je Punkt bis zu ZUSTAENDE zufällig.
Je Zustand zieht olsv (Hülle option_learner_sv_crn, keyed_noise, wie online) DRAWS Suffixe mit der Option, die die Bank dort gewählt hat;
verglichen wird mit dem, was der Simulator danach tatsächlich gemacht hat (Rest des Falls ab dem Options-Event).
Kennzahlen je Punkt, gezogen gegen echt (Verteilung über Zustände bzw. Ziehungen):
  - Restzeit ab der Entscheidung in Tagen: Mittel, SD, Quantile 10/50/90
  - Events im Rest, Angebote (O_Create Offer), verschickte Angebote (O_Sent), Anrufe nach Angeboten (W_Call after offers|start)
  - Ausgang: A_Pending (angenommen) / A_Denied / A_Cancelled / anderes letztes Event
  - Übergänge, die im echten BPIC17-Log nie vorkommen (Token → nächstes Token), je 100 Events
  - Fehler der Restzeit je Zustand: |Mittel der Ziehungen − echt| gegen den naiven Schätzer (Mittel der echten Restzeiten des Punkts)
Ausgabe olsv_realismus.md / .csv hier. Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Prozesse] [Zustände je Punkt] [Ziehungen]
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
F = ROOT/"prescriptive/scripts/results_100k/modellvergleich_20261116/f_zwoelf_seeds"
MODELL = ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_sv_crn.pkl"
PROZESSE = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ZUSTAENDE = int(sys.argv[2]) if len(sys.argv) > 2 else 100
DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
CASE, ACT, LC, TIME = "case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"
DAY = 86400.0
SEP = " ;; "  # Tokens enthalten Leerzeichen (2026-11-23: Trennung mit " " zerlegte sie, alle Übergänge galten als unbekannt)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def token(act, lc):
    return f"{act}|{lc if isinstance(lc, str) and lc else 'complete'}"


def bank_events(seed):
    for pfad in (F/f"bank_seed{seed}/events.pkl", SOURCE/f"bank_reference_seed{seed}/events.pkl"):
        if pfad.exists():
            e = pd.read_pickle(pfad)
            nummer = e[CASE].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
            return e[nummer.between(700, 899)]
    raise FileNotFoundError(seed)


def zustaende():
    import interventions
    from interventions import option_token
    interventions.CANDIDATES_FILE = str(SOURCE/"catalog.yaml")
    punkte = {p.name: p for p in interventions.choice_points("bpic17")}
    rng = np.random.default_rng(23)
    alle = []
    for seed in range(12):
        e = bank_events(seed).sort_values([CASE, TIME], kind="stable")
        for case, t in e.groupby(CASE, sort=False):
            toks = [token(a, l) for a, l in zip(t[ACT], t[LC])]
            for name, p in punkte.items():
                ausloeser = token(p.activity, p.lifecycle)
                optionen = {option_token(o): o for o in p.options}  # Token → Option, wie der Simulator sie der Policy gibt
                for i, tk in enumerate(toks[:-1]):
                    if tk == ausloeser:
                        if toks[i + 1] in optionen:
                            alle.append({"seed": seed, "case": case, "punkt": name, "i": i, "option": toks[i + 1], "option_raw": optionen[toks[i + 1]]})
                        break  # nur der erste Besuch
    d = pd.DataFrame(alle)
    return pd.concat([g.sample(min(len(g), ZUSTAENDE), random_state=int(rng.integers(1 << 30))) for _, g in d.groupby("punkt")])


def merkmale(toks, gaps_tage):
    return {"rest": float(np.sum(gaps_tage[1:])) if len(gaps_tage) > 1 else 0.0, "events": len(toks),
            "angebote": toks.count("O_Create Offer|complete"), "gesendet": toks.count("O_Sent (mail and online)|complete"),
            "anrufe": toks.count("W_Call after offers|start"),
            # Ausgang: welches der drei Antrags-Enden im Rest vorkommt (das letzte Event ist in BPIC17 meist ein W_-Event)
            "ende": next((a for a in ("A_Pending", "A_Denied", "A_Cancelled") if f"{a}|complete" in toks), "keins")}


def job(task):
    nummer, zeilen = task
    import torch
    torch.set_num_threads(1)
    from prescriptive.estimators.psp_model import PSPModel
    model = PSPModel.load(str(MODELL))
    for m in [model] + list(dict.fromkeys(getattr(model, "netze", {}).values())):
        if isinstance(getattr(m.encoder_decoder, "event_log", None), pd.DataFrame):
            m.encoder_decoder.event_log = m.encoder_decoder.event_log.head(0)  # nur im Training gebraucht
    model.set_threads(1)
    events = {}
    out = []
    for j, z in enumerate(zeilen):
        if z["seed"] not in events:
            events[z["seed"]] = bank_events(z["seed"]).sort_values([CASE, TIME], kind="stable")
        t = events[z["seed"]]
        t = t[t[CASE] == z["case"]].reset_index(drop=True)
        prefix = t.iloc[:z["i"] + 1].to_dict("records")
        jetzt = pd.Timestamp(t.loc[z["i"], TIME]).timestamp()
        rest = t.iloc[z["i"] + 1:]
        echte_toks = [token(a, l) for a, l in zip(rest[ACT], rest[LC])]
        zeiten = [pd.Timestamp(x).timestamp() for x in rest[TIME]]
        echte_gaps = [(zeiten[0] - jetzt) / DAY] + list(np.diff(zeiten) / DAY)
        echt = merkmale(echte_toks, echte_gaps)
        echt["rest"] = (zeiten[-1] - jetzt) / DAY
        torch.manual_seed(5000 + 101 * nummer + j)
        gezogen, _ = model.option_suffixes(model.points[z["punkt"]], prefix, [z["option_raw"]], jetzt, DRAWS)
        for k, suffix in enumerate(gezogen[0]):
            toks = [n for n, _, _ in suffix]
            gaps = [g / DAY for _, _, g in suffix]
            if not toks or toks[0] != z["option"]:
                toks, gaps = [z["option"]] + toks, [0.0] + gaps
            m = merkmale(toks, gaps)
            m["rest"] = float(np.sum(gaps))
            out.append({**z, "quelle": "gezogen", "k": k, **m, "toks": SEP.join(toks)})
        out.append({**z, "quelle": "echt", "k": -1, **echt, "toks": SEP.join(echte_toks)})
    return pd.DataFrame(out)


def uebergaenge():
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw, verbose=False).sort_values([CASE, TIME], kind="stable")
    toks = [token(a, l) for a, l in zip(log[ACT], log[LC])]
    faelle = log[CASE].to_numpy()
    return {(toks[i], toks[i + 1]) for i in range(len(toks) - 1) if faelle[i] == faelle[i + 1]}


def verstoesse(kette, erlaubt):
    t = kette.split(SEP) if isinstance(kette, str) and kette else []
    return sum((a, b) not in erlaubt for a, b in zip(t, t[1:])), max(len(t) - 1, 0)


def main():
    z = zustaende().to_dict("records")
    with ProcessPoolExecutor(PROZESSE, mp_context=multiprocessing.get_context("spawn")) as pool:
        d = pd.concat(list(pool.map(job, [(w, z[w::PROZESSE]) for w in range(PROZESSE)])), ignore_index=True)
    erlaubt = uebergaenge()
    v = d.toks.map(lambda s: verstoesse(s, erlaubt))
    d["verstoesse"], d["paare"] = v.str[0], v.str[1]
    d.to_csv(HERE/"olsv_realismus.csv", index=False)
    zeilen = []
    for (punkt, quelle), t in list(d.groupby(["punkt", "quelle"])) + [(("alle", q), t) for q, t in d.groupby("quelle")]:
        zeilen.append({"Punkt": punkt, "Quelle": quelle, "n": len(t), "Restzeit Mittel": t.rest.mean(), "SD": t.rest.std(),
                       "Q10": t.rest.quantile(.1), "Q50": t.rest.quantile(.5), "Q90": t.rest.quantile(.9), "Events": t.events.mean(),
                       "Angebote": t.angebote.mean(), "O_Sent": t.gesendet.mean(), "Anrufe": t.anrufe.mean(),
                       "A_Pending %": 100 * (t.ende == "A_Pending").mean(), "A_Denied %": 100 * (t.ende == "A_Denied").mean(),
                       "A_Cancelled %": 100 * (t.ende == "A_Cancelled").mean(),
                       "kein Ende %": 100 * (t.ende == "keins").mean(),
                       "unbekannte Übergänge je 100": 100 * t.verstoesse.sum() / max(t.paare.sum(), 1)})
    tab = pd.DataFrame(zeilen)
    je = d[d.quelle == "gezogen"].groupby(["seed", "case", "punkt"]).rest.mean().rename("gezogen").to_frame().join(
        d[d.quelle == "echt"].set_index(["seed", "case", "punkt"]).rest.rename("echt"))
    je["naiv"] = je.groupby(level="punkt").echt.transform("mean")
    fehler = je.groupby(level="punkt").apply(lambda t: pd.Series({"MAE olsv (d)": (t.gezogen - t.echt).abs().mean(),
                                                                  "MAE naiv (d)": (t.naiv - t.echt).abs().mean(),
                                                                  "Korrelation": t.gezogen.corr(t.echt)}))
    with pd.option_context("display.width", 300, "display.max_columns", 25):
        text = (f"# Wie realistisch zieht olsv die Suffixe online? (2026-11-23)\n\nZustände: erste Besuche in Bank-Läufen Aufbau F (Seeds 0–11), "
                f"je Punkt bis {ZUSTAENDE}; je Zustand {DRAWS} Ziehungen mit der Option der Bank; echt = Simulator.\n\n"
                f"```\n{tab.round(2).to_string(index=False)}\n```\n\n## Restzeit je Zustand: Fehler des Ziehungsmittels\n\n"
                f"```\n{fehler.round(2).to_string()}\n```\n")
    (HERE/"olsv_realismus.md").write_text(text)
    print(text)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()

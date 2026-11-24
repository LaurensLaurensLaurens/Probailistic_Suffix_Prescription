"""Zufallszahlen je Ereignis statt je Schrittnummer: koppelt das die Optionen besser? (2026-11-19, Nutzer: „teste die Zufallszahlen nach Event-Typ“)

Diagnose ohne Training. Dieselben Simulatorzustände wie punkte_alle.py (je Gruppe 60: konzept, nach V., nachlegen 1. Besuch,
nachlegen später, leads), je Option DRAWS gezogene Suffixe, zwei Bedingungen mit derselben Saat je Zustand:
  schritt = wie online (fortlaufender Zufallsstrom, Zahl Nr. t gehört zu Schritt t)
  event   = params keyed_noise (PSPModel._keyed_draw): Event-Wahl nach (voriges Event, Vorkommen), Zeit nach (gezogenes Event,
            Vorkommen); Dropout-Masken unverändert
Je Modell, Bedingung und Gruppe: Streuung eines Verlaufs (SD der Restzeit je Option, Median über Zustände), Korrelation ρ der beiden Optionen
über die Ziehungen, Rauschen der Differenz je Entscheidung (gepaarter SE, bei DRAWS und auf 300 umgerechnet), mittlerer Vorteil der
Regel-Option, Anteil Zustände mit richtigem Vorzeichen (Simulator: Regel-Option überall besser) und die rauschfreie Streuung der
Schätzungen zwischen Zuständen (Varianz der Zustandsmittel minus mittleres SE², ≥ 0). Gleiche Verteilung je Ziehung: Mittel und SD je Verlauf müssen in beiden Bedingungen gleich sein. Ausgabe crn_schluessel/ (ziehungen.pkl, crn_schluessel.csv, .md).
Aufruf im Ordner process-simulation-engine: .venv/bin/python ../prescriptive/scripts/results_100k/modellvergleich_20261116/crn_schluessel.py
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_ohne_ziehungen as base  # noqa: E402
from leads_meilensteine import milestones  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).resolve().parent/"crn_schluessel"
M = base.ROOT/"prescriptive/models"
MODELS = {"T-Learner (ZZ-TF1)": M/"psp_model_durchlaufzeit_t_learner_zztf1.pkl",
          "ZZ-TF1 (ein Netz)": M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1.pkl"}
DRAWS = 100
GROUPS = ["annehmen_bei_konzept", "annehmen_nach_vervollstaendigung", "nachlegen 1. Besuch", "nachlegen ab 2. Besuch", "handle_leads"]


def group(point, visit):
    return ("nachlegen 1. Besuch" if visit == 1 else "nachlegen ab 2. Besuch") if point == "angebot_nachlegen" else point


def work(task):
    number, states = task
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    torch.set_num_threads(1)
    models = {label: PSPModel.load(path) for label, path in MODELS.items()}

    def set_keyed(model, on):
        # 2026-11-22: auch die Netze von Option-/Punkt-Learner-Hüllen (netze); bis dahin blieb keyed_noise dort aus (Bedingung event = schritt)
        for m in (model, getattr(model, "aktion", None), *dict.fromkeys(getattr(model, "netze", {}).values())):
            if m is not None:
                m.params = {**(m.params or {}), "keyed_noise": on}

    events = {seed: pd.read_pickle(base.SOURCE/base.RUN.format(seed)/"events.pkl") for seed in sorted({s["seed"] for s in states})}
    rows = []
    for index, s in enumerate(states):
        part = events[s["seed"]]
        part = part.loc[part[base.CASE].astype(str).eq(s["case"])].sort_values(base.TIME, kind="stable").reset_index(drop=True)
        stamps = part[base.TIME].map(lambda t: pd.Timestamp(t).timestamp()).to_numpy()
        prefix = part.iloc[:int(np.searchsorted(stamps, s["time"]+1e-6, side="right"))].to_dict("records")
        for label, model in models.items():
            model.set_threads(1)
            for condition in ("schritt", "event"):
                set_keyed(model, condition == "event")
                torch.manual_seed(17000+97*number+index)
                drawn, _ = model.option_suffixes(model.points[s["point"]], prefix, list(s["options"]), s["time"], DRAWS)
                for option, suffixes in zip(s["options"], drawn):
                    for k, suffix in enumerate(suffixes):
                        tokens, gaps = [n for n, _, _ in suffix], [g/base.DAY for _, _, g in suffix]
                        if not model.option_in_encoder():
                            tokens, gaps = [option]+tokens, [0.0]+gaps
                        rows.append({"modell": label, "bedingung": condition, "gruppe": group(s["point"], s["visit"]), "seed": s["seed"],
                                     "case": s["case"], "visit": s["visit"], "rule": option == base.RULE[s["point"]], "k": k,
                                     "rest": milestones(tokens, gaps)["ENDE"], "events": len(tokens)})
            set_keyed(model, False)
    return pd.DataFrame(rows)


def summarize(drawn):
    rows = []
    for (label, condition, g), part in drawn.groupby(["modell", "bedingung", "gruppe"], sort=False):
        wide = part.pivot_table(index=["seed", "case", "visit", "k"], columns="rule", values="rest")
        per_state = []
        for _, t in wide.groupby(level=[0, 1, 2]):
            a, b = t[True].to_numpy(), t[False].to_numpy()
            d = b - a  # Vorteil der Regel-Option (andere Option dauert länger)
            per_state.append({"diff": d.mean(), "se": d.std(ddof=1)/np.sqrt(len(d)), "sd": np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2),
                              "rho": np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan})
        st = pd.DataFrame(per_state)
        between = max(st["diff"].var(ddof=1) - (st.se**2).mean(), 0.0)
        rows.append({"Modell": label, "Bedingung": condition, "Gruppe": g, "Zustände": len(st), "Mittel je Verlauf (d)": part.rest.mean(), "Events je Verlauf": part.events.mean(),
                     "SD je Verlauf (d)": st.sd.median(), "ρ Optionen": st.rho.median(),
                     "Rauschen je Entscheidung bei 100 (d)": st.se.median(), "Rauschen bei 300 (d)": st.se.median()/np.sqrt(3),
                     "Vorteil Regel-Option (d)": st["diff"].mean(), "richtig (100 Ziehungen)": (st["diff"] > 0).mean(),
                     "rauschfreie Streuung zw. Zuständen (d)": np.sqrt(between)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    states = base.pick().to_dict("records")
    order = np.random.default_rng(2).permutation(len(states))
    with ProcessPoolExecutor(6, mp_context=multiprocessing.get_context("spawn")) as pool:
        drawn = pd.concat(list(pool.map(work, [(w, [states[i] for i in order[w::6]]) for w in range(6)])), ignore_index=True)
    drawn.to_pickle(OUT/"ziehungen.pkl")
    table = summarize(drawn)
    table.to_csv(OUT/"crn_schluessel.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        text = table.round(3).to_string(index=False)
    (OUT/"crn_schluessel.md").write_text("# Zufallszahlen je Schritt vs. je Ereignis (Diagnose, 2026-11-19)\n\n```\n" + text + "\n```\n")
    print(text)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()

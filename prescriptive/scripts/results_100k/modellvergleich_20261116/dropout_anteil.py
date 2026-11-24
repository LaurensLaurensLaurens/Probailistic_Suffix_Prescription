"""Wie viel des Entscheidungsrauschens kommt vom MC-Dropout? (2026-11-19, Nutzer: „ja, miss das mal“)

Reine Diagnose, die Policy bleibt unverändert. Dieselben Simulatorzustände wie punkte_alle.py (je Gruppe 60: konzept, nach V.,
nachlegen 1. Besuch, nachlegen später, leads), je Option DRAWS gezogene Suffixe, zwei Bedingungen mit derselben Saat je Zustand:
  an  = wie online (je Ziehung eine Dropout-Maske, für den ganzen Verlauf, gleiche Saat für beide Optionen)
  aus = Masken durch Einsen ersetzt (DropoutUncertaintyLSTMCell._mc_dropout_sample_mask; Encoder und Decoder), Event- und Zeitziehung unverändert
Je Modell, Bedingung und Gruppe: Streuung eines Verlaufs (SD der Restzeit je Option, Median über Zustände), Korrelation ρ der beiden Optionen
über die Ziehungen, Rauschen der Differenz je Entscheidung (gepaarter SE, bei DRAWS und auf 300 umgerechnet), mittlerer Vorteil der
Regel-Option, Anteil Zustände mit richtigem Vorzeichen (Simulator: Regel-Option überall besser) und die rauschfreie Streuung der
Schätzungen zwischen Zuständen (Varianz der Zustandsmittel minus mittleres SE², ≥ 0). Dropout-Anteil an der Streuung eines Verlaufs =
1 − Var(aus) / Var(an). Ausgabe dropout_anteil/ (ziehungen.pkl, dropout_anteil.csv, .md).
Aufruf im Ordner process-simulation-engine: .venv/bin/python ../prescriptive/scripts/results_100k/modellvergleich_20261116/dropout_anteil.py
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

OUT = Path(__file__).resolve().parent/"dropout_anteil"
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
    cell = type(next(m for m in models["ZZ-TF1 (ein Netz)"].net.decoder.modules() if hasattr(m, "_mc_dropout_sample_mask")))
    original = cell._mc_dropout_sample_mask

    def ones(self, B, device):
        return (torch.ones(4, B, self.input_size, device=device), torch.ones(4, B, self.hidden_size, device=device))

    events = {seed: pd.read_pickle(base.SOURCE/base.RUN.format(seed)/"events.pkl") for seed in sorted({s["seed"] for s in states})}
    rows = []
    for index, s in enumerate(states):
        part = events[s["seed"]]
        part = part.loc[part[base.CASE].astype(str).eq(s["case"])].sort_values(base.TIME, kind="stable").reset_index(drop=True)
        stamps = part[base.TIME].map(lambda t: pd.Timestamp(t).timestamp()).to_numpy()
        prefix = part.iloc[:int(np.searchsorted(stamps, s["time"]+1e-6, side="right"))].to_dict("records")
        for label, model in models.items():
            model.set_threads(1)
            for condition in ("an", "aus"):
                cell._mc_dropout_sample_mask = original if condition == "an" else ones
                try:
                    torch.manual_seed(17000+97*number+index)
                    drawn, _ = model.option_suffixes(model.points[s["point"]], prefix, list(s["options"]), s["time"], DRAWS)
                finally:
                    cell._mc_dropout_sample_mask = original
                for option, suffixes in zip(s["options"], drawn):
                    for k, suffix in enumerate(suffixes):
                        tokens, gaps = [n for n, _, _ in suffix], [g/base.DAY for _, _, g in suffix]
                        if not model.option_in_encoder():
                            tokens, gaps = [option]+tokens, [0.0]+gaps
                        rows.append({"modell": label, "dropout": condition, "gruppe": group(s["point"], s["visit"]), "seed": s["seed"],
                                     "case": s["case"], "visit": s["visit"], "rule": option == base.RULE[s["point"]], "k": k,
                                     "rest": milestones(tokens, gaps)["ENDE"]})
    return pd.DataFrame(rows)


def summarize(drawn):
    rows = []
    for (label, condition, g), part in drawn.groupby(["modell", "dropout", "gruppe"], sort=False):
        wide = part.pivot_table(index=["seed", "case", "visit", "k"], columns="rule", values="rest")
        per_state = []
        for _, t in wide.groupby(level=[0, 1, 2]):
            a, b = t[True].to_numpy(), t[False].to_numpy()
            d = b - a  # Vorteil der Regel-Option (andere Option dauert länger)
            per_state.append({"diff": d.mean(), "se": d.std(ddof=1)/np.sqrt(len(d)), "sd": np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2),
                              "rho": np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan})
        st = pd.DataFrame(per_state)
        between = max(st["diff"].var(ddof=1) - (st.se**2).mean(), 0.0)
        rows.append({"Modell": label, "Dropout": condition, "Gruppe": g, "Zustände": len(st),
                     "SD je Verlauf (d)": st.sd.median(), "ρ Optionen": st.rho.median(),
                     "Rauschen je Entscheidung bei 100 (d)": st.se.median(), "Rauschen bei 300 (d)": st.se.median()/np.sqrt(3),
                     "Vorteil Regel-Option (d)": st["diff"].mean(), "richtig (100 Ziehungen)": (st["diff"] > 0).mean(),
                     "rauschfreie Streuung zw. Zuständen (d)": np.sqrt(between)})
    table = pd.DataFrame(rows)
    on = table[table.Dropout == "an"].set_index(["Modell", "Gruppe"])
    off = table[table.Dropout == "aus"].set_index(["Modell", "Gruppe"])
    share = (1 - (off["SD je Verlauf (d)"]**2) / (on["SD je Verlauf (d)"]**2)).rename("Dropout-Anteil an Var. je Verlauf").reset_index()
    return table, share


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    states = base.pick().to_dict("records")
    order = np.random.default_rng(2).permutation(len(states))
    with ProcessPoolExecutor(6, mp_context=multiprocessing.get_context("spawn")) as pool:
        drawn = pd.concat(list(pool.map(work, [(w, [states[i] for i in order[w::6]]) for w in range(6)])), ignore_index=True)
    drawn.to_pickle(OUT/"ziehungen.pkl")
    table, share = summarize(drawn)
    table.to_csv(OUT/"dropout_anteil.csv", index=False)
    share.to_csv(OUT/"dropout_anteil_varianz.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        text = table.round(3).to_string(index=False) + "\n\n" + share.round(3).to_string(index=False)
    (OUT/"dropout_anteil.md").write_text("# MC-Dropout an/aus beim Ziehen (Diagnose, 2026-11-19)\n\n```\n" + text + "\n```\n")
    print(text)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()

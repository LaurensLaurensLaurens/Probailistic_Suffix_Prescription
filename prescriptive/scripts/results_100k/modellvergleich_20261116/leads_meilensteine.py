"""Warum erkennt D den Leads-Punkt kaum? (2026-11-16)

Zeit von der Entscheidung bis zu Meilensteinen des Falls, Option start minus withdraw:
1. Simulator-Wahrheit: randomisierte Erstentscheidungen der RCT-Daten vom 14.11.
2. Gezogene Suffixe an 60 Leads-Zuständen aus den D-Läufen ohne Rückfall, je Option 100 Ziehungen, mit
   - D_stateful (wie online: Zeiten aus der empirischen Abstandstabelle ConditionalGaps)
   - dem Netz desselben D ohne Zusätze (models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl:
     Zeiten aus dem Decoder, keine Zustands- und Übergangsregeln)
Nichts wird geschrieben außer in diesen Ordner. Aufruf im Ordner process-simulation-engine.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d_ohne_ziehungen as base  # Umgebung, Pfade, Auswahl der Zustände
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent/"leads_meilensteine"
PLAIN = base.ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl"
MILESTONES = ["W_Complete application|start", "A_Concept|complete", "A_Accepted|complete", "O_Create Offer|complete",
              "O_Sent (mail and online)|complete", "W_Validate application|start", "ENDE"]
DRAWS = 100


def milestones(tokens, gaps_days):
    """Tage ab Entscheidung bis zum ersten Auftreten jedes Meilensteins (NaN wenn nicht erreicht); ENDE = Summe."""
    out, total = {}, 0.0
    for token, gap in zip(tokens, gaps_days):
        total += gap
        if token in MILESTONES and token not in out:
            out[token] = total
    out["ENDE"] = total
    return {m: out.get(m, np.nan) for m in MILESTONES}


def truth():
    rct = pd.read_pickle(base.ROOT/"prescriptive/scripts/results_100k/d_rct_tf1_training_20261114/rct_data/rct_bpic17_angebot_nachlegen_delta50/log.pkl")
    rct = rct[rct["rct"].astype(bool)].sort_values([base.CASE, base.TIME], kind="stable")
    rows = []
    for case, part in rct.groupby(base.CASE, sort=False):
        tokens = (part[base.ACT].astype(str).str.strip()+"|"+part[base.LIFE].astype(str)).tolist()
        for i in range(len(tokens)-1):
            if tokens[i] == "W_Handle leads|schedule" and tokens[i+1] in ("W_Handle leads|start", "W_Handle leads|withdraw"):
                times = part[base.TIME].to_numpy()
                gaps = np.diff(times[i:]).astype("timedelta64[ms]").astype(float)/86400000
                rows.append({"option": tokens[i+1], **milestones(tokens[i+1:], gaps), "events": len(tokens)-i-1})
                break
    return pd.DataFrame(rows)


def work(task):
    number, states = task
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    torch.set_num_threads(1)
    models = {"D online (Abstandstabelle)": PSPModel.load(base.MODEL), "D-Netz (Decoder-Zeiten)": PSPModel.load(PLAIN)}
    events = {seed: pd.read_pickle(base.SOURCE/base.RUN.format(seed)/"events.pkl") for seed in sorted({s["seed"] for s in states})}
    rows = []
    for index, s in enumerate(states):
        part = events[s["seed"]]
        part = part.loc[part[base.CASE].astype(str).eq(s["case"])].sort_values(base.TIME, kind="stable").reset_index(drop=True)
        stamps = part[base.TIME].map(lambda t: pd.Timestamp(t).timestamp()).to_numpy()
        prefix = part.iloc[:int(np.searchsorted(stamps, s["time"]+1e-6, side="right"))].to_dict("records")
        for label, model in models.items():
            model.set_threads(1)
            torch.manual_seed(7000+97*number+index)
            drawn, _ = model.option_suffixes(model.points[s["point"]], prefix, list(s["options"]), s["time"], DRAWS)
            for option, suffixes in zip(s["options"], drawn):
                for suffix in suffixes:
                    tokens = [name for name, _, _ in suffix]
                    rows.append({"modell": label, "case": s["case"], "option": option, "events": len(tokens),
                                 **milestones(tokens, [gap/base.DAY for _, _, gap in suffix])})
    return pd.DataFrame(rows)


def summarize(frame, label):
    a, b = frame[frame.option == "W_Handle leads|start"], frame[frame.option == "W_Handle leads|withdraw"]
    row = {"Quelle": label, "n start": len(a), "n withdraw": len(b)}
    for m in MILESTONES + ["events"]:
        row[m] = a[m].mean()-b[m].mean()
    for m in ("W_Complete application|start", "O_Create Offer|complete", "ENDE"):
        row[f"erreicht {m[:12]} start/withdraw"] = f"{a[m].notna().mean():.2f}/{b[m].notna().mean():.2f}"
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    states = base.pick()
    states = states[states.point == "handle_leads"].to_dict("records")
    tasks = [(w, states[w::5]) for w in range(5)]
    with ProcessPoolExecutor(5, mp_context=multiprocessing.get_context("spawn")) as pool:
        drawn = pd.concat(list(pool.map(work, tasks)), ignore_index=True)
    real = truth()
    drawn.to_pickle(OUT/"ziehungen.pkl")
    real.to_pickle(OUT/"simulator_rct.pkl")
    table = pd.DataFrame([summarize(real, "Simulator, randomisiert")] +
                         [summarize(part, label) for label, part in drawn.groupby("modell", sort=False)])
    table.to_csv(OUT/"meilensteine.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("Mehr Tage bis zum Meilenstein mit start statt withdraw (positiv = start langsamer):")
        print(table.set_index("Quelle").T.to_string())
        for label, part in [("Simulator", real)] + list(drawn.groupby("modell", sort=False)):
            print(f"\n{label}: mittlere Tage ab Entscheidung")
            print(part.groupby("option")[MILESTONES].mean().round(2).to_string())
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()

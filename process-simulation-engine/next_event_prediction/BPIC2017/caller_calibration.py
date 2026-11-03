"""
Kalibriert den Effekt des Anrufers (W_Call incomplete files) im Simulator auf den bereinigten Effekt im echten Log.

Im gelernten Simulator ist der Anrufer-Effekt auf die Bewilligung etwa halb so groß wie im echten Log
(prescriptive/data/diag/caller_sim_vs_real.py). Hier bekommt jeder Anrufer einen Zuschlag b_r, solange er der Anrufer
des Falls ist (EventTimeModel.caller_offsets): +b_r auf den Logit von O_Accepted (danach folgt immer A_Pending), -b_r auf
die Logits der Enden ohne Bewilligung (A_Denied, A_Cancelled). Nur auf O_Accepted verschob der Zuschlag vor allem den
Zeitpunkt: selbst bei |b| bis 1,9 blieb die Streuung des Effekts gleich.

Ziel je Anrufer: sein bereinigter Effekt im echten Log (Bewilligung minus erwartete Bewilligung), geschrumpft zur 0 nach
seiner Fallzahl (empirischer Bayes: Anrufer mit wenigen Fällen sind unsicher), für Anrufer mit >= 100 Fällen im echten
Log und eigenem Eingang im Simulator; alle anderen bleiben bei 0.
Iteration: RCT simulieren (Anrufer zufällig), Effekt je Anrufer messen (Bewilligung minus Mittel, zentriert),
b_r += Dämpfung * (Ziel - gemessen) / (p (1 - p)). Die Zuschläge werden so zentriert, dass die Bewilligung insgesamt bleibt.

Aufruf (aus process-simulation-engine/):
    python -m next_event_prediction.BPIC2017.caller_calibration --source <Modellordner> --target <neuer Ordner> --real <CSV von caller_sim_vs_real>
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from ..event_time_model import EventTimeModel


def targets(real_csv, callers, min_cases=100):
    real = pd.read_csv(real_csv, index_col=0)
    real = real[(real["n_real"] >= min_cases) & real.index.isin(callers)]
    noise = real["roh_real"] * (1 - real["roh_real"]) / real["n_real"]
    w = real["n_real"]
    effect = real["effekt_real"] - np.average(real["effekt_real"], weights=w)
    tau2 = max(np.average(effect ** 2, weights=w) - np.average(noise, weights=w), 1e-6)
    return effect * tau2 / (tau2 + noise), tau2


def measure(data):
    d = data[(data["stage"] == 0) & (data["treatment"] == 1) & data["call_resource"].notna()]
    g = d.groupby("call_resource")["approved"].agg(["mean", "size"])
    return g["mean"] - d["approved"].mean(), g["size"], float(d["approved"].mean())


if __name__ == "__main__":
    from prescriptive.main import run_rct
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--damping", type=float, default=0.7)
    parser.add_argument("--runs", type=int, default=6, help="Simulated RCT years per iteration (parallel)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tokens", type=str, default="O_Accepted|complete:+1,A_Denied|complete:-1,A_Cancelled|complete:-1",
                        help="Tokens and signs of the offset")
    args = parser.parse_args()
    if os.path.exists(args.target):
        shutil.rmtree(args.target)
    shutil.copytree(args.source, args.target)
    model = EventTimeModel.load(args.target)
    callers = model.encoder.callers
    goal, tau2 = targets(args.real, callers)
    print(f"Ziele für {len(goal)} Anrufer; Streuung echt (bereinigt, ohne Zufall) {np.sqrt(tau2):.3f}, Ziel-Streuung {goal.std():.3f}", flush=True)
    model.caller_offsets = np.zeros(len(callers))
    pairs = [item.rsplit(":", 1) for item in args.tokens.split(",")]
    model.offset_tokens = np.array([model.encoder.token_to_id[name] for name, _ in pairs])
    model.offset_signs = np.array([float(sign) for _, sign in pairs])
    index = {name: i for i, name in enumerate(callers)}
    config = {"event_log_path": "eventlog/eventlog.xes.gz", "model_folder": args.target, "allocation": "learned", "delta": 0.0}
    history = []
    for iteration in range(args.iterations + 1):
        model.save(args.target)
        data = run_rct(31509, args.runs, args.seed, args.runs, config)
        effect, size, p = measure(data)
        effect = effect.reindex(goal.index)
        w = size.reindex(goal.index).fillna(0)
        effect = effect - np.average(effect.fillna(0), weights=w)
        error = goal - effect
        corr = np.corrcoef(goal, effect.fillna(0))[0, 1]
        history.append({"Iteration": iteration, "Bewilligung nach Anruf": p, "Korrelation Ziel/gemessen": corr,
                        "Streuung gemessen": float(np.sqrt(np.average(effect.fillna(0) ** 2, weights=w))), "mittlerer Fehler (abs)": float(np.average(error.abs(), weights=w))})
        print(history[-1], flush=True)
        if iteration == args.iterations:
            break
        for name, e in error.items():
            if np.isfinite(e):
                model.caller_offsets[index[name]] += args.damping * e / (p * (1 - p))
        # Zentrieren (gewichtet nach Anrufen), damit die Bewilligung insgesamt gleich bleibt
        ids = [index[name] for name in goal.index]
        model.caller_offsets[ids] -= np.average(model.caller_offsets[ids], weights=np.maximum(w.to_numpy(), 1))
    with open(os.path.join(args.target, "calibration.json"), "w") as f:
        json.dump({"history": history, "goal": goal.to_dict(), "offsets": {name: float(model.caller_offsets[index[name]]) for name in goal.index}}, f, indent=1)

"""Suffix-Verlust: Verschiebung der gezogenen Restzeit je Wahlpunkt und Arm (2026-11-19, vor der Gewichtung je Punkt).

Echte Testfälle (Testfälle der T-Learner-Arme): je Wahlpunkt und Option der erste Besuch, an dem auf das Entscheidungs-Event die Option
folgt; Prefix bis einschließlich der Option. Je Prefix DRAWS gezogene Suffixe aus dem zuständigen Arm vorher (T-Learner ZZ-TF1) und
nachher (Suffix-Verlust); verglichen mit der echten Restzeit ab der Option (beobachtet, mit Auswahl). Dazu die Zahl der Proben je Punkt
und Arm im ganzen Log (alle Besuche; ≈ Mischung der Trainingsproben im Suffix-Verlust).
Ausgabe je_punkt.csv / je_punkt.md. Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> [Fälle] [Ziehungen] [Modellsatz]
Modellsatz: Kürzel der neuen Arme (Standard sv → ..._t_{aktion,referenz}_sv.pkl).
2026-11-20: zusätzlich rauschfreie Streuung der Fallmittel je Zelle (Varianz der Mittel minus Ziehungsrauschen) und SD der Ziehungen.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ARGS, sys.argv = sys.argv[1:], sys.argv[:1]  # sonde.py liest sys.argv beim Import
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent/"zeitkopf_sonde_20261118"))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CASES = int(ARGS[0]) if len(ARGS) > 0 else 100
DRAWS = int(ARGS[1]) if len(ARGS) > 1 else 100
NEW = ARGS[2] if len(ARGS) > 2 else "sv"
M = ROOT/"prescriptive/models"
BASE = "psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_{arm}{suffix}.pkl"
AKTION = {"A_Accepted|complete", "O_Create Offer|complete", "W_Handle leads|start"}
KURZ = {"annehmen_bei_konzept": "konzept", "annehmen_nach_vervollstaendigung": "nach V.", "angebot_nachlegen": "nachlegen", "handle_leads": "leads"}


def token(event):
    return f"{event['concept:name']}|{event['lifecycle:transition']}"


def cells():
    import sonde
    from prescriptive.estimators.psp_model import PSPModel
    from interventions import option_token
    model = PSPModel.load(str(M/BASE.format(arm="aktion", suffix="")))
    grouped = sonde.load_raw()
    test = sorted(set(map(str, model.test_cases)))
    out, counts = [], []
    for name, point in model.points.items():
        decision = f"{point.activity}|{point.lifecycle}"
        for option in point.options:
            opt = option_token(option)
            arm = "aktion" if opt in AKTION else "referenz"
            prefixes, real = [], []
            for case in test:
                events = grouped.get(case) or []
                for i in range(1, len(events)):
                    if token(events[i - 1]) == decision and token(events[i]) == opt:
                        prefixes.append(events[:i + 1])
                        real.append((events[-1]["time:timestamp"] - events[i]["time:timestamp"]).total_seconds() / 86400)
                        break
                if len(prefixes) >= CASES:
                    break
            out.append({"punkt": KURZ[name], "option": opt, "arm": arm, "prefixes": prefixes, "real": np.array(real)})
            n = sum(1 for events in grouped.values() for i in range(1, len(events))
                    if token(events[i - 1]) == decision and token(events[i]) == opt)
            counts.append({"punkt": KURZ[name], "arm": arm, "Proben im Log (alle Besuche)": n})
    return out, pd.DataFrame(counts)


def work(task):
    """Eine Zelle: einmal kodieren (gleicher Encoder in beiden Modellen), dann vorher und nachher ziehen (je 100 Prefixe auf einmal).
    Ziehungen und Modellsatz kommen mit der Aufgabe (die Arbeitsprozesse sehen sys.argv ohne die Argumente)."""
    cell, draws, new = task
    import torch
    from prescriptive.estimators.psp_model import PSPModel
    out = []
    encoded = None
    for suffix in ("", f"_{new}"):
        model = PSPModel.load(str(M/BASE.format(arm=cell["arm"], suffix=suffix)))
        model.set_threads(4)
        if encoded is None:
            encoded = [model._encode(p)[:4] for p in cell["prefixes"]]
        else:  # gleicher Encoder? Stichprobe von 3 Prefixen
            assert repr([model._encode(p)[:4] for p in cell["prefixes"][:3]]) == repr(encoded[:3]), "verschiedene Kodierung vorher/nachher"
        torch.manual_seed(7)
        means, spreads = [], []
        for start in range(0, len(encoded), 100):
            drawn = model.draw_cases(encoded[start:start + 100], draws)
            rests = [[sum(g for _, _, g in s) / 86400 for s in suffixes] for suffixes in drawn]
            means += [np.mean(r) for r in rests]
            spreads += [np.var(r, ddof=1) for r in rests]
        out.append((np.array(means), np.array(spreads)))
    print(f"fertig: {cell['punkt']} {cell['arm']} ({len(encoded)} Prefixe)", flush=True)
    return out


def main():
    from scipy.stats import spearmanr
    table, counts = cells()
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:
        means = list(pool.map(work, [(cell, DRAWS, NEW) for cell in table]))
    rows = []
    for k, cell in enumerate(table):
        (before, var_before), (after, var_after) = means[k]
        real = cell["real"]

        def between(m, v):  # rauschfreie Streuung der Fallmittel: Varianz der Mittel minus Ziehungsrauschen je Mittel
            return float(np.sqrt(max(0.0, np.var(m, ddof=1) - np.mean(v) / DRAWS)))
        rows.append({"Punkt": cell["punkt"], "Option": cell["option"], "Arm": cell["arm"], "Fälle": len(real), "echt (d)": real.mean(),
                     "gezogen vorher (d)": before.mean(), "gezogen nachher (d)": after.mean(),
                     "Fehler vorher (d)": before.mean() - real.mean(), "Fehler nachher (d)": after.mean() - real.mean(),
                     "Verschiebung (d)": after.mean() - before.mean(),
                     "Rang vorher": spearmanr(before, real)[0], "Rang nachher": spearmanr(after, real)[0],
                     "Streuung Fallmittel vorher (d)": between(before, var_before), "Streuung Fallmittel nachher (d)": between(after, var_after),
                     "SD Ziehungen vorher (d)": float(np.sqrt(np.mean(var_before))), "SD Ziehungen nachher (d)": float(np.sqrt(np.mean(var_after)))})
    out = pd.DataFrame(rows)
    out.to_csv(HERE/f"je_punkt_{NEW}_{CASES}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        text = out.round(2).to_string(index=False)
        text_counts = counts.to_string(index=False)
    (HERE/f"je_punkt_{NEW}_{CASES}.md").write_text(f"# Gezogene Restzeit je Punkt und Arm, vorher / nachher ({NEW})\n\n{CASES} Testfälle je Zelle, {DRAWS} Ziehungen\n\n"
                                           f"```\n{text}\n```\n\nProben im ganzen Log:\n\n```\n{text_counts}\n```\n")
    print(text)
    print(text_counts)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()

"""Zieht _sequence_draws (Training, sequence_loss) wie draw_cases (Bewertung)? (2026-11-19)

T-Learner-Arme (Aktion: Prefix bis einschließlich O_Create Offer nach dem ersten O_Created; Referenz: bis einschließlich O_Sent),
je 40 Testfälle des echten Logs, je Prefix 200 Ziehungen mit beiden Wegen; verglichen wird die gezogene Restzeit (Summe der
Abstände, Tage): Mittel, Streuung, Mittel je Prefix (Korrelation), Anteil an der Schrittgrenze.
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript>
"""
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent/"zeitkopf_sonde_20261118"))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import sonde  # noqa: E402
from prescriptive.estimators import psp_model as pm  # noqa: E402

M = ROOT/"prescriptive/models"
ARMS = {"Aktion": (M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion.pkl", "O_Create Offer"),
        "Referenz": (M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz.pkl", "O_Sent (mail and online)")}
CASES, DRAWS = 40, 200


def prefixes(grouped, test_cases, option):
    out = []
    for case in sorted(test_cases):
        events = grouped.get(case)
        if not events:
            continue
        names = [e["concept:name"] for e in events]
        if "O_Created" not in names:
            continue
        i = names.index("O_Created")
        if i + 1 < len(names) and names[i + 1] == option:
            out.append(events[:i + 2])
        if len(out) >= CASES:
            break
    return out


def main():
    torch.set_num_threads(8)
    grouped = sonde.load_raw()
    for label, (path, option) in ARMS.items():
        model = pm.PSPModel.load(str(path))
        model.set_threads(8)
        ev = model._evaluator()
        encoded = [model._encode(p)[:4] for p in prefixes(grouped, set(map(str, model.test_cases)), option)]
        torch.manual_seed(1)
        old = np.array([[sum(g for _, _, g in s) / pm.DAY for s in suffixes] for suffixes in model.draw_cases(encoded, DRAWS)])
        scaler = model.encoder_decoder.continuous_encoders[pm.TIME_COLUMNS[1]]
        settings = {**pm.SEQUENCE, "draws": DRAWS, "eos_id": int(ev.eos_id), "window": int(model.encoder_decoder.window_size),
                    "suffix": int(model.encoder_decoder.min_suffix_size), "gap_mean": float(scaler.mean_[0]), "gap_scale": float(scaler.scale_[0])}
        cats, nums, static, mask = model._batch(encoded, 1)
        torch.manual_seed(2)
        with torch.no_grad():
            new = pm._sequence_draws(model.net, [cats, nums], static, mask, settings).numpy()
        lengths = np.array([e[3] for e in encoded])
        steps = np.clip(settings["window"] - settings["suffix"] - lengths + 1, 1, None)
        print(f"{label}: {len(encoded)} Prefixe, Schrittgrenze draw_cases {steps.min()}–{steps.max()} (Kappe max_steps {settings['max_steps']})")
        print(f"  draw_cases:      Mittel {old.mean():6.2f} d, SD {old.std():6.2f}, Median {np.median(old):6.2f}")
        print(f"  _sequence_draws: Mittel {new.mean():6.2f} d, SD {new.std():6.2f}, Median {np.median(new):6.2f}")
        print(f"  Mittel je Prefix: Korrelation {np.corrcoef(old.mean(1), new.mean(1))[0, 1]:.3f}, "
              f"mittlere Differenz {np.mean(new.mean(1) - old.mean(1)):+.2f} d (SE {np.std(new.mean(1) - old.mean(1)) / np.sqrt(len(old)):.2f})", flush=True)


if __name__ == "__main__":
    main()

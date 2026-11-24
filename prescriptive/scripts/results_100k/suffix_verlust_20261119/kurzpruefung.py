"""Kurzprüfung nach dem Rauchtest (2026-11-19): gezogene Restzeit des Aktionsarms an echten nachlegen-Prefixen mit Zusatzangebot
(Prefix bis einschließlich O_Create Offer nach dem ersten O_Created), vorher (fertiger T-Learner-Arm) und nach dem Training mit
sequence_loss, gegen die echte Restzeit derselben Testfälle; dazu die Rangkorrelation je Prefix (Signal zwischen Fällen).
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> <Modell nach dem Training> [Fälle] [Ziehungen]
"""
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
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
import sonde  # noqa: E402
from prescriptive.estimators import psp_model as pm  # noqa: E402

BEFORE = ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion.pkl"
AFTER = Path(ARGS[0])
CASES = int(ARGS[1]) if len(ARGS) > 1 else 100
DRAWS = int(ARGS[2]) if len(ARGS) > 2 else 100


def main():
    torch.set_num_threads(8)
    grouped = sonde.load_raw()
    before = pm.PSPModel.load(str(BEFORE))
    prefixes, real = [], []
    for case in sorted(set(map(str, before.test_cases))):
        events = grouped.get(case)
        names = [e["concept:name"] for e in events] if events else []
        if "O_Created" in names:
            i = names.index("O_Created")
            if i + 1 < len(names) and names[i + 1] == "O_Create Offer":
                prefixes.append(events[:i + 2])
                real.append((events[-1]["time:timestamp"] - events[i + 1]["time:timestamp"]).total_seconds() / pm.DAY)
        if len(prefixes) >= CASES:
            break
    real = np.array(real)
    print(f"{len(prefixes)} Testfälle mit Zusatzangebot am ersten nachlegen; echte Restzeit ab O_Create Offer im Mittel {real.mean():.2f} d")
    for label, path in (("vorher (T-Learner-Arm)", BEFORE), ("nachher (sequence_loss)", AFTER)):
        model = pm.PSPModel.load(str(path))
        model.set_threads(8)
        encoded = [model._encode(p)[:4] for p in prefixes]
        torch.manual_seed(7)
        drawn = np.array([[sum(g for _, _, g in s) / pm.DAY for s in suffixes] for suffixes in model.draw_cases(encoded, DRAWS)])
        means = drawn.mean(axis=1)
        print(f"  {label:26s} gezogen im Mittel {drawn.mean():6.2f} d (SD je Ziehung {drawn.std():5.2f}), "
              f"Rangkorrelation Mittel je Prefix ↔ echt {spearmanr(means, real)[0]:+.3f}, Streuung der Mittel je Prefix {means.std():.2f} d", flush=True)


if __name__ == "__main__":
    main()

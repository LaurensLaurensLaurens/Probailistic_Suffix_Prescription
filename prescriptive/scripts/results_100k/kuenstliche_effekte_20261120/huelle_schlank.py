"""Schlanke Kopie einer PSP-Hülle für die Online-Auswertung (2026-11-23). Die Netze vom 4-fachen v3-Log tragen den ganzen kodierten
Log mit (encoder_decoder.event_log, ≈ 0,6 GB je Netz; die Hülle mit 8 Netzen belegt ≈ 10 GB je Prozess). Der Log wird nur beim
Training gelesen (TensorEncoderDecoder.__init__ für window_size "auto" und train_imputers_encoders), nicht beim Kodieren oder Ziehen.
Hier: event_log in Hülle und Netzen durch eine leere Tabelle mit denselben Spalten ersetzen, speichern, dann Gleichheit prüfen –
kurzer Simulatorlauf (Effekte v3, Aufbau wie online_konformanz.py) mit KonformanzPolicy aus Original und schlanker Kopie, gleiche Saat;
Entscheidungen und erwartete Abweichungen müssen identisch sein.
Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> <hülle.pkl> <ziel.pkl> [Fälle]
(online_konformanz liest sys.argv beim Import; die Argumente gehen daher über Umgebungsvariablen an die Arbeitsprozesse.)
"""
import os
from pathlib import Path
import sys
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
if len(sys.argv) > 2:  # Hauptprozess: Argumente in die Umgebung, argv leeren (auch für die gestarteten Arbeitsprozesse)
    os.environ.update(SCHLANK_QUELLE=sys.argv[1], SCHLANK_ZIEL=sys.argv[2], SCHLANK_FAELLE=sys.argv[3] if len(sys.argv) > 3 else "40")
    del sys.argv[1:]
import online_konformanz as ok  # noqa: E402
import pandas as pd  # noqa: E402


def verschlanken(quelle, ziel):
    from prescriptive.estimators.psp_model import PSPModel
    hull = PSPModel.load(str(quelle))
    modelle = [hull] + list(dict.fromkeys(getattr(hull, "netze", {}).values()))
    for m in modelle:
        ed = m.encoder_decoder
        if isinstance(getattr(ed, "event_log", None), pd.DataFrame):
            ed.event_log = ed.event_log.head(0).copy()
    hull.save(str(ziel))
    print(f"schlank gespeichert: {ziel} ({Path(ziel).stat().st_size / 1e6:.0f} MB statt {Path(quelle).stat().st_size / 1e6:.0f} MB)", flush=True)


def entscheidungen(pfad, faelle, seed=7):
    import torch
    torch.set_num_threads(1)
    import kuenstliche_effekte as ke
    ke.aktivieren(ke.laden(ok.EFFEKTE))
    from prescriptive.policies.konformanz_policy import KonformanzPolicy
    policy = KonformanzPolicy(str(pfad), seed=seed, samples=100, first_only=ok.FIRST_ONLY)
    ok.simulator().run(faelle, policy=policy, window=(0, faelle), seed=seed)
    return pd.DataFrame(policy.entscheidungen)


def main():
    quelle, ziel = Path(os.environ["SCHLANK_QUELLE"]), Path(os.environ["SCHLANK_ZIEL"])
    faelle = int(os.environ["SCHLANK_FAELLE"])
    if not ziel.exists():
        verschlanken(quelle, ziel)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        a, b = pool.map(entscheidungen, [quelle, ziel], [faelle, faelle])
    print(f"Entscheidungen: Original {len(a)}, schlank {len(b)}")
    gleich = len(a) == len(b) and len(a) > 0 and a.astype(str).equals(b.astype(str))
    print(a.head(6).to_string())
    print("GLEICH" if gleich else "VERSCHIEDEN", flush=True)
    sys.exit(0 if gleich else 1)


if __name__ == "__main__":
    main()

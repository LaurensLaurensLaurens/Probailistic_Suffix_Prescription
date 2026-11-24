"""Rauchtest sequence_loss (2026-11-19): kurzer Trainingslauf des T-Learner-Aktionsarms mit dem Suffix-Verlust, wie
data/bpic17_psp_t_learner.sh (warm vom fertigen Arm), aber wenige Fälle und Epochen; Modell und Checkpoints nur hier im Ordner
(helpers.training.MODELS umgelenkt), nichts unter prescriptive/models.
Aufruf im Ordner process-simulation-engine: PSP_CATALOG=<catalog_neue_punkte_zuteilung_random.yaml> .venv/bin/python -u <dieses Skript> [Fälle] [Epochen] [Gewicht] [Kalibrierung]
"""
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
cases, epochs = (sys.argv[1] if len(sys.argv) > 1 else "3000"), (sys.argv[2] if len(sys.argv) > 2 else "2")
weight = sys.argv[3] if len(sys.argv) > 3 else "1.0"
calibration = sys.argv[4] if len(sys.argv) > 4 else None  # 2026-11-19 abends: Kalibrierung je Gruppe
sys.path.insert(0, str(ROOT/"prescriptive/scripts"))
sys.path.insert(0, str(ROOT/"prescriptive"))
import helpers.training as training  # noqa: E402
training.MODELS = str(HERE/"rauchtest_modelle")
Path(training.MODELS).mkdir(exist_ok=True)
import train_psp_real  # noqa: E402
tag = f"rauchtest_{time.strftime('%H%M%S')}"
sys.argv = ["train_psp_real.py", "--tag", tag, "--cases", cases, "--epochs", epochs, "--learning-rate", "3e-5", "--point-weight", "3",
            "--catalog-points", "--suffix-size", "60", "--statics-without-future", "--time-from", "gaps", "--counters", "--time-on-event",
            "--teacher-forcing", "1.0", "--option-in-encoder-after-training", "--only-points", "--only-points-validation", "--arm", "aktion",
            "--init-from", str(ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion.pkl"),
            "--sequence-loss", "--sequence-weight", weight]
if calibration:
    sys.argv += ["--sequence-calibration", calibration]
start = time.time()
train_psp_real.main()
print(f"Rauchtest fertig nach {(time.time() - start) / 60:.1f} min", flush=True)

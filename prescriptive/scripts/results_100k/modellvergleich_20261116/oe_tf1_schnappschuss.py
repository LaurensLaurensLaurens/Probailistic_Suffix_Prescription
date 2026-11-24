"""Auswertbare Kopien des laufenden OE-TF1-Trainings (2026-11-16), ohne das Training zu berühren.

Kopiert model.pkl, last.pt und best_val.pt aus models/psp_training_real_alle_s60_token_oe_tf1_e100_20261115 in diesen
Ordner (oe_tf1_schnappschuss/) und baut daraus zwei Modelle wie der Lauf nach dem Training: Netz aus dem Checkpoint,
danach params option_as "encoder" (Optionsabstand je Suffix, keine Übergangssperren). In models/ wird nichts geschrieben.
Aufruf im Ordner process-simulation-engine.
"""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/"prescriptive/models/psp_training_real_alle_s60_token_oe_tf1_e100_20261115"
OUT = Path(__file__).resolve().parent/"oe_tf1_schnappschuss"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PSP_CATALOG", str(ROOT/"prescriptive/data/catalog_neue_punkte_zuteilung_random.yaml"))
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()
import torch
from prescriptive.estimators.psp_model import PSPModel


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("model.pkl", "last.pt", "best_val.pt"):
        if not (OUT/name).exists():  # einmal kopierter Stand bleibt fest
            shutil.copy2(SOURCE/name, OUT/name)
    # model.pkl des Checkpoint-Ordners entsteht vor dem Ende von fit: Testfälle setzt train_psp_real.py erst danach.
    # Derselbe Fall-Split wie bei allen Vollmodellen (helpers.training.test_cases); übernommen vom ursprünglichen OE.
    reference = PSPModel.load(ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_token_oe.pkl")
    info = {}
    for label, checkpoint in (("letzte", "last.pt"), ("beste_val", "best_val.pt")):
        state = torch.load(OUT/checkpoint, weights_only=False, map_location="cpu")
        model = PSPModel.load(OUT/"model.pkl")
        model.net = PSPModel._load_net(str(OUT/checkpoint))
        model.net.eval()
        model.params["option_as"] = "encoder"
        assert model.gap_tokens() and model.option_in_encoder() and not getattr(model, "transition_rules", None)
        model.test_cases, model.trained_on = reference.test_cases, {"echter Log": "bpic17", "outcome": "durchlaufzeit"}
        assert len(model.test_cases) == 6301 and model.seed == reference.seed and model.points.keys() == reference.points.keys()
        path = OUT/f"psp_oe_tf1_{label}.pkl"
        model.save(path)
        info[label] = {"checkpoint": checkpoint, "epoch": state.get("epoch"), "keys": sorted(k for k in state if k != "model_state_dict"),
                       "model": str(path), "params": {k: v for k, v in model.params.items() if k not in ("statics", "checkpoints")}}
    status = json.loads((ROOT/"prescriptive/scripts/results_100k/oe_tf1_100_training_20261115/status.json").read_text())
    best = json.loads((ROOT/"prescriptive/scripts/results_100k/oe_tf1_100_training_20261115/best_checkpoint.json").read_text())
    info["laufstatus"] = {"epochs_completed": status.get("epochs_completed"), "last_epoch": status.get("last_epoch"), "best": best}
    (OUT/"info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(info, indent=1, ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()

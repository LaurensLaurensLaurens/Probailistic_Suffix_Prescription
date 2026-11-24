"""Referenzziehungen bestehender Modelle vor/nach dem Umbau (2026-11-16): gleiche Seeds müssen gleiche Suffixe liefern.
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> vorher|nachher"""
import os, sys, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PSP_CATALOG", str(ROOT/"prescriptive/data/catalog_neue_punkte_zuteilung_random.yaml"))
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()
import pandas as pd, torch
from prescriptive.estimators.psp_model import PSPModel
M = ROOT/"prescriptive/models"
MODELS = {"(c)": "psp_model_durchlaufzeit_real_alle_s60_normal.pkl", "D": "psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl",
          "(e)": "psp_model_durchlaufzeit_real_alle_s60_klassen_event_oe.pkl", "OE": "psp_model_durchlaufzeit_real_alle_s60_token_oe.pkl"}
raw = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
if raw["time:timestamp"].dt.tz is not None:  # wie die Engine: naive UTC
    raw["time:timestamp"] = raw["time:timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
torch.set_num_threads(2)
out = {}
for label, name in MODELS.items():
    model = PSPModel.load(M/name)
    cases = sorted(map(str, model.test_cases))[:3]
    for case in cases:
        part = raw[raw["case:concept:name"].astype(str) == case].sort_values("time:timestamp", kind="stable")
        events = part.to_dict("records")
        # Prefix bis zum ersten O_Created (Wahlpunkt angebot_nachlegen), sonst 8 Events
        cut = next((i+1 for i, e in enumerate(events) if str(e["concept:name"]).strip() == "O_Created"), 8)
        prefix = events[:cut]
        torch.manual_seed(5)
        out[(label, case, "frei")] = model.draw_cases([model._encode(prefix)[:4]], 5)[0]
        if cut < len(events) and str(prefix[-1]["concept:name"]).strip() == "O_Created":
            point = model.points["angebot_nachlegen"]
            torch.manual_seed(6)
            now = pd.Timestamp(prefix[-1]["time:timestamp"]).timestamp()
            out[(label, case, "option")] = model.option_suffixes(point, prefix, list(point.options), now, 4)
path = Path(__file__).with_name(f"referenz_{sys.argv[1]}.pkl")
pickle.dump(out, open(path, "wb"))
print(len(out), "Ziehungsgruppen gespeichert in", path)
if sys.argv[1] == "nachher":
    before = pickle.load(open(Path(__file__).with_name("referenz_vorher.pkl"), "rb"))
    same = {k: repr(before[k]) == repr(out[k]) for k in before}
    print("identisch:", sum(same.values()), "von", len(same), [k for k, v in same.items() if not v])

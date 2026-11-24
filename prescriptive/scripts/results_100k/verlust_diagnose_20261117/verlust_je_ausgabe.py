"""Diagnose 2026-11-17: Verlust je Ausgabe mit Teacher Forcing 1 auf echten Testfällen (Frage: warum fällt der Trainingsverlust von
ZZ-TF1 nicht, anders als bei (c)-TF1?). Je Modell dieselben Prefixe (zufälliger Schnitt, danach ≥ L echte Events), Ziel = die nächsten
L echten Events. Je Ausgabe: Aktivität (Kreuzentropie der mittleren Logits, mittlere Logit-Log-Varianz), Fallzeit und Abstand
(attenuierter Verlust wie im Training mit Log-Varianz auf [−6, 6] begrenzt, MSE, mittlere Log-Varianz, MAE in Tagen).
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> [Fälle]"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PSP_CATALOG", str(ROOT/"prescriptive/data/catalog_neue_punkte_zuteilung_random.yaml"))
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from prescriptive.estimators.psp_model import PSPModel  # noqa: E402

OUT = Path(__file__).resolve().parent
N_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
L = 10
R = ROOT/"prescriptive/scripts/results_100k"
MODELS = {"(c)": ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl",
          "(c)-TF1": ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_tf1.pkl",
          "ZZ ep3": R/"zaehler_zeit_20261116/modelle/psp_zz_ep3.pkl",
          "ZZ final": ROOT/"prescriptive/models/psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit.pkl",
          "ZZ-TF1 ep3": R/"zaehler_zeit_tf1_20261117/modelle/psp_zztf1_ep3.pkl",
          "ZZ-TF1 ep6": R/"zaehler_zeit_tf1_20261117/modelle/psp_zztf1_ep6.pkl"}
if len(sys.argv) > 2:  # andere Modelle: label=pfad ...; Ausgabe dann verlust_je_ausgabe_<erstes Label>.csv
    MODELS = {"(c)": MODELS["(c)"], **dict(a.split("=", 1) for a in sys.argv[2:])}
raw = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
raw["time:timestamp"] = raw["time:timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
raw["concept:name"] = raw["concept:name"].astype(str).str.strip()
reference = PSPModel.load(str(MODELS["(c)"]))
rng = np.random.default_rng(0)
cases = []
grouped = {c: p for c, p in raw.groupby("case:concept:name", sort=False)}
for case in rng.permutation(sorted(map(str, reference.test_cases))):
    part = grouped.get(case)
    if part is None or len(part) < L + 6:
        continue
    events = part.sort_values("time:timestamp", kind="stable").to_dict("records")
    cut = int(rng.integers(5, len(events) - L + 1))
    cases.append((events[:cut], events[:cut + L]))
    if len(cases) >= N_CASES:
        break
rows = []
for label, path in MODELS.items():
    model = PSPModel.load(str(path))
    net, ev = model.net, model._evaluator()
    torch.manual_seed(0)
    prefixes = [model._encode(p) for p, _ in cases]
    fulls = [model._encode(f) for _, f in cases]
    cats, nums, static, mask = model._batch([(e[0], e[1], e[2], e[3]) for e in prefixes], 1)
    fcats, fnums, _, _ = model._batch([(e[0], e[1], e[2], e[3]) for e in fulls], 1)
    S = net.seq_len_pred
    s_cats = [torch.cat([c[:, -L:], torch.zeros(c.shape[0], S - L, dtype=c.dtype)], dim=1) for c in fcats]
    s_nums = [torch.cat([x[:, -L:], torch.zeros(x.shape[0], S - L, dtype=x.dtype)], dim=1) for x in fnums]
    net.train()
    with torch.no_grad():
        predictions = net(prefixes=[cats, nums], static_inputs=static, suffixes=[s_cats, s_nums], teacher_forcing_ratio=1.0, prefix_mask=mask)[0]
    net.eval()
    cat_names = [c[0] for c in net.data_set_categories[0]]
    num_names = [c[0] for c in net.data_set_categories[1]]
    row = {"modell": label}
    for key, value in predictions[0].items():
        if not key.endswith("_mean"):
            continue
        name = key[:-5]
        logits = value.reshape(S, len(cases), -1)[:L]
        target = s_cats[cat_names.index(name)][:, :L].T.long()
        row[f"{name} Kreuzentropie"] = float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)))
        row[f"{name} Treffer"] = float((logits.argmax(-1) == target).float().mean())
        row[f"{name} Logit-Logvar"] = float(predictions[0][f"{name}_var"].reshape(S, len(cases), -1)[:L].mean())
    for key, value in predictions[1].items():
        if not key.endswith("_mean"):
            continue
        name = key[:-5]
        mean = value.reshape(S, len(cases))[:L]
        logvar = predictions[1][f"{name}_var"].reshape(S, len(cases))[:L]
        target = s_nums[num_names.index(name)][:, :L].T.float()
        clamped = logvar.clamp(-6, 6)
        scale = model.encoder_decoder.continuous_encoders[name].scale_[0]
        row[f"{name} attenuiert"] = float((0.5 * (torch.exp(-clamped) * (target - mean) ** 2 + clamped)).mean())
        row[f"{name} MSE"] = float(((target - mean) ** 2).mean())
        row[f"{name} Logvar"] = float(logvar.mean())
        row[f"{name} Logvar P95"] = float(logvar.quantile(0.95))
        row[f"{name} MAE Tage"] = float((target - mean).abs().mean() * scale / 86400)
    rows.append(row)
    print(label, "fertig", flush=True)
frame = pd.DataFrame(rows).set_index("modell")
frame.to_csv(OUT/("verlust_je_ausgabe.csv" if len(sys.argv) <= 2 else "verlust_je_ausgabe_lernkurve.csv"))
pd.set_option("display.width", 250)
print(frame.T.round(3).to_string())

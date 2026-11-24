"""Prüfung von Zählern und Zeit je Event (2026-11-16) an einem kleinen Probemodell.
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> <modell.pkl>"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PSP_CATALOG", str(ROOT/"prescriptive/data/catalog_neue_punkte.yaml"))
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()
import numpy as np
import pandas as pd
import torch
from prescriptive.estimators.psp_model import PSPModel

torch.manual_seed(0)
model = PSPModel.load(sys.argv[1])
net = model.net
ok = {}
ok["params"] = model.counters() and model.time_on_event() and model.option_in_encoder()
ok["Gewichte gelernt (Encoder/Decoder-Zähler, Zeit je Event)"] = (
    float(net.encoder.count_weight.abs().sum()) > 0 and float(net.decoder.count_weight.abs().sum()) > 0
    and float(sum(p.abs().sum() for p in net.decoder.num_condition_weights.values())) > 0)

# 1. Training: Zähler je Decoderschritt = Prefix + echte Vorgänger (Teacher Forcing 1) bzw. eigene Vorhersagen (0)
cats_meta, nums_meta = net.data_set_categories
B, T, S = 4, 12, net.seq_len_pred
gen = torch.Generator().manual_seed(1)
prefix_cats = [torch.randint(1, size, (B, T), generator=gen) for _, size, *_ in cats_meta]
prefix_nums = [torch.rand(B, T, generator=gen) for _ in nums_meta]
suffix_cats = [torch.randint(1, size, (B, S), generator=gen) for _, size, *_ in cats_meta]
suffix_nums = [torch.rand(B, S, generator=gen) for _ in nums_meta]
prefix_cats[net.count_index][0, :3] = 0  # Auffüllung zählt nicht
static_cats, static_nums = net.static_data_set_categories
static = (torch.stack([torch.randint(1, size, (B,), generator=gen) for _, size, *_ in static_cats], dim=1),
          torch.rand(B, len(static_nums), generator=gen))
captured = []
original = net.decoder.forward


def capture(*args, **kwargs):
    captured.append((kwargs.get("counts").clone(), kwargs.get("input"), kwargs.get("pred")))
    return original(*args, **kwargs)


net.decoder.forward = capture
for ratio in (1.0, 0.0):
    captured.clear()
    net.train()
    with torch.no_grad():
        net(prefixes=[prefix_cats, prefix_nums], static_inputs=static, suffixes=[suffix_cats, suffix_nums], teacher_forcing_ratio=ratio)
    expected = net._counts_of(prefix_cats[net.count_index])
    good = torch.allclose(captured[0][0], expected)
    for t in range(1, len(captured)):
        _, inp, pred = captured[t]
        ids = inp[0][net.count_dec_position] if pred else suffix_cats[net.count_index][:, t-1:t]
        expected = expected + net._counts_of(ids)
        good &= torch.allclose(captured[t][0], expected)
    ok[f"Training: Zähler fortgeschrieben (Teacher Forcing {ratio})"] = bool(good) and len(captured) == S
net.decoder.forward = original
net.eval()

# 2. Encoder: kumulierte Zähler, letzte Position = Zähler des ganzen Prefix
enc = net.encoder.counts(prefix_cats[net.count_index])
ok["Encoder: letzte Position = Prefix-Zähler"] = torch.allclose(enc[:, -1], net._counts_of(prefix_cats[net.count_index]))

# 3. Ziehen: Buchführung in inference
with torch.no_grad():
    net.inference(prefix=[prefix_cats, prefix_nums], static_inputs=static)
    first = net._inference_counts.clone()
    step_ids = torch.randint(1, net.count_size, (B, 1), generator=gen)
    hx = (torch.zeros(B, net.hidden_size), torch.zeros(B, net.hidden_size))
    net.inference(last_event=[[step_ids], [torch.rand(B, 1), torch.rand(B, 1)]], hx=hx, z=None)
ok["Ziehen: Start = Prefix-Zähler"] = torch.allclose(first, net._counts_of(prefix_cats[net.count_index]))
ok["Ziehen: + gezogenes Event"] = torch.allclose(net._inference_counts, first + net._counts_of(step_ids))

# 4. Zähler und Event wirken auf die Vorhersage
with torch.no_grad():
    h = torch.randn(B, net.hidden_size, generator=gen)
    a, _ = net.decoder.numeric_prediction(h, torch.full((B,), 3))
    b, _ = net.decoder.numeric_prediction(h, torch.full((B,), 5))
ok["Zeitkopf hängt vom Event ab"] = any(not torch.allclose(a[k], b[k]) for k in a)

# 5. Optionen: eigene erste Abstände (Option im Encoder, Zeit je Event)
raw = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
raw["time:timestamp"] = raw["time:timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
test = sorted(map(str, model.test_cases))
rows = []
for case in test[:400]:
    part = raw[raw["case:concept:name"].astype(str) == case].sort_values("time:timestamp", kind="stable")
    events = part.to_dict("records")
    for name, (activity, lifecycle) in {"angebot_nachlegen": ("O_Created", "complete"), "handle_leads": ("W_Handle leads", "schedule"),
                                        "annehmen_bei_konzept": ("A_Concept", "complete")}.items():
        cut = next((i+1 for i, e in enumerate(events) if str(e["concept:name"]).strip() == activity and e["lifecycle:transition"] == lifecycle), None)
        if cut is None or sum(r["punkt"] == name for r in rows) >= 6:
            continue
        point = model.points[name]
        now = pd.Timestamp(events[cut-1]["time:timestamp"]).timestamp()
        drawn, _ = model.option_suffixes(point, events[:cut], list(point.options), now, 30)
        for option, suffixes in zip(point.options, drawn):
            rows.append({"punkt": name, "fall": case, "option": option, "erster_abstand_h": np.mean([s[0][2] for s in suffixes])/3600,
                         "restzeit_d": np.mean([sum(g for _, _, g in s) for s in suffixes])/86400, "events": np.mean([len(s) for s in suffixes])})
frame = pd.DataFrame(rows)
print(frame.groupby(["punkt", "option"])[["erster_abstand_h", "restzeit_d", "events"]].mean().round(3).to_string())
wide = frame.pivot_table(index=["punkt", "fall"], columns="option", values="erster_abstand_h")
ok["Optionen: erste Abstände verschieden"] = bool((wide.iloc[:, 0] != wide.iloc[:, 1]).fillna(False).any() if wide.shape[1] == 2 else True)

# 6. Checkpoint lädt wieder und rechnet gleich
loaded = PSPModel._load_net(os.path.join(model.params["checkpoints"], "last.pt"))
loaded.eval()
same = all(torch.equal(p, q) for p, q in zip(net.state_dict().values(), loaded.state_dict().values()))
ok["Checkpoint: gleiche Gewichte"] = same and loaded.count_feature == net.count_feature and loaded.numeric_conditioning == net.numeric_conditioning
for key, value in ok.items():
    print(f"{'OK  ' if value else 'FEHLER'} {key}")
print("ALLE OK" if all(ok.values()) else "NICHT ALLE OK")

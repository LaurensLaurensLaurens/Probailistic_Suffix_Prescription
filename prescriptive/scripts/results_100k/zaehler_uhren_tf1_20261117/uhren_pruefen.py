"""Prüfung der Uhren (2026-11-17) an einem kleinen Probemodell (Zähler + Zeit je Event + Uhren).
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> <modell.pkl>"""
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

torch.manual_seed(0)
model = PSPModel.load(sys.argv[1])
net = model.net
clock_days = sys.modules[type(net.encoder).__module__].clock_days
K, mean, std, scale = net.count_size, net.clock_stats[0], net.clock_stats[1], net.clock_scale
ok = {}
ok["params"] = model.counters() and model.time_on_event() and model.clocks() and net.clock_feature == "event_elapsed_time"
ok["Gewichte gelernt (Encoder-Uhren, Decoder-Uhrenteil)"] = (float(net.encoder.clock_weight.abs().sum()) > 0
                                                             and float(net.decoder.count_weight[:, K:].abs().sum()) > 0)
standard = lambda seconds: (torch.as_tensor(seconds, dtype=torch.float32) - mean) / std

# 1. Encoder: Tage seit dem letzten Auftreten, von Hand nachgerechnet
ids = torch.tensor([[0, 0, 5, 7, 5, 9]])
gaps = standard([[0, 0, 0, 86400, 2*86400, 86400]])  # verstrichen an den echten Positionen: 0, 1, 3, 4 Tage
since, seen = clock_days(ids, gaps, K, net.count_ignore, mean, std)
expected = {5: 1.0, 7: 3.0, 9: 0.0}
ok["clock_days: Tage seit letztem Auftreten"] = all(abs(float(since[0, -1, k]) - v) < 1e-4 for k, v in expected.items()) \
    and bool(seen[0, -1, [5, 7, 9]].all()) and not bool(seen[0, -1, 6]) and not bool(seen[0, 1].any())
enc = net.encoder.clocks(ids, gaps)
ok["Encoder-Uhren = log(1 + Tage) / Skala"] = torch.allclose(enc[0, -1], (torch.log1p(since[0, -1]) / scale * seen[0, -1]).float())

# 2. Training: Uhren je Decoderschritt = Uhren über Prefix + zurückgegebene Events (TF 1: echte; TF 0: eigene Vorhersagen)
cats_meta, nums_meta = net.data_set_categories
B, T, S = 4, 12, net.seq_len_pred
gen = torch.Generator().manual_seed(1)
prefix_cats = [torch.randint(1, size, (B, T), generator=gen) for _, size, *_ in cats_meta]
prefix_nums = [torch.rand(B, T, generator=gen) for _ in nums_meta]
suffix_cats = [torch.randint(1, size, (B, S), generator=gen) for _, size, *_ in cats_meta]
suffix_nums = [torch.rand(B, S, generator=gen) for _ in nums_meta]
prefix_cats[net.count_index][0, :3] = 0
static_cats, static_nums = net.static_data_set_categories
static = (torch.stack([torch.randint(1, size, (B,), generator=gen) for _, size, *_ in static_cats], dim=1),
          torch.rand(B, len(static_nums), generator=gen))
captured, original = [], net.decoder.forward


def capture(*args, **kwargs):
    captured.append((kwargs.get("counts").clone(), kwargs.get("input"), kwargs.get("pred")))
    return original(*args, **kwargs)


def clock_part(ids, gaps):
    s, z = clock_days(ids, gaps, K, net.count_ignore, mean, std)
    return (torch.log1p(s[:, -1]) / scale * z[:, -1]).float()


net.decoder.forward = capture
for ratio in (1.0, 0.0):
    captured.clear()
    net.train()
    with torch.no_grad():
        net(prefixes=[prefix_cats, prefix_nums], static_inputs=static, suffixes=[suffix_cats, suffix_nums], teacher_forcing_ratio=ratio)
    ids, gaps = prefix_cats[net.count_index], prefix_nums[net.clock_index]
    good = torch.allclose(captured[0][0][:, K:], clock_part(ids, gaps), atol=1e-5)
    for t in range(1, len(captured)):
        _, inp, pred = captured[t]
        if pred:
            new_ids, new_gaps = inp[0][net.count_dec_position], inp[1][net.clock_dec_position]
        else:
            new_ids, new_gaps = suffix_cats[net.count_index][:, t-1:t], suffix_nums[net.clock_index][:, t-1:t]
        ids = torch.cat([ids, new_ids.reshape(B, 1).long()], dim=1)
        gaps = torch.cat([gaps, new_gaps.reshape(B, 1).float()], dim=1)
        good &= torch.allclose(captured[t][0][:, K:], clock_part(ids, gaps), atol=1e-4)
    ok[f"Training: Uhren fortgeschrieben (Teacher Forcing {ratio})"] = bool(good) and len(captured) == S
net.decoder.forward = original
net.eval()

# 3. Ziehen: Buchführung in inference
with torch.no_grad():
    net.inference(prefix=[prefix_cats, prefix_nums], static_inputs=static)
    first = net._inference_counts
    step_ids = torch.randint(1, net.count_size, (B, 1), generator=gen)
    step_gaps = standard(torch.full((B, 1), 5 * 86400.0))
    hx = (torch.zeros(B, net.hidden_size), torch.zeros(B, net.hidden_size))
    n_num = len(net.dec_feat[1])
    nums = [torch.rand(B, 1) for _ in range(n_num)]
    nums[net.clock_dec_position] = step_gaps
    net.inference(last_event=[[step_ids], nums], hx=hx, z=None)
    after = net._inference_counts
ids = torch.cat([prefix_cats[net.count_index], step_ids], dim=1)
gaps = torch.cat([prefix_nums[net.clock_index], step_gaps], dim=1)
ok["Ziehen: Start = Prefix-Uhren"] = isinstance(first, tuple) and torch.allclose(
    net._count_kwargs(first)["counts"][:, K:], clock_part(prefix_cats[net.count_index], prefix_nums[net.clock_index]), atol=1e-5)
ok["Ziehen: + gezogenes Event mit 5 Tagen Abstand"] = torch.allclose(net._count_kwargs(after)["counts"][:, K:], clock_part(ids, gaps), atol=1e-5)

# 4. Echte Fälle: Optionssuffixe ziehen (Nachlegen, Option im Encoder), Uhr von O_Sent im echten Prefix plausibel
raw = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
raw["time:timestamp"] = raw["time:timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
raw["concept:name"] = raw["concept:name"].astype(str).str.strip()
point = model.points["angebot_nachlegen"]
drawn_ok, n = True, 0
for case in sorted(map(str, model.test_cases))[:200]:
    part = raw[raw["case:concept:name"].astype(str) == case].sort_values("time:timestamp", kind="stable")
    events = part.to_dict("records")
    cut = next((i+1 for i, e in enumerate(events) if e["concept:name"] == "O_Created" and e["lifecycle:transition"] == "complete"), None)
    if cut is None:
        continue
    now = pd.Timestamp(events[cut-1]["time:timestamp"]).timestamp()
    drawn, _ = model.option_suffixes(point, events[:cut], list(point.options), now, 5)
    drawn_ok &= all(len(s) > 0 for option in drawn for s in option)
    n += 1
    if n >= 5:
        break
ok["Ziehen an echten Fällen (5 Fälle × 2 Optionen × 5)"] = bool(drawn_ok) and n == 5

# 5. Checkpoint lädt mit Uhren und gleichen Gewichten
loaded = PSPModel._load_net(os.path.join(model.params["checkpoints"], "last.pt"))
ok["Checkpoint: Uhren-Einstellungen und Gewichte"] = (loaded.clock_feature == net.clock_feature and tuple(loaded.clock_stats) == tuple(net.clock_stats)
                                                      and all(torch.equal(p, q) for p, q in zip(net.state_dict().values(), loaded.state_dict().values())))
for key, value in ok.items():
    print(f"{'OK  ' if value else 'FEHLER'} {key}")
print("ALLE OK" if all(ok.values()) else "NICHT ALLE OK")

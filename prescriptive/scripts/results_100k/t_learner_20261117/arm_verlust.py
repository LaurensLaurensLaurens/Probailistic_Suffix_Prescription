"""Hat das Feintraining der T-Learner-Arme etwas gebracht? (2026-11-18) Roher Verlust je Ausgabe mit Teacher Forcing 1 auf Testfällen,
nur an Prefixen, die am Options-Event der jeweiligen Gruppe enden (Aktion bzw. Referenz, t_learner_policy.ARME), Ziel = die nächsten L
echten Events. Verglichen: Basis ZZ-TF1 (Startpunkt des Feintrainings) gegen das nachtrainierte Arm-Modell, je auf den Prefixen seiner
Gruppe; dazu je Wahlpunkt die Trefferquote des nächsten Events. Wie verlust_diagnose_20261117/verlust_je_ausgabe.py.
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> [Prefixe je Gruppe]
"""
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
from prescriptive.policies.ev_policy.t_learner_policy import ARME  # noqa: E402

OUT = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
L = 10
M = ROOT/"prescriptive/models"
BASIS = M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1.pkl"
ARM = {"aktion": M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion.pkl",
       "referenz": M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz.pkl"}
DECISIONS = {"A_Concept|complete": "Konzept", "W_Complete application|suspend": "nach V.", "O_Created|complete": "Nachlegen",
             "W_Handle leads|schedule": "Leads"}


def prefixes(test_cases):
    raw = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
    raw["time:timestamp"] = raw["time:timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    raw["concept:name"] = raw["concept:name"].astype(str).str.strip()
    out = {arm: [] for arm in ARME}
    rng = np.random.default_rng(0)
    grouped = {c: p for c, p in raw.groupby("case:concept:name", sort=False)}
    for case in rng.permutation(sorted(map(str, test_cases))):
        part = grouped.get(case)
        if part is None:
            continue
        events = part.sort_values("time:timestamp", kind="stable").to_dict("records")
        tokens = [f"{e['concept:name']}|{e.get('lifecycle:transition', 'complete')}" for e in events]
        for i in range(1, len(events) - L):
            if tokens[i - 1] in DECISIONS:
                for arm, options in ARME.items():
                    if tokens[i] in options and len(out[arm]) < N:
                        out[arm].append((DECISIONS[tokens[i - 1]], events[:i + 1], events[:i + 1 + L]))
        if all(len(v) >= N for v in out.values()):
            break
    return out


def measure(model, items):
    net = model.net
    torch.manual_seed(0)
    prefixes_ = [model._encode(p) for _, p, _ in items]
    fulls = [model._encode(f) for _, _, f in items]
    cats, nums, static, mask = model._batch([(e[0], e[1], e[2], e[3]) for e in prefixes_], 1)
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
    row, n = {}, len(items)
    logits = predictions[0]["concept:name_mean"].reshape(S, n, -1)[:L]
    target = s_cats[cat_names.index("concept:name")][:, :L].T.long()
    row["Kreuzentropie"] = float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)))
    hits = (logits.argmax(-1) == target).float()
    row["Treffer 10 Schritte"] = float(hits.mean())
    row["Treffer 1. Schritt"] = float(hits[0].mean())
    points = np.array([p for p, _, _ in items])
    for point in DECISIONS.values():
        if (points == point).any():
            row[f"Treffer 1. Schritt {point}"] = float(hits[0][torch.tensor(points == point)].mean())
    for name in ("event_elapsed_time",):
        mean = predictions[1][f"{name}_mean"].reshape(S, n)[:L]
        tgt = s_nums[num_names.index(name)][:, :L].T.float()
        scale = model.encoder_decoder.continuous_encoders[name].scale_[0]
        row["Abstand MAE (Tage)"] = float((tgt - mean).abs().mean() * scale / 86400)
    return row


def main():
    basis = PSPModel.load(str(BASIS))
    items = prefixes(basis.test_cases)
    rows = []
    for arm, path in ARM.items():
        model = PSPModel.load(str(path))
        for label, m in ((f"Basis ZZ-TF1 auf {arm}", basis), (f"Arm {arm}", model)):
            rows.append({"modell": label, "prefixe": len(items[arm]), **measure(m, items[arm])})
            print(label, "fertig", flush=True)
    frame = pd.DataFrame(rows).set_index("modell")
    frame.to_csv(OUT/"arm_verlust.csv")
    pd.set_option("display.width", 200)
    print(frame.T.round(3).to_string())


if __name__ == "__main__":
    main()

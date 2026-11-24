"""Sonde Zeitkopf (2026-11-18, Wunsch des Nutzers). Frage: Steckt die Information für lange Wartezeiten (Storno-Frist 30,7 Tage nach
dem letzten O_Sent, Antwortzeit nach O_Sent je nach Zahl der Angebote) schon im Decoder-Zustand h, und begrenzt nur der Zeitkopf, der
das Event bloß additiv sieht (Linear(h) + Linear(Embedding(Event)), dropout_uncertainty_decoder.numeric_prediction)?

Aufbau: Testfälle, Schnitt direkt nach dem ersten O_Sent, danach die echten Events mit Teacher Forcing 1 (wie im Training: der
Zeitkopf bekommt das echte Event des Schritts). Je Schritt gespeichert: h (Eingang des Abstandskopfs), echtes Event, echter Abstand,
Tage seit dem letzten O_Sent am vorigen Event (Uhr), Zahl der O_Sent bisher, Kopf-Mittel von Abstand und Fallzeit, P(A_Cancelled).
Vergleich je Modell (Vorhersage des mittleren Abstands in Tagen, außerhalb der Stichprobe über 5 Fallgruppen):
  Kopf          = der trainierte Abstandskopf (additiv)
  additiv       = Ridge auf [h, One-Hot(Event)]   (gleiche Form wie der Kopf, nur neu angepasst)
  Wechselwirk.  = Ridge auf h je Event (eigene Gewichte je Event; Events mit < 40 Schritten: additiv)
  Regel         = 30,7 − Uhr (nur bei A_Cancelled)
Deutung: Wechselwirkung >> additiv ≈ Kopf → Information in h, der Kopf begrenzt. Alle schlecht → Information fehlt in h.
Teil B (Ziehen): je Modell Prefixe × Ziehungen bis EOS; Tage vom letzten O_Sent bis A_Cancelled einmal aus der gezogenen Fallzeit
(wie storno_abstand.py) und einmal aus der Summe der gezogenen Abstände (so bewertet die Policy bei time_from = gaps).
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> [Fälle] [Prefixe Teil B] [Ziehungen Teil B] [label=pfad ...]
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
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from prescriptive.estimators.psp_model import PSPModel  # noqa: E402

torch.set_num_threads(8)
OUT = Path(os.environ.get("SONDE_OUT", Path(__file__).resolve().parent))  # Begleiter: eigener Ordner je Stand
OUT.mkdir(parents=True, exist_ok=True)
N_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
N_DRAW_PREFIXES = int(sys.argv[2]) if len(sys.argv) > 2 else 200
N_DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
M = ROOT/"prescriptive/models"
MODELS = {"ZZ-TF1": M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1.pkl",
          "ZZU-TF1": M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_uhren_tf1.pkl",
          "T-Learner Aktion": M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion.pkl",
          "T-Learner Referenz": M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz.pkl"}
EXTRA = dict(a.split("=", 1) for a in sys.argv[4:])
if EXTRA:
    MODELS = {label: Path(path) for label, path in EXTRA.items()}
SENT = ("O_Sent (mail and online)", "O_Sent (online only)")
CANCEL, VALID = "A_Cancelled|complete", "A_Validating|complete"
DAY = 86400.0
DEADLINE = 30.7
ALPHAS = np.logspace(-2, 3, 6)
BATCH = 100


def token(e):
    return f"{e['concept:name']}|{e.get('lifecycle:transition', 'complete')}"


def load_raw():
    raw = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
    raw["time:timestamp"] = raw["time:timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    raw["concept:name"] = raw["concept:name"].astype(str).str.strip()
    return {c: p.sort_values("time:timestamp", kind="stable").to_dict("records") for c, p in raw.groupby("case:concept:name", sort=False)}


def cut_cases(grouped, test_cases, n, S):
    """Prefix bis einschließlich des ersten O_Sent, Suffix = die nächsten höchstens S echten Events; je Suffix-Schritt die Kennzahlen."""
    rng = np.random.default_rng(0)
    out = []
    for case in rng.permutation(sorted(test_cases)):
        events = grouped.get(case)
        if events is None:
            continue
        hits = [i for i, e in enumerate(events) if e["concept:name"] in SENT]
        if not hits or hits[0] + 1 >= len(events):
            continue
        cut = hits[0] + 1
        start = events[0]["time:timestamp"]
        sec = [(e["time:timestamp"] - start).total_seconds() for e in events]
        rows, last_sent, n_sent = [], sec[hits[0]], 1
        for j in range(cut, min(len(events), cut + S)):
            rows.append({"fall": case, "schritt": j - cut, "event": token(events[j]), "abstand": (sec[j] - sec[j - 1]) / DAY,
                         "fallzeit": sec[j] / DAY, "uhr_sent": (sec[j - 1] - last_sent) / DAY, "sent_bisher": n_sent,
                         "zeit_sent": last_sent / DAY, "fallzeit_vorher": sec[j - 1] / DAY})
            if events[j]["concept:name"] in SENT:
                last_sent, n_sent = sec[j], n_sent + 1
        out.append((case, events[:cut], events[cut:cut + len(rows)], rows))
        if len(out) >= n:
            break
    return out


def teacher_forced(model, cases):
    """Teacher Forcing 1 über die echten Suffixe; je echtem Schritt h, Kopf-Mittel (Tage) und P(A_Cancelled)."""
    net = model.net
    S = net.seq_len_pred
    layer = net.decoder.output_layers["event_elapsed_time_mean"]
    gap_scaler = model.encoder_decoder.continuous_encoders["event_elapsed_time"]
    case_scaler = model.encoder_decoder.continuous_encoders["case_elapsed_time"]
    names = model._evaluator().inverted_suffix_categories[0]
    cancel_id = {str(v): k for k, v in names.items()}[CANCEL]
    records, hidden = [], []
    for b in range(0, len(cases), BATCH):
        chunk = cases[b:b + BATCH]
        enc_p = [model._encode(p) for _, p, _, _ in chunk]
        enc_f = [model._encode(p + s) for _, p, s, _ in chunk]
        cats, nums, static, mask = model._batch([e[:4] for e in enc_p], 1)
        fcats, fnums, _, _ = model._batch([e[:4] for e in enc_f], 1)
        lengths = [len(r) for _, _, _, r in chunk]

        def suffix_of(full):
            rows = []
            for r, L in enumerate(lengths):
                rows.append(torch.cat([full[r, full.shape[1] - L:], torch.zeros(S - L, dtype=full.dtype)]))
            return torch.stack(rows)
        s_cats = [suffix_of(c) for c in fcats]
        s_nums = [suffix_of(x) for x in fnums]
        captured = []
        handle = layer.register_forward_hook(lambda mod, inp, out: captured.append(inp[0].detach().clone()))
        net.train()
        torch.manual_seed(0)
        try:
            with torch.no_grad():
                pred = net(prefixes=[cats, nums], static_inputs=static, suffixes=[s_cats, s_nums], teacher_forcing_ratio=1.0,
                           prefix_mask=mask)[0]
        finally:
            handle.remove()
            net.eval()
        if len(captured) != S:
            raise RuntimeError(f"Hook {len(captured)} mal statt {S}")
        B = len(chunk)
        gap_mu = PSPModel._decode(gap_scaler, pred[1]["event_elapsed_time_mean"].reshape(S, B)).reshape(S, B) / DAY
        gap_lv = pred[1]["event_elapsed_time_var"].reshape(S, B).numpy()
        gap_sigma = np.exp(0.5 * np.clip(gap_lv, -6, 6)) * gap_scaler.scale_[0] / DAY
        case_mu = PSPModel._decode(case_scaler, pred[1]["case_elapsed_time_mean"].reshape(S, B)).reshape(S, B) / DAY
        p_cancel = torch.softmax(pred[0]["concept:name_mean"].reshape(S, B, -1), dim=-1)[:, :, cancel_id].numpy()
        target_ids = s_cats[0].T.numpy()
        for r, (_, _, _, rows) in enumerate(chunk):
            for k, row in enumerate(rows):
                if str(names.get(int(target_ids[k, r]))) != row["event"]:
                    raise RuntimeError(f"Ausrichtung falsch: {names.get(int(target_ids[k, r]))} statt {row['event']}")
                records.append({**row, "kopf": float(gap_mu[k, r]), "kopf_sigma": float(gap_sigma[k, r]),
                                "kopf_fallzeit": float(case_mu[k, r]), "p_storno": float(p_cancel[k, r])})
                hidden.append(captured[k][r].numpy())
        print(f"  {min(b + BATCH, len(cases))}/{len(cases)} Fälle", flush=True)
    return pd.DataFrame(records), np.stack(hidden).astype(np.float32)


def out_of_fold(X, y, groups, folds=5):
    pred = np.zeros(len(y))
    for train, test in GroupKFold(folds).split(X, y, groups):
        pred[test] = RidgeCV(alphas=ALPHAS).fit(X[train], y[train]).predict(X[test])
    return pred


def probes(frame, H):
    groups = frame["fall"].to_numpy()
    y = frame["abstand"].to_numpy()
    onehot = pd.get_dummies(frame["event"]).to_numpy(dtype=np.float32)
    frame["additiv"] = out_of_fold(np.hstack([H, onehot]), y, groups)
    frame["wechselwirkung"] = frame["additiv"]
    for event, index in frame.groupby("event").indices.items():
        if len(index) >= 40 and len(set(groups[index])) >= 5:
            frame.loc[frame.index[index], "wechselwirkung"] = out_of_fold(H[index], y[index], groups[index])
    frame["regel"] = np.where(frame["event"] == CANCEL, DEADLINE - frame["uhr_sent"], np.nan)
    after = frame["uhr_sent"].notna().to_numpy()
    clock_pred = out_of_fold(H[after], frame.loc[after, "uhr_sent"].to_numpy(), groups[after])
    clock = frame.loc[after, "uhr_sent"].to_numpy()
    clock_r2 = 1 - np.sum((clock - clock_pred) ** 2) / np.sum((clock - clock.mean()) ** 2)
    return frame, clock_r2


def segment_sums(frame, column):
    """Je Fall: Summe der Abstände vom letzten O_Sent bis zum ersten A_Cancelled bzw. ersten A_Validating (inkl. dieses Schritts)."""
    out = []
    for case, part in frame.groupby("fall", sort=False):
        part = part.sort_values("schritt")
        for target, name in ((CANCEL, "storno"), (VALID, "validierung")):
            hit = part.index[part["event"] == target]
            if len(hit) == 0:
                continue
            row = part.loc[hit[0]]
            since = part[(part["fallzeit_vorher"] >= row["zeit_sent"] - 1e-9) & (part["schritt"] <= row["schritt"])]
            out.append({"fall": case, "ziel": name, "sent_bisher": int(row["sent_bisher"]), "wert": float(since[column].sum())})
    return pd.DataFrame(out)


def summarise(label, frame, clock_r2):
    rows = []
    methods = ["abstand", "kopf", "additiv", "wechselwirkung", "regel"]
    c = frame[frame["event"] == CANCEL]
    for method in methods:
        values = c[method]
        if values.isna().all():
            continue
        since = c["uhr_sent"] + values
        rows.append({"modell": label, "teil": "A_Cancelled je Schritt", "methode": method, "n": len(c),
                     "Abstand Mittel": values.mean(), "MAE": (values - c["abstand"]).abs().mean() if method != "abstand" else 0.0,
                     "Korrelation": np.corrcoef(values, c["abstand"])[0, 1] if method != "abstand" else 1.0,
                     "seit O_Sent Median": since.median(), "Anteil 29–32 d": since.between(29, 32).mean()})
    mae_all = {m: (frame[m] - frame["abstand"]).abs().mean() for m in ["kopf", "additiv", "wechselwirkung"]}
    for m, v in mae_all.items():
        rows.append({"modell": label, "teil": "alle Schritte", "methode": m, "n": len(frame), "MAE": v})
    for method in ["abstand", "kopf", "additiv", "wechselwirkung"]:
        sums = segment_sums(frame, method)
        for (ziel, many), part in sums.groupby(["ziel", sums["sent_bisher"] >= 2]):
            rows.append({"modell": label, "teil": f"Summe O_Sent → {ziel} ({'≥ 2' if many else '1'} O_Sent)", "methode": method,
                         "n": len(part), "Abstand Mittel": part["wert"].mean(), "seit O_Sent Median": part["wert"].median(),
                         "Anteil 29–32 d": part["wert"].between(29, 32).mean() if ziel == "storno" else np.nan})
    bins = [-0.01, 5, 15, 25, 29, 32, 1e9]
    labels = ["0–5", "5–15", "15–25", "25–29", "29–32", "> 32"]
    frame["uhr_klasse"] = pd.cut(frame["uhr_sent"], bins, labels=labels)
    for klasse, part in frame.groupby("uhr_klasse", observed=True):
        rows.append({"modell": label, "teil": f"P(A_Cancelled) bei Uhr {klasse} d", "methode": "Eventkopf vs. echt", "n": len(part),
                     "P Kopf": part["p_storno"].mean(), "P echt": (part["event"] == CANCEL).mean()})
    rows.append({"modell": label, "teil": "Uhr seit O_Sent aus h (R², Ridge)", "methode": "Sonde", "n": int(frame["uhr_sent"].notna().sum()),
                 "R2": clock_r2})
    c_head = frame[frame["event"] == CANCEL]
    rows.append({"modell": label, "teil": "A_Cancelled: Fallzeit-Kopf seit O_Sent", "methode": "kopf_fallzeit", "n": len(c_head),
                 "seit O_Sent Median": (c_head["kopf_fallzeit"] - c_head["zeit_sent"]).median(),
                 "Anteil 29–32 d": (c_head["kopf_fallzeit"] - c_head["zeit_sent"]).between(29, 32).mean()})
    rows.append({"modell": label, "teil": "σ des Abstandskopfs (Tage, Median)", "methode": "kopf", "n": len(frame),
                 "A_Cancelled": c_head["kopf_sigma"].median(), "alle Schritte": frame["kopf_sigma"].median()})
    return rows


def drawn(label, model, grouped, test_cases):
    """Teil B: Tage vom letzten O_Sent bis A_Cancelled in gezogenen Suffixen, aus der Fallzeit und aus der Abstandssumme."""
    cases = cut_cases(grouped, test_cases, N_DRAW_PREFIXES, 1)
    torch.manual_seed(0)
    encoded = [model._encode(p) for _, p, _, _ in cases]
    suffixes = model.draw_cases([e[:4] for e in encoded], N_DRAWS, completion_steps=200)
    rows = []
    for (_, p, _, _), e, draws in zip(cases, encoded, suffixes):
        start = p[0]["time:timestamp"]
        cut = (p[-1]["time:timestamp"] - start).total_seconds()
        sent = max((ev["time:timestamp"] - start).total_seconds() for ev in p if ev["concept:name"] in SENT)
        for suffix in draws:
            s_time, s_gap, acc = sent, sent, cut
            row = {"storniert": False}
            for name, t, gap in suffix:
                acc += gap
                if name.split("|")[0] in SENT:
                    s_time, s_gap = t, acc
                if name == CANCEL:
                    row = {"storniert": True, "fallzeit": (t - s_time) / DAY, "abstandssumme": (acc - s_gap) / DAY}
                    break
            rows.append(row)
    frame = pd.DataFrame(rows)
    c = frame[frame["storniert"]]
    return [{"modell": label, "teil": "gezogen: seit O_Sent bis A_Cancelled", "methode": m, "n": len(c),
             "Abstand Mittel": c[m].mean(), "seit O_Sent Median": c[m].median(), "Anteil 29–32 d": c[m].between(29, 32).mean(),
             "P echt": frame["storniert"].mean()} for m in ("fallzeit", "abstandssumme")]


def main():
    grouped = load_raw()
    rows = []
    for label, path in MODELS.items():
        model = PSPModel.load(str(path))
        test_cases = set(map(str, model.test_cases))
        cases = cut_cases(grouped, test_cases, N_CASES, model.net.seq_len_pred)
        print(f"{label}: {len(cases)} Fälle, Teacher Forcing", flush=True)
        frame, H = teacher_forced(model, cases)
        frame, clock_r2 = probes(frame, H)
        frame.to_csv(OUT/f"schritte_{label.replace(' ', '_')}.csv.gz", index=False)
        rows.extend(summarise(label, frame, clock_r2))
        if N_DRAW_PREFIXES > 0:
            print(f"{label}: Ziehen ({N_DRAW_PREFIXES} × {N_DRAWS})", flush=True)
            rows.extend(drawn(label, model, grouped, test_cases))
        pd.DataFrame(rows).to_csv(OUT/"sonde.csv", index=False)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

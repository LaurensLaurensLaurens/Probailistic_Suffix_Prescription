"""Sonde „zwei Angebote“ (2026-11-19, Nutzer: „Dann mach das hier“ – Diagnose vor jedem weiteren Training).

Frage: Enthält der Decoder-Zustand h an den Warteschritten der Kundenantwort die Information, wie viele Angebote erstellt bzw.
verschickt sind? Im echten Log (nachlegen_wartezeiten.md) dauert die Antwort nach einem Zusatzangebot länger; alle PSP-Varianten
ziehen dort keinen Unterschied, auch der Zeitkopf je Event (linear in h) mit β-NLL nicht.

Aufbau wie sonde.py (Testfälle, Schnitt nach dem ersten O_Sent, echte Suffixe mit Teacher Forcing 1, h = Eingang des Abstandskopfs).
Ausgewertet nur an den Schritten, die die Wartezeit tragen (W_Call after offers|ate_abort und |resume: 92 % der längsten Abstände
zwischen letztem O_Sent und Validierung). Je Modell, außerhalb der Stichprobe (5 Fallgruppen):
  1. Lesbarkeit: Ridge h → Zahl erstellter Angebote / verschickter O_Sent bisher (R², Treffer „≥ 2“)
  2. Wartezeit: Ridge je Event auf h, auf [h, Angebotszahlen] und nur auf den Angebotszahlen → echter Abstand; verglichen wird der
     vorhergesagte Unterschied „≥ 2 Angebote minus 1 Angebot“ mit dem echten und mit dem des trainierten Kopfs.
Deutung: h-Sonde trifft den echten Unterschied (und Angebotszahl ist aus h lesbar) → Information in h, der Kopf nutzt sie nicht (A).
h-Sonde verfehlt ihn, [h, Angebotszahl] trifft ihn → h enthält sie nicht (B). Angebotszahl nur in der Sonde, nicht als Modelleingabe.
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> [Fälle]
"""
import os
import sys
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
sys.argv = sys.argv[:1]  # sonde.py liest sys.argv beim Import
HERE = Path(__file__).resolve().parent
os.environ["SONDE_OUT"] = str(HERE/"angebote_sonde")
sys.path.insert(0, str(HERE))
import sonde  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from prescriptive.estimators.psp_model import PSPModel  # noqa: E402

M = sonde.M
MODELS = {"T-Learner Aktion": M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion.pkl",
          "T-Learner Referenz": M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz.pkl",
          "Steigung Aktion": M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion_kopf_steigung.pkl",
          "Steigung β-NLL 1 Aktion": M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion_kopf_steigung_bnll1.pkl",
          "ZZ-TF1": M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1.pkl"}
WAIT = ("W_Call after offers|ate_abort", "W_Call after offers|resume")
OUT = HERE/"angebote_sonde"


def oof(X, y, groups):
    pred = np.zeros(len(y))
    for train, test in GroupKFold(5).split(X, y, groups):
        pred[test] = RidgeCV(alphas=sonde.ALPHAS).fit(X[train], y[train]).predict(X[test])
    return pred


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def offers(cases):
    """Je Fall und Suffix-Schritt: Zahl O_Created bis zum vorigen Event (Prefix + Suffix davor)."""
    rows = []
    for case, prefix, suffix, steps in cases:
        created = sum(e["concept:name"] == "O_Created" for e in prefix)
        for row in steps:
            j = row["schritt"]
            rows.append({"fall": case, "schritt": j, "erstellt_bisher": created})
            if suffix[j]["concept:name"] == "O_Created":
                created += 1
    return pd.DataFrame(rows)


def analyse(label, frame, H):
    wait = frame["event"].isin(WAIT).to_numpy()
    f, h = frame[wait].reset_index(drop=True), H[wait]
    groups = f["fall"].to_numpy()
    y = f["abstand"].to_numpy()
    counts = f[["erstellt_bisher", "sent_bisher"]].to_numpy(dtype=float)
    two = (f["sent_bisher"] >= 2).to_numpy()
    out = {"modell": label, "Warteschritte": len(f), "Fälle": f["fall"].nunique(), "Anteil ≥ 2 O_Sent": two.mean()}
    for name, column in (("erstellt", "erstellt_bisher"), ("verschickt", "sent_bisher")):
        target = f[column].to_numpy(dtype=float)
        pred = oof(h, target, groups)
        out[f"R² h → {name}"] = r2(target, pred)
    pred_sent = oof(h, f["sent_bisher"].to_numpy(dtype=float), groups)
    out["Treffer ≥ 2 aus h"] = ((pred_sent >= 1.5) == two).mean()
    onehot = pd.get_dummies(f["event"]).to_numpy(dtype=float)
    preds = {"h je Event": np.zeros(len(f)), "h + Angebote je Event": np.zeros(len(f)), "nur Angebote + Event": oof(np.hstack([counts, onehot]), y, groups)}
    for _, index in f.groupby("event").indices.items():
        preds["h je Event"][index] = oof(h[index], y[index], groups[index])
        preds["h + Angebote je Event"][index] = oof(np.hstack([h[index], counts[index]]), y[index], groups[index])
    preds["Kopf (trainiert)"] = f["kopf"].to_numpy()
    out["echt: Abstand ≥2 − 1 (d)"] = y[two].mean() - y[~two].mean()
    out["echt: Mittel (d)"] = y.mean()
    for name, p in preds.items():
        out[f"{name}: ≥2 − 1 (d)"] = p[two].mean() - p[~two].mean()
        out[f"{name}: R²"] = r2(y, p)
        out[f"{name}: MAE (d)"] = np.abs(y - p).mean()
    for event in WAIT:
        m = (f["event"] == event).to_numpy()
        out[f"echt {event.split('|')[1]}: ≥2 − 1 (d)"] = y[m & two].mean() - y[m & ~two].mean()
        out[f"h je Event {event.split('|')[1]}: ≥2 − 1 (d)"] = preds["h je Event"][m & two].mean() - preds["h je Event"][m & ~two].mean()
        out[f"Kopf {event.split('|')[1]}: ≥2 − 1 (d)"] = preds["Kopf (trainiert)"][m & two].mean() - preds["Kopf (trainiert)"][m & ~two].mean()
    # Abschnitt letztes O_Sent → erstes A_Validating je Fall (wie sonde.segment_sums): echt vs. Kopf, Zahl der Warteschritte darin
    real = sonde.segment_sums(frame, "abstand")
    head = sonde.segment_sums(frame, "kopf")
    seg = real[real.ziel == "validierung"].merge(head[head.ziel == "validierung"], on=["fall", "ziel", "sent_bisher"], suffixes=("_echt", "_kopf"))
    two_seg = seg["sent_bisher"] >= 2
    out["Abschnitt Validierung: Fälle 1 / ≥2"] = f"{(~two_seg).sum()} / {two_seg.sum()}"
    out["Abschnitt echt 1 / ≥2 (d)"] = f"{seg.wert_echt[~two_seg].mean():.2f} / {seg.wert_echt[two_seg].mean():.2f}"
    out["Abschnitt Kopf 1 / ≥2 (d)"] = f"{seg.wert_kopf[~two_seg].mean():.2f} / {seg.wert_kopf[two_seg].mean():.2f}"
    steps = f.groupby("fall").size().rename("warteschritte").reset_index().merge(
        f.groupby("fall")["sent_bisher"].max().rename("max_sent").reset_index(), on="fall")
    out["Warteschritte je Fall 1 / ≥2"] = (f"{steps.warteschritte[steps.max_sent < 2].mean():.2f} / "
                                          f"{steps.warteschritte[steps.max_sent >= 2].mean():.2f}")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grouped = sonde.load_raw()
    rows = []
    for label, path in MODELS.items():
        model = PSPModel.load(str(path))
        cases = sonde.cut_cases(grouped, set(map(str, model.test_cases)), N, model.net.seq_len_pred)
        print(f"{label}: {len(cases)} Fälle, Teacher Forcing", flush=True)
        frame, H = sonde.teacher_forced(model, cases)
        frame = frame.merge(offers(cases), on=["fall", "schritt"], how="left", validate="one_to_one")
        rows.append(analyse(label, frame, H))
        table = pd.DataFrame(rows)
        table.to_csv(OUT/"angebote_sonde.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        text = table.set_index("modell").T.round(3).to_string()
    (OUT/"angebote_sonde.md").write_text("# Sonde: Zahl der Angebote im Decoder-Zustand h (2026-11-19)\n\n```\n" + text + "\n```\n")
    print(text)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()

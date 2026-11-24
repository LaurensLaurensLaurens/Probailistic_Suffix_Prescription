"""Lineare Sonde (2026-11-23, Nutzer: „ja, mach die Sonde“): Wo geht der Betrag verloren – im Encoder oder im Decoder?
Befund 23:10 (konzept_effekt_gezogen.md): k3x4ol zieht die konzept-Effekte unabhängig vom Betrag. Hier: lässt sich „Betrag < 11.000 €“
linear aus den Zuständen des Netzes ablesen?
  - Encoder-Zustand nach dem Options-Event (der Zustand, mit dem der Decoder startet)
  - Decoder-Zustand bei Ziehschritt k (k = 0: Zustand, aus dem das erste Event nach der Option gezogen wird)
Netze von k3x4ol (schlank, keyed_noise) an annehmen_bei_konzept: „vervollständigen“ und „annehmen“. Zustände: erste Besuche in den
Bank-Läufen online_konformanz/bank_seed* (frischer Simulator v3), je Betragsgruppe bis zu N_FAELLE, je Fall N_ZIEHUNGEN gezogene Suffixe.
Zustände über Forward-Hooks am Encoder und Decoder des Netzes abgegriffen (h und c aller Schichten), keine Änderung am Modell.
Sonde: logistische Regression (standardisiert), 5-fach GroupKFold nach Fall, AUC. Dazu: Schritt des ersten gezogenen A_Complete.
Ausgabe sonde_betrag.md / .csv hier. Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Fälle je Gruppe] [Ziehungen]
"""
import os
from pathlib import Path
import sys
import time
import warnings

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parents[3]
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
HUELLE = Path(os.environ.get("DIAG_HUELLE", ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_k3x4_schlank_crn.pkl"))  # 2026-11-24: DIAG_HUELLE / DIAG_NAME für andere Hüllen (static_in_decoder)
NAME = os.environ.get("DIAG_NAME", "sonde_betrag")
N_FAELLE = int(sys.argv[1]) if len(sys.argv) > 1 else 300
N_ZIEHUNGEN = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SCHRITTE = [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20]
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CASE, ACT, LC, TIME = "case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"
PUNKT = "annehmen_bei_konzept"
OPTIONEN = ["W_Complete application|start", "A_Accepted"]
GRENZE = 11000


def zustaende(punkt):
    ausloeser = f"{punkt.activity}|{punkt.lifecycle}"
    out = {True: [], False: []}
    for ordner in sorted((PARENT/"online_konformanz").glob("bank_seed*")):
        e = pd.read_pickle(ordner/"events.pkl").sort_values([CASE, TIME], kind="stable")
        for case, t in e.groupby(CASE, sort=False):
            t = t.reset_index(drop=True)
            toks = [f"{a}|{l}" for a, l in zip(t[ACT], t[LC])]
            if ausloeser not in toks[:-1]:
                continue
            i = toks.index(ausloeser)
            klein = float(t["case:RequestedAmount"].iloc[0]) < GRENZE
            if len(out[klein]) < N_FAELLE:
                out[klein].append((f"{ordner.name}/{case}", t.iloc[:i + 1].to_dict("records"), pd.Timestamp(t.loc[i, TIME]).timestamp()))
    return [(c, p, j, k) for k, liste in out.items() for c, p, j in liste]


def flach(hc):
    """(h, c) des Encoders bzw. Decoders: je Tensor Zeilen x 128 (oberste Schicht, wie an den Decoder übergeben) → Zeilen x 256."""
    def zeilen(t):
        t = t.detach()
        return t.reshape(t.shape[0], -1) if t.dim() == 2 else t.permute(1, 0, 2).reshape(t.shape[1], -1)
    h, c = hc
    return np.concatenate([zeilen(h).numpy(), zeilen(c).numpy()], axis=1)


def sonde(x, y, gruppen):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y)) < 2 or len(np.unique(gruppen)) < 10:
        return np.nan
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)), x, y,
                          groups=gruppen, cv=GroupKFold(5), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    import torch
    from interventions import option_token
    from prescriptive.estimators.psp_model import PSPModel
    start = time.time()
    huelle = PSPModel.load(str(HUELLE))
    huelle.set_threads(8)
    punkt = huelle.points[PUNKT]
    faelle = zustaende(punkt)
    print(f"{len(faelle)} Fälle ({sum(k for *_, k in faelle)} unter 11k), {time.time() - start:.0f} s", flush=True)
    zeilen, abstand = [], []
    for option in OPTIONEN:
        netz = huelle.netze[(PUNKT, option_token(option))].net
        enc, dec = [], []
        hooks = [netz.encoder.register_forward_hook(lambda m, i, o: enc.append(o)),
                 netz.decoder.register_forward_hook(lambda m, i, o: dec.append((len(enc), o[1])))]
        daten = {"Encoder nach Option": ([], [], [])}
        for j, (case, prefix, jetzt, klein) in enumerate(faelle):
            enc.clear()
            dec.clear()
            torch.manual_seed(5000 + j)
            with torch.no_grad():
                drawn, _ = huelle.option_suffixes(punkt, prefix, [option], jetzt, N_ZIEHUNGEN)
            suffixes = drawn[0]
            if len(enc) < 2:
                raise RuntimeError(f"erwartet 2 Encoder-Aufrufe (Option, Rest), gesehen {len(enc)}")
            x = flach(enc[1])  # Zeilen = Ziehungen
            daten["Encoder nach Option"][0].append(x)
            daten["Encoder nach Option"][1].extend([klein] * len(x))
            daten["Encoder nach Option"][2].extend([case] * len(x))
            schritte = [hc for n_enc, hc in dec if n_enc == len(enc)]  # Decoder-Aufrufe nach dem letzten Encoder-Aufruf (Rest)
            rest_len = np.array([len(s) - 1 for s in suffixes])  # ohne das Options-Event
            for k in SCHRITTE:
                if k >= len(schritte):
                    continue
                gueltig = rest_len > k
                if not gueltig.any():
                    continue
                d = daten.setdefault(f"Decoder Schritt {k}", ([], [], []))
                d[0].append(flach(schritte[k])[gueltig])
                d[1].extend([klein] * int(gueltig.sum()))
                d[2].extend([case] * int(gueltig.sum()))
            for s in suffixes:
                toks = [str(t) for t, _, _ in s][1:]
                erst = next((i for i, t in enumerate(toks) if t.split("|")[0] == "A_Complete"), None)
                abstand.append({"option": option, "erster_A_Complete_schritt": erst, "laenge": len(toks)})
            if (j + 1) % 100 == 0:
                print(f"{option}: {j + 1} Fälle ({time.time() - start:.0f} s)", flush=True)
        for h in hooks:
            h.remove()
        for name, (xs, ys, gs) in daten.items():
            x, y, g = np.vstack(xs), np.asarray(ys, dtype=int), np.asarray(gs)
            auc = sonde(x, y, g)
            zeilen.append({"Netz": option, "Zustand": name, "Zeilen": len(y), "Fälle": len(np.unique(g)), "Anteil < 11k": round(float(y.mean()), 3),
                           "AUC Sonde": round(auc, 3) if np.isfinite(auc) else np.nan})
            print(zeilen[-1], flush=True)
    tab = pd.DataFrame(zeilen)
    ab = pd.DataFrame(abstand)
    ab_tab = ab.groupby("option").agg(Suffixe=("laenge", "size"), Länge_Median=("laenge", "median"),
                                      Anteil_mit_A_Complete=("erster_A_Complete_schritt", lambda s: float(s.notna().mean())),
                                      Schritt_erster_A_Complete_Median=("erster_A_Complete_schritt", "median")).round(2)
    tab.to_csv(HERE/f"{NAME}.csv", index=False)
    text = (f"# Lineare Sonde „Betrag < 11.000 €“ in den Zuständen von {HUELLE.name} an konzept ({time.strftime('%Y-%m-%d %H:%M')})\n\n"
            f"{len(faelle)} frische Zustände (Bank-Läufe), {N_ZIEHUNGEN} Ziehungen je Fall und Netz; logistische Regression, 5-fach GroupKFold "
            f"nach Fall; AUC 0,5 = nicht ablesbar, 1,0 = perfekt. Decoder Schritt k: Zustand, aus dem Event k nach der Option gezogen wird "
            f"(nur Ziehungen, die so lang sind).\n\n```\n{tab.to_string(index=False)}\n```\n\n"
            f"Wo im gezogenen Suffix der erste A_Complete steht (Schritte nach der Option):\n\n```\n{ab_tab.to_string()}\n```\n\n"
            f"Laufzeit {time.time() - start:.0f} s\n")
    (HERE/f"{NAME}.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()

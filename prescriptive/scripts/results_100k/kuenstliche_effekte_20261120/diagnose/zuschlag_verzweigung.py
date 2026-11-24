"""Diagnose 2026-11-20: kommt der konzept-Zuschlag an der Verzweigung nach W_Validate application (complete / ate_abort) an?
Zeichnet je Fall mit aktivem Zuschlag an diesen Schritten roh- und Zuschlags-Wahrscheinlichkeit von W_Call incomplete files|schedule
und das gezogene Event auf (Rohlogits einmal abgegriffen, das LSTM wird nicht doppelt weitergeschaltet)."""
import os, sys, warnings
from pathlib import Path
from collections import Counter
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
import wahrheit as w
import numpy as np
import kuenstliche_effekte as ke

ke.aktivieren(ke.laden("kuenstliche_effekte/effekte_v2.yaml"))
ZIEL = "W_Call incomplete files|schedule"
VOR = {"W_Validate application|complete", "W_Validate application|ate_abort"}
rec = []; schritte = Counter(); armed = Counter()
orig = ke._ModellMitZuschlag.step

def step(self, state, tokens, seconds, *a, **k):
    eng = self._engine; case = eng._advancing
    tok = "|".join(eng._tokens[int(np.asarray(tokens).reshape(-1)[0])])
    roh_box = {}
    inner_step = self._inner.step
    def capture(*aa, **kk):
        r = inner_step(*aa, **kk); roh_box["r"] = r; return r
    self._inner.step = capture
    try:
        out = orig(self, state, tokens, seconds, *a, **k)
    finally:
        del self._inner.step
    if case is not None and tok in VOR:
        aktiv = getattr(case, "ablauf_effekte", None)
        j = eng.model.encoder.token_to_id[ZIEL]
        r = roh_box["r"]; pr = np.exp(r - r.max()); pr /= pr.sum(); q = np.exp(out - out.max()); q /= q.sum()
        z = sum(e["zuschlag"][j] for e in aktiv) if aktiv else 0.0
        rec.append({"case": case.case_id, "zuschlag": z, "p_roh": float(pr[0, j]), "p_mit": float(q[0, j])})
    return out

ke._ModellMitZuschlag.step = step
sim = w.simulator()
events, choices = sim.run(800, policy=w.Erzwingen("annehmen_bei_konzept", "A_Accepted", False), seed=5)
import pandas as pd
d = pd.DataFrame(rec)
log = pd.DataFrame(ke.LETZTE_ENGINE.effect_log)
k = log[log.point == "annehmen_bei_konzept"]
print("konzept-Sätze:", len(k), "davon mit Ablauf:", int((k.ablauf != "").sum()))
print("Verzweigungsschritte:", len(d), "davon mit Zuschlag ≠ 0:", int((d.zuschlag != 0).sum()))
if len(d):
    print(d.groupby(d.zuschlag != 0)[["p_roh", "p_mit"]].mean().round(3))
armed_cases = set(k[k.ablauf != ""].case)
print("Fälle mit Ablauf-Effekt, die die Verzweigung erreichen:", len(armed_cases & set(d.case)), "von", len(armed_cases))

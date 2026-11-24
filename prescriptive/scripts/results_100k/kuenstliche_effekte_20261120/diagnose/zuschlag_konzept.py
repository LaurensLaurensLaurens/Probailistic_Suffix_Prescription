"""Diagnose 2026-11-20: warum wirkt der Ablauf-Zuschlag auf A_Incomplete nach konzept nicht? Zählt je Schritt mit aktivem Zuschlag,
ob A_Incomplete möglich war (≥ MIN_P) und wie sich seine Wahrscheinlichkeit durch den Zuschlag ändert, und an welchen Vorgänger-Events."""
import os, sys, warnings
from pathlib import Path
from collections import Counter
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
import wahrheit as w  # setzt Pfade und Katalog
import numpy as np
import kuenstliche_effekte as ke

effekte = ke.aktivieren(ke.laden("kuenstliche_effekte/effekte_v2.yaml"))
stat = Counter(); vorher = Counter(); p_vor, p_nach = [], []
orig_step = ke._ModellMitZuschlag.step

def step(self, state, tokens, seconds, *a, **k):
    roh = self._inner.step(state, tokens, seconds, *a, **k)
    case = self._engine._advancing
    aktiv = getattr(case, "ablauf_effekte", None) if case is not None else None
    if aktiv:
        ids = self._engine.model.encoder.token_to_id
        j = ids["A_Incomplete|complete"]
        z = sum(e["zuschlag"][j] for e in aktiv)
        if z != 0:
            stat["schritte mit Zuschlag"] += 1
            pr = np.exp(roh - roh.max()); pr /= pr.sum()
            if int(np.asarray(tokens).reshape(-1)[0]) in self._engine._control_flow:
                stat["direkt nach Wahlpunkt (ausgelassen)"] += 1
            elif pr[0, j] >= ke.MIN_P:
                stat["A_Incomplete möglich"] += 1
                vorher["|".join(self._engine._tokens[int(np.asarray(tokens).reshape(-1)[0])])] += 1
                p_vor.append(pr[0, j])
    out = orig_step(self, state, tokens, seconds, *a, **k)
    if aktiv and z != 0 and pr[0, j] >= ke.MIN_P and int(np.asarray(tokens).reshape(-1)[0]) not in self._engine._control_flow:
        q = np.exp(out - out.max()); q /= q.sum(); p_nach.append(q[0, j])
    return out

ke._ModellMitZuschlag.step = step
sim = w.simulator()
events, choices = sim.run(600, policy=w.Erzwingen("annehmen_bei_konzept", "A_Accepted", False), seed=5)
print(dict(stat))
print("Vorgänger, an denen A_Incomplete möglich war:", vorher.most_common(8))
if p_vor:
    print(f"P(A_Incomplete) vor / nach Zuschlag: Mittel {np.mean(p_vor):.3f} / {np.mean(p_nach):.3f}")
log = ke.LETZTE_ENGINE.effect_log
print("Sätze konzept mit Ablauf:", sum(1 for s in log if s["point"] == "annehmen_bei_konzept" and s["ablauf"]))

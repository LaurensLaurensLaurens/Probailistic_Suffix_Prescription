"""Diagnose 2026-11-20: ein einzelner Schritt der Zuschlags-Hülle im Detail (Form der Logits, Wert des Zuschlags am Ziel-Event)."""
import os, sys, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
import wahrheit as w
import numpy as np
import kuenstliche_effekte as ke

ke.aktivieren(ke.laden("kuenstliche_effekte/effekte_v2.yaml"))
ZIEL = "W_Call incomplete files|schedule"
gezeigt = [0]
orig_inner = None

class Spion(ke._ModellMitZuschlag):
    pass

orig = ke._ModellMitZuschlag.step
def step(self, state, tokens, seconds, *a, **k):
    eng = self._engine; case = eng._advancing
    tok = "|".join(eng._tokens[int(np.asarray(tokens).reshape(-1)[0])])
    aktiv = getattr(case, "ablauf_effekte", None) if case is not None else None
    j = eng.model.encoder.token_to_id[ZIEL]
    if aktiv and tok in ("W_Validate application|complete", "W_Validate application|ate_abort") and gezeigt[0] < 3 and any(e["zuschlag"][j] != 0 for e in aktiv):
        gezeigt[0] += 1
        st = {kk: (v.copy() if hasattr(v, "copy") else v) for kk, v in state.items()}
        st["h"] = [x.copy() for x in state["h"]]; st["c"] = [x.copy() for x in state["c"]]
        roh = self._inner.step(st, tokens, seconds, *a, **k)
        print("Vorgänger", tok, "Form", roh.shape, "Zuschläge", [float(e["zuschlag"][j]) for e in aktiv], "Länge Zuschlag", len(aktiv[0]["zuschlag"]))
        print("  logit Ziel roh", float(roh[0, j]), "argmax", "|".join(eng._tokens[int(np.argmax(roh[0]))]))
        z = np.zeros(roh.shape[-1]); [z.__iadd__(e["zuschlag"]) for e in aktiv]
        print("  z[j]", z[j], "Summe z", z.sum(), "nonzero", np.nonzero(z)[0][:5], "j", j)
    out = orig(self, state, tokens, seconds, *a, **k)
    if aktiv and tok in ("W_Validate application|complete", "W_Validate application|ate_abort") and gezeigt[0] <= 3:
        q = np.exp(out - out.max()); q /= q.sum()
        print("  Hülle: logit Ziel", float(out[0, j]), "p Ziel", float(q[0, j]), "rest der Effekte", [e["rest"] for e in aktiv])
    return out

ke._ModellMitZuschlag.step = step
sim = w.simulator()
events, choices = sim.run(300, policy=w.Erzwingen("annehmen_bei_konzept", "A_Accepted", False), seed=5)
print("gezeigt", gezeigt[0])

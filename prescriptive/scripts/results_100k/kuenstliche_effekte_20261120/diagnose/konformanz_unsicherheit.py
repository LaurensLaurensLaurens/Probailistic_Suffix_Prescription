"""Unsicherheit beim Conformance-Ansatz (2026-11-23, Nutzer: „Ist Unsicherheit auch das Problem bei dem Conformance-Ansatz?“).
Wie konformanz_signal.py (gleiche Saaten, Bank entscheidet, erste Besuche), aber nur der Option-Learner k3x4ol (schlanke Hülle) mit 100
Ziehungen je Option wie online. Je Zustand aus KonformanzPolicy.bewerten die Abweichungen je Suffix (Optionen x n):
    d_k = Abweichungen_k(andere) − Abweichungen_k(Regel-Option), Δ = Mittel, SE = SD(d) / √n, nicht unterscheidbar: |Δ| < 1,96 · SE
Je Punkt und Teilgruppe: Anteil nicht unterscheidbar, sicher andere besser (Δ < −1,96 SE), sicher Regel besser (Δ > 1,96 SE), mittleres Δ,
und die Wahrheit bei Bank sonst (wahrheit_f.csv, das schätzt das PSP) und bei Regel sonst (wahrheit_f_regel.csv, das zählt für die Policy).
Ausgabe konformanz_unsicherheit.csv / .md hier. Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Prozesse] [Fälle je Prozess]
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import warnings

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parents[3]
if len(sys.argv) > 1:
    os.environ["KU_PROZESSE"] = sys.argv[1]
    os.environ["KU_FAELLE"] = sys.argv[2] if len(sys.argv) > 2 else "60"
    del sys.argv[1:]
PROZESSE = int(os.environ.get("KU_PROZESSE", "6"))
FAELLE = int(os.environ.get("KU_FAELLE", "60"))
MODELL = Path(os.environ.get("KU_MODELL", ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_k3x4_schlank.pkl"))
NAME = os.environ.get("KU_NAME", "konformanz_unsicherheit")  # 2026-11-23 13:25: Ausgabename, z. B. für die keyed_noise-Kopie
Z = 1.96
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PARENT))
import online_konformanz as ok  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


class Aufzeichner:
    def __init__(self, seed):
        from prescriptive.policies.konformanz_policy import KonformanzPolicy
        self.policy = KonformanzPolicy(str(MODELL), seed=seed, samples=100)
        if os.environ.get("KU_ROLLOUT"):  # 2026-11-23: Rollout mit Causal Forest an späteren Wahlpunkten
            from prescriptive.policies.ev_policy.rollout import ForestRollout
            ForestRollout.einschalten(self.policy.model, os.environ["KU_ROLLOUT"], first_only=("angebot_nachlegen",))
        self.orakel = ok.Orakel()
        self.zeilen, self.gesehen = [], set()

    def choose(self, point, case, options, time, engine):
        if point.name not in self.orakel.PUNKTE or len(options) < 2 or (case.case_id, point.name) in self.gesehen:
            return None
        self.gesehen.add((case.case_id, point.name))
        regel, andere, _ = self.orakel.PUNKTE[point.name]
        if regel not in options or andere not in options:
            return None
        from simulation.case_engine_lifecycle_aware import to_seconds
        dauer = (time - to_seconds(case.prefix[0]["time:timestamp"])) / 86400 if case.prefix else 0.0
        zeile = {"case:RequestedAmount": case.attributes.get("case:RequestedAmount"), "case:LoanGoal": case.attributes.get("case:LoanGoal"),
                 "case:ApplicationType": case.attributes.get("case:ApplicationType"), "dauer_tage": dauer}
        _, je_suffix = self.policy.bewerten(point, case.prefix, [regel, andere], time)
        d = je_suffix[1] - je_suffix[0]
        self.zeilen.append({"punkt": point.name, "gruppe": self.orakel.gruppe(zeile, point.name), "case": case.case_id,
                            "regel": je_suffix[0].mean(), "andere": je_suffix[1].mean(), "delta": d.mean(),
                            "se": d.std(ddof=1) / np.sqrt(len(d)), "rho": np.corrcoef(je_suffix[0], je_suffix[1])[0, 1] if je_suffix.std(axis=1).min() > 0 else np.nan})
        return None


def job(k):
    import torch
    torch.set_num_threads(1)
    import kuenstliche_effekte as ke
    ke.aktivieren(ke.laden(ok.EFFEKTE))
    a = Aufzeichner(seed=300 + k)
    ok.simulator().run(400 + FAELLE, policy=a, window=(400, 400 + FAELLE), seed=300 + k)
    return pd.DataFrame(a.zeilen)


def main():
    with ProcessPoolExecutor(PROZESSE, mp_context=multiprocessing.get_context("spawn")) as pool:
        d = pd.concat(list(pool.map(job, range(PROZESSE))), ignore_index=True)
    d.to_csv(HERE/f"{NAME}.csv", index=False)
    bank = pd.read_csv(PARENT/"wahrheit_f.csv").set_index(["Punkt", "Gruppe"])["andere−Regel Abw. (F)"]
    regel = pd.read_csv(PARENT/"wahrheit_f_regel.csv").set_index(["Punkt", "Gruppe"])["andere−Regel Abw. (Regel sonst)"]
    zeilen = []
    for (punkt, gruppe), t in d.groupby(["punkt", "gruppe"]):
        zeilen.append({"Punkt": punkt, "Gruppe": gruppe, "Zustände": len(t), "Δ Mittel": t.delta.mean(), "SE Median": t.se.median(),
                       "ρ Median": t.rho.median(), "nicht unterscheidbar": (t.delta.abs() < Z * t.se).mean(),
                       "sicher andere besser": (t.delta < -Z * t.se).mean(), "sicher Regel besser": (t.delta > Z * t.se).mean(),
                       "Wahrheit Bank sonst": bank.get((punkt, gruppe)), "Wahrheit Regel sonst": regel.get((punkt, gruppe))})
    out = pd.DataFrame(zeilen)
    gesamt = {"Zustände": len(d), "nicht unterscheidbar": float((d.delta.abs() < Z * d.se).mean()),
              "sicher andere besser": float((d.delta < -Z * d.se).mean()), "sicher Regel besser": float((d.delta > Z * d.se).mean()),
              "SE Median": float(d.se.median()), "ρ Median": float(d.rho.median())}
    with pd.option_context("display.width", 300, "display.max_columns", 20, "display.max_colwidth", 60):
        text = (f"# Unsicherheit beim Conformance-Ansatz, {MODELL.name}, 100 Ziehungen je Option (2026-11-23)\n\n"
                f"Δ = erwartete Abweichungen andere − Regel-Option je Fall; nicht unterscheidbar: |Δ| < 1,96 · SE.\n\n"
                f"```\n{out.round(3).to_string(index=False)}\n```\n\nGesamt: {gesamt}\n")
    (HERE/f"{NAME}.md").write_text(text)
    print(text)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()

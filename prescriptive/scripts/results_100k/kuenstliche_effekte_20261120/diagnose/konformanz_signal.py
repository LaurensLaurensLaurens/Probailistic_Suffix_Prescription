"""Diagnose 2026-11-20 abends: hat der PSP auf dem v3-Log (T-Learner k3) die eingebauten Conformance-Effekte gelernt?
Frische Simulatorzustände (Effekte v3, Bank entscheidet), an jedem ersten Besuch eines Wahlpunkts schätzt KonformanzPolicy.bewerten je
Option die erwarteten Abweichungen (50 Ziehungen), für den k3-PSP und zum Vergleich den PSP vom echten Log (svk, kennt die künstlichen
Effekte nicht). Je Punkt und Teilgruppe (wie wahrheit.py): geschätzte andere − Regel-Option gegen die Wahrheit (wahrheit_effekte_v3.csv).
Ausgabe konformanz_signal.csv / .md hier. Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [Prozesse] [Fälle je Prozess]
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
PROZESSE = int(sys.argv[1]) if len(sys.argv) > 1 else 6
FAELLE = int(sys.argv[2]) if len(sys.argv) > 2 else 60
MODELLE = {"k3": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_k3.pkl",
           "k3x4": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_k3x4.pkl",  # 2026-11-22: 4-fache Datenmenge
           "svk (echter Log)": ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_svk_crn.pkl",
           "k3x4ol": ROOT/"prescriptive/models/psp_model_durchlaufzeit_option_learner_k3x4.pkl"}  # 2026-11-22: Netz je Wahlpunkt und Option
# 2026-11-22: SIGNAL_MODELLE (Namen, Komma) wählt Modelle aus, SIGNAL_NAME den Dateinamen der Ausgabe (Standard konformanz_signal).
# Gleiche Saaten und Bank-Entscheidungen → dieselben Zustände wie in früheren Läufen.
if os.environ.get("SIGNAL_MODELLE"):
    MODELLE = {name: MODELLE[name] for name in os.environ["SIGNAL_MODELLE"].split(",")}
NAME = os.environ.get("SIGNAL_NAME", "konformanz_signal")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PARENT))
import online_konformanz as ok  # noqa: E402  (Pfade, Katalog, simulator(), Orakel mit gruppe; liest sys.argv nur in main)
import pandas as pd  # noqa: E402


class Aufzeichner:
    """Bank entscheidet (None); an jedem ersten Besuch eines Punkts schätzen beide PSPs die Abweichungen je Option."""

    def __init__(self, seed):
        from prescriptive.policies.konformanz_policy import KonformanzPolicy
        self.policies = {name: KonformanzPolicy(str(pfad), seed=seed, samples=50) for name, pfad in MODELLE.items()}
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
        satz = {"punkt": point.name, "gruppe": self.orakel.gruppe(zeile, point.name), "case": case.case_id}
        for name, policy in self.policies.items():
            erwartet, _ = policy.bewerten(point, case.prefix, [regel, andere], time)
            satz[f"{name}: Regel"], satz[f"{name}: andere"] = erwartet[0], erwartet[1]
        self.zeilen.append(satz)
        return None


def job(k):
    import torch
    torch.set_num_threads(1)
    import kuenstliche_effekte as ke
    ke.aktivieren(ke.laden(ok.EFFEKTE))
    policy = Aufzeichner(seed=300 + k)
    ok.simulator().run(400 + FAELLE, policy=policy, window=(400, 400 + FAELLE), seed=300 + k)
    return pd.DataFrame(policy.zeilen)


def main():
    with ProcessPoolExecutor(PROZESSE, mp_context=multiprocessing.get_context("spawn")) as pool:
        d = pd.concat(list(pool.map(job, range(PROZESSE))), ignore_index=True)
    d.to_csv(HERE/f"{NAME}.csv", index=False)
    wahr = pd.read_csv(PARENT/"wahrheit_effekte_v3.csv").set_index(["Punkt", "Gruppe"])["n Abweichungen andere−Regel"]
    zeilen = []
    for (punkt, gruppe), teil in d.groupby(["punkt", "gruppe"]):
        z = {"Punkt": punkt, "Gruppe": gruppe, "Zustände": len(teil), "Wahrheit andere−Regel": wahr.get((punkt, gruppe))}
        for name in MODELLE:
            diff = teil[f"{name}: andere"] - teil[f"{name}: Regel"]
            z[f"{name} andere−Regel"] = diff.mean()
            z[f"{name} SE"] = diff.std(ddof=1) / len(diff) ** 0.5 if len(diff) > 1 else float("nan")
            z[f"{name} Regel-Option"] = teil[f"{name}: Regel"].mean()
        zeilen.append(z)
    out = pd.DataFrame(zeilen)
    with pd.option_context("display.width", 300, "display.max_columns", 20, "display.max_colwidth", 60):
        text = out.round(3).to_string(index=False)
    (HERE/f"{NAME}.md").write_text(f"# Gelernte Conformance-Effekte je Teilgruppe (Diagnose 2026-11-20)\n\n"
                                            f"Geschätzte Abweichungen je Fall, andere − Regel-Option, 50 Ziehungen je Option\n\n```\n{text}\n```\n")
    print(text)
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()

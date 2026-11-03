"""
Online-Auswertung mit Wahl an allen Mitarbeiterpunkten aus lauf.mitarbeiterwahl, ohne channel: Bank, Zufall, XGBoost. Eigenes Skript, weil --online-points
aus den Namen einen Ordnernamen baut, der bei so vielen Punkten zu lang wird. Aufruf aus process-simulation-engine/, Tabellen in sys.argv[1].
"""
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = "/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription"
for path in (ROOT, os.path.join(ROOT, "prescriptive"), os.path.join(ROOT, "process-simulation-engine")):
    sys.path.insert(0, path)
from prescriptive.helpers.datasets import add_paths  # noqa: E402


def main():
    add_paths()
    import interventions
    from helpers.parameters import COSTS
    from preprocessing import preprocess
    from prescriptive.evaluation.online_evaluation import evaluate_online
    from prescriptive import profiles
    from simulation.dataset_simulator import Simulator
    raw = pickle.load(open(os.path.join(ROOT, "prescriptive/data/cache/bpic17_raw.pkl"), "rb"))
    log = preprocess("bpic17", raw, verbose=False)
    simulator = Simulator("bpic17", log)
    points = [point["name"] for point in interventions.employee_points("bpic17")]
    print("Wahl an:", points, flush=True)
    table, runs = evaluate_online(simulator, policies=("bank", "random", "xgb"), points=points, n_cases=4000, warmup=700, cooldown=1300, seeds=10, seed=0, workers=14, costs=COSTS)
    os.makedirs(sys.argv[1], exist_ok=True)
    table.to_csv(os.path.join(sys.argv[1], "online.csv"))
    runs.to_csv(os.path.join(sys.argv[1], "online_laeufe.csv"), index=False)
    print(table.round(3).T.to_string(), flush=True)


if __name__ == "__main__":
    main()

"""
Lagebild des PSP auf dem Simulator, je Datensatz mit Engine (BPIC17): Der Simulator gibt die Umgebung vor, die Bank
entscheidet (policy bank, wie in der Evaluation), und das PSP beobachtet alle laufenden Fälle zugleich
(prescriptive/overview.py): alle every_hours zieht es für jeden Fall mit neuen Events Suffixe und hält so eine
Übersicht über alle Prozesse. Ergebnis: das Lagebild (Zeilen je Beobachtung und Fall, Zusammenfassung je Beobachtung),
die Events des Laufs und die Güte der Prognosen gegen den simulierten Verlauf.

    from helpers.overview import simulate_overview
    result = simulate_overview(simulator, "psp_rct_ce_l2_static_bw10", n_cases=2000, every_hours=24)
    result["overview"].summary, result["quality"]["cases"], result["quality"]["portfolio"]
"""
import os
import time

from .datasets import PRESCRIPTIVE, add_paths
from .parameters import N_CASES, SEED

DATA_FOLDER = os.path.join(PRESCRIPTIVE, "data")


def simulate_overview(simulator, model, n_cases=N_CASES, every_hours=24, samples=20, workers=8, seed=SEED, costs=None, data_folder=DATA_FOLDER):
    """
    Ein Lauf des Simulators wie die Bank mit dem Lagebild des PSP-Modells model (Ordner in prescriptive/data, z. B.
    "psp_rct_ce_l2_static_bw10"; evaluation.model_evaluation.find_models). samples: gezogene Suffixe je Fall (und Aktion,
    solange sie offen ist). Liefert {"overview", "events", "quality"}.
    """
    add_paths()
    from simulation.dataset_simulator import SIMULATOR
    from prescriptive import profiles
    from prescriptive.enriched_log import enrich
    from prescriptive.joint import run_policy
    from prescriptive.overview import PSPOverview
    from prescriptive.policies import BankPolicy
    if not SIMULATOR[simulator.dataset]["engine"]:
        raise ValueError(f"{simulator.dataset}: das Lagebild braucht die Engine (alle Fälle gemeinsam), der Datensatz hat nur den Fall-Simulator")
    model_path = os.path.join(data_folder, model, "psp_model.pkl")
    if not os.path.exists(model_path):
        raise ValueError(f"Kein PSP-Modell {model_path}")
    profile, intervention = profiles.get(simulator.dataset), simulator.intervention
    print(f"Lagebild mit {model}: echter Log für Arbeitsstunden, gültige Enden und Anteile der Aktionen ...", flush=True)
    real_log = enrich(simulator.log, n_stages=1, profile=profile, intervention=intervention, costs=costs)
    overview = PSPOverview.for_model(model_path, real_log, intervention, samples=samples, workers=workers, seed=seed, profile=profile)
    print(f"Simulation: {n_cases:,} Fälle wie die Bank, Lagebild alle {every_hours} h ...", flush=True)
    started = time.time()
    try:
        import interventions
        events, _, _ = run_policy(simulator.log, None, BankPolicy(), n_cases, seed, simulator.model_folder, interventions.run_allocation(simulator.dataset),
                               dataset=simulator.dataset, intervention=intervention, observer=overview, observe_hours=every_hours)
    finally:
        overview.close()
    summary = overview.summary
    print(f"Fertig in {time.time() - started:.0f} s: {len(summary)} Beobachtungen, PSP {summary['Rechenzeit (s)'].sum():.0f} s, "
          f"im Mittel {summary['laufende Fälle'].mean():.0f} laufende Fälle, davon neu gezogen {summary['neue Events'].mean():.0f} "
          f"(neue Events) und {summary['neu bedingt'].mean():.0f} (zu wenige passende Suffixe)", flush=True)
    return {"overview": overview, "events": events, "quality": overview.quality(events)}

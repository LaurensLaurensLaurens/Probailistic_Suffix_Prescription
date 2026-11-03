"""
Preprocessing je Datensatz: aus dem Roh-Log (helpers.datasets.load_dataset) das Log für Screening und weitere Schritte.

    from preprocessing import preprocess
    log = preprocess(DATASET, log)

    steps.py     die Maßnahmen: COMMON_STEPS für alle, DATASET_STEPS je Datensatz (aus den Ordnern)
    outcome.py   das Outcome je Fall (OUTCOMES): negative_cycle_days (Standard), negative_days_until
    BPIC2017/    outcome.py: das Outcome profit, auch für die BPIC17-Pipeline in prescriptive/
    BPIC2013/    steps.py: Aktivität mit Support-Ebene, Fallattribute
    BPIC2015/    steps.py: die fünf Logs laden, Gebühren als Zahl, Phasenebene für das Prozessmodell
    BPIC2019/    steps.py: frühe und noch offene Fälle entfernen, Cumulative net worth als Zahl
Was nur ein Datensatz nutzt, steht in dessen Ordner; was mehrere nutzen, hier.

Ablauf von preprocess: COMMON_STEPS, dann DATASET_STEPS[datensatz], optional work_sessions (Lifecycle bereinigt: je
Arbeitsabschnitt ein Paar start -> complete mit einem Mitarbeiter; Standard: die Lifecycles wie im Log, so wie Simulator,
RCT-Daten und Schätzer sie haben), dann die Spalte outcome (Block outcome des Datensatzes im Katalog; die Arbeitszeit ist in beiden Fällen dieselbe). Nach jedem Schritt eine Zeile mit Fällen und Events. Ein neuer Datensatz braucht einen Eintrag in
DATASET_STEPS (Schritte im eigenen Ordner) und, falls nötig, eine Outcome-Funktion in OUTCOMES.
"""
import os

from prescriptive.helpers.datasets import CASE, dataset_info, describe
from .outcome import OUTCOMES, add_outcome
from .steps import COMMON_STEPS, DATASET_STEPS, work_sessions as work_sessions_step


# Der Block des Outcomes: outcome im Katalog, oder mit outcome=Name einer der outcome_optionen des Datensatzes
def outcome_spec(dataset, outcome=None):
    info = dataset_info(dataset)
    if outcome is None:
        return info["outcome"]
    options = info.get("outcome_optionen") or {}
    if outcome not in options:
        raise ValueError(f"Kein Outcome '{outcome}' für {dataset}, möglich: {', '.join(options) or 'keine'} (outcome_optionen im Katalog)")
    return options[outcome]


# Das in diesem Prozess gewählte Outcome (prescriptive/profiles.py set_outcome, scripts/main.py --outcome), sonst das
# Standard-Outcome des Katalogs; für Logs, die nicht durch preprocess mit outcome gehen (simulierte Logs)
def chosen_outcome_spec(dataset):
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    from prescriptive.profiles import OUTCOME_ENV
    return outcome_spec(dataset, os.environ.get(OUTCOME_ENV.format(dataset.upper())) or None)


def preprocess(dataset, log, verbose=True, costs=None, work_sessions=False, outcome=None):
    """
    costs: Kostensätze des Outcomes für alle Datensätze (Standard helpers/parameters.py). work_sessions: Lifecycle
    bereinigt (Arbeitsabschnitte start -> complete); Standard False: die Lifecycles wie im Log (BPIC17: schedule, start,
    suspend, resume, complete, ate_abort, withdraw), so wie Simulator, RCT-Daten und Schätzer sie haben.
    outcome: eine der outcome_optionen des Datensatzes im Katalog statt outcome (z. B. bpic17: gewinn).
    """
    if dataset not in DATASET_STEPS:
        raise ValueError(f"Kein Preprocessing für '{dataset}', möglich: {', '.join(DATASET_STEPS)}")
    for step in COMMON_STEPS + DATASET_STEPS[dataset] + ([work_sessions_step] if work_sessions else []):
        log = step(log)
        if verbose:
            print(f"  {step.__name__:28s} {log[CASE].nunique():>9,} Fälle {len(log):>11,} Events | {(step.__doc__ or '').strip().splitlines()[0]}")
    spec = outcome_spec(dataset, outcome)
    log = add_outcome(log, spec, costs)
    print(f"Preprocessing {dataset}: {describe(log)}\nOutcome ({spec['type']}): {spec.get('beschreibung', spec['type'])}; je Fall im Mittel "
          f"{log.groupby(CASE)['outcome'].first().mean():.2f}")
    return log


def apply_costs(dataset, log, costs, outcome=None):
    """
    Das Outcome des vorverarbeiteten Logs neu, mit den Kostensätzen costs (im Notebook gewählt); ohne Kosten im Outcome
    unverändert. outcome: wie bei preprocess (dasselbe angeben, sonst gilt wieder outcome aus dem Katalog).
    """
    before = log.groupby(CASE)["outcome"].first().mean()
    log = add_outcome(log, outcome_spec(dataset, outcome), costs)
    after = log.groupby(CASE)["outcome"].first().mean()
    if abs(after - before) > 1e-9:
        print(f"Outcome mit den gewählten Kosten {costs}: je Fall im Mittel {after:.2f} (vorher {before:.2f})")
    return log

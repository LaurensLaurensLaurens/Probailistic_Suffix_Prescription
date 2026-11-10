"""
Ein Beispiel, wie das Screening einen Fall sieht: für einen Kandidaten (eine Intervention aus dem Katalog oder
"Mitarbeiter: <Tätigkeit>") ein zufälliger Fall mit Entscheidungspunkt aus dem vorverarbeiteten Log.

    from helpers.screening import candidate_example, show_example
    example = candidate_example("bpic17", log, "Mitarbeiter: O_Accepted", seed=None, raw_log=raw)   # raw: der Roh-Log
    show_example(example)

Je Beispiel:
    summary   Kandidat, Art, Entscheidungs-Event, Fall, Fallattribute, gewählte Option T und Outcome Y
    features  alle Merkmale X des Falls in der Reihenfolge, in der sie in das Propensity-Modell (und die Outcome-Modelle)
              gehen, auch die Zählmerkmale mit 0: Confounder (Fall oder Prefix), Gruppe, Typ, Wert, bei Kalendermerkmalen lesbar
    original  mit raw_log: die Events des Falls im Roh-Log mit allen Spalten, vorne je Event, was das Preprocessing daraus
              macht (entfernt, wird start, wird complete, bleibt)
    events    die Events desselben Falls nach dem Preprocessing mit allen Spalten, vorne Teil: Prefix (bekannt),
              Entscheidung, danach (nicht in X)
show_example zeigt beide untereinander: erst das Original, dann das vorverarbeitete.
"""
import numpy as np
import pandas as pd

from helpers.datasets import ACTIVITY, CASE, LIFECYCLE, RESOURCE, TIMESTAMP, dataset_info
from preprocessing.steps import COMMON_STEPS, DATASET_STEPS, work_item_activities, work_sessions
from .candidates import build_candidate, prepared_log


def candidate_example(dataset, log, name=None, seed=None, case_id=None, raw_log=None, sessions=False):
    """
    name: Kandidat (None: die Standard-Intervention des Datensatzes); seed None: jedes Mal ein anderer Fall. raw_log: der
    Roh-Log (helpers.datasets.load_dataset) für den Vergleich; sessions: ob log mit work_sessions vorverarbeitet ist
    (Standard nein: die Lifecycles wie im Log, preprocess).
    """
    info = dataset_info(dataset)
    name = name or info["standard_intervention"]
    # Für den Kandidaten 
    # Für den gewählten Kandidaten werden aus dem vorverarbeiteten Log alle Fälle mit Entscheidungspunkt bestimmt,
    candidate = build_candidate(dataset, log, name)
    if candidate.skip:
        raise ValueError(f"{name}: nicht testbar ({candidate.skip})")
    if case_id is None:
        case_id = candidate.X.index[np.random.default_rng(seed).integers(len(candidate.X))]
    if case_id not in candidate.X.index:
        raise ValueError(f"Fall {case_id} hat für {name} keinen Entscheidungspunkt")
    at = int(candidate.decision_pos[case_id])

    # Der Fall nach dem Preprocessing, alle Spalten; Nr. = Position im Fall (wie im Screening)
    events = prepared_log(log)
    events = events[events[CASE] == case_id].rename(columns={"_pos": "Nr."}).set_index("Nr.")
    decision = events.loc[at]
    events[TIMESTAMP] = pd.to_datetime(events[TIMESTAMP]).dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    events = events.astype(object).where(events.notna(), "")
    events.insert(0, "Teil", np.where(events.index < at, "Prefix", np.where(events.index == at, "Entscheidung", "danach, nicht in X")))

    summary = {"Kandidat": name, "Art": candidate.kind, "Fall": case_id,
               "Entscheidungs-Event": f"Nr. {at}: {decision[ACTIVITY]}" + (f" ({decision[LIFECYCLE]})" if LIFECYCLE in decision.index else ""),
               # Fallattribute (gelten für den ganzen Fall, stehen auch in X)
               **{column: candidate.X.loc[case_id, column] for column in info["screening"].get("case_attributes", []) + info["screening"].get("numeric_attributes", [])
                  if column in candidate.X},
               "Option T": candidate.T[case_id], "Outcome Y": round(float(candidate.Y[case_id]), 3),
               "Fälle mit Entscheidungspunkt": f"{len(candidate.X):,}", "Merkmale": candidate.X.shape[1]}
    original = original_events(dataset, raw_log, case_id, sessions) if raw_log is not None else None
    return {"dataset": dataset, "summary": pd.Series(summary, name="Wert").to_frame(), "events": events, "original": original,
            "features": feature_table(candidate.X, case_id, info["screening"]), "candidate": candidate}


# Die Events eines Falls im Roh-Log mit allen Spalten und was das Preprocessing aus jedem macht: dieselben Schritte wie
# preprocess, nur für diesen Fall, mit der Zeilennummer des Roh-Logs; die Arbeitsschritte wie im ganzen Log
def original_events(dataset, raw_log, case_id, sessions=False):
    raw = raw_log[raw_log[CASE] == case_id].copy()
    raw["_row"] = np.arange(len(raw))
    done = raw
    for step in COMMON_STEPS + DATASET_STEPS[dataset]:
        done = step(done)
    if sessions:
        whole = raw_log[[ACTIVITY, LIFECYCLE]].assign(**{ACTIVITY: raw_log[ACTIVITY].astype(str).str.strip()}) if LIFECYCLE in raw_log else raw_log
        done = work_sessions(done, work_item_activities(whole))
    after = done.set_index("_row")
    shown = raw.copy()
    shown[TIMESTAMP] = pd.to_datetime(shown[TIMESTAMP], utc=True).dt.tz_localize(None)
    shown = shown.sort_values([TIMESTAMP, "_row"], kind="stable")
    before_lifecycle = shown[LIFECYCLE] if LIFECYCLE in shown else pd.Series("complete", index=shown.index)

    def fate(row, activity, lifecycle):
        if row not in after.index:
            return "entfernt"
        new_activity, new_lifecycle = after.loc[row, ACTIVITY], after.loc[row, LIFECYCLE]
        if str(new_activity) != str(activity).strip():
            return f"wird {new_activity} ({new_lifecycle})"
        return "bleibt" if new_lifecycle == lifecycle else f"wird {new_lifecycle}"

    table = shown.drop(columns="_row")
    table.insert(0, "Preprocessing", [fate(row, activity, lifecycle) for row, activity, lifecycle in zip(shown["_row"], shown[ACTIVITY], before_lifecycle)])
    table[TIMESTAMP] = table[TIMESTAMP].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    table.index = pd.RangeIndex(len(table), name="Nr.")
    return table.astype(object).where(table.notna(), "")


WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]


# Alle Merkmale eines Falls, wie sie in die Modelle gehen (helpers.screening.features), mit Gruppe und Typ
def feature_table(X, case_id, spec):
    case_attributes = set(spec.get("case_attributes", [])) | set(spec.get("numeric_attributes", []))
    rows = []
    for number, column in enumerate(X.columns, 1):
        value = X.loc[case_id, column]
        if column in case_attributes:
            group = "Fallattribut (gilt für den ganzen Fall)"
        elif column.startswith("last:"):
            group = "Event-Attribut: letzter Wert bis zum Entscheidungspunkt"
        elif column.startswith("n:"):
            group = "Verlauf: so oft kam die Aktivität im Prefix vor (jedes Event, auch start)"
        elif column in ("previous_activity", "events"):
            group = "Verlauf: " + ("Aktivität vor dem Entscheidungs-Event" if column == "previous_activity" else "Events im Prefix")
        elif column == "elapsed_h":
            group = "Zeit: Stunden seit Fallbeginn"
        elif column in ("prev_worker", "prev_worker_in_pool", "pool_size"):
            group = "Zuteilung: " + {"prev_worker": "wer zuletzt am Fall gearbeitet hat", "prev_worker_in_pool": "ist er jetzt wählbar (1 = ja)",
                                     "pool_size": "wählbare Mitarbeiter"}[column]
        else:
            group = "Kalender am Entscheidungspunkt"
        readable = ""
        if column == "weekday" and pd.notna(value):
            readable = WEEKDAYS[int(value)]
        elif column == "month" and pd.notna(value):
            readable = f"{int(value - 1) // 12}-{int(value - 1) % 12 + 1:02d}"
        elif column == "hour" and pd.notna(value):
            readable = f"{int(value)} Uhr"
        # Confounder: was die Wahl und das Outcome beeinflussen kann; Fall = steht ab Fallbeginn fest, Prefix = aus dem
        # Verlauf bis zum Entscheidungspunkt
        rows.append({"Nr.": number, "Merkmal": column, "Confounder": "Fall" if column in case_attributes else "Prefix", "Gruppe": group,
                     "Typ": "kategorial" if isinstance(X[column].dtype, pd.CategoricalDtype) else "Zahl",
                     "Wert": "" if pd.isna(value) else value, "lesbar": readable})
    return pd.DataFrame(rows).set_index("Nr.")


def show_example(example):
    """
    Im Notebook: Angaben zum Fall, alle Merkmale X (wie sie in das Propensity-Modell gehen), dann derselbe Fall
    untereinander: erst im Roh-Log (mit raw_log), dann nach dem Preprocessing, jeweils mit allen Spalten. Ungekürzt.
    """
    try:
        from IPython.display import Markdown, display
    except ImportError:
        Markdown, display = str, print
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
        display(example["summary"])
        display(Markdown(f"**Merkmale X: so gehen sie in das Propensity-Modell** ({len(example['features'])} Merkmale, Ziel: Option T)"))
        display(example["features"])
        processed = example["events"]
        if example.get("original") is not None:
            original = example["original"]
            removed = int((original["Preprocessing"] == "entfernt").sum())
            display(Markdown(f"**Original (Roh-Log)**: {len(original)} Events, alle Spalten; Spalte Preprocessing: was daraus wird "
                             f"({removed} entfernt)"))
            display(original)
        display(Markdown(f"**Nach dem Preprocessing**: {len(processed)} Events, alle Spalten; Teil: Prefix = bekannt am Entscheidungspunkt"))
        display(processed)

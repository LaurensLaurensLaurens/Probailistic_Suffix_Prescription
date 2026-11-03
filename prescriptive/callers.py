"""
Mitarbeiter am Arbeitsschritt der Mitarbeiterwahl (lauf.mitarbeiterwahl im Katalog, BPIC17: der Anruf "W_Call incomplete
files"), unabhängig von der Intervention.

Wenn der Arbeitsschritt im Fall zum ersten Mal beginnt, wählt die Policy unter den Mitarbeitern, die jetzt kompetent, im
Dienst und frei sind. Ein eigenes Token
haben die Mitarbeiter mit genug solchen Arbeitsschritten in den Trainingsdaten (top_resources); alle übrigen stehen im
Token "other". Funktionen: first_starts, top_resources.

Das Arbeitstempo der Mitarbeiter (Tempo-Klassen) steht in helpers/preprocessing/work_time.py.
"""
import pandas as pd

OTHER = "other"  # Token für alle Mitarbeiter am Arbeitsschritt außerhalb von top_resources


# Erster Beginn (lifecycle, Standard start) des Arbeitsschritts work_item je Fall, nicht durch einen Systemnutzer:
# DataFrame case_id -> resource, time (leer ohne Lifecycle- oder Mitarbeiter-Spalte). An dieser Stelle wählt die Policy
# den Mitarbeiter.
# Wird aufgerufen von enriched_log.enrich -> first_starts (vor dem Filtern auf complete)
def first_starts(df, work_item, system_users=(), lifecycle="start"):
    if "lifecycle:transition" not in df.columns or "org:resource" not in df.columns:
        return pd.DataFrame(columns=["resource", "time"])
    system_users = set(system_users)
    df = df.sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    result = {}
    for case, activity, event_lifecycle, resource, time in zip(df["case:concept:name"], df["concept:name"], df["lifecycle:transition"], df["org:resource"], df["time:timestamp"]):
        if case not in result and activity == work_item and event_lifecycle == lifecycle and resource not in system_users:
            result[case] = (resource, time)
    return pd.DataFrame.from_dict(result, orient="index", columns=["resource", "time"])


# Die Mitarbeiter, die in einem Event-Log (enriched_log.enrich) mindestens min_calls Arbeitsschritte übernommen haben (eigenes Token im PSP)
# Wird aufgerufen von main.py (__main__) -> top_resources (Block PSP)
def top_resources(data, min_calls):
    counts = data.groupby("case_nr")["call_resource"].first().dropna().value_counts()
    return list(counts[counts >= min_calls].index)

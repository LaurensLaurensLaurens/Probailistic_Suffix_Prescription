"""
Ein Event-Log für alles in prescriptive/: RCT-Daten, Bank- und Policy-Läufe, Offline-Auswertung und der laufende Fall
einer Live-Entscheidung haben dasselbe Format. enrich() erzeugt es aus Events im XES-Schema (Ausgabe der Engine oder
echter Log), für jeden Datensatz gleich; was es je Datensatz braucht, steht in dessen Profil (profiles.py):
Fallattribute, Angebote (offers), Arbeitsschritte (work_item_prefix), Systemnutzer, Outcome und Intervention.
Ohne Angabe gelten Profil und Intervention dieses Prozesses (PROFILE, INTERVENTION); das Notebook mit mehreren
Datensätzen gibt sie je Datensatz mit (enrich(..., profile=, intervention=), with_current_costs(..., profile=)).

Regel: Die Spalten des Logs behalten ihre XES-Namen und Werte, enrich fügt nur abgeleitete Spalten hinzu (einmal
berechnet, gleich für alle). Jedes Modell wählt die Spalten, die es braucht:
- Outcome und Auswertung (joint.summarize, offline_evaluation.py): stage, treatment, outcome und seine Bestandteile
- Mitarbeiter am Arbeitsschritt der Mitarbeiterwahl: call_resource, call_start (callers.top_resources)
- PSP: psp.psp_view wählt seine Spalten, setzt die Tokens und bringt die Angebote auf log1p
- Sequenzen und Zählmerkmale der Schätzer (PSP, XGBoost, LSTM): event, die Aktivität mit ihrem Lifecycle
Eine neue Information ist eine neue Spalte in enrich, kein neues Format.

Lifecycles: Das Log behält alle Lifecycle-Events (BPIC17: schedule, start, suspend, resume, complete, ate_abort,
withdraw der Arbeitsschritte W_...), wie die Engine sie simuliert. Entscheidungspunkt, Aktion, Angebote und Outcome
zählen nur complete-Events (wie bisher); die Schätzer sehen über event auch die übrigen.

Spalten (eine Zeile je Event, Fälle nach Startzeit sortiert):
  aus dem Log   case:concept:name, concept:name, org:resource, time:timestamp (naiv, in UTC), lifecycle:transition,
                die Fallattribute des Profils, die Angebotsattribute (nur am Angebots-Event, nur mit offers)
  abgeleitet    event          Aktivität mit Lifecycle als ein Token (event_token): "W_Validate application:suspend";
                               complete-Events nur mit der Aktivität ("A_Validating", "W_Validate application")
                case_nr        Nummer des Falls in der Reihenfolge der Startzeit
                event_nr       Nummer des Events im Fall
                elapsed_hours  Stunden seit dem ersten Event des Falls
                offer:...      nur mit offers: das aktuelle Angebot, Attribute des zuletzt erstellten Angebots (vor dem ersten leer)
                offer:Count    nur mit offers: Zahl der bisher erstellten Angebote
                stage          am Entscheidungspunkt: der wievielte des Falls (0-basiert), sonst leer
                treatment      am Entscheidungspunkt: 1, wenn in der Runde die Aktion gewählt wurde, sonst 0
                working_hours  Arbeitsstunden des Falls (helpers/preprocessing/work_time.py; 0 ohne Arbeitsschritte)
                cycle_days     Durchlaufzeit des Falls in Tagen, vom ersten bis zum letzten complete-Event
                outcome und seine Bestandteile, je Outcome-Typ (OUTCOME_COLUMNS), auf allen Zeilen:
                  profit (BPIC17)              approved, interest_income, calls (helpers/preprocessing/BPIC2017/outcome.py)
                  negative_days_until          approved (Ziel-Event erreicht), resolution_days
                call_resource, call_start
                               Mitarbeiter und Zeit des ersten Beginns des Arbeitsschritts der Mitarbeiterwahl
                               (lauf.mitarbeiterwahl im Katalog, der erste Punkt; BPIC17: W_Call incomplete files), unabhängig
                               von der Intervention; leer ohne Mitarbeiterwahl
  RCT-Daten zusätzlich rct (Entscheidung und Mitarbeiter zufällig, main.run_rct).
"""
import os
from collections import defaultdict

import pandas as pd

import interventions as intervention_catalog
from prescriptive import profiles
from prescriptive.callers import first_starts
from preprocessing.BPIC2017 import outcome as bpic17_outcome
from prescriptive.costs.repair import outcome as repair_outcome
from preprocessing.work_time import working_hours

CASE = "case:concept:name"
ACTIVITY = "concept:name"
RESOURCE = "org:resource"
TIMESTAMP = "time:timestamp"
LIFECYCLE = "lifecycle:transition"

# Intervention: der gewählte Interventionspunkt aus dem Katalog (interventions/), Standard die standard_intervention des
# Profils (BPIC17: "channel": nach O_Created O_Sent (mail and online) = 0 oder O_Sent (online only) = 1). Die Aktion wird
# nach der Regel der Intervention aus dem Log abgeleitet (rule, siehe enrich). Unabhängig davon die Mitarbeiterwahl
# (lauf.mitarbeiterwahl, interventions.employee_points): beim ersten Beginn ihres Arbeitsschritts wählt die Policy den
# Mitarbeiter (Spalten call_resource, call_start).
# Gewählt mit main.py / offline_evaluation.py --intervention (set_intervention); die Umgebungsvariable gibt die Wahl an die
# Arbeitsprozesse weiter. Schlüssel: key, name, actions, decision_activity, rule, action_activity, action_activities.
INTERVENTION_ENV = "PRESCRIPTIVE_INTERVENTION"
# Log-Profil (prescriptive/profiles.py, gewählt mit --profile; Standard bpic17): Fallattribute, Outcome und
# Standard-Intervention des Logs
PROFILE = profiles.current()
INTERVENTION = intervention_catalog.get(os.environ.get(INTERVENTION_ENV) or PROFILE["intervention"])


# Wird aufgerufen von main.py / offline_evaluation.py (__main__) -> set_intervention, vor allem anderen
def set_intervention(name):
    os.environ[INTERVENTION_ENV] = intervention_catalog.get(name)["key"]
    INTERVENTION.clear()
    INTERVENTION.update(intervention_catalog.get(name))

# Aktivität mit Lifecycle als ein Token (Spalte event). complete-Events behalten den Namen der Aktivität, damit die
# Tokens der Schätzer (z. B. "A_Validating|call" im PSP) und die Namen im Katalog gleich bleiben. Das Trennzeichen ist
# nicht das "|" der PSP-Tokens.
EVENT = "event"
LIFECYCLE_SEPARATOR = ":"


# Wird aufgerufen von enrich, ensure_events und helpers/screening/features.py -> event_token
def event_token(activities, lifecycles):
    activities = pd.Series(activities).astype(str).reset_index(drop=True)
    lifecycles = pd.Series(lifecycles).fillna("complete").astype(str).reset_index(drop=True)
    return activities.where(lifecycles == "complete", activities + LIFECYCLE_SEPARATOR + lifecycles).to_numpy(dtype=object)


# Ist das Token ein complete-Event? (auch für die Tokens der Schätzer, z. B. "A_Validating|call")
# Wird aufgerufen von helpers/preprocessing/BPIC2017/outcome.py -> suffix_outcome
def is_complete_token(token):
    return LIFECYCLE_SEPARATOR not in str(token)


# Spalte event für Logs von vor den Lifecycles (nur complete-Events; gespeicherte RCT-Daten)
# Wird aufgerufen von with_current_costs (jedes gespeicherte Log)
def ensure_events(log):
    if EVENT not in log.columns:
        log[EVENT] = event_token(log[ACTIVITY], log[LIFECYCLE] if LIFECYCLE in log.columns else None)
    return log


# Zeilen vor dem ersten Beginn des Arbeitsschritts der Intervention nach dem Entscheidungspunkt (call_start): was die
# Policy weiß, wenn sie den Mitarbeiter wählt. Das start-Event selbst gehört nicht dazu, es passiert erst mit dem
# gewählten Mitarbeiter. Fälle ohne call_start: alle Zeilen.
# Wird aufgerufen von xgb_estimator.training_frames und lstm_estimator._stage_data -> before_call_start
def before_call_start(log, intervention=None):
    intervention = intervention or INTERVENTION
    start = naive_utc(log["call_start"])
    times = naive_utc(log[TIMESTAMP])
    lifecycle = log[LIFECYCLE] if LIFECYCLE in log.columns else pd.Series("complete", index=log.index)
    is_start = ((log[ACTIVITY] == intervention.get("action_work_item")) & (lifecycle == "start") & (times == start)).to_numpy()
    first_start = pd.Series(log["event_nr"].where(is_start).to_numpy(), index=log.index).groupby(log[CASE]).transform("min")
    before = (times < start) | ((times == start) & (first_start.isna() | (log["event_nr"] < first_start)))
    return before.to_numpy() | start.isna().to_numpy()

# Fallattribute des Profils (BPIC17: LoanGoal, ApplicationType kategorial, RequestedAmount numerisch)
CASE_CATEGORICAL, CASE_NUMERIC = list(PROFILE["static_categorical"]), list(PROFILE["static_numeric"])
CASE_ATTRS = CASE_CATEGORICAL + CASE_NUMERIC
# Angebote (offers im Profil, sonst keine): Attribute, die an einem Event stehen und ab dann gelten. BPIC17: das Angebot an
# O_Create Offer. Das aktuelle Angebot (offer:...) ist, was die Bank zu jedem Zeitpunkt über das Angebot weiß.
# Wird aufgerufen beim Import und von enrich -> current_offer (je Profil)
def current_offer(profile):
    return {name: "offer:" + name for name in profile["dynamic_numeric"]["attributes"]} if profile["dynamic_numeric"] else {}


OFFERS = PROFILE["dynamic_numeric"]
OFFER_ACTIVITY = OFFERS["activity"] if OFFERS else None
CURRENT_OFFER = current_offer(PROFILE)
OFFER_COUNT = "offer:Count"


# Zeitstempel naiv in UTC (die Engine schreibt naive Zeiten in UTC, der echte Log hat eine Zeitzone)
# Wird aufgerufen von enrich und psp.encode_live_prefix -> naive_utc
def naive_utc(values):
    values = pd.Series(values) if not isinstance(values, pd.Series) else values
    if not pd.api.types.is_datetime64_any_dtype(values):
        values = pd.to_datetime(values, utc=True, format="ISO8601")
    if values.dt.tz is not None:
        values = values.dt.tz_convert("UTC").dt.tz_localize(None)
    return values


# Systemnutzer des Datensatzes im Log (Muster screening.system_users im Katalog, wie in der Engine)
def system_users(df, dataset=None):
    from resources.datasets import system_users as dataset_system_users
    return dataset_system_users(dataset or PROFILE["key"], df) if RESOURCE in df.columns else set()


# Outcome negative_days_until (derzeit von keinem Datensatz genutzt): negative Tage bis zum ersten bzw. letzten Event mit prefix (wie
# prescriptive/helpers/preprocessing/outcome.py); approved = ein solches Event kommt vor. Nur complete-Events zählen.
def _until_columns(df, profile, costs=None, complete=None):
    spec = profile["outcome_spec"]
    complete = complete if complete is not None else pd.Series(True, index=df.index)
    elapsed = df["elapsed_hours"].where(complete)
    resolved = df[ACTIVITY].astype(str).str.startswith(spec["prefix"]) & complete
    first = df["elapsed_hours"].where(resolved).groupby(df["case_nr"]).transform("min" if spec.get("which", "first") == "first" else "max")
    df["approved"] = resolved.groupby(df["case_nr"]).transform("any").astype(float)
    df["outcome"] = -(first.fillna(elapsed.groupby(df["case_nr"]).transform("max")) / 24)
    df["resolution_days"] = -df["outcome"]
    return df


# negative_cycle_days (Standard für BPIC17 und BPIC19): negative Durchlaufzeit in Tagen vom ersten bis zum letzten Event des Falls
# (alle Events, wie helpers/preprocessing/outcome.py); bei BPIC17 dazu die Bestandteile des Gewinns (bewilligt, Anrufe,
# Arbeitszeit) als Spalten für die Auswertung
def _cycle_columns(df, profile, costs, complete):
    if profile["key"] == "bpic17":
        df = bpic17_outcome.enriched_columns(df, current_offer(profile), costs, complete)
    return _cycle_outcome(df)


def _cycle_outcome(df, costs=None):
    df["outcome"] = -df["elapsed_hours"].groupby(df["case_nr"]).transform("max") / 24
    return df


# Outcome und seine Bestandteile je Outcome-Typ (outcome im Katalog): Spalten auf dem angereicherten Log.
# complete: welche Zeilen complete-Events sind (nur sie zählen)
OUTCOME_COLUMNS = {
    "profit": lambda df, profile, costs, complete: bpic17_outcome.enriched_columns(df, current_offer(profile), costs, complete),
    "negative_days_until": _until_columns,
    "negative_cycle_days": _cycle_columns,
    "negative_repair_cost": lambda df, profile, costs, complete: repair_outcome.enriched_columns(df, profile["outcome_spec"]),
}
# Outcome neu aus seinen Bestandteilen mit den aktuellen bzw. gewählten Kostensätzen (gespeicherte Logs, die mit einem
# anderen Outcome erzeugt sein können)
CURRENT_COSTS = {"profit": bpic17_outcome.with_current_costs, "negative_cycle_days": _cycle_outcome}
# Ebenso, aus dem Block des Outcomes im Katalog (Werkstatt: Tarif und Satz je Tag)
CURRENT_SPEC_COSTS = {"negative_repair_cost": repair_outcome.enriched_columns}
# Was vom Outcome am Prefix schon feststeht (Name -> Wert), Eingang von PSP (suffix_outcome) und XGBoost; BPIC17: der
# Zinsertrag des aktuellen Angebots. frame: eine Zeile oder je Fall die letzte Zeile des angereicherten Logs
OUTCOME_INPUTS = {"profit": lambda frame: {"income": bpic17_outcome.offer_income(frame, CURRENT_OFFER)}}


# Wird aufgerufen von psp.encode_live_prefix und xgb_estimator -> outcome_inputs
def outcome_inputs(frame):
    inputs = OUTCOME_INPUTS.get(PROFILE["outcome"])
    return inputs(frame) if inputs else {}


# Events im XES-Schema (DataFrame oder Liste von dicts, alle Lifecycle-Events) -> angereichertes Event-Log (siehe oben)
# Das Event-Log eines Falls bis jetzt in der laufenden Simulation (engine.events), angereichert wie im Training (enrich).
# Für die Policies, die an einem Wahlpunkt den Fall bis jetzt brauchen (policies/xgb_policy.py, policies/lstm_policy.py).
# Sortiert engine.events nur seit dem letzten Aufruf nach Fällen ein; ein neuer Lauf (neue Liste engine.events) beginnt neu.
class LiveCaseLogs:
    def __init__(self):
        self.source, self.cursor, self.events = None, 0, defaultdict(list)

    def __call__(self, engine, case_id):
        if engine.events is not self.source:
            self.source, self.cursor, self.events = engine.events, 0, defaultdict(list)
        for event in engine.events[self.cursor:]:
            self.events[event[CASE]].append(event)
        self.cursor = len(engine.events)
        return enrich(self.events[case_id], n_stages=1)


# Wird aufgerufen von main.py -> run_rct / policy_run -> enrich (simulierte Läufe)
# sowie von LiveCaseLogs -> enrich (laufender Fall einer Live-Entscheidung der Policies)
# profile, intervention: ohne Angabe die dieses Prozesses (PROFILE, INTERVENTION); costs: Kostensätze des Outcomes statt
# der Standardwerte (helpers/parameters.py), z. B. die im Notebook gewählten
def enrich(events, n_stages=1, profile=None, intervention=None, costs=None):
    profile, intervention = profile or PROFILE, intervention or INTERVENTION
    offers, offer_columns = profile["dynamic_numeric"], current_offer(profile)
    df = (events if isinstance(events, pd.DataFrame) else pd.DataFrame(events)).copy()
    df[TIMESTAMP] = naive_utc(df[TIMESTAMP])
    # Die Arbeitszeit aus allen Lifecycle-Events (start/resume bis suspend/complete), ebenso der erste Beginn des
    # Arbeitsschritts der Mitarbeiterwahl (Mitarbeiter und Zeit; dort wählt die Policy)
    users = system_users(df, profile["key"])
    hours = working_hours(df, profile["work_item_prefix"], users)
    points = intervention_catalog.employee_points(profile["key"])
    calls = (first_starts(df, points[0]["activity"], users, points[0]["lifecycle"]) if points
             else pd.DataFrame(columns=["resource", "time"]))
    # Fehlen Spalten (z. B. ein Log ohne Angebot oder ohne Lifecycle), werden sie leer angelegt
    case_attributes = list(profile["static_categorical"]) + list(profile["static_numeric"])
    df = df.reindex(columns=list(dict.fromkeys(list(df.columns) + [LIFECYCLE, RESOURCE] + case_attributes + list(offer_columns))))
    df[LIFECYCLE] = df[LIFECYCLE].fillna("complete")
    df = df.reset_index(drop=True)
    df[EVENT] = event_token(df[ACTIVITY], df[LIFECYCLE])
    # Entscheidungspunkt, Aktion, Angebote und Outcome zählen nur complete-Events
    complete = df[LIFECYCLE] == "complete"

    # Fälle nach Startzeit nummerieren. SCOPE teilt Training und Inferenz in der Reihenfolge
    # von case_nr auf, dadurch wird der Split zeitlich.
    case_start = df.groupby(CASE)[TIMESTAMP].transform("min")
    order = pd.DataFrame({"start": case_start, "case": df[CASE], "time": df[TIMESTAMP]}).sort_values(["start", "case", "time"], kind="stable").index
    df, case_start = df.loc[order].reset_index(drop=True), case_start.loc[order].reset_index(drop=True)
    complete = complete.loc[order].reset_index(drop=True)
    df["case_nr"] = pd.factorize(df[CASE])[0]
    df["event_nr"] = df.groupby("case_nr").cumcount()
    df["elapsed_hours"] = (df[TIMESTAMP] - case_start).dt.total_seconds() / 3600

    # Das aktuelle Angebot gilt bis zum nächsten Angebot weiter. Vor dem ersten Angebot bleibt es leer.
    if offers:
        for name, column in offer_columns.items():
            df[column] = pd.to_numeric(df[name], errors="coerce").groupby(df["case_nr"]).ffill()
        df[OFFER_COUNT] = ((df[ACTIVITY] == offers["activity"]) & complete).groupby(df["case_nr"]).cumsum().astype(float)

    # Beispiel (BPIC17, call, ein Fall, n_stages = 1)
    # Event	is_decision	round_nr	called	stage	treatment
    # A_Create Application	            F	−1	F	NaN	0
    # O_Create Offer	                F	−1	F	NaN	0
    # A_Validating	                    T	0	T	0	1
    # O_Returned	                    F	0	T	NaN	0
    # A_Incomplete	                    F	0	T	NaN	0
    # A_Validating	                    T → F (Runde 1 ≥ n_stages)	1	F	NaN	0
    # A_Pending	                        F	1	F	NaN	0
    # stage = wievielter Entscheidungspunkt des Falls (0-basiert), nur auf der Zeile des Entscheidungspunkts.
    # treatment = 1 auf dieser Zeile, wenn in der Runde die Aktion gewählt wurde.
    # Entscheidung ist ein complete-Event, Aktion eines mit dem Lifecycle action_lifecycle der Intervention (Standard
    # complete); die übrigen Lifecycle-Events dazwischen gehören zur Runde.
    decision_lifecycle = intervention.get("decision_lifecycle", "complete")
    is_decision = (df[ACTIVITY] == intervention["decision_activity"]) & (complete if decision_lifecycle == "complete" else df[LIFECYCLE] == decision_lifecycle)
    action_lifecycle = (df[LIFECYCLE] == (intervention.get("action_lifecycle") or "complete")).to_numpy()
    if intervention["rule"] == "next_event":
        # Entscheidungspunkt nur, wo direkt danach eine der Optionen kommt (interventions.decision_labels); Aktion: Option 1
        options = [intervention_catalog.option_token(option) for option in intervention["action_activities"]]
        tokens = df[ACTIVITY].astype(str) + "|" + (df[LIFECYCLE].astype(str) if LIFECYCLE in df else "complete")
        following = tokens.groupby(df["case_nr"]).shift(-1)
        is_decision = is_decision & following.isin(options)
        called = following == options[1]
    round_nr = is_decision.groupby(df["case_nr"]).cumsum() - 1
    if intervention["rule"] == "next_event":
        pass
    elif intervention["rule"] == "follows_before_next_decision":
        # Called ist True zwischen zwei Entscheidungs-Events
        called = ((df[ACTIVITY] == intervention["action_activity"]) & action_lifecycle).groupby([df["case_nr"], round_nr]).transform("any")
    elif intervention["rule"] == "follows_later":
        # Die Aktion kommt nach dem Entscheidungs-Event noch vor (z. B. ein weiteres Angebot)
        is_action = pd.Series(intervention_catalog.is_action(df[ACTIVITY], intervention) & action_lifecycle, index=df.index).astype(int)
        called = is_action.groupby(df["case_nr"]).transform("sum") - is_action.groupby(df["case_nr"]).cumsum() > 0
    else:
        raise ValueError(f"Intervention {intervention['key']}: unbekannte Regel {intervention['rule']}")
    is_decision = is_decision & (round_nr < n_stages)
    df["stage"] = round_nr.where(is_decision)
    df["treatment"] = (is_decision & called).astype(int)

    # Arbeitszeit (oben berechnet) und Durchlaufzeit vom ersten bis zum letzten complete-Event, dann das Outcome des Datensatzes
    df["working_hours"] = df[CASE].map(hours).fillna(0.0)
    complete_hours = df["elapsed_hours"].where(complete).groupby(df["case_nr"])
    df["cycle_days"] = (complete_hours.transform("max") - complete_hours.transform("min")).fillna(0.0) / 24
    df = OUTCOME_COLUMNS[profile["outcome"]](df, profile, costs, complete)
    df["call_resource"] = df[CASE].map(calls["resource"])
    df["call_start"] = naive_utc(df[CASE].map(calls["time"]))
    return df


# Outcome neu aus seinen Bestandteilen, mit den aktuellen Kostensätzen (gespeicherte Logs); ohne Kosten unverändert
# Wird aufgerufen von main.py (__main__), offline_evaluation.py, dr_learner.py, aipw.py, cost_sweep.py -> with_current_costs
# Ältere Logs (nur complete-Events, ohne Spalte event) bekommen dabei die Spalte event.
def with_current_costs(log, profile=None, costs=None):
    log = ensure_events(log)
    profile = profile or PROFILE
    if profile["outcome"] in CURRENT_SPEC_COSTS:
        return CURRENT_SPEC_COSTS[profile["outcome"]](log, profile["outcome_spec"])
    recompute = CURRENT_COSTS.get(profile["outcome"])
    return recompute(log, costs) if recompute else log


# Mehrere Event-Logs (enrich) zu einem: Fälle wieder nach Startzeit nummeriert (case_nr); alle anderen Spalten gelten je
# Fall und bleiben. Die Fall-IDs müssen verschieden sein (z. B. R0_..., R3_...).
# Wird aufgerufen von main.py (__main__) -> concat_logs (--rct-extend-from)
def concat_logs(*logs):
    log = pd.concat(logs, ignore_index=True)
    start = log.groupby(CASE)[TIMESTAMP].transform("min")
    order = pd.DataFrame({"start": start, "case": log[CASE], "time": log[TIMESTAMP]}).sort_values(["start", "case", "time"], kind="stable").index
    log = log.loc[order].reset_index(drop=True)
    log["case_nr"] = pd.factorize(log[CASE])[0]
    return log


with_current_profit = with_current_costs  # alter Name (Skripte in data/)

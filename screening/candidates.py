"""
Datensatz je Kandidat (Control Flow und Variablen): eine Zeile je Fall, der den Entscheidungspunkt erreicht, mit den
Merkmalen X (helpers.screening.features), der gewählten Option T und dem Outcome Y (Spalte outcome aus dem Preprocessing).

    data = build_candidates(DATASET, log, art="control_flow")   # bzw. art="variable"; {Kandidat: CandidateData}
    one = build_candidate(DATASET, log, "Mitarbeiter: O_Accepted")  # ein Kandidat, Control Flow oder Variable

Die Art des Kandidaten (Feld test im Katalog, kind) bestimmt, was die Option ist (KINDS):
    action          (Standard, wenn test eine rule hat) folgt die Aktion laut Regel? T = 0/1 (events.actions)
    attribute_bins  Wert von attribute am Entscheidungs-Event (decision bzw. decision_contains, lifecycle Standard
                    complete), optional geteilt durch ratio_to, in Klassen (bins: Grenzen, labels: Namen)
    case_attribute  Wert des Fallattributs attribute; entschieden zu Fallbeginn
    resource        wer das Entscheidungs-Event ausführt (resource_column, Standard org:resource), ohne Systemnutzer
    lag_bins        Stunden vom Entscheidungs-Event bis zur Aktion (wie action), in Klassen (bins: median teilt am
                    Median); nur Fälle mit Aktion
    skip            nicht testbar (der Grund steht im Ergebnis)
exclude_features: Merkmale, die die Wahl selbst enthalten (z. B. das aktuelle Angebot bei der Wahl der Laufzeit).
max_cases im Block screening: zufällige Auswahl der Entscheidungsfälle.

Die Mitarbeiter-Kandidaten stehen im Katalog wie alle anderen: die inhaltlich begründeten (BPIC17 z. B. caller) und
"Mitarbeiter: <Tätigkeit>" für jede Tätigkeit, bei der die Wahl des Mitarbeiters eine eigene Entscheidung ist
(resource_decision_points: kein Systemschritt, genug Fälle, nicht nur derselbe Mitarbeiter wie bei der letzten Arbeit
davor, nicht der Beginn des Falls).

Pool der Mitarbeiter-Kandidaten (pool, POOLS): unter wem je Fall gewählt wird; die Auswertung vergleicht je Fall nur
diese Mitarbeiter (helpers.screening.evaluation). None: alle Mitarbeiter mit genug Fällen, wie die übrigen Variablen.
    kompetent   wer die Tätigkeit laut Berechtigungsmodell des Datensatzes darf (process-simulation-engine/resources/
                datasets.py, RESOURCES: BPIC17 OrdinoR, BPIC13 Teams, BPIC19 wer sie im Log ausgeführt hat), ohne
                Systemnutzer; hängt nur von der Tätigkeit ab, ist also für alle Fälle derselbe. Mit lauf.berechtigung
                erfahrung im Katalog wie die Engine: wer den Arbeitsschritt im echten Log mindestens 30-mal begonnen oder
                fortgesetzt hat (resources/case_allocation.experienced_employees)
    im Dienst   kompetent und zum Zeitpunkt der Entscheidung im Dienst (Dienstplan der Engine: an dem Tag von der Stunde
                des ersten bis zu der des letzten Events)
    frei        im Dienst und zu dem Zeitpunkt in keinem anderen Arbeitsabschnitt (start/resume bis suspend/complete/
                ate_abort/withdraw, wie die Engine); so wählt die Engine (free_candidates)
Der gewählte Mitarbeiter gehört immer zum Pool seines Falls (ResourcePool, Anteil außerhalb in der Spalte Pool).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from helpers.datasets import ACTIVITY, CASE, LIFECYCLE, RESOURCE, TIMESTAMP, dataset_info, intervention_table
from .events import actions, decision_events, prefix_until
from .features import features


@dataclass
class CandidateData:
    name: str
    X: pd.DataFrame  # Merkmale am Entscheidungspunkt, Index: Fall
    T: pd.Series  # die gewählte Option (Control Flow: 1 = Aktion, 0 = keine)
    Y: pd.Series  # Outcome des Falls
    lag_hours: pd.Series  # Stunden vom Entscheidungs-Event bis zur ersten Aktion (Fälle mit Aktion)
    system: pd.Series  # je Aktions- bzw. Entscheidungs-Event: von einem Systemnutzer?
    skip: str = None  # Grund, warum der Kandidat nicht testbar ist
    pool: pd.Series = None  # Mitarbeiter-Kandidaten: je Fall die Mitarbeiter, unter denen gewählt wird (frozenset); sonst None
    pool_name: str = None  # welcher Pool (POOLS)
    pool_outside: float = 0.0  # Anteil der Fälle, deren gewählter Mitarbeiter laut Regel nicht im Pool wäre
    decision_pos: pd.Series = None  # je Fall die Position des Entscheidungs-Events im Fall (0 = erstes Event)
    kind: str = None  # Art des Kandidaten (KINDS)

    def __repr__(self):
        if self.skip:
            return f"CandidateData({self.name}: nicht testbar, {self.skip})"
        return f"CandidateData({self.name}: {len(self.X)} Fälle, Optionen {self.T.value_counts().to_dict()}, {self.X.shape[1]} Merkmale)"


POOLS = ("kompetent", "im Dienst", "frei")

# Das ist für alle Beispiele
# {
#   'call':              CandidateData(21870 Fälle, Optionen {1: 14999, 0: 6871}, 46 Merkmale),
#   'second_offer':      CandidateData(31509 Fälle, Optionen {0: 22950, 1: 8559},  31 Merkmale),
#   'channel':           CandidateData(31509 Fälle, Optionen {0: 31094, 1: 415},   31 Merkmale),
#   'call_after_offers': CandidateData(31509 Fälle, Optionen {1: 31362, 0: 147},   31 Merkmale),
# }
def build_candidates(dataset, log, art="control_flow", seed=0, pool=None):
    log, spec = prepared_log(log), dataset_info(dataset)["screening"]
    outcome = log.groupby(CASE)["outcome"].first()
    tests = candidate_tests(dataset, art)
    pools = _pool(dataset, log, pool)
    return {name: candidate(name, log, outcome, test, spec, seed, pools, pool) for name, test in tests.items()}


# Ein Kandidat (Name aus dem Katalog, Control Flow oder Variable, oder "Mitarbeiter: <Tätigkeit>")
# Das ist für ein Beispiel ganz am Anfang des runs
def build_candidate(dataset, log, name, seed=0, pool=None):
    log, spec = prepared_log(log), dataset_info(dataset)["screening"]
    tests = {**candidate_tests(dataset, "control_flow"), **candidate_tests(dataset, "variable")}
    if name not in tests:
        raise ValueError(f"Unbekannter Kandidat '{name}' für {dataset}, möglich: {', '.join(tests)}")
    return candidate(name, log, log.groupby(CASE)["outcome"].first(), tests[name], spec, seed, _pool(dataset, log, pool), pool)


# Das Log, wie das Screening es liest: nach Fall und Zeit sortiert, _pos = Position des Events im Fall
def prepared_log(log):
    if "outcome" not in log:
        raise ValueError("Das Log hat keine Spalte outcome: erst helpers.preprocessing.preprocess(DATASET, log)")
    log = log.sort_values([CASE, TIMESTAMP], kind="stable").reset_index(drop=True)
    log["_pos"] = log.groupby(CASE).cumcount()
    return log


# Die Kandidaten einer Art aus dem Katalog: Name -> test
def candidate_tests(dataset, art):
    table = intervention_table(dataset, full=True)
    return {row["name"]: row["test"] or {} for _, row in table[table["art"] == art].iterrows()}


def _pool(dataset, log, pool):
    if pool is not None and pool not in POOLS:
        raise ValueError(f"Unbekannter Pool {pool}, möglich: {', '.join(POOLS)} oder None")
    return ResourcePool(dataset, log, pool) if pool is not None else None


# Wer eine Tätigkeit darf, laut Berechtigungsmodell des Datensatzes (wie die Engine), ohne Systemnutzer: Tätigkeit -> frozenset
def competent_resources(dataset, log):
    from helpers.datasets import add_paths
    add_paths()
    import interventions
    from next_event_prediction.datasets import model_spec
    from resources.datasets import dataset_permissions, main_system_user, resource_spec, system_users
    users = system_users(dataset, log)
    prefix = model_spec(dataset).get("work_item_prefix")
    if interventions.run_permissions(dataset) == "erfahrung" and prefix:
        # Wie die Engine mit lauf.berechtigung erfahrung: wer den Arbeitsschritt oft genug begonnen oder fortgesetzt hat
        from resources.case_allocation import experienced_employees, work_counts
        counts, cache = work_counts(log, prefix), {}

        def experienced(activity):
            if activity not in cache:
                cache[activity] = frozenset(experienced_employees(counts, activity, users)) if str(activity).startswith(prefix) else frozenset()
            return cache[activity]
        return experienced
    permissions = dataset_permissions(resource_spec(dataset)["permissions"], log, users, main_system_user(log, users))
    cache = {}

    def allowed(activity):
        if activity not in cache:
            cache[activity] = frozenset(str(r) for r in permissions.get_eligible_resources(activity)) - users
        return cache[activity]
    return allowed


class ResourcePool:
    """
    Der Pool je Fall zum Zeitpunkt der Entscheidung (POOLS): kompetent, dazu im Dienst (Dienstplan der Engine,
    resources/daily_roster.py), dazu frei (in keinem anderen Arbeitsabschnitt, _sessions). Der tatsächlich gewählte Mitarbeiter gehört immer zum Pool (er war ja wählbar); outside: Anteil der
    Fälle, in denen er es laut Regel nicht wäre.
    """

    def __init__(self, dataset, log, level):
        from helpers.datasets import add_paths
        add_paths()
        from resources.datasets import system_users
        self.level, self.competent = level, competent_resources(dataset, log)
        if level in ("im Dienst", "frei"):
            from resources.daily_roster import DailyRoster
            self.roster = DailyRoster(log, system_users(dataset, log))
        if level == "frei":
            self.sessions = _sessions(log)

    def __call__(self, activity, times, chosen):
        """times, chosen: je Fall (Index) Zeit der Entscheidung und gewählter Mitarbeiter -> (Pools, Anteil außerhalb)."""
        members = sorted(self.competent(activity))
        allowed = np.ones((len(times), len(members)), dtype=bool)
        if self.level in ("im Dienst", "frei"):
            hours = pd.to_datetime(times).dt.floor("h")
            duty = {hour: self.roster.on_duty(hour.to_pydatetime()) for hour in hours.unique()}
            allowed &= np.array([[member in duty[hour] for member in members] for hour in hours])
        if self.level == "frei":
            seconds = (pd.to_datetime(times) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
            for j, member in enumerate(members):
                starts, latest_end = self.sessions.get(member, (np.zeros(0), np.zeros(0)))
                if not len(starts):
                    continue  # nie gearbeitet (z. B. in einem kurzen simulierten Log): nie belegt
                # belegt: ein Abschnitt hat vorher begonnen und endet erst danach
                before = np.searchsorted(starts, seconds, side="left") - 1
                busy = (before >= 0) & (latest_end[np.clip(before, 0, None)] > seconds)
                allowed[:, j] &= ~busy
        pools, outside = [], 0
        for row, person in zip(allowed, chosen):
            pool = frozenset(m for m, ok in zip(members, row) if ok)
            if person not in pool:
                outside += 1
                pool = pool | {person}
            pools.append(pool)
        return pd.Series(pools, index=times.index, dtype=object), outside / max(len(times), 1)


# Arbeitsabschnitte je Mitarbeiter, wie in der Engine und in helpers/preprocessing/work_time.py: je Fall und Tätigkeit
# von start bzw. resume bis zum direkt folgenden suspend, complete, ate_abort oder withdraw (Mitarbeiter des öffnenden
# Events); geht mit den Lifecycles wie im Log und mit work_sessions (start -> complete). Beginne sortiert und das
# größte Ende bis dahin (in Sekunden)
OPENING, CLOSING = ("start", "resume"), ("suspend", "complete", "ate_abort", "withdraw")


def _sessions(log):
    if LIFECYCLE not in log:
        raise ValueError('Pool "frei" braucht Lifecycle-Events (start, complete, ...) im Log')
    work = log[log[ACTIVITY].isin(log.loc[log[LIFECYCLE].isin(OPENING), ACTIVITY].unique()) & log[LIFECYCLE].isin(OPENING + CLOSING)]
    work = work.sort_values([CASE, ACTIVITY, TIMESTAMP], kind="stable")
    seconds = (pd.to_datetime(work[TIMESTAMP]) - pd.Timestamp("1970-01-01")).dt.total_seconds()
    group = work.groupby([CASE, ACTIVITY], sort=False)
    closes = group[LIFECYCLE].shift(-1).isin(CLOSING) & work[LIFECYCLE].isin(OPENING)
    pairs = pd.DataFrame({"resource": work.loc[closes, RESOURCE].to_numpy(), "start": seconds[closes].to_numpy(),
                          "end": seconds.groupby([work[CASE], work[ACTIVITY]], sort=False).shift(-1)[closes].to_numpy()}).sort_values("start")
    return {resource: (part["start"].to_numpy(), np.maximum.accumulate(part["end"].to_numpy())) for resource, part in pairs.groupby("resource")}


# Wahlpunkte der Mitarbeiter (resource_decision_points): Kriterien, nach denen die Kandidaten "Mitarbeiter: <Tätigkeit>"
# im Katalog ausgewählt sind
MAX_SYSTEM_SHARE = 0.5  # Systemschritt: Systemnutzer (bzw. niemand) führen mehr als diesen Anteil der Events aus
MIN_CASES = 100  # so viele Fälle mit der Tätigkeit mindestens
MAX_INHERITED = 0.6  # höchstens so oft derselbe Mitarbeiter wie bei der letzten Arbeit davor (sonst Teil einer Sitzung)
MAX_CASE_START = 0.5  # höchstens so oft das erste Event des Falls (sonst legt der Mitarbeiter den Fall an, er wird nicht zugeteilt)
BOOKKEEPING = ("schedule", "ate_abort", "withdraw")  # Lifecycle-Events, die beim Zuteilen gebucht werden, keine Arbeit


def resource_decision_points(dataset, log):
    """
    Je Tätigkeit: ist die Wahl des Mitarbeiters eine eigene Entscheidung? Gezählt am ersten Beginn (start) der Tätigkeit im
    Fall, wo es ihn gibt, sonst am ersten complete-Event (wie die Kandidaten "Mitarbeiter: <Tätigkeit>" im Katalog).
    Spalten: Fälle, Systemanteil, Mitarbeiter (mit mindestens 100 Fällen), "wie davor" (Anteil der Fälle, in denen der
    Mitarbeiter derselbe ist wie bei der letzten Arbeit davor, ohne BOOKKEEPING-Events), Fallbeginn (Anteil erstes Event),
    im Katalog (Name des Mitarbeiter-Kandidaten dort) und Wahlpunkt (alle Kriterien oben erfüllt).
    """
    log = prepared_log(log)
    spec = dataset_info(dataset)["screening"]
    system = log[RESOURCE].isna() | log[RESOURCE].astype(str).str.contains(spec["system_users"])
    work = log[RESOURCE].where(~log[LIFECYCLE].isin(BOOKKEEPING)) if LIFECYCLE in log else log[RESOURCE]
    previous = work.groupby(log[CASE]).transform(lambda s: s.shift(1).ffill())
    table = intervention_table(dataset, full=True)
    catalog = {(t.get("decision"), t.get("lifecycle", "complete")): name for name, t in zip(table["name"], table["test"])
               if isinstance(t, dict) and t.get("kind") == "resource"}
    rows = []
    for activity, group in log.groupby(ACTIVITY, sort=True):
        lifecycle = "start" if (group[LIFECYCLE] == "start").any() else "complete"
        first = group[group[LIFECYCLE] == lifecycle].groupby(CASE).head(1).index
        own = first[~system[first].to_numpy()]
        counts = log.loc[own, RESOURCE].value_counts()
        rows.append({"Tätigkeit": activity, "Lifecycle": lifecycle, "Fälle": len(first),
                     "Systemanteil": float(system[group.index[group[LIFECYCLE] == lifecycle]].mean()),
                     "Mitarbeiter": len(counts), "mit mindestens 100 Fällen": int((counts >= 100).sum()),
                     "wie davor": float((log.loc[own, RESOURCE] == previous[own]).mean()) if len(own) else np.nan,
                     "Fallbeginn": float((log.loc[first, "_pos"] == 0).mean()),
                     "im Katalog": catalog.get((activity, lifecycle), "")})
    rows = pd.DataFrame(rows)
    rows["Wahlpunkt"] = ((rows["Systemanteil"] <= MAX_SYSTEM_SHARE) & (rows["Fälle"] >= MIN_CASES) & (rows["Mitarbeiter"] >= 2)
                         & (rows["wie davor"] <= MAX_INHERITED) & (rows["Fallbeginn"] <= MAX_CASE_START))
    return rows.set_index("Tätigkeit")

# Aufgerufen von Build_Candidates
def candidate(name, log, outcome, test, spec, seed=0, pools=None, pool_name=None):
    if test.get("skip"):
        return CandidateData(name, pd.DataFrame(), pd.Series(dtype=object), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=bool), skip=test["skip"])
    kind = test.get("kind", "action")
    if kind not in KINDS:
        raise ValueError(f"{name}: unbekannte Art {kind}, möglich: {', '.join(KINDS)}")
    decision_pos, T, prefix, lag, system = KINDS[kind](log, test, spec)
    X = features(log, prefix, decision_pos, spec)
    X = X.drop(columns=[column for column in test.get("exclude_features", ()) if column in X])
    # Fälle mit Outcome und Option; bei max_cases eine zufällige Auswahl
    T = T.reindex(X.index)
    keep = (outcome.reindex(X.index).notna() & T.notna()).to_numpy()
    if spec.get("max_cases") and keep.sum() > spec["max_cases"]:
        rows = np.random.default_rng(seed).choice(np.flatnonzero(keep), spec["max_cases"], replace=False)
        keep = np.isin(np.arange(len(X)), rows)
    X = X[keep]
    data = CandidateData(name, X, T[keep], outcome.reindex(X.index).astype(float), lag, system,
                         decision_pos=decision_pos.reindex(X.index), kind=kind)
    # Mitarbeiter-Kandidaten mit Pool: je Fall die wählbaren Mitarbeiter zur Zeit des Entscheidungs-Events
    if kind == "resource" and pools is not None and test.get("resource_column", RESOURCE) == RESOURCE:
        at = log[log["_pos"] == log[CASE].map(data.decision_pos)].set_index(CASE)[TIMESTAMP].reindex(X.index)
        data.pool, data.pool_outside = pools(test["decision"], at, data.T)
        data.pool_name = pool_name
    if kind == "resource":
        data.X = with_allocation(log, data, spec)
    return data


# Mitarbeiter-Kandidaten: Merkmale, die die Zuteilung erklären (Confounder der Wahl des Mitarbeiters). Wer zuletzt am Fall
# gearbeitet hat (ohne Buchungs-Events und Systemnutzer; Engine und Bank lassen den Fall oft bei ihm), mit Pool dazu, ob er
# jetzt wählbar ist, und wie groß der Pool ist
ALLOCATION_FEATURES = ("prev_worker", "prev_worker_in_pool", "pool_size")


def with_allocation(log, data, spec):
    work = log[RESOURCE].where(~log[LIFECYCLE].isin(BOOKKEEPING)) if LIFECYCLE in log else log[RESOURCE]
    system = log[RESOURCE].isna() | log[RESOURCE].astype(str).str.contains(spec["system_users"])
    work = work.where(~system)
    before = (log["_pos"] < log[CASE].map(data.decision_pos)).to_numpy()
    previous = work[before].groupby(log.loc[before, CASE]).last().reindex(data.X.index)
    X = data.X.copy()
    X["prev_worker"] = previous.fillna("<keiner>").astype(str).astype("category")
    if data.pool is not None:
        X["prev_worker_in_pool"] = np.array([float(person in pool) for person, pool in zip(previous, data.pool)])
        X["pool_size"] = data.pool.map(len).astype(float).to_numpy()
    return X


# ------------------------------------------------------------------ Arten: (Entscheidungs-Position, Option, Prefix, Verzögerung, Systemnutzer)

def _bins(values, bins, labels):
    labels = np.asarray(labels, dtype=object)
    return pd.Series(np.where(values.notna(), labels[np.digitize(values.fillna(0).to_numpy(), bins)], None), index=values.index)


def action(log, test, spec):
    decision_pos, treated, lag, system, prefix = actions(log, test, spec)
    return decision_pos, treated.astype(int), prefix, lag, system


def attribute_bins(log, test, spec):
    rows = decision_events(log, test)
    values = pd.to_numeric(rows[test["attribute"]], errors="coerce")
    if test.get("ratio_to"):
        values = values / pd.to_numeric(rows[test["ratio_to"]], errors="coerce").replace(0, np.nan)
    decision_pos = rows["_pos"]
    return decision_pos, _bins(values, test["bins"], test["labels"]), prefix_until(log, decision_pos), pd.Series(dtype=float), pd.Series(dtype=bool)


def case_attribute(log, test, spec):
    first = log.groupby(CASE)[test["attribute"]].first()
    return pd.Series(0, index=first.index), first.astype(str), log.iloc[0:0], pd.Series(dtype=float), pd.Series(dtype=bool)


def resource(log, test, spec):
    rows = decision_events(log, test)
    who = rows[test.get("resource_column", RESOURCE)].astype(str)
    system = who.str.contains(spec["system_users"])
    decision_pos = rows["_pos"]
    return decision_pos, who.where(~system), prefix_until(log, decision_pos), pd.Series(dtype=float), system


def lag_bins(log, test, spec):
    decision_pos, treated, lag, system, prefix = actions(log, test, spec)
    threshold = [float(lag.median())] if test["bins"] == "median" else test["bins"]
    return decision_pos, _bins(lag.reindex(decision_pos.index), threshold, test["labels"]), prefix, lag, system


KINDS = {"action": action, "attribute_bins": attribute_bins, "case_attribute": case_attribute, "resource": resource, "lag_bins": lag_bins}

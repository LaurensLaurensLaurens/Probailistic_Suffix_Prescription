"""
XGBoost-Modell der Policy xgb (policies/xgb_policy.py), als S-Learner; gespeichert in models/xgb_model.pkl: je Wahlpunkt (interventions.choice_points) ein
Regressionsmodell, das den Outcome des Falls aus dem Fall bis zum Wahlpunkt und der Option schätzt. Die Policy nimmt die
Option mit dem höchsten geschätzten Outcome.

Proben (fit) aus einem Event-Log mit der Spalte outcome (je Fall gleich, z. B. nach helpers/preprocessing), jedes
Erreichen eines Wahlpunkts eine Probe (die Engine fragt die Policy auch jedes Mal):
- control_flow (BPIC17 channel): jedes complete-Event der Aktivität des Punkts, nach dem direkt eine der Optionen kommt;
  Merkmale bis einschließlich dieses Events, Option = die folgende Aktivität
- employee (BPIC17 W_Call incomplete files, start): jeder Beginn des Arbeitsschritts durch einen Mitarbeiter (kein
  Systemnutzer); Merkmale bis vor dieses Event, Option = der Mitarbeiter, jeder mit eigener Kategorie (ein Mitarbeiter,
  der im Training an diesem Wahlpunkt nie vorkam, ist bei der Abfrage eine fehlende Kategorie)

Merkmale (Prefix.row, für Training und Abfrage dieselben): Fallattribute (screening im Katalog: case_attributes,
numeric_attributes), Zahl der Events je Token (Aktivität|Lifecycle), letztes Token, Zahl der Events, Stunden seit
Fallbeginn und seit dem letzten Event, Wochentag und Stunde, die aktuellen Werte der Event-Attribute (screening:
event_numeric; BPIC17 das letzte Angebot) und wie oft sie gesetzt wurden.
Mit params {"option_rates": ...} (nur Werkstatt mit dem Outcome kosten, prescriptive/costs/repair) an der Mitarbeiterwahl dazu
der Stundensatz der Option nach dem Tarif des gewählten Outcomes: "jetzt" der Satz zum Zeitpunkt der Wahl (option_rate),
"jetzt_48h" dazu sein Mittel über die nächsten 48 Stunden (option_rate_48h). Im Training der Satz des gewählten Mitarbeiters,
in der Abfrage der jeder Option; nach einem Tarifwechsel also die neuen Sätze, ohne neues Training.

    model = XGBModel("bpic17").fit(log)          # log: Event-Log mit Spalte outcome
    model.save("prescriptive/models/xgb_model.pkl")
    XGBModel.load(path).values(point, events, options, now)   # geschätzter Outcome je Option
"""
import os
import pickle
import re
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

CASE, ACTIVITY, LIFECYCLE, RESOURCE, TIMESTAMP = "case:concept:name", "concept:name", "lifecycle:transition", "org:resource", "time:timestamp"
VALIDATION_SHARE = 0.1  # Anteil der Fälle für das frühe Stoppen
PARAMS = {"n_estimators": 2000, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.9,
          "min_child_weight": 20, "early_stopping_rounds": 50, "tree_method": "hist", "enable_categorical": True, "n_jobs": 8}


def _seconds(values):
    """Zeitstempel als Sekunden seit 1970 (UTC; ohne Zeitzone gelten sie als UTC, wie in der Engine)."""
    times = pd.to_datetime(pd.Series(values))
    if times.dt.tz is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    return (times - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()


class Prefix:
    """Was ein Fall bis jetzt hatte, Event für Event fortgeschrieben (add); row liefert die Merkmale zu einem Zeitpunkt."""

    def __init__(self, attributes, event_numeric):
        self.attributes, self.event_numeric = attributes, event_numeric
        self.counts, self.n, self.first, self.last, self.last_token = Counter(), 0, None, None, None
        self.current, self.n_set = {name: np.nan for name in event_numeric}, 0

    def add(self, token, seconds, values):
        self.counts[token] += 1
        self.n += 1
        self.first = seconds if self.first is None else self.first
        self.last, self.last_token = seconds, token
        set_here = {name: value for name, value in values.items() if value is not None and not pd.isna(value)}
        if set_here:
            self.current.update(set_here)
            self.n_set += 1

    def row(self, now):
        moment = datetime.fromtimestamp(now, tz=timezone.utc)
        return {**self.attributes, **{f"now:{name}": value for name, value in self.current.items()}, "n_set": float(self.n_set),
                "events": float(self.n), "last_token": self.last_token,
                "elapsed_hours": (now - self.first) / 3600 if self.first is not None else 0.0,
                "since_last_hours": (now - self.last) / 3600 if self.last is not None else 0.0,
                "weekday": float(moment.weekday()), "hour": moment.hour + moment.minute / 60,
                **{f"n:{token}": float(count) for token, count in self.counts.items()}}


class XGBModel:
    def __init__(self, dataset, points=None, seed=42, params=None):
        """
        dataset: Schlüssel im Katalog; points: die Wahlpunkte (Standard: interventions.choice_points des Datensatzes);
        params: {"option_rates": "jetzt" oder "jetzt_48h"} für die Stundensätze der Option (siehe oben).
        """
        import interventions
        screening = interventions.catalog()[dataset]["screening"]
        self.dataset, self.seed = dataset, seed
        self.points = {point.name: point for point in (points if points is not None else interventions.choice_points(dataset))}
        self.categorical, self.numeric = list(screening["case_attributes"]), list(screening.get("numeric_attributes") or [])
        self.event_numeric = list(screening.get("event_numeric") or [])
        self.system_users = screening.get("system_users")
        self.models, self.columns, self.categories, self.options = {}, {}, {}, {}
        self.option_rates = (params or {}).get("option_rates")
        if self.option_rates not in (None, "jetzt", "jetzt_48h"):
            raise ValueError(f"option_rates {self.option_rates}: möglich jetzt, jetzt_48h")

    # ------------------------------------------------------------------ Merkmale

    def _attributes(self, first):
        """Fallattribute aus dem ersten Event des Falls (dict Spalte -> Wert)."""
        return {**{name: str(first.get(name)) for name in self.categorical},
                **{name: float(first.get(name)) if first.get(name) is not None else np.nan for name in self.numeric}}

    def _rates(self, options, now):
        """Je Option die Stundensätze als Merkmale (option_rates) zum Zeitpunkt now (Sekunden seit 1970), Tarif des gewählten Outcomes."""
        from preprocessing import chosen_outcome_spec
        from prescriptive.costs.repair import rates
        spec = chosen_outcome_spec(self.dataset)
        if spec["type"] != "negative_repair_cost":
            raise ValueError(f"XGBoost mit option_rates braucht das Outcome kosten der Werkstatt, gewählt ist {spec['type']}")
        at, n = pd.Timestamp(now, unit="s"), len(options)
        columns = {"option_rate": rates.rate(list(options), [at] * n, spec["tarif"])}
        if self.option_rates == "jetzt_48h":
            columns["option_rate_48h"] = rates.labor_cost(list(options), [at] * n, [at + pd.Timedelta(hours=48)] * n, spec["tarif"]) / 48
        return [dict(zip(columns, values)) for values in zip(*columns.values())]

    def _frame(self, rows, point, options):
        """Merkmale (rows: Prefix.row) und Option je Zeile als Tabelle mit den Spalten und Kategorien des Trainings."""
        X = pd.DataFrame(rows).reindex(columns=self.columns[point.name])
        for column in [c for c in X.columns if c.startswith("n:")]:
            X[column] = X[column].fillna(0.0)
        for column, categories in self.categories.items():
            if column in X:
                X[column] = pd.Categorical(X[column], categories=categories)
        X["option"] = pd.Categorical(options, categories=self.options[point.name])
        # XGBoost erlaubt [, ] und < nicht in Merkmalsnamen (BPIC13: "Accepted/In Progress [1st]")
        X.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt") for c in X.columns]
        return X

    # ------------------------------------------------------------------ Training

    def samples(self, log):
        """Je Wahlpunkt die Proben des Logs: Liste von (Fall, Merkmale, Option)."""
        from interventions import option_at
        log = log.sort_values([CASE, TIMESTAMP], kind="stable")
        seconds = _seconds(log[TIMESTAMP])
        lifecycle = log[LIFECYCLE].to_numpy() if LIFECYCLE in log else np.full(len(log), "complete")
        resource = log[RESOURCE].to_numpy() if RESOURCE in log else np.full(len(log), None)
        activity, case = log[ACTIVITY].to_numpy(), log[CASE].to_numpy()
        values = log.reindex(columns=self.event_numeric).to_numpy(dtype=float)
        first_rows = log.groupby(CASE, sort=False).head(1).set_index(CASE)
        system = re.compile(self.system_users) if self.system_users else None
        control = {(point.activity, point.lifecycle): point for point in self.points.values() if point.kind == "control_flow"}
        employee = {(point.activity, point.lifecycle): point for point in self.points.values() if point.kind == "employee"}
        samples = {name: [] for name in self.points}
        bounds = np.flatnonzero(np.r_[True, case[1:] != case[:-1], True])
        for start, end in zip(bounds[:-1], bounds[1:]):
            prefix = Prefix(self._attributes(first_rows.loc[case[start]]), self.event_numeric)
            for i in range(start, end):
                point = employee.get((activity[i], lifecycle[i]))
                if point is not None and isinstance(resource[i], str) and not (system and system.search(resource[i])):
                    # Mitarbeiterwahl: der Fall vor dem Beginn, so wie die Engine fragt. An einem complete-Event (Engine ohne
                    # Lifecycles) fragt sie am vorigen Event: die Zeit bis hierher hängt schon von der Wahl ab
                    decided = seconds[i - 1] if point.lifecycle == "complete" and i > start else seconds[i]
                    row = prefix.row(decided)
                    if getattr(self, "option_rates", None):
                        row.update(self._rates([resource[i]], decided)[0])
                    samples[point.name].append((case[i], row, resource[i]))
                prefix.add(f"{activity[i]}|{lifecycle[i]}", seconds[i], dict(zip(self.event_numeric, values[i])))
                point = control.get((activity[i], lifecycle[i]))
                option = option_at(activity[i + 1], lifecycle[i + 1], point.options) if point is not None and i + 1 < end else None
                if option is not None:
                    # Control Flow: der Fall bis einschließlich des Entscheidungs-Events, Option = das folgende Event
                    samples[point.name].append((case[i], prefix.row(seconds[i]), option))
        return samples

    def fit(self, log):
        """Ein Modell je Wahlpunkt mit Proben; log: Event-Log mit Spalte outcome. Liefert self (summary: je Punkt Proben, Bäume, RMSE)."""
        from xgboost import XGBRegressor
        outcome = log.groupby(CASE)["outcome"].first()
        self.categories = {name: sorted(log[name].dropna().astype(str).unique()) for name in self.categorical}
        self.categories["last_token"] = sorted((log[ACTIVITY].astype(str) + "|" + log[LIFECYCLE].astype(str)).unique())
        self.summary = {}
        for name, rows in self.samples(log).items():
            point = self.points[name]
            if not rows:
                continue
            cases, features, options = zip(*rows)
            # Control Flow: die Optionen des Punkts; Mitarbeiter: jeder, der hier im Training vorkam
            self.options[name] = list(point.options) if point.kind == "control_flow" else sorted(set(options))
            self.columns[name] = sorted({column for row in features for column in row})
            X = self._frame(list(features), point, list(options))
            y = outcome.reindex(list(cases)).to_numpy(dtype=float)
            # Frühes Stoppen auf einem Teil der Fälle (alle Proben eines Falls auf derselben Seite)
            unique = sorted(set(cases))
            validation = set(np.array(unique, dtype=object)[np.random.default_rng(self.seed).random(len(unique)) < VALIDATION_SHARE])
            held = np.array([c in validation for c in cases])
            model = XGBRegressor(**PARAMS, random_state=self.seed)
            model.fit(X[~held], y[~held], eval_set=[(X[held], y[held])], verbose=False)
            self.models[name] = model
            rmse = float(np.sqrt(np.mean((model.predict(X[held]) - y[held]) ** 2)))
            self.summary[name] = {"Proben": len(y), "Fälle": len(set(cases)), "Optionen": len(self.options[name]),
                                  "Bäume": model.best_iteration + 1, "RMSE Validierung": rmse, "Streuung Outcome": float(np.std(y[held]))}
        return self

    # ------------------------------------------------------------------ Abfrage

    def knows(self, point):
        return point.name in self.models

    def predict(self, point, samples, options):
        """Geschätzter Outcome je Probe (samples: Merkmale aus samples bzw. Prefix.row) mit der Option je Probe."""
        return self.models[point.name].predict(self._frame(list(samples), point, list(options)))

    def values(self, point, events, options, now):
        """
        Geschätzter Outcome je Option (Array in der Reihenfolge von options). events: die Events des Falls bis jetzt
        (Liste von dicts wie engine.events, in Reihenfolge); now: Zeitpunkt der Wahl in Sekunden seit 1970.
        """
        prefix = Prefix(self._attributes(events[0]), self.event_numeric)
        times = _seconds([event[TIMESTAMP] for event in events])
        for event, seconds in zip(events, times):
            prefix.add(f"{event[ACTIVITY]}|{event.get(LIFECYCLE, 'complete')}", seconds, {name: event.get(name) for name in self.event_numeric})
        rows = [prefix.row(now)] * len(options)
        if getattr(self, "option_rates", None) and point.kind == "employee":
            rows = [{**row, **rates} for row, rates in zip(rows, self._rates(options, now))]
        return self.predict(point, rows, options)

    def set_threads(self, n):
        for model in self.models.values():
            model.set_params(n_jobs=n)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

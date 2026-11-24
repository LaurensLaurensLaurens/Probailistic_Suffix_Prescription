"""One event-cost definition for observed logs, PSP suffixes and online scores."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CASE, ACTIVITY, LIFECYCLE, TIMESTAMP = "case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp"
DEFAULT_CONFIG = Path(__file__).with_name("event_costs.yaml")
OPENING = {"start", "resume"}
CLOSING = {"suspend", "complete", "ate_abort", "withdraw"}
COMPONENTS = ["base_eur", "lifecycle_eur", "time_eur", "active_hours", "total_eur",
              "unclosed_segments", "orphan_closures", "reopened_segments", "unconfigured_events"]


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name}: expected a finite, nonnegative number")
    return float(value)


def seconds(value):
    """Absolute seconds; naive timestamps follow the project's UTC convention."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Missing event timestamp")
    return stamp.value / 1e9


class EventCosts:
    def __init__(self, config=None):
        if isinstance(config, EventCosts):
            config = config.to_dict()
        elif config is None or isinstance(config, (str, Path)):
            with Path(config or DEFAULT_CONFIG).open() as stream:
                config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError("Event-cost configuration must be a mapping")
        if set(config) - {"version", "currency", "unknown_activity", "defaults", "events"}:
            raise ValueError("Unknown event-cost configuration fields")
        if config.get("version", 1) != 1 or config.get("currency", "EUR") != "EUR":
            raise ValueError("Expected event-cost version 1 with currency EUR")
        unknown = config.get("unknown_activity", "default")
        if unknown not in ("default", "error"):
            raise ValueError("unknown_activity must be default or error")
        defaults = self._tariff(config.get("defaults", {}), {"base_eur": 0., "eur_per_work_hour": 0.,
                                                            "lifecycle_eur": {"default": 0.}})
        events = config.get("events", {})
        if not isinstance(events, dict):
            raise ValueError("events must map activity names to tariffs")
        normalized = {}
        for name, tariff in events.items():
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError("Activity names must be nonempty strings without surrounding spaces")
            normalized[name] = self._tariff(tariff, defaults)
        self._data = {"version": 1, "currency": "EUR", "unknown_activity": unknown,
                      "defaults": defaults, "events": normalized}
        self.fingerprint = hashlib.sha256(json.dumps(self._data, sort_keys=True, allow_nan=False).encode()).hexdigest()

    @staticmethod
    def _tariff(data, fallback):
        if not isinstance(data, dict) or set(data) - {"base_eur", "eur_per_work_hour", "lifecycle_eur"}:
            raise ValueError("Tariffs allow base_eur, eur_per_work_hour and lifecycle_eur")
        fees = data.get("lifecycle_eur", {})
        if not isinstance(fees, dict) or any(not isinstance(k, str) or not k or k != k.lower().strip() for k in fees):
            raise ValueError("lifecycle_eur must map lowercase lifecycle names to fees")
        return {"base_eur": _number(data.get("base_eur", fallback["base_eur"]), "base_eur"),
                "eur_per_work_hour": _number(data.get("eur_per_work_hour", fallback["eur_per_work_hour"]), "eur_per_work_hour"),
                "lifecycle_eur": {**fallback["lifecycle_eur"], **{k: _number(v, k) for k, v in fees.items()}}}

    def to_dict(self):
        return deepcopy(self._data)

    def tariff(self, activity):
        if activity not in self._data["events"] and self._data["unknown_activity"] == "error":
            raise ValueError(f"No event tariff for {activity!r}")
        return self._data["events"].get(activity, self._data["defaults"])

    def state(self, events=()):
        state = CostState(self)
        for event in events:
            if pd.isna(event[ACTIVITY]):
                raise ValueError("Missing event activity")
            lifecycle = event.get(LIFECYCLE)
            lifecycle = "complete" if lifecycle is None or pd.isna(lifecycle) else str(lifecycle).lower().strip()
            state.push(str(event[ACTIVITY]).strip(), lifecycle,
                       seconds(event[TIMESTAMP]))
        return state

    def event_costs(self, log):
        """Cost ledger in the original row order; active time is booked once at closing."""
        if not {CASE, ACTIVITY, TIMESTAMP} <= set(log.columns):
            raise ValueError("Event log needs case, activity and timestamp columns")
        frame = log[[CASE, ACTIVITY, TIMESTAMP]].copy()
        frame[LIFECYCLE] = log[LIFECYCLE].fillna("complete").to_numpy() if LIFECYCLE in log else "complete"
        frame["_row"] = np.arange(len(frame))
        frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP], utc=True, errors="raise")
        if frame[[CASE, ACTIVITY, TIMESTAMP]].isna().any().any():
            raise ValueError("Case, activity and timestamp must not be missing")
        frame = frame.sort_values([CASE, TIMESTAMP], kind="stable")
        frame["_seconds"] = frame[TIMESTAMP].dt.as_unit("ns").astype("int64") / 1e9
        values = np.zeros((len(frame), len(COMPONENTS)), dtype=float)
        for _, group in frame.groupby(CASE, sort=False, observed=True):
            state = CostState(self)
            opening_rows = {}
            for activity, lifecycle, at, row in group[[ACTIVITY, LIFECYCLE, "_seconds", "_row"]].itertuples(index=False, name=None):
                activity, lifecycle = str(activity).strip(), str(lifecycle).lower().strip()
                before = (state.base, state.lifecycle, state.time, state.hours, state.orphan_closures,
                          state.reopened_segments, state.unconfigured_events)
                state.push(activity, lifecycle, float(at))
                if lifecycle in OPENING:
                    opening_rows[activity] = row
                elif lifecycle in CLOSING:
                    opening_rows.pop(activity, None)
                values[row, :4] = [state.base - before[0], state.lifecycle - before[1],
                                   state.time - before[2], state.hours - before[3]]
                values[row, 6:] = [state.orphan_closures - before[4], state.reopened_segments - before[5],
                                  state.unconfigured_events - before[6]]
            # Incomplete intervals accrue to the observed horizon. This correction
            # belongs to that activity's opening row, never to an unrelated final
            # event. It adds no synthetic closing event or lifecycle fee.
            for activity, (hours, cost) in state.open_charges().items():
                row = opening_rows[activity]
                values[row, 2] += cost
                values[row, 3] += hours
                values[row, 5] += 1
        values[:, 4] = values[:, :3].sum(axis=1)
        return pd.DataFrame(values, index=log.index, columns=COMPONENTS)

    def case_costs(self, log):
        """Positive components per case. Open intervals accrue through the last observed event."""
        ledger = self.event_costs(log)
        ledger[CASE] = log[CASE].to_numpy()
        return ledger.groupby(CASE, sort=False, observed=True)[COMPONENTS].sum()

    def label_log(self, log):
        """Copy a log and replace its outcome with negative total event costs."""
        labelled = log.copy()
        ledger = self.event_costs(log)
        labelled["event_cost_eur"] = ledger["total_eur"].to_numpy()
        totals = labelled.groupby(CASE, sort=False, observed=True)["event_cost_eur"].sum()
        labelled["case_event_cost_eur"] = labelled[CASE].map(totals)
        labelled["outcome"] = -labelled["case_event_cost_eur"]
        return labelled


class CostState:
    """Active intervals are keyed by activity, independently of intervening events."""
    def __init__(self, costs):
        self.costs = costs
        self.opened = {}
        self.base = self.lifecycle = self.time = self.hours = 0.
        self.last = None
        self.orphan_closures = self.reopened_segments = self.unconfigured_events = 0

    def copy(self):
        state = CostState(self.costs)
        state.__dict__.update(self.__dict__)
        state.opened = self.opened.copy()
        return state

    def _close(self, activity, at):
        began = self.opened.pop(activity)
        hours = max(at - began, 0.) / 3600
        self.hours += hours
        self.time += hours * self.costs.tariff(activity)["eur_per_work_hour"]

    def push(self, activity, lifecycle, at):
        if not np.isfinite(at) or (self.last is not None and at < self.last - 1e-6):
            raise ValueError("Events must have finite, nondecreasing timestamps")
        tariff = self.costs.tariff(activity)
        self.base += tariff["base_eur"]
        self.lifecycle += tariff["lifecycle_eur"].get(lifecycle, tariff["lifecycle_eur"]["default"])
        self.unconfigured_events += int(activity not in self.costs._data["events"])
        if lifecycle in OPENING:
            if activity in self.opened:
                self.reopened_segments += 1
                self._close(activity, at)
            self.opened[activity] = at
        elif lifecycle in CLOSING:
            if activity in self.opened:
                self._close(activity, at)
            elif tariff["eur_per_work_hour"] > 0:
                self.orphan_closures += 1
        self.last = at

    def open_charges(self, at=None):
        at = self.last if at is None else at
        if at is None:
            at = 0.
        if not np.isfinite(at) or (self.last is not None and at < self.last - 1e-6):
            raise ValueError("Cost horizon cannot precede the last event")
        return {activity: (max(at - began, 0.) / 3600,
                           max(at - began, 0.) / 3600 * self.costs.tariff(activity)["eur_per_work_hour"])
                for activity, began in self.opened.items()}

    def summary(self, at=None):
        pending = self.open_charges(at)
        extra_hours = sum(hours for hours, _ in pending.values())
        extra_cost = sum(cost for _, cost in pending.values())
        return {"base_eur": self.base, "lifecycle_eur": self.lifecycle, "time_eur": self.time + extra_cost,
                "active_hours": self.hours + extra_hours, "total_eur": self.base + self.lifecycle + self.time + extra_cost,
                "unclosed_segments": len(self.opened), "orphan_closures": self.orphan_closures,
                "reopened_segments": self.reopened_segments, "unconfigured_events": self.unconfigured_events}

    def total(self, at=None):
        return self.summary(at)["total_eur"]

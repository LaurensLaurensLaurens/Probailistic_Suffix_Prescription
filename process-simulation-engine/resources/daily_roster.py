"""
Who is on duty when, read directly from the event log.

An employee is on duty on a day from the hour of their first event to the hour of their last event
of that day. Mined availability patterns that treat an employee as present in every hour they were ever
seen in put about 78 employees on duty on a weekday morning where 24 are active in the real log (BPIC17),
so simulated cases never have to wait for anybody.
"""
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import pandas as pd


class DailyRoster:
    def __init__(self, event_log: pd.DataFrame, system_users: Iterable[str] = ()):
        """system_users: arbeiten rund um die Uhr (resources/datasets.py)."""
        self.system_users = set(system_users)
        log = event_log[event_log["org:resource"].notna() & ~event_log["org:resource"].isin(self.system_users)]
        timestamps = pd.to_datetime(log["time:timestamp"])
        if timestamps.dt.tz is not None:
            timestamps = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
        hours = timestamps.dt.hour.groupby([timestamps.dt.date, log["org:resource"]]).agg(["min", "max"])
        self._shifts: Dict[object, Dict[str, Tuple[int, int]]] = {}
        for (date, resource), (first, last) in zip(hours.index, hours.values):
            self._shifts.setdefault(date, {})[resource] = (int(first), int(last))
        self.first_date, self.last_date = min(self._shifts), max(self._shifts)
        self.resources: List[str] = sorted(set(event_log["org:resource"].dropna()) | self.system_users)

    def _date(self, current_time: datetime):
        date = current_time.date()
        # Days outside the log count as the same weekday 52 weeks earlier or later
        while date > self.last_date:
            date -= timedelta(days=364)
        while date < self.first_date:
            date += timedelta(days=364)
        return date

    def is_available(self, resource_id: str, current_time: datetime) -> bool:
        if resource_id in self.system_users:
            return True  # the system works around the clock
        shift = self._shifts.get(self._date(current_time), {}).get(resource_id)
        return shift is not None and shift[0] <= current_time.hour <= shift[1]

    def on_duty(self, current_time: datetime) -> set:
        """Alle, die zu dieser Zeit im Dienst sind (wie is_available), ohne die Systemnutzer."""
        hour = current_time.hour
        return {resource for resource, (first, last) in self._shifts.get(self._date(current_time), {}).items() if first <= hour <= last}


class AlwaysOnDuty:
    """
    Alle Mitarbeiter des Logs sind rund um die Uhr im Dienst (die synthetische Werkstatt, simulation/repair_workshop.py). DailyRoster sähe dort
    einen Mechaniker nur an Tagen mit eigenem Start oder Ende im Dienst: seine Reparaturen dauern Tage, dazwischen hat er keine Events.
    """

    def __init__(self, event_log: pd.DataFrame, system_users: Iterable[str] = ()):
        self.system_users = set(system_users)
        self.resources: List[str] = sorted(set(event_log["org:resource"].dropna()) | self.system_users)

    def is_available(self, resource_id: str, current_time: datetime) -> bool:
        return True

    def on_duty(self, current_time: datetime) -> set:
        return set(self.resources) - self.system_users

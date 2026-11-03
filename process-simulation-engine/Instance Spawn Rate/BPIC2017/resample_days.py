"""
BPIC17: Ankünfte aus neu gezogenen echten Tagen (ARRIVALS "resample_days", ../case_arrivals.py).

Each simulated calendar day reuses the arrival times of one real day with the same weekday,
drawn from the days at most `weeks` weeks away from the same calendar date. This keeps the
level and trend over the year, the weekday pattern, the time-of-day pattern and the
day-to-day variation of the log, while different seeds still give different arrivals.

This replaces the earlier learned arrival model (global segments, weekday clusters and
interarrival KDEs). Against the full BPIC17 log that model produced 96 arrivals per day
instead of 86 and missed the time-of-day pattern; resampled days match both.

period beschränkt die echten Tage auf einen Zeitraum (ist er kürzer als ein Jahr, laufen die simulierten Tage in
Schritten von ganzen Wochen durch ihn statt 52 Wochen; ohne period wie bisher).
"""
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd


def generate_arrivals(
    df: pd.DataFrame,
    num_cases: int,
    start_date: datetime,
    seed: int = 42,
    weeks: int = 3,
    period=None,
) -> List[datetime]:
    """
    Generate num_cases arrival timestamps starting at start_date.

    Args:
        df: Event log with the columns "case:concept:name" and "time:timestamp".
        num_cases: Number of arrivals to generate.
        start_date: Start of the simulation. The first day is partial if it starts during the day.
        seed: Seed for the choice of real days.
        weeks: A simulated day draws its real day from at most this many weeks around the
            same calendar date. 0 replays the real days exactly.
        period: (erster, letzter) Tag, aus dem die echten Tage stammen; None: der ganze Log.

    Dates outside the period covered by the log are mapped into it in steps of 52 weeks,
    which keeps weekday and season. Timestamps are tz-naive UTC, like the log.
    """
    arrivals = pd.to_datetime(df.groupby("case:concept:name")["time:timestamp"].min())
    if arrivals.dt.tz is not None:
        arrivals = arrivals.dt.tz_convert("UTC").dt.tz_localize(None)
    if arrivals.empty:
        raise ValueError("Event log contains no cases; cannot resample arrivals.")
    if period is not None:
        first, last = pd.Timestamp(period[0]), pd.Timestamp(period[1])
        arrivals = arrivals[(arrivals >= first) & (arrivals < last + pd.Timedelta(days=1))]
    arrivals = arrivals.sort_values()
    log_start = arrivals.min().floor("D")
    n_log_days = (arrivals.max().floor("D") - log_start).days + 1
    # Umlauf außerhalb des Zeitraums: 52 Wochen (Wochentag und Jahreszeit bleiben), bei kürzerem Zeitraum ganze Wochen
    wrap = 364 if n_log_days >= 364 else max(7, (n_log_days // 7) * 7)
    # Per real day (index since log start): arrival times as offsets from midnight
    day_index = (arrivals.dt.floor("D") - log_start).dt.days
    offsets_per_day = {int(i): (group - group.dt.floor("D")).tolist() for i, group in arrivals.groupby(day_index)}

    rng = np.random.default_rng(seed)
    sim_start = pd.Timestamp(start_date)
    current_day = sim_start.floor("D")
    timestamps: List[datetime] = []
    while len(timestamps) < num_cases:
        i = (current_day - log_start).days
        while i >= n_log_days:
            i -= wrap
        while i < 0:
            i += wrap
        candidates = [i + o for o in range(-7 * weeks, 7 * weeks + 1, 7) if 0 <= i + o < n_log_days]
        source = int(rng.choice(candidates)) if candidates else i % n_log_days
        for offset in offsets_per_day.get(source, []):
            ts = current_day + offset
            if ts >= sim_start:
                timestamps.append(ts.floor("us").to_pydatetime())
        current_day += pd.Timedelta(days=1)
    return timestamps[:num_cases]

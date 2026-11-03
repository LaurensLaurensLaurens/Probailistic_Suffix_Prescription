"""
Beispielfall je Datensatz (scripts/main.py): fehlt example_cases/<Jahr>/beispielfall.html, wird ein zufälliger Fall des
vorverarbeiteten Logs abgelegt, immer derselbe (seed): alle Events und alle Spalten (ohne die in diesem Fall leeren) als
scrollbare Tabelle in beispielfall.html und als beispielfall.csv. Vorhandene bleiben unverändert.

    from helpers.example_case import example_case
    path = example_case("bpic13", log)
"""
import html
import os

import numpy as np

from .datasets import ACTIVITY, CASE, LIFECYCLE, PROJECT, RESOURCE, TIMESTAMP

FOLDER = os.path.join(PROJECT, "example_cases")
FIRST = [CASE, ACTIVITY, LIFECYCLE, TIMESTAMP, RESOURCE]  # die wichtigsten Spalten zuerst
# Scrollt in beide Richtungen; Kopfzeile und Nummer der Events bleiben stehen
PAGE = """<!doctype html><html lang="de"><head><meta charset="utf-8"><title>{title}</title><style>
:root {{ color-scheme: light dark; --surface: Canvas; --line: #8884; --head: #8882; }}
body {{ margin: 0; padding: 16px; height: 100vh; box-sizing: border-box; display: flex; flex-direction: column; background: var(--surface); font: 13px system-ui, sans-serif; }}
h1 {{ font-size: 15px; margin: 0 0 10px; }}
.scroll {{ flex: 1; overflow: auto; border: 1px solid var(--line); }}
table {{ border-collapse: separate; border-spacing: 0; }}
th, td {{ padding: 4px 10px; white-space: nowrap; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); text-align: left; background: var(--surface); }}
thead th {{ position: sticky; top: 0; z-index: 2; }}
tbody th {{ position: sticky; left: 0; z-index: 1; }}
thead th:first-child {{ left: 0; z-index: 3; }}
</style></head><body><h1>{title}</h1><div class="scroll">{table}</div></body></html>
"""


def example_case(dataset, log, folder=FOLDER, seed=0):
    """Pfad der Tabelle des Beispielfalls von dataset (bpic13 liegt in example_cases/2013); fehlt sie, wird der Fall aus log gezogen."""
    target = os.path.join(folder, "20" + dataset[-2:])
    path = os.path.join(target, "beispielfall.html")
    if os.path.exists(path):
        print(f"{dataset}: Beispielfall vorhanden ({path})", flush=True)
        return path
    cases = np.sort(log[CASE].unique())
    case = cases[np.random.default_rng(seed).integers(len(cases))]
    events = log[log[CASE] == case].sort_values(TIMESTAMP, kind="stable").dropna(axis=1, how="all")
    first = [column for column in FIRST if column in events]
    events = events[first + [column for column in events if column not in first]].reset_index(drop=True)
    os.makedirs(target, exist_ok=True)
    events.to_csv(os.path.join(target, "beispielfall.csv"), index=False)
    events.index = range(1, len(events) + 1)
    days = (events[TIMESTAMP].max() - events[TIMESTAMP].min()).total_seconds() / 86400
    title = html.escape(f"{dataset.upper()}, Fall {case}: {len(events)} Events, {events.shape[1]} Spalten, Durchlaufzeit {days:.1f} Tage")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PAGE.format(title=title, table=events.to_html(na_rep="", border=0)))
    print(f"{dataset}: Beispielfall {case} mit {len(events)} Events ({path})", flush=True)
    return path

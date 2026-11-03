"""
Die Simulation für Logs nur mit complete-Events (BPIC19), angetrieben vom Next-Event-Modell. Das Gegenstück zur Engine mit
Lifecycles (simulation/case_engine_lifecycle_aware.py), mit derselben Schnittstelle (build_engine, run, set_policy).

Ein Event hat hier keine Dauer: Niemand beginnt oder beendet etwas, also ist niemand belegt, es gibt keine Warteschlange, keinen
Dienstplan und keine Zuteilung, und jeder Fall läuft für sich. lauf.ankunft_faktor hat deshalb keine Wirkung (Last 1 ist Pflicht).
Die Ereignis-Warteschlange bleibt nur für die gemeinsame Uhr (Reihenfolge der Events, Beobachter wie das Lagebild).

Ist ein Event geschehen, zieht _advance je Fall in dieser Reihenfolge:
1. die nächste Aktivität (EventTimeModel.step), gegeben den Verlauf und wer bisher am Fall gearbeitet hat
2. wer sie ausführt (EventTimeModel.resource_logits), gegeben die Aktivität; nur wer sie im Training ausgeführt hat
3. die Zeit bis dahin (EventTimeModel.time_logits), gegeben Aktivität und Person, als Zeitklasse; innerhalb der Klasse
   bestimmt der Kalender der Aktivität den Zeitpunkt (_calendar_seconds)

Der Kalender ist nötig, weil die Zeitklassen bei langen Abständen breiter als ein Tag sind (um 30 Tage 1,6 Tage), der Prozess aber
Terminen folgt: 75 % der Clear Invoice fallen auf Donnerstag, Vendor creates invoice hat nur ein Datum. Das Ziehen in der Klasse ändert
nicht, wie wahrscheinlich sie ist.

Wahlpunkte (set_policy, interventions.ChoicePoint), die Policy wie in der Engine mit Lifecycles: policy.choose(point, case, options,
time, engine) gibt eine Option zurück oder None (das Gezogene):
- control_flow: direkt nach dem Entscheidungs-Event; zieht das Modell eine Option, wählt die Policy, welche. Person und Zeit
  werden danach zur gewählten Aktivität gezogen.
- employee: wurde für die Aktivität des Punkts eine Person gezogen (kein Systemnutzer), wählt die Policy unter allen, die die
  Aktivität im Training ausgeführt haben. Gefragt wird am vorigen Event, vor der Zeit, denn die hängt von der Person ab. Die Merkmale
  der Schätzer stehen deshalb auf demselben Zeitpunkt (estimators/xgb_model.py samples).
"""
import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from next_event_prediction.data_prep import BOS, END, sample_seconds
from next_event_prediction.event_time_model import EventTimeModel
from simulation.case_engine_lifecycle_aware import HOUR, raise_load, to_datetime, to_seconds

ARRIVAL, EVENT, OBSERVE = 0, 1, 2
THURSDAY = 3  # Wochentag des 1.1.1970, Tag 0 der Sekunden seit 1970


@dataclass
class Case:
    case_id: str
    attributes: Dict[str, object]
    attribute_state: object  # Zustand des Falls in den Fallattributen (die Zahlen an den Events)
    model_state: Dict[str, np.ndarray]
    rng: np.random.Generator  # je Fall eigene Zufallszahlen: der Verlauf hängt nicht von der Reihenfolge der Fälle ab
    last_token: int = BOS
    last_time: float = 0.0
    n_events: int = 0
    resource: Optional[str] = None  # wer das letzte Event ausgeführt hat
    pending: Optional[Tuple[int, str]] = None  # nächstes Event und wer es ausführt
    prefix: List[Dict] = field(default_factory=list)  # die Events bis jetzt, wie in engine.events; bekommen die Policies


class CompleteEventEngine:
    def __init__(self, model: EventTimeModel, allocator, attribute_generator, arrival_timestamps, calendar: np.ndarray, seed: int = 42,
                 min_probability: float = 0.0):
        """
        allocator: Resources (resources/datasets.py system_resources), hier nur die Systemnutzer. attribute_generator: CaseAttributes
        (case_attributes/datasets.py). arrival_timestamps: aufsteigend. calendar: je Token, Wochentag und Stunde, wie oft das Event
        dann im Log vorkommt (calendar_table). min_probability: was das Modell seltener gibt, wird nie gezogen.
        """
        self.model, self.attribute_generator = model, attribute_generator
        self.system_users = allocator.system_users
        self.arrival_timestamps, self.calendar = arrival_timestamps, calendar
        self.seed, self.min_probability = seed, min_probability
        self._tokens = [tuple(token.split("|")) for token in model.encoder.tokens]
        self.observer: Optional[Callable] = None
        self.observe_seconds: Optional[float] = None
        # Wer ein Event ausführen kann, ohne Systemnutzer: die Optionen eines Mitarbeiter-Wahlpunkts
        resources = model.encoder.resources
        self._performers = [[resources[r] for r in np.flatnonzero(row) if resources[r] not in self.system_users] for row in model.allowed_resources]
        self.set_policy(None, [])
        self._reset()

    def _reset(self) -> None:
        self.queue: List[tuple] = []
        self._sequence = itertools.count()
        self.cases: Dict[str, Case] = {}
        self.events: List[Dict] = []
        self.choices: List[Dict] = []
        self.truncated_cases = set()
        # Dieselben Größen wie die Engine mit Lifecycles: hier wartet niemand auf einen Mitarbeiter
        self.waiting_seconds: Dict[str, float] = {}
        self.waits: List[Dict] = []
        self.forced_assignments = 0

    def run(self, num_cases: int) -> List[Dict]:
        """Die ersten num_cases Ankünfte simulieren. Liefert das Event-Log als Liste von Zeilen."""
        self._reset()
        for number, timestamp in enumerate(self.arrival_timestamps[:num_cases]):
            self._schedule(to_seconds(timestamp), ARRIVAL, f"Case_{number}")
        if self.observer is not None and self.observe_seconds and num_cases:
            self._schedule(to_seconds(self.arrival_timestamps[0]) + self.observe_seconds, OBSERVE, None)
        while self.queue:
            time, _, kind, case_id = heapq.heappop(self.queue)
            if kind == ARRIVAL:
                self._arrive(case_id, time)
            elif kind == EVENT:
                self._happen(self.cases[case_id], time)
            else:
                self.observer(time, self)
                if self.queue:
                    self._schedule(time + self.observe_seconds, OBSERVE, None)
        self.events = self.attribute_generator.finish(self.events)
        return self.events

    def set_policy(self, policy, points: List, window: Optional[Tuple[float, float]] = None) -> None:
        """
        Die Policy und ihre Wahlpunkte (interventions.choice_points); gefragt wird nur für Fälle mit Ankunftsnummer in
        window [erste, letzte) (Standard alle).
        """
        self.policy, self.window = policy, window or (0, float("inf"))
        ids = self.model.encoder.token_to_id
        self._control_flow: Dict[int, Tuple[object, List[int]]] = {}  # Token des Entscheidungs-Events -> Punkt, Tokens der Optionen
        self._employee: Dict[int, object] = {}  # Token des Events -> Punkt
        for point in points:
            token = ids.get(point.activity + "|" + point.lifecycle)
            if point.kind == "control_flow":
                options = [ids.get(option if "|" in option else option + "|complete") for option in point.options]
                if token is None or None in options:
                    raise ValueError(f"Wahlpunkt {point.name}: Entscheidungs-Event oder Optionen kennt das Modell nicht")
                self._control_flow[token] = (point, options)
            elif point.kind == "employee":
                if token is None or point.lifecycle != "complete":
                    raise ValueError(f"Wahlpunkt {point.name}: kein complete-Event, das das Modell kennt")
                self._employee[token] = point
            else:
                raise ValueError(f"Wahlpunkt {point.name}: unbekannte Art {point.kind}")

    def _ask(self, point, case: Case, options: List[str], time: float) -> Optional[str]:
        """Die Wahl der Policy unter options, oder None (das Gezogene): nur für Fälle im Fenster window."""
        if self.policy is None or not self.window[0] <= int(case.case_id.rsplit("_", 1)[1]) < self.window[1]:
            return None
        choice = self.policy.choose(point, case, options, time, self)
        self.choices.append({"case": case.case_id, "point": point.name, "kind": point.kind, "choice": choice,
                             "index": None if choice is None else options.index(choice), "options": len(options), "time": time})
        return choice

    def _schedule(self, time: float, kind: int, case_id: Optional[str]) -> None:
        heapq.heappush(self.queue, (time, next(self._sequence), kind, case_id))

    def _arrive(self, case_id: str, time: float) -> None:
        number = int(case_id.rsplit("_", 1)[1])
        attributes, attribute_state = self.attribute_generator.new_case(case_id, number, to_datetime(time))
        categorical, numeric = self.model.encoder.encode_attributes(attributes, time)
        state = self.model.start(np.array([categorical]), np.array([numeric]), np.array([time]))
        case = Case(case_id, attributes, attribute_state, state, np.random.default_rng([self.seed, number]), last_time=time)
        self.cases[case_id] = case
        self._advance(case, time)

    def _advance(self, case: Case, now: float) -> None:
        """Nächstes Event eines Falls: Aktivität, Person, Zeit (siehe oben). Die Policy kann Aktivität und Person ersetzen."""
        encoder = self.model.encoder
        logits = self.model.step(case.model_state, np.array([case.last_token]), np.array([now - case.last_time]),
                                 resources=np.array([encoder.resource_id(case.resource)]))
        case.last_time = now
        token = self._control_flow_choice(case, int(self.model.draw(logits, case.rng, self.min_probability)[0]), now)
        if token == END or case.n_events >= self.model.max_length - 1:
            if token != END:
                self.truncated_cases.add(case.case_id)
            self.cases.pop(case.case_id, None)
            return
        drawn = self.model.draw(self.model.resource_logits(case.model_state, np.array([token])), case.rng, self.min_probability)[0]
        resource = self._employee_choice(case, token, encoder.resources[int(drawn)], now)
        logits = self.model.time_logits(case.model_state, np.array([token]), np.array([encoder.resource_id(resource)]))
        seconds = self._calendar_seconds(int(self.model.draw(logits, case.rng, self.min_probability)[0]), token, now, case.rng)
        case.pending = (token, resource)
        self._schedule(now + seconds, EVENT, case.case_id)

    def _control_flow_choice(self, case: Case, drawn: int, now: float) -> int:
        # Wahlpunkt nur, wenn das Modell nach dem Entscheidungs-Event eine der Optionen gezogen hat
        entry = self._control_flow.get(case.last_token)
        if entry is None or drawn not in entry[1]:
            return drawn
        chosen = self._ask(entry[0], case, list(entry[0].options), now)
        return drawn if chosen is None else entry[1][entry[0].options.index(chosen)]

    def _employee_choice(self, case: Case, token: int, drawn: str, now: float) -> str:
        # Ein Systemschritt bleibt einer. Ohne Prefix (das Event wäre das erste des Falls) kennt die Policy den Fall noch nicht
        point = self._employee.get(token)
        if point is None or not case.prefix or drawn in self.system_users or not self._performers[token]:
            return drawn
        chosen = self._ask(point, case, self._performers[token], now)
        return drawn if chosen is None else chosen

    def _calendar_seconds(self, time_bin: int, token: int, now: float, rng: np.random.Generator) -> float:
        """
        Sekunden bis zum Event aus der Zeitklasse: unter den Stunden der Klasse nach dem Kalender der Aktivität, gewichtet mit dem
        Anteil der Stunde, der in der Klasse liegt. Kürzere Klassen und Klassen, in denen das Event nie vorkommt: wie sonst auch.
        """
        edges = self.model.encoder.time_edges
        low, high = now + edges[time_bin], now + edges[time_bin + 1]
        if high - low >= HOUR:
            hours = np.arange(int(low // HOUR), int(high // HOUR) + 1)
            overlap = np.minimum((hours + 1) * HOUR, high) - np.maximum(hours * HOUR, low)
            weights = self.calendar[token, (hours // 24 + THURSDAY) % 7, hours % 24] * overlap
            if weights.sum() > 0:
                chosen = rng.choice(len(hours), p=weights / weights.sum())
                return float(max(hours[chosen] * HOUR, low) + rng.random() * overlap[chosen] - now)
        return float(sample_seconds(np.array([time_bin]), rng, edges)[0])

    def _happen(self, case: Case, time: float) -> None:
        token, resource = case.pending
        activity, lifecycle = self._tokens[token]
        case.resource = resource
        # Zahlen am Event: gelten weiter, wechseln je Aktivität wie im Log (case_attributes/datasets.py event_numeric)
        values = self.attribute_generator.event_attributes(case.attribute_state, activity, case.rng)
        event = {"case:concept:name": case.case_id, "concept:name": activity, "org:resource": resource, "time:timestamp": to_datetime(time),
                 "lifecycle:transition": lifecycle, **case.attributes, **values}
        self.events.append(event)
        case.prefix.append(event)
        case.last_token, case.n_events, case.pending = token, case.n_events + 1, None
        self._advance(case, time)


def calendar_table(event_log, encoder) -> np.ndarray:
    """Je Token, Wochentag und Stunde, wie oft das Event im Log vorkommt (für _calendar_seconds)."""
    tokens = (event_log["concept:name"].astype(str) + "|" + event_log["lifecycle:transition"].astype(str)).map(encoder.token_to_id)
    known = tokens.notna()
    moments = event_log.loc[known, "time:timestamp"]
    table = np.zeros((encoder.vocabulary_size, 7, 24))
    np.add.at(table, (tokens[known].astype(int).to_numpy(), moments.dt.dayofweek.to_numpy(), moments.dt.hour.to_numpy()), 1)
    return table


# Aufgerufen von Simulator.run. allocation und min_experience gibt es nur für die Schnittstelle: ohne Belegung teilt niemand zu
def build_engine(event_log, num_cases: int, seed: int = 42, model_folder: Optional[str] = None, allocation: str = "random",
                 min_experience: int = 0, dataset: str = "bpic19", load: float = 1.0) -> CompleteEventEngine:
    """Die Engine mit dem Modell, den Ankünften und Fallattributen des Datensatzes (aus dem Log nach dem Preprocessing)."""
    import os
    import sys

    from case_attributes import dataset_case_attributes
    from next_event_prediction.train import MIN_PROBABILITY
    from resources import system_resources

    if load != 1:
        raise ValueError(f"{dataset}: ohne Belegung hat die Ankunftslast keine Wirkung, lauf.ankunft_faktor muss 1 sein, nicht {load}")
    spawn_rate = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Instance Spawn Rate")
    if spawn_rate not in sys.path:
        sys.path.insert(0, spawn_rate)
    from case_arrivals import dataset_arrivals

    start = event_log["time:timestamp"].min().floor("D").to_pydatetime()
    model = EventTimeModel.load(model_folder)
    if not model.encoder.performers:
        raise ValueError(f"{dataset}: das Modell in {model_folder} zieht keine Mitarbeiter (next_event_prediction/datasets.py performers)")
    arrivals, sources = dataset_arrivals(dataset, event_log, num_cases, seed, start_date=start)
    arrivals, numbers = raise_load(arrivals, 1, seed)  # nach Zeit sortiert, numbers: welche Ankunft
    attributes = dataset_case_attributes(dataset, event_log, seed=seed)
    attributes.arrival_cases([sources[k] for k in numbers])
    return CompleteEventEngine(model, system_resources(dataset, event_log), attributes, arrivals, calendar_table(event_log, model.encoder),
                               seed=seed, min_probability=MIN_PROBABILITY)

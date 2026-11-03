"""
Discrete-event simulation of the loan application process, driven by the next-event model.

All cases run in one event queue and compete for the employees. For every case the model draws, at
the moment an event has happened, the next event and the time until it:

- Working time and pauses (mostly waiting for the customer) come from the model.
- Waiting for an employee comes from the simulation: when a work item is started or resumed, the
  case needs an employee who may do this activity, is on duty and is not working on another case.
  If there is none, the case waits until one becomes free or comes on duty. Who gets which task is
  decided by the allocation policy (resources/case_allocation.py).

An employee stays with a case from "start" or "resume" of a work item until its "suspend",
"complete", "ate_abort" or "withdraw", and performs the events of the case in between. That is how
the real log looks: 99 % of these closing events are done by the employee who started the session.

Choice points (set_policy, interventions.ChoicePoint): at them the engine asks the policy, policy.choose(point, case,
options, time, engine), every time a case reaches a point (_ask). The policy returns one of the options or None (as the bank).
- control_flow (in _advance): directly after the decision event of an intervention (rule next_event; BPIC17 channel:
  O_Created) the model draws the next event as always. If it is one of the options (O_Sent (mail and online) or O_Sent
  (online only)), the policy chooses which one happens; otherwise the case goes on as drawn.
- employee (in _assign): when the work item of the point (BPIC17: W_Call incomplete files, start) gets an employee,
  the allocation of the engine draws one (in the pipeline a selection strategy, lauf.zuteilung); the policy may replace
  it by another one who may do it, is on duty and is free; None: the drawn one.
Every choice is logged in engine.choices.

Usage (from process-simulation-engine/), simulates as many cases as the real log has:
    python -m simulation.case_engine_lifecycle_aware --output simulation/output/engine_log.csv
    python -m evaluation.evaluation --simulated simulation/output/engine_log.csv
"""
import heapq
import itertools
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from next_event_prediction.data_prep import BOS, END, sample_seconds
from next_event_prediction.event_time_model import EventTimeModel
from interventions import WAIT
from resources.case_allocation import Allocation, Task

OPENING = ("start", "resume")  # an employee takes the work item
CLOSING = ("suspend", "complete", "ate_abort", "withdraw")  # the employee lets go of it
ARRIVAL, EVENT, RETRY, OBSERVE = 0, 1, 2, 3
HOUR = 3600.0
# A case that has waited this long gets an employee who may do the activity even if none is on duty.
# Without it a case would wait forever once the simulation runs past the period the duty times cover.
MAX_WAIT = 14 * 24 * HOUR
# Why a task had to wait
ALL_BUSY, NOBODY_ON_DUTY, HELD_BACK = "alle beschäftigt", "niemand im Dienst", "zurückgestellt"


# Wird aufgerufen von _happen (Zeitstempel im Log, angehaltene Fälle) und _on_duty_at (Dienstplan)
def to_datetime(seconds: float) -> datetime:
    """Seconds since 1970 as a tz-naive UTC datetime, as in the event log."""
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)


# Wird aufgerufen von run (Ankunftszeiten)
def to_seconds(timestamp: datetime) -> float:
    return timestamp.replace(tzinfo=timezone.utc).timestamp()


@dataclass
class Case:
    case_id: str
    attributes: Dict[str, object]  # loan goal, application type, requested amount
    attribute_state: object  # state of the case in the attribute generator (for the offer attributes)
    model_state: Dict[str, np.ndarray]
    last_token: int = BOS
    last_time: float = 0.0  # time of the last event
    n_events: int = 0
    sessions: Dict[str, str] = field(default_factory=dict)  # open work item -> employee working on it
    resource: Optional[str] = None  # who performed the last event
    session_holder: Optional[str] = None  # employee of the work session the last event happened in (None outside sessions)
    offer_attributes: Dict[str, object] = field(default_factory=dict)
    pending: Optional[Tuple[int, float]] = None  # next event (token) and when it is due
    # Who worked on the case so far and who had each work item last (for the allocation "learned")
    handlers: Set[str] = field(default_factory=set)
    item_holders: Dict[str, str] = field(default_factory=dict)
    last_opener: Optional[str] = None  # who started or resumed a work item of the case last
    # Every case draws from its own random numbers. Its course then does not depend on the order in
    # which the cases happen to be processed, so two runs that differ only in the allocation of
    # employees simulate the same cases.
    rng: Optional[np.random.Generator] = None
    # Die Events des Falls bis jetzt, wie in engine.events (Spalten des Event-Logs); den Policies mitgegeben (_ask)
    prefix: List[Dict] = field(default_factory=list)

    def holder(self, activity: str) -> Optional[str]:
        """
        Holder of the work session an event of this activity happens in: the session of the work item itself, otherwise
        the one opened last, outside any session None (as data_prep.session_holders in training).
        """
        if activity in self.sessions:
            return self.sessions[activity]
        return next(reversed(self.sessions.values())) if self.sessions else None


# Simulator.run -> build_engine -> CaseEngine
class CaseEngine:
    # Wird aufgerufen von Simulator.run -> build_engine -> CaseEngine.__init__
    def __init__(self, model: EventTimeModel, allocator, attribute_generator, arrival_timestamps: List[datetime],
                 seed: int = 42, min_probability: float = 0.0, allocation: Optional[Allocation] = None,
                 experience: Optional[Dict[Tuple[str, str], int]] = None, min_experience: int = 0):
        """
        Args:
            model: Trained next-event model.
            allocator: Resources (resources/datasets.py, dataset_resources): its permissions say who may do an activity,
                its availability who is on duty, its system_users are no employees (system_user does the system steps).
            attribute_generator: CaseAttributes (case_attributes/datasets.py); draws the case attributes and the dynamic attributes.
            arrival_timestamps: When the cases arrive.
            min_probability: Events and times the model gives less than this are never drawn.
            allocation: Decides who works on which task (default: a random free employee).
            experience: How often each employee started or resumed each work item in the real log
                ((employee, activity) -> count, resources.case_allocation.work_counts).
            min_experience: An employee may do an activity only with at least this experience, among those
                the permission model allows (if that leaves nobody, all permitted ones).
        """
        self.model = model
        self.allocator = allocator
        # Systemnutzer (resources/datasets.py): nehmen keinem Mitarbeiter Arbeit ab; Systemschritte macht system_user
        self.system_users, self.system_user = allocator.system_users, allocator.system_user
        # Arbeitsschritte mit Mitarbeitern: Tätigkeiten mit diesem Präfix (next_event_prediction/datasets.py; BPIC17 W_)
        self.work_item_prefix = model.encoder.work_item_prefix
        self.attribute_generator = attribute_generator
        self.arrival_timestamps = sorted(arrival_timestamps)
        self.min_probability = min_probability
        # Beobachter über allen laufenden Fällen (z. B. das Lagebild des PSP, prescriptive/overview.py): observer(time, engine)
        # alle observe_seconds, ab der ersten Ankunft. Er liest nur (events, cases) und zieht keine Zufallszahlen der
        # Engine, der Verlauf der Fälle bleibt also derselbe wie ohne ihn.
        self.observer: Optional[Callable] = None
        self.observe_seconds: Optional[float] = None
        self.experience, self.min_experience = experience or {}, min_experience
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)  # the selection strategies draw from Python's random module
        self.allocation = allocation or Allocation()
        self.allocation.bind(self)
        self._tokens = [tuple(token.split("|")) for token in model.encoder.tokens]
        self._eligible: Dict[str, List[str]] = {}
        self._on_duty: Dict[int, Set[str]] = {}
        # Wahlpunkte und Policy (set_policy): ohne sie läuft jeder Fall wie bei der Bank
        self.set_policy(None, [])
        self._reset()

    # ------------------------------------------------------------------ running
    # _reset() setzt alles zurück, was zu einem einzelnen Simulationslauf gehört.
    # Wird aufgerufen von Simulator.run -> build_engine -> __init__ -> _reset
    # sowie vor jedem Lauf von run
    def _reset(self) -> None:
        self.queue: List[tuple] = []
        self._sequence = itertools.count()
        self.cases: Dict[str, Case] = {}
        self.busy: Dict[str, Tuple[str, str, float]] = {}  # employee -> (case, work item, since when)
        self.waiting: List[Task] = []  # tasks waiting for an employee, longest-waiting first
        self.held: Dict[Task, Set[str]] = {}  # Aufgaben, für die die Policy warten wollte: wer da frei war (interventions.WAIT)
        self.retry_scheduled = False
        self.allocation.reset()
        self.events: List[Dict] = []
        self.choices: List[Dict] = []  # jede Frage an die Policy (_ask)
        self.truncated_cases: Set[str] = set()
        self.waiting_seconds = {ALL_BUSY: 0.0, NOBODY_ON_DUTY: 0.0, HELD_BACK: 0.0}  # total waiting for an employee, by reason
        self.waits: List[Dict] = []  # each wait for an employee: case, work item, lifecycle, seconds, reason (_commit)
        self.forced_assignments = 0  # cases that got an employee off duty after MAX_WAIT

    # Wird aufgerufen von main.py -> generate_eval_engine -> start_cases -> run
    def run(self, num_cases: int) -> List[Dict]:
        """Simulate the first num_cases arrivals together. Returns the event log as a list of records."""
        # Alle cases laufen gleichzetg in einer Ereignis Warteschlange
        self._reset()
        # Alle Case arrivals auf den Heap pushen
        for number, timestamp in enumerate(self.arrival_timestamps[:num_cases]):
            self._schedule(to_seconds(timestamp), ARRIVAL, f"Application_{number}")
        if self.observer is not None and self.observe_seconds and num_cases:
            self._schedule(to_seconds(self.arrival_timestamps[0]) + self.observe_seconds, OBSERVE, None)
        self._process()
        # Was erst am Fallende feststeht (BPIC17: Selected, CreditScore und Accepted der Angebote)
        self.events = self.attribute_generator.finish(self.events)
        return self.events

    # Wird aufgerufen von simulation/dataset_simulator.py Simulator.run (je Lauf)
    def set_policy(self, policy, points: List, window: Optional[Tuple[float, float]] = None) -> None:
        """
        The policy (with choose(point, case, options, time, engine); case.prefix: the events of the case so far) and its
        choice points (interventions.choice_points).
        Asked only for cases whose arrival number lies in window [first, last) (default: all).
        """
        self.policy, self.window = policy, window or (0, float("inf"))
        ids = self.model.encoder.token_to_id
        self._control_flow: Dict[int, Tuple[object, List[int]]] = {}  # Token des Entscheidungs-Events -> Punkt, Tokens der Optionen
        self._employee: Dict[Tuple[str, str], object] = {}  # (Arbeitsschritt, Lifecycle) -> Punkt
        for point in points:
            if point.kind == "control_flow":
                after = ids.get(point.activity + "|" + point.lifecycle)
                options = [ids.get(option if "|" in option else option + "|complete") for option in point.options]
                if after is None or None in options:
                    raise ValueError(f"Wahlpunkt {point.name}: Entscheidungs-Event oder Optionen kennt das Modell nicht")
                self._control_flow[after] = (point, options)
            elif point.kind == "employee":
                if not self._is_work_item(point.activity) or point.lifecycle not in OPENING:
                    raise ValueError(f"Wahlpunkt {point.name}: kein Beginn eines Arbeitsschritts (start/resume), dort teilt die Engine niemanden zu")
                self._employee[(point.activity, point.lifecycle)] = point
            else:
                raise ValueError(f"Wahlpunkt {point.name}: unbekannte Art {point.kind}")

    # Wird aufgerufen von _advance (control_flow) und _assign (employee)
    def _ask(self, point, case: Case, options: List[str], time: float) -> Optional[str]:
        """
        Die Wahl der Policy unter options, oder None (wie die Bank). Gefragt wird bei jedem Erreichen eines Wahlpunkts, nur
        für Fälle im Fenster window (die anderen laufen mit, damit die Auslastung stimmt, und folgen der Bank).
        # MIt window soll fairness sichergestellt werden, weil am anfang und am Ende
        """
        if self.policy is None or not self.window[0] <= int(case.case_id.rsplit("_", 1)[1]) < self.window[1]:
            return None
        choice = self.policy.choose(point, case, options, time, self)
        self.choices.append({"case": case.case_id, "point": point.name, "kind": point.kind, "choice": choice,
                             "index": options.index(choice) if choice in options else None, "options": len(options), "time": time})
        return choice

    # Wird aufgerufen von run (Ankünfte), _advance (nächstes Event), _commit (Event nach Wartezeit) und _schedule_retry
    def _schedule(self, time: float, kind: int, case_id: Optional[str]) -> None:
        # Pusht das nächste event auf den heap
        # Kind ist arrrival oder Even etc. 
        # self._sequence st die Reihenfole bei gleicher Zeit
        heapq.heappush(self.queue, (time, next(self._sequence), kind, case_id))

    # Wird aufgerufen von main.py -> generate_eval_engine -> start_cases -> run -> _process
    # Gibt wie in der alten Engine das Event weiter an einen Handler je nach Casearrival oder normales event
    def _process(self) -> None:
        # Die Queue enthält die Events, die bearbeitet werden müssen
        # Am Anfang sind die case Arribals schon alle auf den Heap gepusht worden
        while self.queue:
            # Welches Event haben wir als nächtes?
            time, _, kind, case_id = heapq.heappop(self.queue)
            if kind == ARRIVAL:
                # Beim Arrival wird nur nachgeguckt, welche Art von Event das nächste Event ist und diesem zugeorndet, wann es ankommt 
                # Dann wird es gescheduled
                self._arrive(case_id, time)
            elif kind == EVENT:
                self._due(self.cases[case_id], time)
            elif kind == OBSERVE:
                self.observer(time, self)
                # Weiter beobachten, solange noch etwas geschieht
                if self.queue:
                    self._schedule(time + self.observe_seconds, OBSERVE, None)
            # Retry
            else:
                self.retry_scheduled = False
                # Be assign wird die Ressource gewählt
                self._assign(time)

    # ------------------------------------------------------------------ cases

    # Wird aufgerufen von main.py -> generate_eval_engine -> start_cases -> run -> _process -> _arrive (je Ankunft)
    # Also das handlung eines event, wenn es ein case arrival ist
    def _arrive(self, case_id: str, time: float) -> None:
        # Bei case arrival sollen erst de case attribute gezogen werden
        # Mit Ankunftszeit: die Fallattribute kommen aus echten Fällen derselben Jahreszeit
        # case nr
        number = int(case_id.rsplit("_", 1)[1])
        attributes, attribute_state = self.attribute_generator.new_case(case_id, number, to_datetime(time))
        # encoden
        categorical, numeric = self.model.encoder.encode_attributes(attributes, time)
        state = self.model.start(np.array([categorical]), np.array([numeric]), np.array([time]))
        # Speichert den gesamten case state. Auf die Dauer auch die case history etc.
        case = Case(case_id, attributes, attribute_state, state, last_time=time, rng=np.random.default_rng([self.seed, number]))
        self.cases[case_id] = case

        #"""Draw the next event of the case and the time until it."""
        self._advance(case, time)

    # Wird aufgerufen von run -> _process -> _arrive -> _advance (erstes Event eines Falls),
    # und von _process -> _due -> _happen -> _advance (jedes weitere Event)
    # Zu diesem Zeitpunkt wurde die Ressource bereits gewählt, wenn nicht von _arrive aufgerufen wird
    # HIER WIRD NUR GESCHEDULED, NICHT SCHON AUSGEFÜHRT
    def _advance(self, case: Case, now: float) -> None:
        """
        Draw the next event of the case (given the history, also who worked on it) and then the time until it, given the
        event and who performs it. At a control-flow choice point the policy may replace the drawn option (set_policy).
        """
        token = case.last_token
        # Das aktuelle Angebot des Falls (bei einem Modell mit Angeboten als Eingang): die Attribute des letzten O_Create Offer
        offers = np.array([self.model.encoder.encode_offer(case.offer_attributes)]) if self.model.use_offers else None
        # Wer das Event ausgeführt hat (Inhaber seiner Sitzung, wie im Training): Teil des Prefix
        logits = self.model.step(case.model_state, np.array([token]), np.array([now - case.last_time]),
                                 resources=np.array([self.model.encoder.resource_id(case.session_holder)]), offers=offers)
        case.last_time = now
        # Hier wird mittels der Logits die nächste Aktivität gezogen
        drawn = int(self.model.draw(logits, case.rng, self.min_probability)[0])
        # Hier wird durch die Policy eine Wahl getroffen
        # HIEr KANN NOCH KEIN MITARBITER GEWÄHLT WERDEN, WEIL UNTEN NUR DE ZEIT BIS ZU DIESEM EVENT GEZOGEN WIRD
        # DA SICH ABER TOKEN AUF AKTIVITÄTEN BEZIEHEN IST DIES NUR FÜR CONTROL FLOW
        # Wahlpunkt? Das letzte Event (token) ist das Entscheidungs-Event eines Punkts control_flow (point[0]) und das
        # Modell hat eine seiner Optionen gezogen (drawn in point[1]); sonst hat es keine wirkliche Wahl
        point_control_flow = self._control_flow.get(token)
        if point_control_flow is not None and drawn in point_control_flow[1]:
            chosen = self._ask(point_control_flow[0], case, list(point_control_flow[0].options), now)
            if chosen is not None:
                drawn = point_control_flow[1][point_control_flow[0].options.index(chosen)]
        token = drawn
        # Es gibt eine maximale Länge
        if token == END or case.n_events >= self.model.max_length - 1:
            # Wenn die Länge überstiegen worden ist ... künstliches kürzen
            if token != END:
                self.truncated_cases.add(case.case_id)
            # Resource freigeben, Case beenden
            self._finish(case, now)
            return

        # Hier wird die Zeit gezogen. WICHTIG: WIR ZIEHEN HIER NUR DIE ZEIT, BIS DAS EVENT ANKOMMT. Das ist reine Zeit bis zum mächsten Event. 
        # Das hat noch nicht mit Wartezeit zu tun für Bearbeitungsschritte und Ressourcen.
        # -> Das ist eine von drei Möglichkeiten
        # 1. "W_" ... Das nächste Event startet einen Arbeitsschritt bei W_. Die gezogene Zeit ist die Warteschlange, bis jemand ihn beginnt (im echten Log Median 12,9 h). Danach sucht _due einen Mitarbeiter, und der Fall wartet zusätzlich, bis einer frei und im Dienst ist (Konkurrenz in der Engine).
        #    (Bis Oktober 2026 war ein start sofort fällig und die gezogene Zeit wurde verworfen: Gewartet wurde nur durch Konkurrenz, im Median 1 Minute.)
        # 2. "W_ ... lifecycle:Resume" ... Das nächste Event setzt einen Arbeitsschritt fort. Die gezogene Zeit ist die Pause, meist Warten auf den Kunden. Dafür braucht es keinen Mitarbeiter. Ist die Pause vorbei, sucht _due einen Mitarbeiter. Ist keiner frei, kommt das Warten auf ihn obendrauf.
        # 3. (suspend, complete, A_…, O_…). Das nächste Event passiert innerhalb einer laufenden Bearbeitung. Die gezogene Zeit ist Bearbeitungszeit. Der Mitarbeiter ist dann schon zugeteilt, er steht in case.sessions, seit der Arbeitsschritt gestartet wurde.
        # Who performs the next event, if already known: the holder of its work session (as data_prep.session_holders).
        # Not for start/resume: the time before them is waiting, and the employee is assigned only then.
        next_activity, next_lifecycle = self._tokens[token]
        waiting = self._is_work_item(next_activity) and next_lifecycle in OPENING
        performer = None if waiting else case.holder(next_activity)  # Warten: noch niemand; Arbeiten: der Inhaber
        time_bin = self.model.draw(self.model.time_logits(case.model_state, np.array([token]), np.array([self.model.encoder.resource_id(performer)])),
                                   case.rng, self.min_probability)
        #Aus den Time bins, die das LSTM vorhersagt, wird zuföllig gezogen. Diese Zeit wir dnur verwendet, wenn wir im Fall 3 sind.
        # SECONDS IST HIER JE NACH EVENT ENTWEDER DIE BEARBEITUNGSZEIT ODER DIE ZEIT, DIE WIR AUF EINEN KUNDEN WARTEN
        # Es st also due Zeit bis zum nächsten Event
        # Wenn Start, dann Warteschlabge bis jemand Schritt aufnimmt
        # Wenn resume, dann Pause, meist warten auf den kunden
        # Wenn Suspend, complete, dann bearbeitungszeit des Mitarbeiters -> Inhaber der Sitzung als Mitarbeiter, weil der selbe Mitarbeiter, der die Sitzung öffnet se häufig auch schließt
        seconds = float(sample_seconds(time_bin, case.rng, self.model.encoder.time_edges)[0])
        # Auch vor einem start gilt die gezogene Zeit (Warteschlange wie im echten Log), danach wird ein Mitarbeiter gesucht
        due = now + seconds
        # Die Zeile merkt sich im Fall, welches Event als Nächstes kommt und wann es fällig ist.
        case.pending = (token, due)
        # Schedule für das Case handling
        self._schedule(due, EVENT, case.case_id)

    # Wird aufgerufen von run -> _process -> _due (wenn das nächste Event eines Falls fällig ist)
    def _due(self, case: Case, time: float) -> None:
        """The next event of the case is due: it happens now, or the case waits for an employee."""
        # Kommt noch aus arrival und enthält Daten
        # Due ist ein Zetpunk
        # token das event
        token, due = case.pending
        activity, lifecycle = self._tokens[token]
        # Die Zeile prüft in _due, ob das fällige Event einen Mitarbeiter braucht, der noch nicht zugeteilt ist
        # W_ ist ein Arbeitsschritt
        # lifecycle in OPENING, also start oder resume. Aber hier ist ein Mitarbeter gebunden
        # activity not in case.sessions: Für diesen Arbeitsschritt ist in diesem Fall noch kein Mitarbeiter eingetragen.
        # Das hei0t hier werden Bearbeitunhsschrtte behandelt, bei denen der Fall einen Arbeitsschritt starten oder fortsetzen will, jedoch noch niemand zugeteilt ist.
        if self._is_work_item(activity) and lifecycle in OPENING and activity not in case.sessions:
            # Kein Mitarbeter darf diese Aktivität ausführen, daher überneimmt das System
            if not self.eligible_for(activity):
                case.sessions[activity] = self.system_user  # nobody but the system does this activity
            else:
                # Task speichert die umstönd des offenen Tasks
                task = Task(case.case_id, activity, lifecycle, time, case.resource)
                # Task ist ein Arbeitsauftrag, der auf einen Mitarbeiter wartet. 
                self.waiting.append(task)
                # Fragt die Zuteilungsstrategie, ob ein Mitarbeiter sie übernimmt, sonst wird es später nochmal probiert.
                # Versucht also, eine Ressource zuzuordnen
                self._assign(time, new_task=task)
                # Wenn nach der Zuteilung noch immer in Warteschlange, dann soll der Grund warum aktualisiert werden
                if task in self.waiting:
                    on_duty = self.on_duty_eligible(activity, time)
                    task.reason = NOBODY_ON_DUTY if not on_duty else HELD_BACK if any(r not in self.busy for r in on_duty) else ALL_BUSY
                    return
        self._happen(case, token, time)

    # Wird aufgerufen von run -> _process -> _due -> _happen (Event findet statt und kommt in den Log)
    # Hier wird das Event wirklich in Echtzet durchgeführt
    # WENN ANDERE VARIABLEN GEWÄHLT WERDEN SOLLEN, dann hier
    def _happen(self, case: Case, token: int, time: float) -> None:
        activity, lifecycle = self._tokens[token]
        released = None
        # The model gets the employee of the session the event happens in, as in training (data_prep.session_holders)
        case.session_holder = case.holder(activity)
        # Gehört das Event zu einem Arbeitsschritt, an dem gerade ein Mitarbeiter sitzt?
        if activity in case.sessions:
            # case.sessions[activity] schlägt nach, wer gerade an diesem Arbeitsschritt sitzt. Zum Beispiel ergibt
            # case.resource = ... speichert diesen Namen im Fall. case.resource heißt: „Wer hat das letzte Event dieses Falls ausgeführt?“
            case.resource = case.sessions[activity]
            # Nur Mitarbeter releasen, wenn lifecycle durch ist
            if lifecycle in CLOSING:
                released = case.sessions.pop(activity)
        elif case.sessions:
            # Events while an employee works on the case are done by that employee
            # case.sessions: von _happen. Sie greift bei einem Event, das zu keinem offenen Arbeitsschritt gehört, etwa O_Accepted oder A_Pending, während am Fall gerade gearbeitet wird. 
            # Alle Mitarbeter, die gerade an einem Arbeitsschritt dieses Fall sitzen
            case.resource = next(reversed(case.sessions.values()))
        elif case.resource is None or (self._is_work_item(activity) and lifecycle == "withdraw"):
            case.resource = self.system_user

        # Case Offer Attribute ziehen. Die Angebotsattribute stehen wie im echten Log nur an diesem Event; was erst feststeht,
        # wenn der Verlauf bekannt ist, kommt in run am Fallende dazu (attribute_generator.finish)
        offer = {}
        if activity == self.model.encoder.dynamic_attributes_activity and lifecycle == "complete":
            case.offer_attributes, offer = self.attribute_generator.dynamic_attributes(case.attribute_state)

        # Geschene Event notieren
        event = {
            "case:concept:name": case.case_id, "concept:name": activity, "org:resource": case.resource,
            "time:timestamp": to_datetime(time), "lifecycle:transition": lifecycle, **case.attributes, **offer,
        }
        self.events.append(event)
        case.prefix.append(event)
        case.last_token, case.n_events, case.pending = token, case.n_events + 1, None
        if released is not None:
            self._release(released, time)
            
        # Isz dann wieder das Ziehen des nächsten Events
        self._advance(case, time)

    # Wird aufgerufen von ... -> _happen -> _advance -> _finish (Fallende: END gezogen oder abgeschnitten)
    def _finish(self, case: Case, time: float) -> None:
        for resource in case.sessions.values():
            self._release(resource, time)
        case.sessions.clear()
        self.cases.pop(case.case_id, None)

    # ------------------------------------------------------------------ employees

    def _is_work_item(self, activity: str) -> bool:
        return activity.startswith(self.work_item_prefix)

    def eligible_for(self, activity: str) -> List[str]:
        """Employees who may do the activity: permitted by the permission model and, with min_experience, experienced."""
        if activity not in self._eligible:
            permitted = [r for r in sorted(self.allocator.permissions.get_eligible_resources(activity)) if r not in self.system_users]
            experienced = [r for r in permitted if self.experience.get((r, activity), 0) >= self.min_experience]
            self._eligible[activity] = experienced or permitted
        return self._eligible[activity]

    def _on_duty_at(self, time: float) -> Set[str]:
        hour = int(time // HOUR)
        if hour not in self._on_duty:
            # The duty times are full hours
            moment = to_datetime(hour * HOUR)
            self._on_duty[hour] = {r for r in self.allocator.availability.resources
                                   if r not in self.system_users and self.allocator.availability.is_available(r, moment)}
        return self._on_duty[hour]

    # Können die Aktivität geundsätzlivh machen, sind aber u.U. gerade beschäftigt.
    def on_duty_eligible(self, activity: str, time: float) -> List[str]:
        """Employees who may do the activity and are on duty, busy or not."""
        on_duty = self._on_duty_at(time)
        return [r for r in self.eligible_for(activity) if r in on_duty]
    # Können die Aktivität machen und sind nicht beschäftigt.
    def free_candidates(self, activity: str, time: float) -> List[str]:
        """Employees who may do the activity, are on duty and are not working on a case."""
        return [r for r in self.on_duty_eligible(activity, time) if r not in self.busy]

    def free_on_duty(self, time: float) -> List[str]:
        return [r for r in self._on_duty_at(time) if r not in self.busy]

    # Wird aufgerufen von Policies, die über den Fall hinaus vorausschauen (prescriptive/policies/psp_lookahead_policy.py)
    def other_cases(self, case_id: str, activity: str) -> List[Dict]:
        """
        What a policy may know about the other running cases when choosing an employee for case_id at activity: per case
        its events so far (prefix, read only), whether it has the work item open right now (then it needs nobody new for
        it) and whether it is waiting for an employee for it right now.
        """
        waiting = {task.case_id for task in self.waiting if task.activity == activity}
        return [{"case": other.case_id, "prefix": other.prefix, "open": activity in other.sessions, "waiting": other.case_id in waiting}
                for other in self.cases.values() if other.case_id != case_id and other.prefix]

    # Wird aufgerufen von
    # - _process -> _due -> _assign(time, new_task=task): ein start/resume ist fällig und braucht einen Mitarbeiter
    # - _happen / _finish -> _release -> _assign(time, freed=resource): ein Mitarbeiter ist frei geworden
    # - _process (RETRY) -> _assign(time): neue volle Stunde, evtl. neue Mitarbeiter im Dienst (_schedule_retry)
    def _assign(self, time: float, new_task: Optional[Task] = None, freed: Optional[str] = None) -> None:
        """
        Who takes which waiting task: the allocation of the engine draws (when a task comes on and whom it gets; in the
        pipeline a selection strategy); at an employee choice point the policy may replace the drawn employee by another
        one who may do the work item and is free now. Commit the answers.
        """
        if not self.waiting:
            return
        for task in list(self.waiting):
            # since: seit wann die Aufgabe wartet, für die Wartezeit und „wer wartet am längsten“
            # Hier wird eine Ressource erzwungen, die eigentlich frei hat, damit wir nicht unendlich lange warten
            # ABER: Der Fall greift tatsächlich nie. Es kommt nie dazu.
            if time - task.since < MAX_WAIT:
                break  # die Liste ist nach Wartedauer sortiert
            anybody = [r for r in self.eligible_for(task.activity) if r not in self.busy]
            if anybody:
                self.forced_assignments += 1
                self._commit(task, anybody[self.rng.integers(len(anybody))], time, new_task)

        # HIER IST DIE RESOURCEN-ZUORDNUNGSLOGIK: die Zuteilung der Engine zieht (in der Pipeline eine Strategie aus
        # resources/selection_strategies.py, lauf.zuteilung im Katalog)
        # Erst Allocation strategx wie gewählt, wenn policy vorhanden, dann durch policy überschrieben
        for task, resource in self.allocation.assign(list(self.waiting), time, new_task=new_task, freed=freed):
            if task in self.waiting and resource not in self.busy:
                # Wahlpunkt employee: die Strategie hat resource gezogen; ist eine Policy gesetzt, darf sie unter allen, die den
                # Arbeitsschritt jetzt machen können und frei sind, einen anderen wählen (wie beim Control Flow das gezogene Event).
                # Wählt sie einen, den die Strategie einem weiteren wartenden Task geben wollte, wartet dieser auf die nächste Zuteilung.
                point = self._employee.get((task.activity, task.opening)) if self.policy is not None else None
                if point is not None:
                    free = [r for r in self.free_candidates(task.activity, time) if freed in (None, r)]
                    # Wollte die Policy hier warten, fragt die Engine erst wieder, wenn ein Mitarbeiter frei wird (freed) oder zur
                    # vollen Stunde jemand frei ist, der es damals nicht war (Dienstbeginn, _schedule_retry); nicht bei neuen Aufgaben
                    if task in self.held and freed is None and set(free) <= self.held[task]:
                        continue
                    chosen = self._ask(point, self.cases[task.case_id], free, time)
                    if chosen == WAIT:
                        # Die Policy lässt den Fall warten (interventions.WAIT): er bleibt in der Warteschlange
                        self.held[task] = set(free)
                        continue
                    self.held.pop(task, None)
                    resource = chosen if chosen is not None else resource
                # Sofortiges zuordnen nach Strategie, wenn eine Resource frei wird
                # Bei Comitt wird die Resource auch auf Busy gestellt
                self._commit(task, resource, time, new_task)
        if self.waiting:
            self._schedule_retry(time)

    def _commit(self, task: Task, resource: str, time: float, new_task: Optional[Task]) -> None:
        case = self.cases[task.case_id]
        # Aus Warteschlange entfernen
        self.waiting.remove(task)
        self.held.pop(task, None)
        # Resource busy stellen
        self.busy[resource] = (task.case_id, task.activity, time)
        # Resource zuordnen
        case.sessions[task.activity] = resource
        # Wer am Fall gearbeitet hat und wer welchen Arbeitsschritt zuletzt hatte (für die Zuteilung "learned")
        case.handlers.add(resource)
        case.item_holders[task.activity] = resource
        case.last_opener = resource
        # Für self.strategy.notify_assignment(resource, activity)
        self.allocation.committed(resource, task.activity)
        if task is not new_task:
            # The task waited. Its event goes through the queue, so that it does not run inside another case's event.#
            # Aufzeichne, wie lange der Task schon wartet
            self.waiting_seconds[task.reason or ALL_BUSY] += time - task.since
            self.waits.append({"case": task.case_id, "activity": task.activity, "lifecycle": task.opening, "seconds": time - task.since,
                               "reason": task.reason or ALL_BUSY})
            # Die Zeile steht in _commit und läuft, wenn eine Aufgabe nach dem Warten endlich einen Mitarbeiter bekommt. Sie verschiebt den Fälligkeitszeitpunkt des anstehenden Events auf jetzt. Welches Event es ist, bleibt gleich.
            case.pending = (case.pending[0], time)
            # Das Event soll dann sofort ausgeführt werden,nachdem ein Mitarbeiter zugerondet worden ust
            self._schedule(time, EVENT, task.case_id)

    def _release(self, resource: str, time: float) -> None:
        if self.busy.pop(resource, None) is not None:
            self.allocation.released(resource)
            self._assign(time, freed=resource)

    # Der punkt ist, dass eben nicht nur, wenn eine Resource aus einem Finished Task kommen, neue Resourcen available werden, 
    # sondern auch, wenn eine neue Stunde anbricht und Mitarbeiter erscheinen
    def _schedule_retry(self, time: float) -> None:
        """Look again at the next full hour, when other employees come on duty."""
        if not self.retry_scheduled:
            self.retry_scheduled = True
            self._schedule((time // HOUR + 1) * HOUR, RETRY, None)


DEFAULT_MODEL_FOLDER = "models/next_event/bpic17"

# Aufgerufen von Simulator.run -> build_engine
# The permission model lets 133 to 143 employees do each main work item, although few of them do most of it.
# Employees who started or resumed an activity fewer times than this in the real log may not do it.
MIN_EXPERIENCE = 30


# Die Ankünfte nach Zeit sortiert und je Ankunft ihre Nummer in arrivals. Mehr Last: factor Ankünfte statt einer, die Kopien zur
# selben Tageszeit, um bis zu 30 Minuten versetzt (run nimmt die ersten num_cases); weniger Last (factor < 1): jede Ankunft bleibt
# mit Wahrscheinlichkeit factor
# Wird aufgerufen von build_engine
def raise_load(arrivals, factor: float, seed: int):
    rng = np.random.default_rng(seed)
    if factor < 1:
        pairs = [(time, k) for k, time in enumerate(arrivals) if rng.random() < factor]
    else:
        pairs = list(zip(arrivals, range(len(arrivals))))
    for copy in range(int(np.ceil(factor)) - 1):
        share = min(factor - 1 - copy, 1.0)
        pairs += [(time + timedelta(minutes=float(rng.uniform(-30, 30))), k) for k, time in enumerate(arrivals) if rng.random() < share]
    pairs.sort(key=lambda pair: pair[0])
    return [time for time, _ in pairs], [k for _, k in pairs]


def build_engine(event_log, num_cases: int, seed: int = 42, model_folder: str = DEFAULT_MODEL_FOLDER,
                 allocation: str = "bank", min_experience: int = MIN_EXPERIENCE,
                 dataset: str = "bpic17", load: float = 1.0) -> CaseEngine:
    """
    Engine with everything it needs: the trained next-event model, the arrivals, the case attribute generator, the
    permissions and the duty roster of the dataset (each from the log after the preprocessing). The simulation starts
    on the first day of the log, so that calendar and roster match it. load: Ankunftslast (raise_load), 1 wie im Log.
    """
    import os
    import sys

    from case_attributes import dataset_case_attributes
    from next_event_prediction.train import MIN_PROBABILITY
    from resources import dataset_resources
    from resources.case_allocation import create_allocation, work_counts

    # Absolut, damit es aus jedem Arbeitsverzeichnis geht (Notebook in prescriptive/, Arbeitsprozesse)
    spawn_rate = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Instance Spawn Rate")
    if spawn_rate not in sys.path:
        sys.path.insert(0, spawn_rate)
    from case_arrivals import dataset_arrivals

    # Mitternacht des ersten Tages im echten Log iszt der Starttag
    start = event_log["time:timestamp"].min().tz_localize(None).floor("D").to_pydatetime()

    model = EventTimeModel.load(model_folder)
    work_items = model.encoder.work_item_prefix
    # Bei weniger Last entsprechend mehr ziehen, damit nach dem Ausdünnen num_cases bleiben
    arrivals, sources = dataset_arrivals(dataset, event_log, num_cases if load >= 1 else int(np.ceil(1.05 * num_cases / load)), seed, start_date=start)
    arrivals, numbers = raise_load(arrivals, load, seed)
    attributes = dataset_case_attributes(dataset, event_log, seed=seed)
    attributes.arrival_cases(None if sources is None else [sources[k] for k in numbers])
    # Baut die Case Engine Logik mit pausieren, fortsetzen etc.
    return CaseEngine(
        model=model,
        # Mitarbeiter je Datensatz (resources/datasets.py): Berechtigungen und Dienstplan
        allocator=dataset_resources(dataset, event_log),
        # Fallattribute je Datensatz (case_attributes/datasets.py); BPIC17: der Generator
        attribute_generator=attributes,
        # Ankünfte je Datensatz (Instance Spawn Rate/case_arrivals.py); BPIC17: echte Tage neu gezogen, ab dem ersten Tag
        arrival_timestamps=arrivals,
        seed=seed,
        min_probability=MIN_PROBABILITY,
        allocation=create_allocation(allocation, event_log, dataset),
        experience=work_counts(event_log, work_items),
        min_experience=min_experience,
    )


if __name__ == "__main__":
    import argparse
    import os
    import time

    import pandas as pd

    from next_event_prediction.train import prepared_log

    parser = argparse.ArgumentParser()
    parser.add_argument("--event-log", default="eventlog/eventlog.xes.gz")
    parser.add_argument("--model", default=DEFAULT_MODEL_FOLDER, help="Folder of the trained next-event model")
    parser.add_argument("--cases", type=int, default=None, help="Number of cases (default: as many as the real log)")
    parser.add_argument("--output", default=os.path.join("simulation", "output", "engine_log.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allocation", choices=["bank", "learned", "random", "round_robin", "shortest_queue", "batch", "drl", "pmsp"], default="bank",
                        help="Who works on which task (see resources/case_allocation.py)")
    args = parser.parse_args()

    # Der Log nach dem Preprocessing (prescriptive/helpers/preprocessing), wie für das Modell
    real_log = prepared_log("bpic17", args.event_log)
    n_cases = args.cases or real_log["case:concept:name"].nunique()
    engine = build_engine(real_log, n_cases, args.seed, args.model, allocation=args.allocation)
    started = time.time()
    log = pd.DataFrame(engine.run(n_cases))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    log.to_csv(args.output, index=False)
    print(f"{log['case:concept:name'].nunique()} cases, {len(log)} events in {time.time() - started:.0f} s, written to {args.output}")
    waiting = ", ".join(f"{reason} {seconds / 3600 / n_cases:.2f}" for reason, seconds in engine.waiting_seconds.items())
    print(f"Waiting for an employee, hours per case: {waiting}; cut off: {len(engine.truncated_cases)} cases")

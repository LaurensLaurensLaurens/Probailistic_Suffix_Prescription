"""
Turns the event log into token sequences with case attributes, split by time. The same for every dataset: which case
attributes and event attributes (offers) are inputs, and whether there are work items with employees, says datasets.py (NEXT_EVENT).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .datasets import model_spec

PAD, BOS, END = 0, 1, 2
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<END>"]
SEPARATOR = "|"

# Who performed an event: every employee has its own index, this one stands for no employee yet or one not seen in training
UNKNOWN_RESOURCE = "<UNKNOWN>"


# Time between two events is modelled in bins: bin 0 is "less than a second", the others are
# log-spaced between one second and max_gap_days (datasets.py; at 180 days each about 5 % longer than the one before).
# The bins have to be this fine: in BPIC17 cancellations come 30.5 to 30.9 days after the last offer, and
# with bins that span 25 to 35 days the model cannot place them.
N_TIME_BINS = 321


def time_bin_edges(max_gap_days: float) -> np.ndarray:
    return np.concatenate([[0.0], np.logspace(0, np.log10(max_gap_days * 86400), N_TIME_BINS)])

@dataclass
class Trace:
    case_id: str
    events: List[Tuple[str, str]]  # (activity, lifecycle) in log order
    attributes: Dict[str, object]
    gaps: List[float] = None  # seconds since the previous event (0 for the first event)
    start: float = 0.0  # time of the first event, in seconds since 1970 (UTC)
    resources: Optional[List[str]] = None  # who performed each event
    offers: Optional[List[Optional[Tuple[float, ...]]]] = None  # offer attributes (datasets.py) at each offer event, else None


def offer_states(events: List[Tuple[str, str]], offers: Optional[List[Optional[Tuple[float, ...]]]], dynamic_attributes_activity: Optional[str]) -> List[Optional[Tuple[float, ...]]]:
    """Per event the current offer of the case after the event (the last dynamic_attributes_activity up to it), None before the first."""
    current, states = None, []
    for (activity, lifecycle), offer in zip(events, offers or [None] * len(events)):
        if activity == dynamic_attributes_activity and lifecycle == "complete" and offer is not None:
            current = offer
        states.append(current)
    return states


def session_holders(events: List[Tuple[str, str]], resources: List[Optional[str]], prefix: Optional[str]) -> List[Optional[str]]:
    """
    Per event the employee of the work session it happens in, as the simulation engine knows it: for a work
    item (activity starting with prefix, datasets.py work_item_prefix; BPIC17 W_...) with an open session its
    holder (the session opens with start/resume and closes with
    suspend/complete/ate_abort/withdraw); otherwise, while some session of the case is open, the holder of
    the one opened last; outside any session None. Who performed events outside sessions is not used: the
    engine cannot know it. Without prefix there are no sessions: always None.
    """
    if not prefix:
        return [None] * len(events)
    opened: Dict[str, Optional[str]] = {}
    holders = []
    for (activity, lifecycle), resource in zip(events, resources):
        if activity.startswith(prefix) and lifecycle in ("start", "resume"):
            opened.pop(activity, None)
            opened[activity] = resource
        holder = opened.get(activity) if activity in opened else (next(reversed(opened.values())) if opened else None)
        holders.append(holder)
        if activity.startswith(prefix) and lifecycle in ("suspend", "complete", "ate_abort", "withdraw"):
            opened.pop(activity, None)
    return holders


def time_bin(seconds, edges: np.ndarray) -> np.ndarray:
    """Index of the time bin for a duration in seconds (edges: Encoder.time_edges)."""
    return np.clip(np.searchsorted(edges, seconds, side="right") - 1, 0, N_TIME_BINS - 1)


def sample_seconds(bins: np.ndarray, rng: np.random.Generator, edges: np.ndarray) -> np.ndarray:
    """A duration from each bin: uniform in the first bin, log-uniform in the others (edges: Encoder.time_edges)."""
    low, high = edges[bins], edges[bins + 1]
    u = rng.random(len(bins))
    return np.where(bins == 0, u * high, np.exp(np.log(np.maximum(low, 1.0)) + u * (np.log(high) - np.log(np.maximum(low, 1.0)))))

# Trace(
#   case_id    = "Application_1000334415",
#   attributes = {"case:LoanGoal": "Other, see explanation",
#                 "case:ApplicationType": "New credit",
#                 "case:RequestedAmount": 5000.0},
#   start      = 1473957557.758,          # erstes Event in Sekunden seit 1970 → 15.09.2016, 16:39 UTC

#   # je Event: (Aktivität, Lifecycle) – Lücke zum Event davor (gaps, hier in h) – Mitarbeiter (resources) – Angebot (offers)
#   events / gaps / resources / offers =
#      0  ("A_Create Application",   "complete")    0.00 h   User_1     None
#      1  ("A_Submitted",            "complete")    0.00 h   User_1     None
#      2  ("W_Handle leads",         "schedule")    0.00 h   User_1     None
#      3  ("W_Handle leads",         "withdraw")    0.01 h   User_1     None
#      4  ("W_Complete application", "schedule")    0.00 h   User_1     None
#      5  ("A_Concept",              "complete")    0.00 h   User_1     None
#      6  ("W_Complete application", "start")      17.61 h   User_132   None      ← Warteschlange bis jemand beginnt
#      7  ("W_Complete application", "suspend")     0.04 h   User_132   None
#      8  ("W_Complete application", "resume")      1.09 h   User_91    None      ← anderer Mitarbeiter macht weiter
#      9  ("A_Accepted",             "complete")    0.13 h   User_91    None
#     10  ("O_Create Offer",         "complete")    0.06 h   User_91    (5000.0, 36.0, 152.82, 5000.0)
#     11  ("O_Created",              "complete")    0.00 h   User_91    None
#     12  ("O_Sent (mail and online)","complete")   0.00 h   User_91    None
#     13  ("W_Complete application", "complete")    0.00 h   User_91    None
#     …   (insgesamt 40 Events)
#     37  ("A_Denied",               "complete")  167.51 h   User_68    None      ← eine Woche später abgelehnt
#     38  ("O_Refused",              "complete")    0.00 h   User_68    None
#     39  ("W_Validate application", "ate_abort")   0.00 h   User_68    None
# )
# load_traces
def load_traces(df: pd.DataFrame, dataset: str) -> List[Trace]:
    """
    One trace per case, ordered by case start. Events keep their order in the log. The case attributes and the offer
    attributes are those of the dataset (datasets.py).
    """
    spec = model_spec(dataset)
    categorical, numeric, offer = spec["static_categorical"], spec["static_numeric"], spec["dynamic_numeric"]
    df = df.sort_values(["case:concept:name", "time:timestamp"], kind="stable")
    case_start = df.groupby("case:concept:name")["time:timestamp"].min().sort_values(kind="stable")
    activities = df.groupby("case:concept:name")["concept:name"].agg(list)
    lifecycles = df.groupby("case:concept:name")["lifecycle:transition"].agg(list)
    attributes = df.groupby("case:concept:name")[list(categorical) + list(numeric)].first()
    seconds = df["time:timestamp"].astype("int64") / (1e9 if str(df["time:timestamp"].dtype).startswith("datetime64[ns") else 1e6 if "[us" in str(df["time:timestamp"].dtype) else 1e3)
    gaps = seconds.groupby(df["case:concept:name"]).diff().fillna(0.0).groupby(df["case:concept:name"]).agg(list)
    starts = seconds.groupby(df["case:concept:name"]).min()
    resources = df.groupby("case:concept:name")["org:resource"].agg(list) if "org:resource" in df.columns else None
    offers = None
    if offer and all(column in df.columns for column in offer["attributes"]):
        # Angebotsattribute nur an den Angebots-Events (complete)
        values = df[offer["attributes"]].apply(pd.to_numeric, errors="coerce")
        is_offer = (df["concept:name"] == offer["activity"]) & (df["lifecycle:transition"] == "complete")
        rows = pd.Series([tuple(row) if offer else None for row, offer in zip(values.to_numpy(), is_offer)], index=df.index)
        offers = rows.groupby(df["case:concept:name"]).agg(list)
    return [
        Trace(case_id, list(zip(activities[case_id], lifecycles[case_id])), attributes.loc[case_id].to_dict(), gaps[case_id], float(starts[case_id]),
              resources[case_id] if resources is not None else None, offers[case_id] if offers is not None else None)
        for case_id in case_start.index
    ]


def calendar_features(seconds: np.ndarray) -> np.ndarray:
    """Hour of the day, weekday (Monday = 0) and day of the month (first = 0) of times given in seconds since 1970."""
    times = np.asarray(seconds, dtype="float64").astype("int64").astype("datetime64[s]")
    days = times.astype("datetime64[D]")
    hour = (times - days).astype("int64") // 3600
    weekday = (days.astype("int64") + 3) % 7
    day_of_month = (days - times.astype("datetime64[M]").astype("datetime64[D]")).astype("int64")
    return np.stack([hour, weekday, day_of_month], axis=-1)


def month_number(seconds) -> np.ndarray:
    """Months since 1970, to tell in which month a case started."""
    return np.asarray(seconds, dtype="float64").astype("int64").astype("datetime64[s]").astype("datetime64[M]").astype("int64")


def split_traces(traces: List[Trace], train_share: float = 0.8, validation_share: float = 0.1):
    """Split by time: the first cases train, the last ones test. Validation is the end of the training part."""
    n_train_all = int(len(traces) * train_share)
    n_validation = int(n_train_all * validation_share)
    return traces[: n_train_all - n_validation], traces[n_train_all - n_validation: n_train_all], traces[n_train_all:]


class Encoder:
    """Maps events to token ids and case attributes to model inputs. Fitted on the training traces."""

    def __init__(self, traces: List[Trace], dataset: str, month_traces: Optional[List[Trace]] = None):
        """
        traces: fix the vocabulary and the attribute values.
        dataset: welche statischen und dynamischen Attribute Eingang sind (datasets.py). Ohne numerisches Attribut ist der
        numerische Eingang immer 0.
        month_traces: the cases whose start months the model will be trained on (default: traces).
        """
        spec = model_spec(dataset)
        self.dataset = dataset
        if spec.get("dynamic_categorical"):
            raise NotImplementedError(f"{dataset}: dynamic_categorical ist noch nicht als Eingang des Modells umgesetzt (datasets.py)")
        self.categorical_attributes, self.numeric_attributes = list(spec["static_categorical"]), list(spec["static_numeric"])
        # dynamic_numeric (datasets.py): an welchem Event die Werte gesetzt werden und welche (BPIC17: das Angebot)
        self.dynamic_attributes_activity = spec["dynamic_numeric"]["activity"] if spec["dynamic_numeric"] else None
        self.dynamic_attributes = list(spec["dynamic_numeric"]["attributes"]) if spec["dynamic_numeric"] else []
        # Warten oder Arbeiten: Die Zeit vor einem start oder resume eines Arbeitsschritts ist Warten auf einen Mitarbeiter, keine Bearbeitungszeit 
        self.work_item_prefix = spec["work_item_prefix"]
        # Nur complete-Events: der Mitarbeiter jedes Events ist Eingang, das Modell zieht ihn selbst (datasets.py performers)
        self.performers = bool(spec.get("performers"))
        self.max_gap_days = spec["max_gap_days"]
        tokens = sorted({activity + SEPARATOR + lifecycle for trace in traces for activity, lifecycle in trace.events})
        # Aktvtät und Lifecycle als Tokens
        self.tokens = SPECIAL_TOKENS + tokens
        # The process changes over the year, so the month in which a case starts is an attribute.
        # Months outside the training period count as its first or last month.
        months = month_number([trace.start for trace in (month_traces or traces)])
        self.first_month, self.n_months = int(months.min()), int(months.max() - months.min() + 1)
        self._index_tokens()
        # Index 0 of each categorical attribute is reserved for values not seen in training
        # Das ist Behandlung von Missing values
        self.categories = {
            name: ["<UNKNOWN>"] + sorted({str(trace.attributes[name]) for trace in traces}) for name in self.categorical_attributes
        }
        # Das ist eine Transformation
        if self.numeric_attributes:
            amounts = np.log1p([float(trace.attributes[name]) for trace in traces for name in self.numeric_attributes])
            self.numeric_mean, self.numeric_std = float(amounts.mean()), float(amounts.std() + 1e-8)
        else:
            self.numeric_mean, self.numeric_std = 0.0, 1.0
        # Every employee of the traces gets its own index; index 0 is unknown (no employee yet, or not seen in training)
        self.resources = [UNKNOWN_RESOURCE] + sorted({r for trace in traces if trace.resources for r in trace.resources if isinstance(r, str)})
        self._index_resources()
        # Angebotsattribute: log1p, standardisiert mit Mittel und Streuung aller Angebote der Trainingsfälle
        values = np.log1p(np.clip(np.array([offer for trace in traces if trace.offers for offer in trace.offers if offer is not None], dtype=float), 0, None))
        if len(values):
            self.offer_mean, self.offer_std = np.nanmean(values, axis=0).tolist(), (np.nanstd(values, axis=0) + 1e-8).tolist()
        else:
            self.offer_mean, self.offer_std = [0.0] * len(self.dynamic_attributes), [1.0] * len(self.dynamic_attributes)

    @property
    def time_edges(self) -> np.ndarray:
        """Grenzen der Zeitklassen (time_bin, sample_seconds)."""
        if getattr(self, "_edges", None) is None or self._edges_days != self.max_gap_days:
            self._edges, self._edges_days = time_bin_edges(self.max_gap_days), self.max_gap_days
        return self._edges

    @property
    def n_offer_features(self) -> int:
        return len(self.dynamic_attributes) + 1  # dazu: gibt es schon ein Angebot

    def encode_offer(self, offer) -> np.ndarray:
        """
        Modelleingang für das aktuelle Angebot: 1 und die standardisierten log1p-Werte von dynamic_attributes, ohne Angebot
        nur Nullen. offer: Tupel in der Reihenfolge von dynamic_attributes, dict mit diesen Spalten oder None; fehlende Werte 0.
        """
        if offer is None or (isinstance(offer, dict) and not offer):
            return np.zeros(self.n_offer_features)
        values = np.array([offer.get(name, np.nan) for name in self.dynamic_attributes] if isinstance(offer, dict) else offer, dtype=float)
        encoded = (np.log1p(np.clip(values, 0, None)) - np.array(self.offer_mean)) / np.array(self.offer_std)
        return np.concatenate([[1.0], np.nan_to_num(encoded)])

    def _index_resources(self) -> None:
        self.resource_to_id = {resource: i for i, resource in enumerate(self.resources)}

    def resource_id(self, resource: Optional[str]) -> int:
        return self.resource_to_id.get(resource, 0) if isinstance(resource, str) else 0

    def _index_tokens(self) -> None:
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.activities = sorted({token.split(SEPARATOR)[0] for token in self.tokens[len(SPECIAL_TOKENS):]})
        # Activity index of every token (-1 for the special tokens)
        self.token_activity = np.array([
            self.activities.index(token.split(SEPARATOR)[0]) if i >= len(SPECIAL_TOKENS) else -1 for i, token in enumerate(self.tokens)
        ])

    @property
    def vocabulary_size(self) -> int:
        return len(self.tokens)

    @property
    def category_sizes(self) -> List[int]:
        return [len(values) for values in self.categories.values()] + [self.n_months]

    def encode_events(self, events: List[Tuple[str, str]]) -> List[int]:
        """Token ids of the events. Events not seen in training are skipped."""
        ids = (self.token_to_id.get(activity + SEPARATOR + lifecycle) for activity, lifecycle in events)
        return [i for i in ids if i is not None]

    def is_start_of_work(self, token_id: int) -> bool:
        """True for the event in which somebody picks up a work item. The time before it is waiting for a resource."""
        return bool(self.work_item_prefix) and self.tokens[token_id].startswith(self.work_item_prefix) and self.tokens[token_id].endswith(SEPARATOR + "start")

    def decode_events(self, ids: List[int]) -> List[Tuple[str, str]]:
        return [tuple(self.tokens[i].split(SEPARATOR)) for i in ids if i > END]

    def encode_attributes(self, attributes: Dict[str, object], start: Optional[float] = None) -> Tuple[List[int], List[float]]:
        """Model inputs for the case attributes. With start (seconds since 1970) the month of the case start is appended."""
        categorical = [
            self.categories[name].index(str(attributes[name])) if str(attributes[name]) in self.categories[name] else 0
            for name in self.categories
        ]
        if start is not None:
            categorical.append(int(np.clip(month_number(start) - self.first_month, 0, self.n_months - 1)))
        numeric = [(np.log1p(float(attributes[name])) - self.numeric_mean) / self.numeric_std for name in self.numeric_attributes]
        return categorical, numeric or [0.0]

"""
Fallattribute der simulierten Fälle je Datensatz: wie der Simulator die Attribute eines neuen Falls bildet.
dataset_case_attributes ist der Einstieg für alle Simulatoren (die Engine simulation/case_engine.py und
simulation/dataset_simulator.py). Welche Spalten es sind, sagt next_event_prediction/datasets.py
(categorical, numeric: die Fallattribute, die das Next-Event-Modell als Eingang hat).

Die Engine kennt nur die Schnittstelle CaseAttributes (new_case, dynamic_attributes, event_attributes, finish, arrival_cases),
welche Spalten ein Datensatz hat, steht in CASE_ATTRIBUTES.

Methoden:
    generator          BPIC2017/: AttributeSimulationEngine (setup_case_attributes): eigene Modelle für die Fallattribute (nach
                       Jahreszeit) und die Angebote, hinter GeneratedCaseAttributes
    from_arrival_case  die Attribute des echten Falls, von dem die Ankunft stammt (Instance Spawn Rate/case_arrivals.py,
                       real_starts): Attribute und Startdatum bleiben zusammen
    from_random_case   die Attribute eines zufälligen echten Falls (mit Zurücklegen)
Dazu event_numeric: Zahlen an jedem Event, die nur selten wechseln (RealCaseAttributes.event_attributes). Sie beginnen mit dem
Wert des echten Falls und gelten weiter. Nur wenn sie im echten Fall wechseln, wechseln sie auch im simulierten: die Wechsel
ballen sich in wenigen Fällen. Dann je Aktivität so oft wie in solchen Fällen im Log, um ein Verhältnis aus dem Log.
"""
import numpy as np

CASE = "case:concept:name"

# Wie der Simulator die Fallattribute eines Datensatzes bildet
CASE_ATTRIBUTES = {
    # LoanGoal, ApplicationType, RequestedAmount und die Angebote (OfferedAmount, NumberOfTerms, MonthlyCost,
    # FirstWithdrawalAmount, CreditScore, Selected, Accepted) aus gelernten Verteilungen und Modellen.
    # attributes und dynamic: Spalte im Log -> Feld im Zustand des Generators, bei der Ankunft bzw. am Event
    # dynamic_numeric.activity (next_event_prediction/datasets.py); open: Spalten, die im Log erst am Fallende feststehen
    "bpic17": {"method": "generator",
               "attributes": {"case:LoanGoal": "loan_goal", "case:ApplicationType": "application_type", "case:RequestedAmount": "requested_amount"},
               "dynamic": {"CreditScore": "credit_score", "OfferedAmount": "offered_amount", "FirstWithdrawalAmount": "first_withdrawal_amount",
                           "NumberOfTerms": "number_of_terms", "MonthlyCost": "monthly_cost", "Selected": "selected", "Accepted": "accepted"},
               "open": {"CreditScore": float("nan"), "Selected": None, "Accepted": None}},
    # impact, product, org line: wie der Verlauf hängen sie im Ausschnitt am Startdatum, deshalb vom selben echten Fall
    "bpic13": {"method": "from_arrival_case"},
    # Ausgabenbereich, Positionstyp, Art der Rechnungsprüfung, Dokumenttyp, Firma, Lieferant: vom selben echten Fall.
    # Der Wert wechselt nur in 1,8 % der Fälle, vor allem an Record Service Entry Sheet und Record Goods Receipt
    "bpic19": {"method": "from_arrival_case", "event_numeric": ["Cumulative net worth (EUR)"]},
    # Gerätetyp: unabhängig vom Ankunftstag, also von einem zufälligen Fall
    "repair": {"method": "from_random_case"},
}


def attribute_spec(dataset):
    if dataset not in CASE_ATTRIBUTES:
        raise ValueError(f"Keine Fallattribute für '{dataset}' festgelegt, möglich: {', '.join(CASE_ATTRIBUTES)}")
    return CASE_ATTRIBUTES[dataset]


def dataset_case_attributes(dataset, log, seed=42):
    """generator: GeneratedCaseAttributes; sonst RealCaseAttributes mit den Fallattributen des Next-Event-Modells."""
    spec = attribute_spec(dataset)
    if spec["method"] == "generator":
        from .BPIC2017 import setup_case_attributes
        return GeneratedCaseAttributes(setup_case_attributes(log, seed=seed), spec)
    if spec["method"] in ("from_arrival_case", "from_random_case"):
        from next_event_prediction.datasets import model_spec
        model = model_spec(dataset)
        return RealCaseAttributes(log, model["static_categorical"] + model["static_numeric"], spec["method"], seed, spec.get("event_numeric", ()))
    raise ValueError(f"{dataset}: unbekannte Methode {spec['method']}, möglich: generator, from_arrival_case, from_random_case")


class CaseAttributes:
    """Was die Engine von den Fallattributen eines Datensatzes braucht. Ohne dynamische Attribute reicht new_case."""

    # Wird aufgerufen von build_engine: je Ankunftsnummer der echte Fall, von dem die Ankunft stammt (None: keiner)
    def arrival_cases(self, cases):
        pass

    # Wird aufgerufen von CaseEngine._arrive
    def new_case(self, case_id, number, time):
        """Die Attribute des neuen Falls (Spalte -> Wert) und sein Zustand, den die anderen Methoden zurückbekommen."""
        raise NotImplementedError

    # Wird aufgerufen von CaseEngine._happen, am complete-Event von dynamic_numeric.activity (next_event_prediction/datasets.py)
    def dynamic_attributes(self, state):
        """Die Werte für das Modell und die Spalten für dieses Event im Log."""
        return {}, {}

    # Wird aufgerufen von der Engine ohne Lifecycles, an jedem Event (simulation/case_engine_lifecycle_unaware.py)
    def event_attributes(self, state, activity, rng):
        """Die Spalten dieses Events im Log; ändert state."""
        return {}

    # Wird aufgerufen von CaseEngine.run, am Ende des Laufs
    def finish(self, events):
        """Die Events, ergänzt um Werte, die erst am Fallende feststehen."""
        return events


class GeneratedCaseAttributes(CaseAttributes):
    """Attribute aus einem Generator (BPIC2017/), die Spalten aus CASE_ATTRIBUTES."""

    def __init__(self, generator, spec):
        self.generator, self.spec = generator, spec

    def new_case(self, case_id, number, time):
        state = self.generator.start_new_case(case_id, time)
        return {column: getattr(state, field) for column, field in self.spec["attributes"].items()}, state

    def dynamic_attributes(self, state):
        self.generator.populate_offer_attributes(state)
        values = {column: getattr(state, field) for column, field in self.spec["dynamic"].items()}
        return values, {**values, **self.spec["open"]}

    def finish(self, events):
        return self.generator.final_offer_attributes(events)


class RealCaseAttributes(CaseAttributes):
    """Fallattribute echter Fälle (je Spalte der erste vorhandene Wert im Fall)."""

    def __init__(self, log, columns, method, seed=42, event_numeric=()):
        self.columns, self.method, self.event_numeric = list(columns), method, list(event_numeric)
        self.table = log.groupby(CASE)[self.columns].first()
        self.rng = np.random.default_rng(seed)
        self.sources = None
        self.first_values = log.groupby(CASE)[self.event_numeric].first() if self.event_numeric else None
        self.varying = log.groupby(CASE)[self.event_numeric].nunique().gt(1) if self.event_numeric else None
        self.changes = {column: self._changes(log[log[CASE].isin(self.varying.index[self.varying[column]])], column) for column in self.event_numeric}

    # Je Aktivität, wie oft der Wert in den Fällen des Logs an ihr wechselt, und die Verhältnisse neu / alt dieser Wechsel
    @staticmethod
    def _changes(log, column):
        ordered = log.sort_values([CASE, "time:timestamp"], kind="stable")
        before = ordered.groupby(CASE)[column].shift()
        known = before.notna() & ordered[column].notna()
        ratio = (ordered[column] / before.where(before != 0))[known]
        changed = ratio.notna() & (ratio != 1)
        activity = ordered.loc[known, "concept:name"].astype(str)
        return {name: (float(changed[part.index].mean()), ratio[part.index][changed[part.index]].to_numpy())
                for name, part in activity.groupby(activity) if changed[part.index].any()}

    def arrival_cases(self, cases):
        self.sources = cases

    def new_case(self, case_id, number, time):
        source = self.sources[number] if self.method == "from_arrival_case" else self.table.index[self.rng.integers(len(self.table))]
        if not self.event_numeric:
            return self.table.loc[source].to_dict(), None
        return self.table.loc[source].to_dict(), {"values": self.first_values.loc[source].to_dict(), "varying": self.varying.loc[source].to_dict()}

    def event_attributes(self, state, activity, rng):
        if not self.event_numeric:
            return {}
        for column in self.event_numeric:
            share, ratios = self.changes[column].get(activity, (0.0, None))
            if state["varying"][column] and rng.random() < share:
                state["values"][column] *= ratios[rng.integers(len(ratios))]
        return dict(state["values"])

    def draw(self, n_cases, source_cases=None):
        """Je neuem Fall ein dict der Attribute: from_arrival_case die der source_cases, sonst zufällige echte Fälle."""
        if self.method == "from_arrival_case":
            if source_cases is None:
                raise ValueError("from_arrival_case braucht die echten Fälle der Ankünfte (case_arrivals.real_starts)")
            rows = self.table.loc[list(source_cases)]
        else:
            rows = self.table.iloc[self.rng.integers(len(self.table), size=n_cases)]
        return rows.to_dict("records")

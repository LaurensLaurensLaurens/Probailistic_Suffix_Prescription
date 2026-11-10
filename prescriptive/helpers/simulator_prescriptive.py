"""
Prescriptive-KPIs des Simulators: bildet er die Aspekte des echten Logs ab, auf die es für die Ansätze (XGBoost, LSTM,
PSP, Vorausschau, ERM) und ihre Online-Auswertung ankommt? Ergänzt die allgemeinen KPIs (evaluate_simulator,
process-simulation-engine/evaluation/evaluation.py) um Kennzahlen je Wahlpunkt des Laufs (simulator.points).

Drei Logs mit denselben Ankünften: echt (beobachtet), Simulator wie die Bank (beobachtet, soll die Lage des echten Logs
treffen) und Simulator mit Zufall an allen Wahlpunkten (policies/rct_policy.py wie die Policy random der
Online-Auswertung: die Wirkung der Optionen im Simulator, wie in den RCT-Daten, aus denen die Modelle lernen). Die Proben
an den Wahlpunkten wie beim Training (estimators/xgb_model.py samples): je Erreichen des Punkts der Fall, seine Merkmale
(Prefix) und die Option; das Outcome ist das des Falls (Spalte outcome, bei der Durchlaufzeit −Tage).

A Outcome (worauf die Policies optimiert und bewertet werden): Mittel, Median, schlechteste 10 % (CVaR), Abstand der
  Verteilungen (Wasserstein in Std des echten Outcomes); Durchlaufzeit Mittel, P90, P95 in Tagen wie in der Online-Auswertung.
B Lage an den Control-Flow-Wahlpunkten (echt gegen Bank): Fälle mit Wahl, Wahlen je Fall (Schleifen), Zeitpunkt der Wahl,
  Unterscheidbarkeit der Lage (AUC eines Klassifikators echt gegen Bank auf den Merkmalen, kreuzvalidiert; 0,5: gleich),
  Anteil je Option, Wahlen außerhalb des Trägers und Kontrast im Outcome je Option gegen die häufigste (beobachtet). Trifft
  die Bank hier den echten Log, gibt der Simulator wieder, was ein Modell aus dem echten Log lernen würde.
C Träger und Wirkung an den Control-Flow-Wahlpunkten (Zufall): ein Propensity-Modell auf dem echten Log (Option aus den
  Merkmalen, XGBoost) sagt, ob eine Option in einer Lage im echten Log vorkommt (Propensity ≥ EPS: im Träger).
  C1 Zufallswahlen einer Option außerhalb des Trägers: so viel der RCT-Daten und der Online-Wahlen ist Extrapolation.
  C2 Wirkung der Option im Simulator (randomisiert, gegen die häufigste Option) im Träger und außerhalb.
  C3 Dieselbe Wirkung im Träger auf dem echten Log (IPW mit der kreuzgefitteten Propensity, Hajek).
  Weicht C2 außerhalb des Trägers stark von C2 im Träger ab, nimmt der Simulator dort eine Wirkung an, die der echte Log
  nicht stützt (eine Policy kann sie ausnutzen, ohne dass es sie gibt). Weicht C2 im Träger von C3 ab, widerspricht er dem
  echten Log auch dort (unter der Annahme, dass die Merkmale die Konfundierung erfassen).
D Mitarbeiter-Wahlpunkte: Wahlen je Fall, Wartezeit vor dem Start (Median, Zeit seit dem letzten Event), Unterscheidbarkeit
  der Lage und der Mitarbeitereffekt (_employee_effects): die Restlaufzeit ab dem Start (bei anderen Outcomes das Outcome),
  um die Lage bereinigt (Vorhersage aus den Merkmalen ohne den Mitarbeiter), gemittelt je Mitarbeiter. Erst die
  Reliabilität (Split-Half): ergeben zwei zufällige Hälften der Fälle dieselbe Rangfolge der Mitarbeiter? Ist sie im
  echten Log klein, gibt es dort keinen verlässlichen Mitarbeitereffekt, und der Simulator wird nicht bewertet. Sonst die
  Streuung zwischen den Mitarbeitern (um das Rauschen bereinigt; Zufall: die Wirkung der Mitarbeiterwahl im Simulator) und
  die Korrelation der Effekte mit echt, um das Rauschen korrigiert (roh geteilt durch die Wurzel der beiden
  Reliabilitäten). Echt ist der Effekt beobachtet: was die Merkmale nicht erfassen (z. B. eine Spezialisierung auf
  schwierige Fälle), steckt mit darin.

Bewertung je Zeile: ok, prüfen, kritisch nach den Schwellen unten (Kontraste und Wirkungen nie schlechter als ok, solange
die Abweichung innerhalb von 2 Standardfehlern liegt); "–": nur zur Information. Die Übersicht hat je Bereich die
schlechteste Bewertung und die kritischen Kennzahlen.

    from helpers.simulator_prescriptive import prescriptive_kpis
    table, details = prescriptive_kpis(simulator, log, n_cases=10000, seed=0)
"""
from collections import Counter

import numpy as np
import pandas as pd

from .datasets import CASE, TIMESTAMP, add_paths
from .parameters import N_CASES, SEED

EPS = 0.05  # Propensity im echten Log, ab der eine Option in einer Lage vorkommt (Träger)
MIN_EFFECT = 30  # Proben je Mitarbeiter für seinen Effekt
MIN_HALF = 10  # Proben je Mitarbeiter in jeder Hälfte (Split-Half-Reliabilität)
SPLITS = 20  # zufällige Hälften der Fälle für die Reliabilität
MIN_RELIABILITY = 0.3  # darunter ist der Mitarbeitereffekt im echten Log nicht verlässlich: keine Bewertung des Simulators
MIN_OPTION = 20  # Proben einer Option im echten Log für das Propensity-Modell (seltener: kein Träger)
MAX_DISCRIMINATOR = 20000  # Proben je Log für die Unterscheidbarkeit
THREADS = 4  # Threads von XGBoost
RANDOM_SEED = 7  # Zufall der Policy (wie random in evaluation/online_evaluation.py: RCTPolicy([seed, 7]))

# Schwellen der Bewertung: ok bis zur ersten, prüfen bis zur zweiten, darüber kritisch
SHARE_PP = (5.0, 10.0)  # Anteile: Abstand in Prozentpunkten
RELATIVE = (10.0, 25.0)  # Mittel, Quantile, Zeiten: Abweichung in % des echten Werts
EFFECT_SD = (0.10, 0.25)  # Kontraste und Wirkungen: Abstand in Std des echten Outcomes am Wahlpunkt
DISTANCE_SD = (0.10, 0.25)  # Wasserstein-Abstand in Std des echten Outcomes
AUC = (0.60, 0.75)  # Unterscheidbarkeit der Lage
FACTOR = (1.5, 2.5)  # Streuung zwischen Mitarbeitern: Faktor zwischen Simulator und echt (in beide Richtungen)
CORRELATION = (0.7, 0.4)  # Korrelation der Mittel je Mitarbeiter: ok ab, prüfen ab

REAL, BANK, RANDOM = "echt", "Simulator (Bank)", "Simulator (Zufall)"
GRADES = ("–", "ok", "prüfen", "kritisch")


def simulate_logs(simulator, n_cases=N_CASES, seed=SEED, costs=None):
    """Bank und Zufall an allen Wahlpunkten, mit denselben Ankünften (seed), vorbereitet wie der echte Log (mit outcome)."""
    from preprocessing import chosen_outcome_spec
    from prescriptive.policies.rct_policy import RCTPolicy
    from .simulator_check import prepare_simulated
    spec = chosen_outcome_spec(simulator.dataset)
    print(f"Prescriptive-KPIs: {n_cases:,} Fälle wie die Bank ...", flush=True)
    bank = prepare_simulated(simulator.run(n_cases, seed=seed)[0], spec, costs)
    print(f"Prescriptive-KPIs: {n_cases:,} Fälle mit Zufall an allen Wahlpunkten ...", flush=True)
    randomized = prepare_simulated(simulator.run(n_cases, policy=RCTPolicy([seed, RANDOM_SEED]), seed=seed)[0], spec, costs)
    return bank, randomized


def prescriptive_kpis(simulator, log, n_cases=N_CASES, seed=SEED, costs=None, logs=None):
    """
    Die Kennzahlen (Zeilen: Bereich, Kennzahl; Spalten: echt, Simulator (Bank), Simulator (Zufall), Abweichung, Abweichung
    als, Bewertung) und Details: Übersicht (je Bereich), zeitpunkt (Outcome je Option nach Zeitpunkt der Wahl), mitarbeiter
    (Mittel je Mitarbeiter), unterscheidbar (Merkmale, an denen sich die Lage echt und Bank am meisten unterscheidet).
    log: der echte Log nach dem Preprocessing (mit outcome); logs: (bank, zufall) schon simuliert und vorbereitet
    (simulate_logs), sonst hier simuliert mit n_cases und seed.
    """
    add_paths()
    from preprocessing import chosen_outcome_spec
    from prescriptive.estimators.xgb_model import XGBModel
    cycle = chosen_outcome_spec(simulator.dataset)["type"] == "negative_cycle_days"  # Mitarbeitereffekt auf die Restlaufzeit
    bank, randomized = logs or simulate_logs(simulator, n_cases, seed, costs)
    model = XGBModel(simulator.dataset, simulator.points)  # nur für die Proben an den Wahlpunkten und ihre Merkmale
    sources = {REAL: log, BANK: bank, RANDOM: randomized}
    print("Prescriptive-KPIs: Proben an den Wahlpunkten ...", flush=True)
    samples = {name: _decisions(model, frame) for name, frame in sources.items()}
    cases = {name: frame[CASE].nunique() for name, frame in sources.items()}
    rows, details = _Rows(), {"zeitpunkt": [], "mitarbeiter": [], "unterscheidbar": []}
    _outcome_rows(rows, sources)
    for point in simulator.points:
        data = {name: samples[name].get(point.name) for name in sources}
        if any(data[name] is None for name in sources):
            rows.add(_area(point), "Proben (echt, Bank, Zufall)", *(0 if data[name] is None else len(data[name][0]) for name in sources))
            continue
        print(f"Prescriptive-KPIs: {point.name} ...", flush=True)
        if point.kind == "control_flow":
            _control_flow_rows(rows, details, point, data, cases, model, seed)
        else:
            _employee_rows(rows, details, point, data, cases, model, seed, cycle)
    table = pd.DataFrame(rows.rows).set_index(["Bereich", "Kennzahl"])
    return table, {"Übersicht": _overview(table), **{name: pd.DataFrame(parts) for name, parts in details.items()}}


# ------------------------------------------------------------------ Zeilen und Bewertung

def _grade(value, ok, check, higher_is_worse=True):
    if value is None or not np.isfinite(value):
        return "–"
    if higher_is_worse:
        return "ok" if value <= ok else "prüfen" if value <= check else "kritisch"
    return "ok" if value >= ok else "prüfen" if value >= check else "kritisch"


def _grade_effect(delta, se, sd):
    """Abstand zweier Kontraste in Std sd des Outcomes; innerhalb von 2 Standardfehlern ok."""
    if not np.isfinite(delta) or not np.isfinite(sd) or sd <= 0:
        return "–"
    noise = 2 * se if np.isfinite(se) else 0.0
    size = abs(delta)
    return "ok" if size <= max(noise, EFFECT_SD[0] * sd) else "prüfen" if size <= max(noise, EFFECT_SD[1] * sd) else "kritisch"


class _Rows:
    def __init__(self):
        self.rows = []

    def add(self, area, name, real=np.nan, bank=np.nan, randomized=np.nan, deviation=np.nan, unit="", grade="–", se=np.nan):
        self.rows.append({"Bereich": area, "Kennzahl": name, REAL: _float(real), BANK: _float(bank), RANDOM: _float(randomized),
                          "Abweichung": _float(deviation), "Std.-Fehler der Abweichung": _float(se), "Abweichung als": unit, "Bewertung": grade})

    def share(self, area, name, real, bank, randomized):
        """Anteile in %: Abweichung der Bank in Prozentpunkten."""
        deviation = bank - real
        self.add(area, name, real, bank, randomized, deviation, "Prozentpunkte Bank − echt", _grade(abs(deviation), *SHARE_PP))

    def relative(self, area, name, real, bank, randomized, floor=0.0):
        """Mittel, Quantile, Zeiten: Abweichung der Bank in % des echten Werts (mindestens floor als Nenner)."""
        base = max(abs(real), floor) if np.isfinite(real) else np.nan
        deviation = 100 * (bank - real) / base if base and np.isfinite(base) else np.nan
        self.add(area, name, real, bank, randomized, deviation, "% von echt (Bank)", _grade(abs(deviation), *RELATIVE))

    def effect(self, area, name, real, bank, randomized, compared, sd, unit):
        """Kontraste (Wert, Standardfehler); compared: die beiden verglichenen (Wert, Standardfehler), Abweichung zweiter − erster."""
        (first, first_se), (second, second_se) = compared
        deviation, se = second - first, float(np.hypot(first_se, second_se))
        self.add(area, name, real[0], bank[0], randomized[0], deviation, unit, _grade_effect(deviation, se, sd), se)

    def distinguishable(self, area, auc, top):
        """AUC echt gegen Bank; in Abweichung als das Merkmal mit dem größten Anteil an der Trennung (Details: unterscheidbar; Gewinn der Bäume, total_gain)."""
        unit = "AUC − 0,5" + (f"; v. a. {top[0]['Merkmal']} ({top[0]['Anteil an der Trennung (%)']:.0f} %)" if top else "")
        self.add(area, "Lage unterscheidbar: AUC echt gegen Bank (0,5 = gleich)", 0.5, auc, np.nan, auc - 0.5, unit, _grade(auc, *AUC))


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _area(point):
    return f"Control Flow: {point.name}" if point.kind == "control_flow" else point.name


NONE = (np.nan, np.nan)


def _overview(table):
    rank = {grade: k for k, grade in enumerate(GRADES)}
    rows = {}
    for area, part in table.groupby(level=0, sort=False):
        grades = part["Bewertung"]
        worst = max(grades, key=lambda grade: rank.get(grade, 0))
        rows[area] = {"Bewertung": worst, "kritisch": int((grades == "kritisch").sum()), "prüfen": int((grades == "prüfen").sum()),
                      "ok": int((grades == "ok").sum()),
                      "kritische Kennzahlen": "; ".join(part.index.get_level_values(1)[grades.to_numpy() == "kritisch"])}
    overview = pd.DataFrame.from_dict(rows, orient="index")
    overview.index.name = "Bereich"
    return overview


# ------------------------------------------------------------------ Proben und Merkmale

def _decisions(model, log):
    """Je Wahlpunkt (Tabelle: case, option, y, elapsed in Tagen, since_last in Stunden; Merkmale je Probe) oder None."""
    outcome = log.groupby(CASE)["outcome"].first()
    result = {}
    for name, samples in model.samples(log).items():
        if not samples:
            result[name] = None
            continue
        cases, features, options = zip(*samples)
        frame = pd.DataFrame({"case": list(cases), "option": list(options)})
        frame["y"] = outcome.reindex(frame["case"]).to_numpy(dtype=float)
        frame["elapsed"] = [row["elapsed_hours"] / 24 for row in features]
        frame["since_last"] = [row["since_last_hours"] for row in features]
        result[name] = (frame, list(features))
    return result


def _layout(model, features):
    """Spalten (alle Schlüssel der Merkmale) und Kategorien der kategorialen Merkmale (Fallattribute, letztes Event)."""
    columns = sorted({key for row in features for key in row})
    categorical = set(model.categorical) | {"last_token"}
    categories = {column: sorted({str(row[column]) for row in features if row.get(column) is not None}) for column in columns if column in categorical}
    return columns, categories


def _clean(name):
    # XGBoost erlaubt [, ] und < nicht in Merkmalsnamen (wie estimators/xgb_model.py)
    return str(name).replace("[", "(").replace("]", ")").replace("<", "lt")


def _matrix(features, columns, categories):
    X = pd.DataFrame.from_records(features).reindex(columns=columns)
    for column in columns:
        if column in categories:
            X[column] = pd.Categorical(X[column].map(lambda value: None if value is None else str(value)), categories=categories[column])
        else:
            X[column] = pd.to_numeric(X[column], errors="coerce")
            if column.startswith("n:"):
                X[column] = X[column].fillna(0.0)
    X.columns = [_clean(column) for column in columns]
    return X


def _folds(groups, seed):
    """Zwei Hälften nach Fällen (alle Proben eines Falls in derselben)."""
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    side = dict(zip(unique, np.random.default_rng(seed).random(len(unique)) < 0.5))
    return np.array([side[group] for group in groups], dtype=int)


def _classifier(X, labels, groups, seed):
    """XGBoost-Klassifikator mit frühem Stoppen auf einem Fünftel der Fälle."""
    from xgboost import XGBClassifier
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    held = set(unique[np.random.default_rng(seed + 1).random(len(unique)) < 0.2])
    held = np.array([group in held for group in groups])
    params = {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 20, "subsample": 0.8,
              "colsample_bytree": 0.8, "tree_method": "hist", "enable_categorical": True, "n_jobs": THREADS,
              "early_stopping_rounds": 30, "random_state": seed}
    model = XGBClassifier(**params)
    model.fit(X[~held], labels[~held], eval_set=[(X[held], labels[held])], verbose=False)
    return model


def _auc(labels, scores):
    from scipy.stats import rankdata
    labels = np.asarray(labels).astype(bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if not n1 or not n0:
        return np.nan
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _discriminate(model, real, bank, seed, top=5):
    """
    Wie gut sich die Lage am Wahlpunkt echt und Bank unterscheiden lässt: AUC eines Klassifikators auf den Merkmalen
    (kreuzvalidiert über zwei Hälften der Fälle), dazu die Merkmale mit dem größten Anteil an der Trennung (total_gain der Bäume) und ihr Mittel bzw. häufigster Wert.
    """
    (real_frame, real_rows), (bank_frame, bank_rows) = real, bank
    rng = np.random.default_rng(seed)
    take = lambda n: np.sort(rng.choice(n, min(n, MAX_DISCRIMINATOR), replace=False))
    ir, ib = take(len(real_rows)), take(len(bank_rows))
    features = [real_rows[i] for i in ir] + [bank_rows[i] for i in ib]
    labels = np.r_[np.zeros(len(ir), dtype=int), np.ones(len(ib), dtype=int)]
    groups = np.r_["echt:" + real_frame["case"].astype(str).to_numpy()[ir], "sim:" + bank_frame["case"].astype(str).to_numpy()[ib]]
    columns, categories = _layout(model, features)
    X = _matrix(features, columns, categories)
    folds = _folds(groups, seed)
    scores, gains = np.zeros(len(labels)), Counter()
    for fold in (0, 1):
        train, test = folds != fold, folds == fold
        classifier = _classifier(X[train], labels[train], groups[train], seed)
        scores[test] = classifier.predict_proba(X[test])[:, 1]
        gains.update(classifier.get_booster().get_score(importance_type="total_gain"))
    original = dict(zip(X.columns, columns))
    rows = []
    total = sum(gains.values()) or 1.0
    for rank, (name, gain) in enumerate(gains.most_common(top), 1):
        column = X[name]
        if isinstance(column.dtype, pd.CategoricalDtype):
            describe = lambda part: f"{part.mode().iloc[0]} ({100 * (part == part.mode().iloc[0]).mean():.0f} %)" if part.notna().any() else ""
            values = describe(column[labels == 0]), describe(column[labels == 1])
        else:
            values = round(float(column[labels == 0].mean()), 3), round(float(column[labels == 1].mean()), 3)
        rows.append({"Rang": rank, "Merkmal": original.get(name, name), "Anteil an der Trennung (%)": round(100 * gain / total, 1),
                     REAL: values[0], BANK: values[1]})
    return _auc(labels, scores), rows


class _Propensity:
    """P(Option | Merkmale) im echten Log; kreuzgefittet für die Proben des echten Logs, auf allen Proben für die Simulation."""

    def __init__(self, model, point, real, seed):
        frame, features = real
        counts = frame["option"].value_counts()
        self.options = list(point.options)
        self.present = [option for option in self.options if counts.get(option, 0) >= MIN_OPTION]
        self.columns, self.categories = _layout(model, features)
        self.cross = np.zeros((len(frame), len(self.options)))
        self.model = None
        if len(self.present) < 2:
            # Höchstens eine Option kommt genug vor: sie überall, die anderen nirgends
            for option in self.present:
                self.cross[:, self.options.index(option)] = 1.0
            return
        X = _matrix(features, self.columns, self.categories)
        labels = frame["option"].map({option: k for k, option in enumerate(self.present)}).to_numpy()
        keep = ~pd.isna(labels)
        labels = np.where(keep, labels, -1).astype(int)
        groups = frame["case"].astype(str).to_numpy()
        self.model = _classifier(X[keep], labels[keep], groups[keep], seed)
        folds = _folds(groups, seed)
        for fold in (0, 1):
            train, test = keep & (folds != fold), folds == fold
            try:
                classifier = _classifier(X[train], labels[train], groups[train], seed)
            except ValueError:  # eine Option fehlt in der Hälfte: das Modell auf allen Proben
                classifier = self.model
            self.cross[test] = self._expand(classifier.predict_proba(X[test]))

    def _expand(self, proba):
        full = np.zeros((len(proba), len(self.options)))
        full[:, [self.options.index(option) for option in self.present]] = proba
        return full

    def predict(self, features):
        if self.model is None:
            full = np.zeros((len(features), len(self.options)))
            for option in self.present:
                full[:, self.options.index(option)] = 1.0
            return full
        return self._expand(self.model.predict_proba(_matrix(features, self.columns, self.categories)))


# ------------------------------------------------------------------ Kennzahlen

def _cycle_days(log):
    times = log.groupby(CASE)[TIMESTAMP]
    return ((times.max() - times.min()).dt.total_seconds() / 86400).to_numpy(dtype=float)


def _cvar(values, share=0.1):
    values = np.sort(np.asarray(values, dtype=float))
    return float(values[:max(1, int(np.ceil(share * len(values))))].mean())


def _outcome_rows(rows, sources):
    from scipy.stats import wasserstein_distance
    area = "Outcome"
    outcome = {name: frame.groupby(CASE)["outcome"].first().to_numpy(dtype=float) for name, frame in sources.items()}
    days = {name: _cycle_days(frame) for name, frame in sources.items()}
    get = lambda values, function: [function(values[name]) for name in (REAL, BANK, RANDOM)]
    rows.relative(area, "Outcome je Fall: Mittel", *get(outcome, np.mean))
    rows.relative(area, "Outcome je Fall: Median", *get(outcome, np.median))
    rows.relative(area, "Outcome: schlechteste 10 % (CVaR)", *get(outcome, _cvar))
    sd = outcome[REAL].std()
    distance = {name: wasserstein_distance(outcome[REAL], outcome[name]) / sd for name in (BANK, RANDOM)}
    rows.add(area, "Outcome: Abstand der Verteilung (Wasserstein / Std echt)", 0.0, distance[BANK], distance[RANDOM], distance[BANK],
             "Std echt (Bank)", _grade(distance[BANK], *DISTANCE_SD))
    rows.relative(area, "Durchlaufzeit: Mittel (Tage)", *get(days, np.mean))
    rows.relative(area, "Durchlaufzeit: P90 (Tage)", *get(days, lambda values: np.quantile(values, 0.9)))
    rows.relative(area, "Durchlaufzeit: P95 (Tage)", *get(days, lambda values: np.quantile(values, 0.95)))


def _difference(frame, option, reference):
    """Mittel des Outcomes mit option minus mit reference (Wert, Standardfehler)."""
    a, b = frame.loc[frame["option"] == option, "y"], frame.loc[frame["option"] == reference, "y"]
    if len(a) < 2 or len(b) < 2:
        return NONE
    return float(a.mean() - b.mean()), float(np.sqrt(a.var() / len(a) + b.var() / len(b)))


def _ipw_difference(frame, propensity, options, option, reference, inside):
    """Hajek-IPW auf dem echten Log in den Proben inside: Mittel mit option minus mit reference (Wert, Standardfehler)."""
    result = []
    for chosen in (option, reference):
        mask = inside & (frame["option"] == chosen).to_numpy()
        if mask.sum() < 2:
            return NONE
        weights = 1 / propensity[mask, options.index(chosen)]
        y = frame["y"].to_numpy()[mask]
        mean = float((weights * y).sum() / weights.sum())
        result.append((mean, float(np.sqrt((weights ** 2 * (y - mean) ** 2).sum()) / weights.sum())))
    (a, a_se), (b, b_se) = result
    return a - b, float(np.hypot(a_se, b_se))


def _control_flow_rows(rows, details, point, data, cases, model, seed):
    area = _area(point)
    frames = {name: data[name][0] for name in data}
    real, bank, randomized = frames[REAL], frames[BANK], frames[RANDOM]
    per_source = lambda function: [function(name, frames[name]) for name in (REAL, BANK, RANDOM)]
    # B: Lage am Wahlpunkt
    rows.share(area, "Fälle mit Wahl (%)", *per_source(lambda name, frame: 100 * frame["case"].nunique() / cases[name]))
    rows.relative(area, "Wahlen je Fall mit Wahl", *per_source(lambda name, frame: len(frame) / frame["case"].nunique()))
    rows.relative(area, "Zeitpunkt der Wahl: Median (Tage seit Fallbeginn)", *per_source(lambda name, frame: frame["elapsed"].median()), floor=1.0)
    auc, top = _discriminate(model, data[REAL], data[BANK], seed)
    rows.distinguishable(area, auc, top)
    details["unterscheidbar"] += [{"Bereich": area, **row} for row in top]
    options = list(point.options)
    reference = real["option"].value_counts().idxmax()
    for option in options:
        rows.share(area, f"Anteil {option} (%)", *per_source(lambda name, frame: 100 * (frame["option"] == option).mean()))
    # C: Träger des echten Logs
    propensity = _Propensity(model, point, data[REAL], seed)
    p = {REAL: propensity.cross, BANK: propensity.predict(data[BANK][1]), RANDOM: propensity.predict(data[RANDOM][1])}
    index = {option: k for k, option in enumerate(options)}
    chosen_p = lambda name: p[name][np.arange(len(frames[name])), frames[name]["option"].map(index).to_numpy()]
    rows.share(area, "Wahlen außerhalb des Trägers des echten Logs (%)", *[100 * float(np.mean(chosen_p(name) < EPS)) for name in (REAL, BANK, RANDOM)])
    sd = float(real["y"].std())
    k_reference = index[reference]
    for option in options:
        if option == reference:
            continue
        label = f"{option} − {reference}"
        observed = {name: _difference(frames[name], option, reference) for name in frames}
        rows.effect(area, f"Kontrast {label}: beobachtet (Zufall: randomisiert)", observed[REAL], observed[BANK], observed[RANDOM],
                    (observed[REAL], observed[BANK]), sd, "Differenz Bank − echt")
        picked = (randomized["option"] == option).to_numpy()
        outside = 100 * float(np.mean(p[RANDOM][picked, index[option]] < EPS)) if picked.any() else np.nan
        rows.add(area, f"Zufallswahlen {option} außerhalb des Trägers (%)", randomized=outside)
        inside = {name: (p[name][:, index[option]] >= EPS) & (p[name][:, k_reference] >= EPS) for name in (REAL, RANDOM)}
        ipw = _ipw_difference(real, p[REAL], options, option, reference, inside[REAL])
        effect_in = _difference(randomized[inside[RANDOM]], option, reference)
        effect_out = _difference(randomized[~inside[RANDOM]], option, reference)
        rows.effect(area, f"Wirkung {label} im Träger (echt: IPW, Zufall: randomisiert)", ipw, NONE, effect_in, (ipw, effect_in), sd,
                    "Differenz Zufall − echt")
        rows.effect(area, f"Wirkung {label} außerhalb des Trägers (Zufall)", NONE, NONE, effect_out, (effect_in, effect_out), sd,
                    "Differenz außerhalb − im Träger")
    # Detail: Outcome je Option nach Zeitpunkt der Wahl (Quartile der echten Zeitpunkte)
    edges = np.unique(np.quantile(real["elapsed"], [0, 0.25, 0.5, 0.75, 1.0]))
    if len(edges) >= 3:
        bins = np.r_[-np.inf, edges[1:-1], np.inf]
        labels = [f"bis {edges[1]:.1f}"] + [f"{a:.1f} bis {b:.1f}" for a, b in zip(edges[1:-2], edges[2:-1])] + [f"ab {edges[-2]:.1f}"]
        for option in options:
            for k, label in enumerate(labels):
                row = {"Bereich": area, "Option": option, "Zeitpunkt der Wahl (Tage)": label}
                for name, frame in frames.items():
                    part = frame.loc[(frame["option"] == option) & (pd.cut(frame["elapsed"], bins, labels=False) == k), "y"]
                    row[f"Proben {name}"], row[f"Outcome {name}"] = len(part), round(float(part.mean()), 3) if len(part) else np.nan
                details["zeitpunkt"].append(row)


def _regressor(X, y, groups, seed):
    """XGBoost-Regression mit frühem Stoppen auf einem Fünftel der Fälle."""
    from xgboost import XGBRegressor
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    held = set(unique[np.random.default_rng(seed + 2).random(len(unique)) < 0.2])
    held = np.array([group in held for group in groups])
    model = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", enable_categorical=True, n_jobs=THREADS, early_stopping_rounds=30, random_state=seed)
    model.fit(X[~held], y[~held], eval_set=[(X[held], y[held])], verbose=False)
    return model


def _employee_effects(model, data, seed, cycle):
    """
    Effekt je Mitarbeiter an einem Wahlpunkt in einem Log: das Ziel (Durchlaufzeit: −Restlaufzeit ab dem Start in Tagen,
    sonst das Outcome) minus seine Vorhersage aus den Merkmalen ohne den Mitarbeiter (XGBoost, kreuzgefittet über zwei
    Hälften der Fälle), gemittelt je Mitarbeiter mit mindestens MIN_EFFECT Proben. Dazu die Streuung zwischen den
    Mitarbeitern (um das Rauschen der Mittel bereinigt) und die Reliabilität der Mittel: Korrelation der Mittel aus zwei
    zufälligen Hälften der Fälle (SPLITS mal gemittelt), hochgerechnet auf alle Proben (Spearman-Brown). Liefert
    (Mittel, Varianz und Proben je Mitarbeiter, Streuung, Reliabilität).
    """
    frame, features = data
    target = (frame["y"] + frame["elapsed"]).to_numpy(dtype=float) if cycle else frame["y"].to_numpy(dtype=float)
    columns, categories = _layout(model, features)
    X = _matrix(features, columns, categories)
    groups = frame["case"].astype(str).to_numpy()
    folds = _folds(groups, seed)
    predicted = np.zeros(len(frame))
    for fold in (0, 1):
        train, test = folds != fold, folds == fold
        predicted[test] = _regressor(X[train], target[train], groups[train], seed).predict(X[test])
    table = pd.DataFrame({"resource": frame["option"].to_numpy(), "case": groups, "effect": target - predicted})
    stats = table.groupby("resource")["effect"].agg(["mean", "var", "count"])
    stats = stats[stats["count"] >= MIN_EFFECT]
    if len(stats) < 5:
        return stats, np.nan, np.nan
    spread = float(np.sqrt(max(stats["mean"].var(ddof=1) - (stats["var"] / stats["count"]).mean(), 0.0)))
    kept = table[table["resource"].isin(stats.index)]
    cases = kept["case"].unique()
    rng = np.random.default_rng(seed)
    correlations = []
    for _ in range(SPLITS):
        half = dict(zip(cases, rng.random(len(cases)) < 0.5))
        side = kept["case"].map(half).to_numpy(dtype=bool)
        a = kept[side].groupby("resource")["effect"].agg(["mean", "count"])
        b = kept[~side].groupby("resource")["effect"].agg(["mean", "count"])
        common = a.index[a["count"] >= MIN_HALF].intersection(b.index[b["count"] >= MIN_HALF])
        if len(common) >= 5:
            correlations.append(np.corrcoef(a.loc[common, "mean"], b.loc[common, "mean"])[0, 1])
    r = float(np.mean(correlations)) if correlations else np.nan
    reliability = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan
    return stats, spread, reliability


def _employee_rows(rows, details, point, data, cases, model, seed, cycle):
    area = _area(point)
    frames = {name: data[name][0] for name in data}
    per_source = lambda function: [function(name, frames[name]) for name in (REAL, BANK, RANDOM)]
    rows.relative(area, "Wahlen je Fall", *per_source(lambda name, frame: len(frame) / cases[name]))
    rows.relative(area, "Wartezeit vor dem Start: Median (h)", *per_source(lambda name, frame: frame["since_last"].median()), floor=1.0)
    auc, top = _discriminate(model, data[REAL], data[BANK], seed)
    rows.distinguishable(area, auc, top)
    details["unterscheidbar"] += [{"Bereich": area, **row} for row in top]
    # Mitarbeitereffekt: um die Lage bereinigt; ob er im echten Log verlässlich ist, und ob der Simulator ihn wiedergibt
    effects = {name: _employee_effects(model, data[name], seed, cycle) for name in (REAL, BANK, RANDOM)}
    target = "Tage Restlaufzeit" if cycle else "Outcome"
    rows.add(area, f"Mitarbeiter mit mindestens {MIN_EFFECT} Proben", *(len(effects[name][0]) for name in (REAL, BANK, RANDOM)))
    real_reliability = effects[REAL][2]
    reliable = bool(np.isfinite(real_reliability) and real_reliability >= MIN_RELIABILITY)
    note = "" if reliable else f"; echt nicht verlässlich (Reliabilität < {MIN_RELIABILITY}): keine Bewertung"
    rows.add(area, "Mitarbeitereffekt: Reliabilität (Split-Half; 1 = stabil, 0 = nur Rauschen)", *(effects[name][2] for name in (REAL, BANK, RANDOM)),
             unit="Information")
    random_reliability = effects[RANDOM][2]
    simulated = bool(np.isfinite(random_reliability) and random_reliability >= MIN_RELIABILITY)
    if reliable and not simulated:
        note = f"; Simulator ohne verlässlichen Effekt (Reliabilität Zufall < {MIN_RELIABILITY})"
    real_sd, random_sd = effects[REAL][1], effects[RANDOM][1]
    factor = random_sd / real_sd if np.isfinite(random_sd) and np.isfinite(real_sd) and real_sd > 0 else np.nan
    if not reliable or not np.isfinite(factor):
        grade = "–"
    else:
        grade = "kritisch" if factor <= 0 else _grade(max(factor, 1 / factor), *FACTOR)
    rows.add(area, f"Mitarbeitereffekt: Streuung zwischen Mitarbeitern ({target}, um Lage und Rauschen bereinigt)", real_sd, effects[BANK][1], random_sd,
             factor, "Faktor Zufall / echt" + note, grade)
    raw, corrected = {}, {}
    for name in (BANK, RANDOM):
        common = effects[REAL][0].index.intersection(effects[name][0].index)
        raw[name] = float(np.corrcoef(effects[REAL][0].loc[common, "mean"], effects[name][0].loc[common, "mean"])[0, 1]) if len(common) >= 5 else np.nan
        both = min(real_reliability, effects[name][2]) if np.isfinite(real_reliability) and np.isfinite(effects[name][2]) else np.nan
        # Nur mit verlässlichen Effekten auf beiden Seiten: sonst teilt die Korrektur durch fast 0
        corrected[name] = float(np.clip(raw[name] / np.sqrt(real_reliability * effects[name][2]), -1, 1)) if np.isfinite(raw[name]) and both >= MIN_RELIABILITY else np.nan
    rows.add(area, "Mitarbeitereffekt: Korrelation mit echt (roh)", 1.0, raw[BANK], raw[RANDOM], unit="Information")
    if not reliable:
        grade = "–"
    elif not simulated:
        grade = "kritisch"  # echt ein verlässlicher Effekt, im Simulator keiner
    else:
        grade = _grade(corrected[RANDOM], *CORRELATION, higher_is_worse=False)
    rows.add(area, "Mitarbeitereffekt: Korrelation mit echt (um Rauschen korrigiert)", 1.0, corrected[BANK], corrected[RANDOM], corrected[RANDOM],
             "Korrelation (Zufall)" + note, grade)
    resources = sorted(set().union(*(effects[name][0].index for name in frames)))
    for resource in resources:
        row = {"Bereich": area, "Mitarbeiter": resource}
        for name in frames:
            stats = effects[name][0]
            row[f"Proben {name}"] = int(stats.loc[resource, "count"]) if resource in stats.index else 0
            row[f"Effekt {name}"] = round(float(stats.loc[resource, "mean"]), 3) if resource in stats.index else np.nan
        details["mitarbeiter"].append(row)

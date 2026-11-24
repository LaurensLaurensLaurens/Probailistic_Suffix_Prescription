"""
Kausale Schätzer nach Bozorgi et al. (2023), "Prescriptive Process Monitoring based on Causal Effect Estimation", als
Vergleich zum PSP: S-Learner, X-Learner und Causal Forest. Je Control-Flow-Wahlpunkt (interventions.choice_points) ein
Modell für den Effekt der Aktion gegenüber der Referenz auf den Outcome des Falls (CATE):

    τ(x) = E[Y(Aktion) − Y(Referenz) | X = x]      Referenz = erste Option des Punkts, Aktion = jede weitere

Das Paper kennt ein binäres Treatment je Fall; hier ist jeder Wahlpunkt ein eigenes Treatment, und jedes Erreichen eines
Punkts ist eine Probe (wie beim XGBoost-Modell: die Engine fragt die Policy auch jedes Mal). Proben und Merkmale sind
die von estimators/xgb_model.py (XGBModel.samples, Prefix.row), damit sich die Schätzer nur im Verfahren unterscheiden.
Mitarbeiter-Wahlpunkte bekommen kein Modell (viele Optionen ohne Überdeckung).

learner:
    "s"   S-Learner: ein Outcome-Modell μ(x, t) mit der Option als Merkmal, τ(x) = μ(x, 1) − μ(x, 0)
    "x"   X-Learner (Künzel et al. 2019): μ0, μ1 je Arm; unterstellte Effekte D1 = Y − μ0(x) auf den Fällen mit Aktion,
          D0 = μ1(x) − Y auf den Fällen mit Referenz; darauf τ1, τ0; τ(x) = g(x)·τ0(x) + (1 − g(x))·τ1(x) mit der
          Propensity g(x) = P(Aktion | x)
    "cf"  Causal Forest (Athey, Tibshirani, Wager 2019) in der Umsetzung von EconML (CausalForestDML): Outcome und
          Treatment werden mit kreuzweise angepassten Modellen bereinigt (Falten nach Fällen), darauf ein ehrlicher Wald;
          Baumtiefe und Blattgröße wählt EconML (tune)
Basis-Modelle sind überall XGBoost mit den Einstellungen von xgb_model.PARAMS; frühes Stoppen auf VALIDATION_SHARE der
Fälle (alle Proben eines Falls auf derselben Seite). Beim Causal Forest steht die Baumzahl der Hilfsmodelle fest
(aus einem Vorlauf mit frühem Stoppen), weil EconML sie ohne Validierungsdaten anpasst.

Merkmale als Zahlenmatrix (alle drei Verfahren dieselbe): kategoriale Merkmale als 0/1-Spalten je Wert, fehlende Zahlen
durch den Median des Trainings ersetzt und mit einer 0/1-Spalte "fehlt" markiert, im Training konstante Spalten entfernt.

    model = CATEModel("bpic17", learner="x").fit(log)      # log: Event-Log mit Spalte outcome
    model.save("prescriptive/models/cate_model_x.pkl")
    CATEModel.load(path).effects(point, [Prefix.row, ...])   # τ je Probe und Aktion (Spalten: Optionen ab der zweiten)
    CATEModel.load(path).values(point, events, options, now) # je Option: 0 für die Referenz, τ für die Aktionen
"""
import numpy as np
import pandas as pd

from prescriptive.estimators.xgb_model import ACTIVITY, CASE, LIFECYCLE, PARAMS, TIMESTAMP, VALIDATION_SHARE, Prefix, XGBModel, _seconds

LEARNERS = {"s": "S-Learner", "x": "X-Learner", "cf": "Causal Forest"}
MIN_HELD = 30  # so viele Validierungsproben braucht das frühe Stoppen, sonst FALLBACK_TREES Bäume
FALLBACK_TREES = 200
PROPENSITY_CLIP = (0.01, 0.99)
FOREST = {"n_estimators": 500, "min_samples_leaf": 5, "max_samples": 0.45, "honest": True, "inference": False, "cv": 5, "n_jobs": 8}
FOREST_TUNED = ("max_depth", "min_weight_fraction_leaf", "min_var_fraction_leaf")  # wählt CausalForestDML.tune
MISSING = ":fehlt"


def _fit_xgb(kind, X, y, held, seed):
    """XGBoost (kind regressor oder classifier) mit frühem Stoppen auf den Zeilen held; zu wenige dort: FALLBACK_TREES Bäume."""
    from xgboost import XGBClassifier, XGBRegressor
    model_class = XGBRegressor if kind == "regressor" else XGBClassifier
    params = {key: value for key, value in PARAMS.items() if key != "enable_categorical"}
    enough = held.sum() >= MIN_HELD and (~held).sum() >= MIN_HELD and (kind == "regressor" or len(np.unique(y[held])) > 1)
    if not enough:
        params.update(n_estimators=FALLBACK_TREES, early_stopping_rounds=None)
        return model_class(**params, random_state=seed).fit(X, y)
    return model_class(**params, random_state=seed).fit(X[~held], y[~held], eval_set=[(X[held], y[held])], verbose=False)


def _trees(model):
    return int(getattr(model, "best_iteration", None) + 1) if getattr(model, "best_iteration", None) is not None else int(model.n_estimators)


def _fixed_xgb(kind, trees, seed):
    """XGBoost ohne frühes Stoppen mit fester Baumzahl (für EconML, das seine Hilfsmodelle ohne Validierungsdaten anpasst)."""
    from xgboost import XGBClassifier, XGBRegressor
    params = {key: value for key, value in PARAMS.items() if key not in ("enable_categorical", "early_stopping_rounds")}
    params["n_estimators"] = max(int(trees), 20)
    return (XGBRegressor if kind == "regressor" else XGBClassifier)(**params, random_state=seed)


class _SLearner:
    """Ein Outcome-Modell mit dem Treatment als Merkmal."""

    def fit(self, X, t, y, held, cases, seed):
        self.model = _fit_xgb("regressor", np.column_stack([X, t]), y, held, seed)
        return self

    def effect(self, X):
        ones = np.ones(len(X), dtype=X.dtype)
        return self.model.predict(np.column_stack([X, ones])) - self.model.predict(np.column_stack([X, 0 * ones]))

    def info(self):
        return {"Bäume": _trees(self.model)}

    def set_threads(self, n):
        self.model.set_params(n_jobs=n)


class _XLearner:
    """X-Learner nach Künzel et al. (2019) mit der Propensity als Gewicht."""

    def fit(self, X, t, y, held, cases, seed):
        treated = t == 1
        self.mu0 = _fit_xgb("regressor", X[~treated], y[~treated], held[~treated], seed)
        self.mu1 = _fit_xgb("regressor", X[treated], y[treated], held[treated], seed)
        d1 = y[treated] - self.mu0.predict(X[treated])       # unterstellter Effekt bei den Fällen mit Aktion
        d0 = self.mu1.predict(X[~treated]) - y[~treated]     # und bei den Fällen mit Referenz
        self.tau1 = _fit_xgb("regressor", X[treated], d1, held[treated], seed)
        self.tau0 = _fit_xgb("regressor", X[~treated], d0, held[~treated], seed)
        self.propensity = _fit_xgb("classifier", X, t.astype(int), held, seed)
        return self

    def effect(self, X):
        g = np.clip(self.propensity.predict_proba(X)[:, 1], *PROPENSITY_CLIP)
        return g * self.tau0.predict(X) + (1 - g) * self.tau1.predict(X)

    def info(self):
        return {"Bäume": "/".join(str(_trees(model)) for model in (self.mu0, self.mu1, self.tau0, self.tau1, self.propensity))}

    def set_threads(self, n):
        for model in (self.mu0, self.mu1, self.tau0, self.tau1, self.propensity):
            model.set_params(n_jobs=n)


class _CausalForest:
    """Causal Forest (EconML CausalForestDML): Bereinigung von Outcome und Treatment kreuzweise nach Fällen, dann der Wald."""

    def fit(self, X, t, y, held, cases, seed):
        from econml.dml import CausalForestDML
        self.trees = (_trees(_fit_xgb("regressor", X, y, held, seed)), _trees(_fit_xgb("classifier", X, t.astype(int), held, seed)))
        self.model = CausalForestDML(model_y=_fixed_xgb("regressor", self.trees[0], seed), model_t=_fixed_xgb("classifier", self.trees[1], seed),
                                     discrete_treatment=True, random_state=seed, **FOREST)
        groups = pd.factorize(np.asarray(cases, dtype=object))[0]
        # Baumtiefe und Blattgröße wählt EconML selbst (tune: kleine Wälder, bewertet auf zurückgehaltenen Proben)
        self.model.tune(y, t.astype(int), X=X.astype(np.float64), groups=groups)
        self.tuned = {key: getattr(self.model, key) for key in FOREST_TUNED}
        self.model.fit(y, t.astype(int), X=X.astype(np.float64), groups=groups)
        return self

    def effect(self, X):
        return np.asarray(self.model.effect(X.astype(np.float64))).reshape(-1)

    def info(self):
        return {"Bäume": f"{self.trees[0]}/{self.trees[1]} (Hilfsmodelle), {FOREST['n_estimators']} (Wald)", "Wald": self.tuned}

    def set_threads(self, n):
        forest = getattr(self.model, "model_cate", None)
        for holder in [self.model, forest] + list(getattr(forest, "estimators_", [])):
            if holder is not None and hasattr(holder, "n_jobs"):
                holder.n_jobs = n


_CLASSES = {"s": _SLearner, "x": _XLearner, "cf": _CausalForest}


class CATEModel(XGBModel):
    def __init__(self, dataset, points=None, seed=42, learner="x"):
        if learner not in LEARNERS:
            raise ValueError(f"learner {learner}: möglich {', '.join(LEARNERS)}")
        super().__init__(dataset, points, seed)
        self.learner, self.design = learner, {}

    # ------------------------------------------------------------------ Merkmale

    def _table(self, rows, name):
        X = pd.DataFrame(rows).reindex(columns=self.columns[name])
        for column in [c for c in X.columns if c.startswith("n:")]:
            X[column] = X[column].fillna(0.0)
        return X

    def _design(self, rows, name):
        """Die Kodierung eines Wahlpunkts aus den Trainingsproben festlegen (Spalten, Mediane, Kategorien) und die Matrix liefern."""
        X = self._table(rows, name)
        categorical = [column for column in X.columns if column in self.categories]
        numeric = [column for column in X.columns if column not in categorical]
        values = X[numeric].apply(pd.to_numeric, errors="coerce")
        design = {"numeric": numeric, "medians": {c: float(np.nan_to_num(values[c].median())) for c in numeric},
                  "missing": [c for c in numeric if values[c].isna().any()],
                  "levels": {c: sorted(X[c].dropna().astype(str).unique()) for c in categorical}}
        self.design[name] = design
        matrix, names = self._encode(X, design)
        keep = np.flatnonzero(matrix.std(axis=0) > 0)
        design["keep"], design["names"] = keep, [names[i] for i in keep]
        return matrix[:, keep]

    @staticmethod
    def _encode(X, design):
        values = X[design["numeric"]].apply(pd.to_numeric, errors="coerce")
        parts = [values.fillna(design["medians"]).to_numpy(dtype=np.float32)]
        names = list(design["numeric"])
        if design["missing"]:
            parts.append(values[design["missing"]].isna().to_numpy(dtype=np.float32))
            names += [column + MISSING for column in design["missing"]]
        for column, levels in design["levels"].items():
            text = X[column].astype(str).to_numpy()
            parts.append((text[:, None] == np.array(levels, dtype=object)[None, :]).astype(np.float32))
            names += [f"{column}={level}" for level in levels]
        return np.column_stack(parts), names

    def matrix(self, point, rows):
        """Merkmale (rows: Prefix.row) als Zahlenmatrix mit den Spalten des Trainings."""
        design = self.design[point.name]
        return self._encode(self._table(list(rows), point.name), design)[0][:, design["keep"]]

    # ------------------------------------------------------------------ Training

    def _targets(self, name, outcome, held):
        """Trainingsziel je Probe; die ERM-Variante transformiert hier den Outcome (gleiche Skalierung für alle Arme)."""
        return outcome

    def fit(self, log, samples=None):
        """
        Je Control-Flow-Wahlpunkt mit Proben und je Aktion ein Effektmodell gegen die Referenz (die Proben beider Optionen).
        log: Event-Log mit Spalte outcome; samples: schon gezogene Proben (XGBModel.samples), sonst aus log.
        """
        outcome = log.groupby(CASE)["outcome"].first()
        self.categories = {name: sorted(log[name].dropna().astype(str).unique()) for name in self.categorical}
        self.categories["last_token"] = sorted((log[ACTIVITY].astype(str) + "|" + log[LIFECYCLE].astype(str)).unique())
        self.summary = {}
        for name, rows in (samples if samples is not None else self.samples(log)).items():
            point = self.points[name]
            if not rows or point.kind != "control_flow" or len(point.options) < 2:
                continue
            cases, features, options = zip(*rows)
            self.options[name] = list(point.options)
            self.columns[name] = sorted({column for row in features for column in row})
            X = self._design(list(features), name)
            y = outcome.reindex(list(cases)).to_numpy(dtype=float)
            arm = np.array([self.options[name].index(option) for option in options])
            unique = sorted(set(cases))
            validation = set(np.array(unique, dtype=object)[np.random.default_rng(self.seed).random(len(unique)) < VALIDATION_SHARE])
            held = np.array([c in validation for c in cases])
            y = self._targets(name, y, held)
            cases = np.array(cases, dtype=object)
            self.models[name] = {}
            for k in range(1, len(point.options)):
                rows_k = (arm == 0) | (arm == k)
                if min((arm[rows_k] == 0).sum(), (arm[rows_k] == k).sum()) < MIN_HELD:
                    continue  # eine Option kommt im Training praktisch nicht vor: kein Effekt schätzbar
                t = (arm[rows_k] == k).astype(np.float32)
                model = _CLASSES[self.learner]().fit(X[rows_k], t, y[rows_k], held[rows_k], cases[rows_k], self.seed)
                self.models[name][k] = model
                effect = model.effect(X[rows_k][held[rows_k]])
                self.summary[f"{name}: {point.options[k]}" if len(point.options) > 2 else name] = {
                    "Proben": int(rows_k.sum()), "Fälle": len(set(cases[rows_k])), "Anteil Aktion": float(t.mean()), "Merkmale": X.shape[1],
                    **model.info(), "Effekt Mittel (Validierung)": float(effect.mean()), "Effekt Streuung": float(effect.std()),
                    "Anteil Effekt > 0": float((effect > 0).mean())}
            if not self.models[name]:
                del self.models[name]
        return self

    # ------------------------------------------------------------------ Abfrage

    def effects(self, point, samples):
        """τ je Probe (Zeilen) und Aktion (Spalten: die Optionen des Punkts ab der zweiten; ohne Modell 0)."""
        X = self.matrix(point, samples)
        out = np.zeros((len(X), len(self.options[point.name]) - 1))
        for k, model in self.models[point.name].items():
            out[:, k - 1] = model.effect(X)
        return out

    def predict(self, point, samples, options):
        """Je Probe der Wert ihrer Option relativ zur Referenz (0 für die Referenz, τ für eine Aktion)."""
        effects = np.column_stack([np.zeros(len(samples)), self.effects(point, samples)])
        index = [self.options[point.name].index(option) for option in options]
        return effects[np.arange(len(index)), index]

    def values(self, point, events, options, now):
        """
        Je Option (Array in der Reihenfolge von options) ihr Wert relativ zur Referenz: 0 für die Referenz, τ für eine
        Aktion; eine Option, die der Punkt nicht kennt, −inf. events: die Events des Falls bis jetzt (wie engine.events).
        """
        prefix = Prefix(self._attributes(events[0]), self.event_numeric)
        for event, seconds in zip(events, _seconds([event[TIMESTAMP] for event in events])):
            prefix.add(f"{event[ACTIVITY]}|{event.get(LIFECYCLE, 'complete')}", seconds, {name: event.get(name) for name in self.event_numeric})
        relative = np.r_[0.0, self.effects(point, [prefix.row(now)])[0]]
        known = self.options[point.name]
        return np.array([relative[known.index(option)] if option in known else -np.inf for option in options])

    def set_threads(self, n):
        for models in self.models.values():
            for model in models.values():
                model.set_threads(n)

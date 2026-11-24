"""
s_learner, x_learner, causal_forest: Policies nach den kausalen Schätzern von Bozorgi et al. (2023), als Vergleich zum
PSP (psp_policy.py im selben Ordner). Das Modell (estimators/cate_model.py, CATEModel) schätzt an einem
Control-Flow-Wahlpunkt aus dem Prefix des Falls den Effekt τ jeder Aktion gegenüber der Referenz (der ersten Option
des Punkts) auf den Outcome. Die Policy behandelt, wenn der geschätzte Effekt über der Schwelle liegt:

    Aktion mit dem größten τ, falls τ > threshold; sonst die Referenz

Im Paper kommt die Schwelle aus der Net-Value-Kurve (Nutzen gegen Kosten des Treatments). Hier hat keine Option eigene
Kosten, der Outcome ist die Durchlaufzeit: threshold 0, also die Option mit dem besseren geschätzten Outcome, wie bei
psp und xgb. Mit threshold > 0 (in Einheiten des Outcomes, hier Tage) greift die Policy nur bei größeren Effekten ein.

Die drei Policies unterscheiden sich nur im Modell: models/cate_model_<s|x|cf>... (scripts/train_cate_real.py).
Hat das Modell den Wahlpunkt nicht (z. B. Mitarbeiterwahl), None (wie die Bank).

first_only (Wahlpunkte, Standard keine): wie bei psp_policy.py, dort nur die erste Entscheidung je Fall.
"""
import numpy as np

ACTIVITY, LIFECYCLE = "concept:name", "lifecycle:transition"  # wie helpers/datasets.py


class CATEPolicy:
    def __init__(self, model_path, seed=42, threshold=0.0, first_only=()):
        from prescriptive.estimators.cate_model import CATEModel
        self.model = CATEModel.load(model_path)
        if getattr(self.model, "objective", None) == "entropic_risk":
            raise ValueError("CATEPolicy braucht ein EV-Modell; für ERM-Modelle CATEERMPolicy verwenden")
        self.model.set_threads(1)  # ein Thread je Arbeitsprozess (die Läufe laufen parallel)
        self.threshold = float(threshold)
        self.first_only = set(first_only or ())
        self.rng = np.random.default_rng(seed)  # nur für Gleichstände

    # Wird aufgerufen von case_engine._ask -> choose (je Wahlpunkt)
    def choose(self, point, case, options, time, engine):
        if not self.model.knows(point) or self._repeated(point, case.prefix):
            return None
        # je Option: 0 für die Referenz, τ für eine Aktion (Prefix mit dem Entscheidungs-Event)
        values = self.model.values(point, case.prefix, options, time)
        reference = self.model.options[point.name][0]
        values = np.where(np.array([option == reference for option in options]), values, values - self.threshold)
        best = np.flatnonzero(values >= values.max())
        return options[int(best[0] if len(best) == 1 else self.rng.choice(best))]

    def _repeated(self, point, prefix):
        """first_only: steht das Entscheidungs-Event des Punkts (zusammengelegte Punkte a+b: einer der Teile) schon vor dem aktuellen im Prefix?"""
        if not self.first_only or not set(point.name.split("+")) & self.first_only:
            return False
        return sum(1 for event in prefix if event.get(ACTIVITY) == point.activity and event.get(LIFECYCLE, "complete") == point.lifecycle) > 1

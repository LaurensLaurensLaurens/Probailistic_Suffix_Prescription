"""
s_learner_erm, x_learner_erm, causal_forest_erm: ERM-Vergleichsmethoden zum PSP-ERM.
Das neu trainierte CATEERMModel (estimators/cate_erm_model.py) schätzt je Aktion den Effekt auf den
erwarteten exponentiellen Verlust exp(γ · Durchlaufzeit - b), mit demselben b für alle Aktionen eines
Wahlpunkts. Die kleinste Vorhersage minimiert auch das entropische Risikomaß bei dem trainierten γ:

    Referenz = 0, Aktion = τ_Z; die Option mit dem kleinsten Wert wählen.

Die Werte sind keine Tage. Keine Schwelle in Tagen; Gleichstände zufällig wie bei CATEPolicy.
first_only und nicht modellierte Wahlpunkte wie bei der EV-Policy. Nicht trainierte/unbekannte Optionen
werden nicht gewählt. γ kommt aus dem Modell; optional gamma prüft, ob es zum Vergleich mit PSP-ERM passt.
Ein EV-Modell oder ein anderes γ wird abgelehnt. Training: scripts/train_cate_real.py --erm-gamma 0.1.
"""
import numpy as np

from prescriptive.policies.ev_policy.cate_policy import CATEPolicy


class CATEERMPolicy(CATEPolicy):
    def __init__(self, model_path, seed=42, gamma=None, first_only=()):
        from prescriptive.estimators.cate_erm_model import CATEERMModel
        self.model = CATEERMModel.load(model_path)
        if not isinstance(self.model, CATEERMModel):
            raise ValueError("CATE-ERM braucht ein ERM-Modell aus train_cate_real.py --erm-gamma, kein EV-Modell")
        if gamma is not None and (not np.isfinite(gamma) or gamma != self.model.gamma):
            raise ValueError(f"CATE-ERM: Modell auf γ={self.model.gamma} trainiert, angefragt ist γ={gamma}; "
                             "für ein anderes γ neu trainieren")
        self.gamma = self.model.gamma
        self.model.set_threads(1)
        self.first_only = set(first_only or ())
        self.rng = np.random.default_rng(seed)
        self.stats = {"ERM γ (1/Tag)": self.gamma}

    def choose(self, point, case, options, time, engine):
        if not self.model.knows(point) or self._repeated(point, case.prefix):
            return None
        values = self.model.values(point, case.prefix, options, time)
        supported = np.flatnonzero(np.isfinite(values))
        if not len(supported):
            return None
        best = supported[values[supported] <= values[supported].min()]
        return options[int(best[0] if len(best) == 1 else self.rng.choice(best))]

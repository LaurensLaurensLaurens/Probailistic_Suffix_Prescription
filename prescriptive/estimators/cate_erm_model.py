"""
S-Learner, X-Learner und Causal Forest mit entropischem Risikoziel, als Vergleich zu PSP-ERM.
Wie CATEModel, aber das Ziel ist der exponentielle Verlust statt des Outcomes:

    L = -outcome (Durchlaufzeit in Tagen), Z = exp(γ L - b)
    τ_Z(x, a) = E[Z(a) - Z(Referenz) | X=x]

Für festes γ > 0 minimiert die Aktion mit dem kleinsten E[Z | x, a] auch
ρ_γ(L | x, a) = log(E[exp(γ L) | x, a]) / γ. Deshalb genügen die Effekte τ_Z zur Wahl.
Sie sind keine ERM-Differenzen in Tagen. Referenz = 0, kleiner ist besser.

b ist je Wahlpunkt ein gemeinsamer, gespeicherter Log-Skalierungsfaktor: log(mean(exp(γ L)))
auf dem inneren Trainingsteil. Die Ziele haben dort Mittel 1; alle Arme (auch bei mehr als zwei Optionen)
verwenden denselben Faktor. Berechnung mit abgezogenem Maximum. Kein Abschneiden von Verlusten;
bei für XGBoost (Float32) nicht darstellbaren Zielen wird mit einer Fehlermeldung abgebrochen.

Alle Merkmale, Proben, Fall-Splits und Lernverfahren stammen aus cate_model.py. γ ist Teil des
trainierten Modells; ein anderes γ erfordert neues Training. Ohne Effektmodell für eine Aktion: +inf,
damit die ERM-Policy diese Aktion nicht wählt. Ohne Modell für den Wahlpunkt: wie CATEModel.knows.

    model = CATEERMModel("bpic17", learner="x", gamma=0.1).fit(log)
    model.save("prescriptive/models/cate_erm_model_x_gamma0.1.pkl")
"""
import numpy as np

from prescriptive.estimators.cate_model import CATEModel


class CATEERMModel(CATEModel):
    objective = "entropic_risk"

    def __init__(self, dataset, points=None, seed=42, learner="x", gamma=0.1):
        gamma = float(gamma)
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError(f"CATE-ERM: γ muss endlich und größer 0 sein, nicht {gamma}")
        super().__init__(dataset, points, seed, learner)
        self.gamma, self.log_scales = gamma, {}

    def _targets(self, name, outcome, held):
        losses = -np.asarray(outcome, dtype=float)
        if not np.isfinite(losses).all() or (losses < 0).any():
            raise ValueError(f"CATE-ERM {name}: outcome muss die endliche negative Durchlaufzeit in Tagen sein")
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = self.gamma * losses
        if not np.isfinite(scaled).all():
            raise ValueError(f"CATE-ERM {name}: γ · Durchlaufzeit ist nicht darstellbar; kleineres γ verwenden")
        training = ~held if (~held).any() else np.ones(len(losses), dtype=bool)
        top = scaled[training].max()
        centered = scaled - top
        with np.errstate(under="ignore"):
            log_mean = np.log(np.exp(centered[training]).mean())
        log_targets = centered - log_mean
        limits = np.log([np.finfo(np.float32).tiny, np.finfo(np.float32).max])
        if (log_targets < limits[0]).any() or (log_targets > limits[1]).any():
            raise ValueError(f"CATE-ERM {name}: exponentielle Ziele bei γ={self.gamma} haben zu große Dynamik für Float32; "
                             "kleineres γ verwenden (Verluste werden nicht abgeschnitten)")
        self.log_scales[name] = float(top + log_mean)
        return np.exp(log_targets)

    def fit(self, log, samples=None):
        self.log_scales = {}
        super().fit(log, samples)
        for name, models in self.models.items():
            for k in models:
                key = f"{name}: {self.options[name][k]}" if len(self.options[name]) > 2 else name
                self.summary[key].update({"ERM γ (1/Tag)": self.gamma, "ERM Log-Skalierung": self.log_scales[name],
                                          "Effektziel": "skalierter exponentieller Verlust (kleiner ist besser)"})
        return self

    def effects(self, point, samples):
        """Effekte auf E[exp(γ L - b)] je Aktion; ohne trainiertes Modell +inf statt eines scheinbaren Nulleffekts."""
        effects = super().effects(point, samples)
        for k in range(1, len(self.options[point.name])):
            if k not in self.models[point.name]:
                effects[:, k - 1] = np.inf
        return effects

    def values(self, point, events, options, now):
        """0 für die Referenz, τ_Z für trainierte Aktionen, +inf für unbekannte oder nicht trainierte Aktionen."""
        values = super().values(point, events, options, now)
        known = self.options[point.name]
        return np.array([value if option in known else np.inf for option, value in zip(options, values)])

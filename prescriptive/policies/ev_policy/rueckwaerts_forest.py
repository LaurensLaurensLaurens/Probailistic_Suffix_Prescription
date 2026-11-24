"""
Causal Forest mit Rückwärtsinduktion (2026-11-24, Nutzer: „ja mach beides“ – Forest, der seine eigenen späteren Entscheidungen
einrechnet). Der einfache Conformance-Forest (cate_model_cf_konformanz_simv3.pkl) schätzt die Wirkung einer Option so, wie es mit der
Bank an den übrigen Punkten käme; die Punkt-Effekte überlappen aber, und die beste Kombination hängt davon ab, wer später entscheidet.

Rückwärts über die Wahlpunkte (im Fall: leads → konzept → nach Vervollständigung → nachlegen): für den Punkt der Stufe k wird im Simulator
ein Log erzeugt, in dem dort die Option zufällig gewählt wird (gleichverteilt über die zulässigen Optionen; angebot_nachlegen nur beim
ersten Besuch, spätere Besuche Bank), an den schon trainierten späteren Punkten die Forests der früheren Stufen entscheiden und an den
übrigen die Bank. Darauf ein Causal Forest nur für diesen Punkt (Outcome −Abweichungen). Am Ende entscheidet je Punkt sein Forest.

StufenPolitik     Log-Erzeugung einer Stufe (zufällig am Stufenpunkt, Forests an den fertigen Punkten, sonst Bank)
PunktForestPolicy je Punkt ein eigener Forest (CATEPolicy je Punkt), für die Online-Auswertung
"""
import numpy as np

from prescriptive.policies.ev_policy.cate_policy import CATEPolicy

ACTIVITY, LIFECYCLE = "concept:name", "lifecycle:transition"


def erster_besuch(point, prefix):
    """Steht das Entscheidungs-Event des Punkts zum ersten Mal im Prefix (wie CATEPolicy._repeated, umgekehrt)?"""
    return sum(1 for e in prefix if e.get(ACTIVITY) == point.activity and e.get(LIFECYCLE, "complete") == point.lifecycle) <= 1


class PunktForestPolicy:
    """Je Wahlpunkt ein eigener Forest; Punkte ohne Forest entscheidet die Bank (None)."""

    def __init__(self, forests, seed=42, first_only=()):
        self.policies = {name: CATEPolicy(str(path), seed=seed + i, threshold=0.0, first_only=first_only)
                         for i, (name, path) in enumerate(sorted(forests.items()))}
        for name, policy in self.policies.items():
            if name not in policy.model.models:
                raise ValueError(f"Forest {forests[name]} kennt den Punkt {name} nicht")

    def choose(self, point, case, options, time, engine):
        policy = self.policies.get(point.name)
        return None if policy is None else policy.choose(point, case, options, time, engine)


class StufenPolitik(PunktForestPolicy):
    """Zufällig am Stufenpunkt (first_only-Punkte nur beim ersten Besuch), Forests an den fertigen Punkten, sonst Bank."""

    def __init__(self, stufenpunkt, forests, seed=42, first_only=()):
        if stufenpunkt in forests:
            raise ValueError(f"{stufenpunkt} hat schon einen Forest")
        super().__init__(forests, seed=seed, first_only=first_only)
        self.stufenpunkt, self.first_only = stufenpunkt, set(first_only or ())
        self.rng = np.random.default_rng(seed)
        self.zufall = 0

    def choose(self, point, case, options, time, engine):
        if point.name == self.stufenpunkt:
            if len(options) < 2 or (point.name in self.first_only and not erster_besuch(point, case.prefix)):
                return None
            self.zufall += 1
            return options[int(self.rng.integers(len(options)))]
        return super().choose(point, case, options, time, engine)

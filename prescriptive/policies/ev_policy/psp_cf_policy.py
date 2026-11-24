"""
psp_cf: PSP entscheidet, bei nicht unterscheidbaren Optionen der Causal Forest (2026-11-23, Nutzer: „mach es so, dass wenn dieser Fall
eintritt, dann soll ein Causal Forest entscheiden: nicht unterscheidbar, wenn |Δ| < 1,96 · SE“).

Wie PSPPolicy (psp_policy.py): je Option samples gezogene Suffixe (outcome_samples, dieselben Ziehungen und Zufallszahlen wie
PSPPolicy.values – bei sicheren Entscheidungen also dieselbe Wahl wie der reine PSP-Arm). Dann für die beste gegen die zweitbeste Option:
    d_k = Outcome_k(beste) − Outcome_k(zweitbeste)   (gepaart über den Ziehungsindex k, gemeinsame Zufallszahlen)
    Δ = Mittel d_k,  SE = SD(d_k) / √n
    nicht unterscheidbar, wenn |Δ| < z · SE (z = 1,96: das 95-%-Intervall des Unterschieds enthält die 0)
Dann entscheidet der Causal Forest (estimators/cate_model.py, CATEModel learner "cf", models/cate_model_cf_durchlaufzeit_real_alle.pkl):
die Option mit dem größten geschätzten Wert (Referenz 0, Aktion τ; wie CATEPolicy mit threshold 0). Kennt der Forest den Punkt oder
eine Option nicht, bleibt die PSP-Wahl.
first_only wie PSPPolicy. Je Entscheidung ein Eintrag in self.entscheidungen (PSP-Werte, Δ, SE, unsicher, PSP-, Forest- und
endgültige Wahl) für die Auswertung.
"""
import numpy as np

from prescriptive.policies.ev_policy.psp_policy import PSPPolicy

Z = 1.96


class PSPCausalForestPolicy(PSPPolicy):
    def __init__(self, model_path, forest_path, seed=42, samples=300, first_only=(), z=Z):
        super().__init__(model_path=model_path, seed=seed, samples=samples, first_only=first_only)
        from prescriptive.estimators.cate_model import CATEModel
        self.forest = CATEModel.load(forest_path)
        self.forest.set_threads(1)
        self.z = float(z)
        self.entscheidungen = []

    def choose(self, point, case, options, time, engine):
        if not self.model.knows(point) or self._repeated(point, case.prefix):
            return None
        if len(options) == 1:
            return options[0]
        from prescriptive.estimators.conditional_gaps import SuffixSamplingError
        try:
            samples = self.model.outcome_samples(point, case.prefix, options, time)  # Optionen x n, wie PSPModel.values
        except SuffixSamplingError:
            self.sampling_fallbacks = getattr(self, "sampling_fallbacks", 0) + 1
            return None
        values = samples.mean(axis=1)
        best = np.flatnonzero(values >= values.max())
        psp = int(best[0] if len(best) == 1 else self.rng.choice(best))  # wie PSPPolicy.choose
        ordnung = [int(i) for i in np.argsort(-values, kind="stable")]
        zweite = next(i for i in ordnung if i != psp)  # die beste andere Option
        d = samples[psp] - samples[zweite]
        delta = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
        unsicher = abs(delta) < self.z * se
        wahl, forest = psp, None
        if unsicher and self.forest.knows(point) and all(o in self.forest.options[point.name] for o in options):
            werte = self.forest.values(point, case.prefix, options, time)
            forest = int(np.argmax(werte))
            wahl = forest
        self.entscheidungen.append({"point": point.name, "case": case.case_id, "time": time, "delta": delta, "se": se, "unsicher": unsicher,
                                    "psp": options[psp], "forest": options[forest] if forest is not None else None, "wahl": options[wahl],
                                    **{f"wert {o}": float(v) for o, v in zip(options, values)}})
        return options[wahl]

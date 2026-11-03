"""
psp_erm: wie psp (policies/psp_policy.py), aber die Option mit dem kleinsten entropischen Risikomaß (Entropic Risk Measure)
statt dem besten Mittel. Je Option zieht das PSP samples Suffixe (alle Optionen mit denselben Zufallszahlen); der Verlust
je Suffix ist L = -Outcome, bei der Durchlaufzeit (negative_cycle_days) also die Durchlaufzeit in Tagen. Daraus

    ρ_γ(L) = (1/γ) · ln E[exp(γ · L)]     (E: Mittel über die gezogenen Suffixe)

γ > 0 ist die Risikoaversion in 1/Tag: Für γ → 0 ist ρ das Mittel (wie psp), mit wachsendem γ zählen lange Suffixe
überproportional, für γ → ∞ ist ρ der längste gezogene Suffix. Bei normalverteiltem L ist ρ = Mittel + γ · Varianz / 2,
γ = 0,1 schlägt also bei einer Streuung von 5 Tagen gut 1 Tag auf, bei 20 Tagen 20 Tage. ρ verschiebt sich mit L
(ρ(L + c) = ρ(L) + c): ob die Durchlaufzeit ab Fallbeginn oder die Restzeit zählt, ändert die Wahl nicht. Gerechnet mit
abgezogenem Maximum (kein Überlauf bei großem γ · L). Gleichstand: eine zufällige der besten.
Mit wenigen Suffixen und großem γ hängt ρ an den wenigen längsten Ziehungen und streut entsprechend.

stats: je Lauf, wie oft gefragt wurde und wie oft das ERM anders gewählt hat als das Mittel (die Policy psp mit denselben
Suffixen); online_evaluation übernimmt es in die Zeile des Laufs.

Das Modell wie psp: models/psp_model.pkl (MODEL_PATH) oder model_path (helpers/training.model_path).
"""
import os

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "psp_model.pkl")
SAMPLES = 30  # gezogene Suffixe je Option, wie psp
GAMMA = 0.1  # Risikoaversion γ in 1/Tag (scripts/main.py --erm-gamma)


def entropic_risk(losses, gamma):
    """ρ_γ je Zeile von losses (Optionen x Ziehungen): (1/γ) · ln Mittel(exp(γ · L)), mit abgezogenem Maximum."""
    scaled = gamma * np.asarray(losses, dtype=float)
    top = scaled.max(axis=1, keepdims=True)
    return (top[:, 0] + np.log(np.exp(scaled - top).mean(axis=1))) / gamma


class PSPERMPolicy:
    def __init__(self, model_path=MODEL_PATH, seed=42, samples=SAMPLES, gamma=GAMMA):
        import torch

        from prescriptive.estimators.psp_model import PSPModel
        if not gamma > 0:
            raise ValueError(f"psp_erm: γ muss größer 0 sein, nicht {gamma}")
        self.model = PSPModel.load(model_path)
        self.model.set_threads(1)  # ein Thread je Arbeitsprozess (die Läufe laufen parallel)
        self.model.samples_per_option = samples
        self.gamma = gamma
        torch.manual_seed(seed)  # die gezogenen Suffixe, je Lauf wiederholbar
        self.rng = np.random.default_rng(seed)  # nur für Gleichstände
        self.stats = {"ERM γ (1/Tag)": gamma, "Fragen (ERM)": 0, "ERM anders als das Mittel": 0}

    # Wird aufgerufen von case_engine._ask -> choose (je Wahlpunkt)
    def choose(self, point, case, options, time, engine):
        if not self.model.knows(point):
            return None
        losses = -self.model.outcome_samples(point, case.prefix, options, time)
        risk = entropic_risk(losses, self.gamma)
        best = np.flatnonzero(risk <= risk.min())
        self.stats["Fragen (ERM)"] += 1
        self.stats["ERM anders als das Mittel"] += int(not np.isin(np.argmin(losses.mean(axis=1)), best))
        return options[int(best[0] if len(best) == 1 else self.rng.choice(best))]

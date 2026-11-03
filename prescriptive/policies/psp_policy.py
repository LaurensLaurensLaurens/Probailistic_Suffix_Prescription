"""
psp: Policy nach dem PSP-Modell (estimators/psp_model.py), wie die Policies xgb und lstm, nur mit dem PSP als Modell.
Fragt die Engine an einem Wahlpunkt (simulation/case_engine.py set_policy), bekommt die Policy vom Simulator den Prefix
des Falls (case.prefix) und die Optionen. Das PSP hängt je Option das Options-Event an den Prefix (das gewählte Event bzw.
das start-Event mit dem gewählten Mitarbeiter), zieht samples Suffixe (alle Optionen mit denselben Zufallszahlen) und
schätzt den Outcome als Mittel über die Suffixe; die Policy gibt die Option mit dem höchsten zurück, bei Gleichstand eine
zufällige von ihnen. Hat das Modell den Wahlpunkt nicht, None (das gezogene Event bzw. die Zuteilungsstrategie).

Das Modell: models/psp_model.pkl (MODEL_PATH) oder model_path, gespeichert mit PSPModel.save.
"""
import os

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "psp_model.pkl")
SAMPLES = 30  # gezogene Suffixe je Option (das Ziehen ist der teure Teil der Online-Auswertung)


class PSPPolicy:
    def __init__(self, model_path=MODEL_PATH, seed=42, samples=SAMPLES):
        import torch

        from prescriptive.estimators.psp_model import PSPModel
        self.model = PSPModel.load(model_path)
        self.model.set_threads(1)  # ein Thread je Arbeitsprozess (die Läufe laufen parallel)
        self.model.samples_per_option = samples
        torch.manual_seed(seed)  # die gezogenen Suffixe, je Lauf wiederholbar
        self.rng = np.random.default_rng(seed)  # nur für Gleichstände

    # Wird aufgerufen von case_engine._ask -> choose (je Wahlpunkt)
    def choose(self, point, case, options, time, engine):
        if not self.model.knows(point):
            return None
        # Prefix: beim Control Flow mit dem Entscheidungs-Event (O_Created), beim Mitarbeiter bis vor das start
        values = self.model.values(point, case.prefix, options, time)
        best = np.flatnonzero(values >= values.max())
        return options[int(best[0] if len(best) == 1 else self.rng.choice(best))]

"""
lstm: Policy nach dem LSTM-Modell (estimators/lstm_model.py, S-Learner je Wahlpunkt), genau wie die Policy xgb
(policies/xgb_policy.py), nur mit dem LSTM als Modell. Fragt die Engine an einem Wahlpunkt (simulation/case_engine.py
set_policy), bekommt die Policy vom Simulator den Prefix des Falls (case.prefix: die Events bis jetzt) und die Optionen
(Control Flow: die Aktivitäten, Mitarbeiter: die jetzt kompetenten, freien). Das Modell schätzt je Option den Outcome des
Falls, die Policy gibt die Option mit dem höchsten zurück; liegen mehrere gleichauf (z. B. Mitarbeiter, die im Training
nie am Wahlpunkt waren), eine zufällige von ihnen. Hat das Modell für den Wahlpunkt kein Netz, None (das gezogene Event
bzw. die Zuteilungsstrategie).

Das Modell: models/lstm_model.pkl (MODEL_PATH) oder model_path, gespeichert mit LSTMModel.save.
"""
import os

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "lstm_model.pkl")


class LSTMPolicy:
    def __init__(self, model_path=MODEL_PATH, seed=42):
        from prescriptive.estimators.lstm_model import LSTMModel
        self.model = LSTMModel.load(model_path)
        self.model.set_threads(1)  # ein Thread je Arbeitsprozess (die Läufe laufen parallel)
        self.rng = np.random.default_rng(seed)  # nur für Gleichstände

    # Wird aufgerufen von case_engine._ask -> choose (je Wahlpunkt)
    def choose(self, point, case, options, time, engine):
        if not self.model.knows(point):
            return None
        # Prefix: beim Control Flow mit dem Entscheidungs-Event (O_Created), beim Mitarbeiter bis vor das start
        values = self.model.values(point, case.prefix, options, time)
        best = np.flatnonzero(values >= values.max())
        return options[int(best[0] if len(best) == 1 else self.rng.choice(best))]

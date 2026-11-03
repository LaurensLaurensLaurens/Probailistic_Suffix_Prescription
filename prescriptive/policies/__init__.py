"""
Die Policies der Simulation, je Policy eine Datei. Jede hat eine Methode
    choose(point, case, options, time, engine) -> eine der options oder None (wie die Bank),
die die Engine an den Wahlpunkten fragt (interventions.choice_points: Control Flow der Intervention und Mitarbeiterwahl;
simulation/case_engine.py set_policy); case.prefix sind die Events des Falls bis jetzt.

rct_policy.py     rct: zufällige Option an jedem Wahlpunkt (Trainingsdaten), mit delta je Fall wie die Bank
xgb_policy.py     xgb: Option mit dem höchsten geschätzten Outcome nach XGBoost (models/xgb_model.pkl)
lstm_policy.py    lstm: wie xgb, mit dem LSTM als Modell (models/lstm_model.pkl)
psp_policy.py     psp: wie xgb, mit dem PSP als Modell: Outcome aus gezogenen Suffixen (models/psp_model.pkl)
psp_lookahead_policy.py
                  psp_lookahead: Mitarbeiterwahl nach dem PSP mit Blick auf die anderen laufenden Fälle (engine.other_cases):
                  den besten Mitarbeiter nicht nehmen, wenn ein Fall, der ihn bald braucht, ihn im Erwartungswert besser
                  nutzt; sind alle besser bei anderen aufgehoben oder lohnt es sich selbst, warten (interventions.WAIT)
psp_erm_policy.py psp_erm: wie psp, aber die Option mit dem kleinsten entropischen Risikomaß der gezogenen Durchlaufzeiten
                  (Risikoaversion γ) statt dem besten Mittel
"""
from prescriptive.policies.lstm_policy import LSTMPolicy
from prescriptive.policies.psp_erm_policy import PSPERMPolicy
from prescriptive.policies.psp_lookahead_policy import PSPLookaheadPolicy
from prescriptive.policies.psp_policy import PSPPolicy
from prescriptive.policies.rct_policy import RCTPolicy
from prescriptive.policies.xgb_policy import XGBPolicy

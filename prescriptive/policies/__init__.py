"""
Die Policies der Simulation, je Policy eine Datei. Jede hat eine Methode
    choose(point, case, options, time, engine) -> eine der options oder None (wie die Bank),
die die Engine an den Wahlpunkten fragt (interventions.choice_points: Control Flow der Intervention und Mitarbeiterwahl;
simulation/case_engine.py set_policy); case.prefix sind die Events des Falls bis jetzt.

rct_policy.py     rct: zufällige Option an jedem Wahlpunkt (Trainingsdaten), mit delta je Fall wie die Bank
xgb_policy.py     xgb: Option mit dem höchsten geschätzten Outcome nach XGBoost (models/xgb_model.pkl)
lstm_policy.py    lstm: wie xgb, mit dem LSTM als Modell (models/lstm_model.pkl)
ev_policy/        Policies, die die Option mit dem besten geschätzten Erwartungswert des Outcomes nehmen:
  psp_policy.py   psp: wie xgb, mit dem PSP als Modell: Outcome aus gezogenen Suffixen (models/psp_model.pkl)
  cate_policy.py  s_learner, x_learner, causal_forest: die kausalen Schätzer nach Bozorgi et al. (2023) als Vergleich zum PSP;
                  die Aktion, wenn ihr geschätzter Effekt gegenüber der Referenz über der Schwelle liegt (estimators/cate_model.py)
psp_lookahead_policy.py
                  psp_lookahead: Mitarbeiterwahl nach dem PSP mit Blick auf die anderen laufenden Fälle (engine.other_cases):
                  den besten Mitarbeiter nicht nehmen, wenn ein Fall, der ihn bald braucht, ihn im Erwartungswert besser
                  nutzt; sind alle besser bei anderen aufgehoben oder lohnt es sich selbst, warten (interventions.WAIT)
accepted_policy/
  psp_accepted_policy.py psp_annahme: das PSP nach der Angebotsannahme: die Option mit dem höchsten Anteil gezogener Suffixe
                  mit O_Accepted (Ziel wie screening/results_accepted)
erm_policy/
  psp_erm_policy.py psp_erm: wie psp, aber die Option mit dem kleinsten entropischen Risikomaß der gezogenen Durchlaufzeiten
                  (Risikoaversion γ) statt dem besten Mittel
  cate_policy.py s_learner_erm, x_learner_erm, causal_forest_erm: Effekte auf den exponentiellen Verlust,
                  die Option mit dem kleinsten Wert (estimators/cate_erm_model.py, neues Training je γ)
"""
from prescriptive.policies.erm_policy.cate_policy import CATEERMPolicy
from prescriptive.policies.erm_policy.psp_erm_policy import PSPERMPolicy
from prescriptive.policies.psp_lookahead_policy import PSPLookaheadPolicy
from prescriptive.policies.ev_policy.cate_policy import CATEPolicy
from prescriptive.policies.ev_policy.psp_policy import PSPPolicy
from prescriptive.policies.rct_policy import RCTPolicy
from prescriptive.policies.xgb_policy import XGBPolicy


def __getattr__(name):
    # LSTM erst beim Zugriff laden, damit die übrigen Policies unabhängig von ihrer Verfügbarkeit importierbar sind.
    if name == "LSTMPolicy":
        from prescriptive.policies.lstm_policy import LSTMPolicy
        return LSTMPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

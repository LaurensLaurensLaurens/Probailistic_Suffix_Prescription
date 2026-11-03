"""
Der Code der Modelle, die die Policies fragen (prescriptive/policies/), je Modell eine Datei; die gespeicherten Modelle
liegen in prescriptive/models/.

xgb_model.py        XGBModel: XGBoost als S-Learner je Wahlpunkt, Outcome je Option direkt geschätzt (policies/xgb_policy.py)
lstm_model.py       LSTMModel: dasselbe mit einem LSTM über die Event-Sequenz (policies/lstm_policy.py)
dr_learner.py       DR-Learner (AIPW als Schätzer), offline bewertet
"""

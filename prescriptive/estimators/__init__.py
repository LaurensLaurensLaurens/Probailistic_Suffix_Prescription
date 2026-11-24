"""
Der Code der Modelle, die die Policies fragen (prescriptive/policies/), je Modell eine Datei; die gespeicherten Modelle
liegen in prescriptive/models/.

xgb_model.py        XGBModel: XGBoost als S-Learner je Wahlpunkt, Outcome je Option direkt geschätzt (policies/xgb_policy.py)
lstm_model.py       LSTMModel: dasselbe mit einem LSTM über die Event-Sequenz (policies/lstm_policy.py)
dr_learner.py       DR-Learner (AIPW als Schätzer), offline bewertet
cate_model.py       CATEModel: S-Learner, X-Learner und Causal Forest nach Bozorgi et al. (2023), Effekt der Aktion gegenüber
                    der Referenz je Control-Flow-Wahlpunkt (policies/ev_policy/cate_policy.py)
cate_erm_model.py   CATEERMModel: dieselben Verfahren auf exponentiellem Verlust für das entropische Risikomaß
                    (policies/erm_policy/cate_policy.py), eigenes Training je γ
"""

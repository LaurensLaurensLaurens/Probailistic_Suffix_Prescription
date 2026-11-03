"""
Helpers für prescriptive_guide.ipynb, in der Reihenfolge des Notebooks:

    datasets.py        Datensatz wählen und laden (Pfade, Katalog prescriptive/catalog.yaml, Roh-Log)
    preprocessing/     Preprocessing je Datensatz
        steps.py       die Maßnahmen (für alle Datensätze und je Datensatz)
        outcome.py     Outcome je Fall (Spalte outcome); auch die Gewinn-Logik der BPIC17-Pipeline (prescriptive/helpers/preprocessing/BPIC2017/outcome.py)
    screening/         Screening der Control-Flow-Kandidaten
        data.py        Datensatz je Kandidat (Merkmale X, Aktion T, Outcome Y)
        causal.py      Schätzer mit XGBoost und Cross-Fitting (Propensity, AIPW, DR-Learner, BLP)
        evaluation.py  Kennzahlen und Ergebnis je Kandidat
    simulator_check.py Simulator (process-simulation-engine/simulation/dataset_simulator.py) gegen den echten Log
    rct.py             RCT-Daten auf dem Simulator (Trainingsdaten der Schätzer), wie main.py gespeichert
    cases.py           ein zufälliger echter Fall je Datensatz (mit Entscheidungspunkt)
    training.py        Training der Schätzer (PSP, XGBoost, LSTM) und ihre Kennzahlen
    overview.py        Lagebild des PSP auf dem Simulator
    report.py          Ausgabe von scripts/main.py: Tabellen, Diagramme, bericht.html
    parameters.py      Standardwerte der Parameter, die für alle Datensätze gelten
Die Auswertungen der Modelle (offline, online) liegen in prescriptive/evaluation/.
"""

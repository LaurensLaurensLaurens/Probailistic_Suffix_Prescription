"""
Screening der Kandidaten eines Datensatzes (Control Flow und Variablen), eine Offline-Auswertung auf dem beobachteten Log:

    events.py       Entscheidungs-Event und Aktion je Fall
    features.py     Merkmale X am Entscheidungspunkt
    candidates.py   Datensatz je Kandidat (CandidateData mit X, Option T, Outcome Y), je Art (kind) des Kandidaten
    causal.py       Schätzer mit XGBoost und Cross-Fitting (Propensity, AIPW, DR-Learner, BLP)
    evaluation.py   die drei Kennzahlen, Ergebnis-Regel, Tabelle
    example.py      ein Beispiel: ein zufälliger Fall eines Kandidaten, wie das Screening ihn sieht

    from helpers.screening import screen_control_flow, screen_variables
    results = screen_control_flow(DATASET, log)
    variables = screen_variables(DATASET, log)

    # oder in Schritten:
    data = build_candidates(DATASET, log, art="variable")    # {Kandidat: CandidateData}
    results = screen(data, show_options=True)
"""
from .candidates import KINDS, CandidateData, build_candidate, build_candidates
from .example import candidate_example, show_example
from .evaluation import evaluate, screen, screen_control_flow, screen_variables, show_screening

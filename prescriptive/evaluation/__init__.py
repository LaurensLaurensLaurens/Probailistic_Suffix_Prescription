"""
Auswertungen der Modelle.

Für prescriptive_guide.ipynb und scripts/main.py (Wahl des Mitarbeiters am Arbeitsschritt der Intervention):
model_evaluation.py     offline: AIPW auf dem echten Log an einem wählbaren Interventionspunkt, gegen Bank, Zufall, DR-Learner
online_evaluation.py    online: das Modell wählt live im Simulator (Engine), gegen Bank und Zufall

Für prescriptive/main.py (Entscheidung am Entscheidungspunkt), auf gehaltenen RCT-Fällen, schneller als die gemeinsame
Simulation:
offline_evaluation.py   selbstnormiertes IPW auf den echten Ergebnissen der RCT-Fälle (alle Datensätze mit RCT-Daten);
                        dazu init_worker, mit dem auch model_evaluation.py und das Lagebild das PSP in Arbeitsprozesse laden
oracle_evaluation.py    der Simulator als Wahrheit: je Fall und Aktion simulierte Fortsetzungen (nur ohne Engine, BPIC13)
"""

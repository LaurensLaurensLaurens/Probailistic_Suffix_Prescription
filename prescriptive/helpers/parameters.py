"""
Parameter, die für alle Datensätze gelten: die Standardwerte. Im Notebook wählbar (Zelle "Simulator aufstellen") und
von dort weitergegeben; ohne Angabe gelten diese Werte (auch für die Pipeline in prescriptive/). Was je Datensatz
anders ist, steht im Katalog (prescriptive/catalog.yaml).
"""
# Kostensätze des Outcomes. Sie wirken dort, wo das Outcome sie nutzt (profit, BPIC17 als Option gewinn); die Durchlaufzeit
# (negative_cycle_days, Standard) kennt keine Kosten. Die Kosten der Aktion selbst (BPIC17: call_cost je Anruf) sind datensatzspezifisch und
# bleiben im Katalog.
HOURLY_RATE = 50.0  # € je Arbeitsstunde (Vollkosten eines Sachbearbeiters)
DAILY_COST = 10.0  # € je Tag Durchlaufzeit (Kapitalbindung, Aufwand, Zufriedenheit des Kunden)
COSTS = {"hourly_rate": HOURLY_RATE, "daily_cost": DAILY_COST}

# Simulation (Simulator-Check, Evaluation)
N_CASES = 2000  # simulierte Fälle je Lauf
SEED = 0  # Zufall der Simulation

# RCT-Daten auf dem Simulator (Trainingsdaten der Schätzer, helpers/rct.py; dieselben Standardwerte wie main.py)
RCT_CASES = None  # Fälle je Lauf (simuliertes Jahr); None: so viele, wie der echte Log Fälle hat
RCT_RUNS = 3  # Läufe (simulierte Jahre), je mit eigenem seed
RCT_SEED = 42  # Lauf r zieht mit seed RCT_SEED + 1000 + r
DELTA = 0.0  # Anteil der ersten Entscheidungspunkte, die wie die Bank statt zufällig entschieden werden (SCOPE: confounding "point")
WORKERS = 8  # parallele Simulationsprozesse (Engine: ein Lauf je Prozess)

# Training der Modelle auf den RCT-Daten (helpers/training.py)
TRAIN_SEED = 42  # Zufall des Trainings

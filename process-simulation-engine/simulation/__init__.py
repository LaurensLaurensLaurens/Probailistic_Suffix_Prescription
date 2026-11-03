"""
Die Simulatoren.

dataset_simulator.py               Einstieg: Simulator(datensatz, log).run(n, policy, points, window), wählt die Engine des Datensatzes
case_engine_lifecycle_aware.py     die Simulation mit Arbeitsschritten: alle Fälle konkurrieren, Dienstplan, Berechtigungen, Zuteilung (BPIC17)
case_engine_lifecycle_unaware.py   die Simulation nur mit complete-Events: jeder Fall für sich, das Modell zieht auch die Mitarbeiter (BPIC19)
repair_engine.py                   die synthetische Reparaturwerkstatt: kein gelerntes Modell, die Regeln sind der Prozess; dieselbe Schnittstelle
repair_workshop.py                 die Werkstatt als Datensatz repair (Log eventlog/repair/, Katalog): größer, Gerätetypen, Mechaniker je Typ
                                   verschieden, Wahl bei W_repair; die Wahrheit für Policies aus dem gelernten Simulator
"""

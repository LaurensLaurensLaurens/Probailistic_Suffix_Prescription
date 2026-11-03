"""
BPIC17: was es nur für die Mitarbeiter dieses Datensatzes gibt (der einzige, der in der Engine mit Mitarbeitern läuft).

resource_permissions/    OrdinoR (ordinor.py) mit dem gespeicherten Modell ordinor_fullrecall.pkl; dazu die Analysen, mit
                         denen die Methode gewählt wurde (benchmark_permissions.py, Berichte, visualizations/, Notizen)
resource_optimization/   pmsp: Zuteilung aller wartenden Aufgaben zusammen (CP-SAT)
allocation/              die Zuteilungen außer der Grundklasse: allocations.py (bank, batch, drl, pmsp),
                         learned_allocation.py, batch_policies.py,
                         drl_allocation/ (Zustand), processing_time_estimator.py (erwartete Arbeitszeit)
"""

"""
Die Mitarbeiter: wer eine Tätigkeit darf, wer wann im Dienst ist, wer welche Aufgabe bekommt (Engine
simulation/case_engine.py, Auswertungen). Hier steht, was mehrere Datensätze nutzen; was nur einer nutzt, im Ordner des
Datensatzes. Was ein Datensatz wählt, steht in datasets.py (RESOURCES).

datasets.py               je Datensatz (RESOURCES); dataset_resources(): Berechtigungen, Dienstplan, Systemnutzer
daily_roster.py           Dienstplan: im Dienst von der ersten bis zur letzten Stunde mit eigenen Events am Tag (AlwaysOnDuty: immer, die Werkstatt)
resource_permissions/     Berechtigungen performed (wer es im Log getan hat; BPIC19) und die Datenvorbereitung
case_allocation.py        Grundklasse der Zuteilung (zufällig unter den freien Berechtigten), create_allocation
selection_strategies.py   Auswahl aus dem Pool: random, round_robin, shortest_queue (Engine und Resources.select)
BPIC2017/                 resource_permissions/ (OrdinoR, gespeichertes Modell, Analysen), resource_optimization/
                          (pmsp), allocation/ (bank, learned, batch, drl)
BPIC2013/                 resource_permissions.py: Berechtigungen über die Support-Teams
"""
from .datasets import RESOURCES, Resources, dataset_resources, resource_spec, system_resources, system_users

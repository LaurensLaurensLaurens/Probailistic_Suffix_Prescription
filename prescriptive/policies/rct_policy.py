"""
rct: die Trainingsdaten der Modelle (helpers/rct.py). An jedem Wahlpunkt eine zufällige Option (Control Flow: eine der
Aktivitäten, Mitarbeiter: einer der jetzt kompetenten, freien), damit jede Option in jeder Lage vorkommt (Positivity).
Wie delta in SCOPE (confounding_type "point"): je Fall entscheidet mit Wahrscheinlichkeit delta an allen Wahlpunkten
die Bank (None: das gezogene Event bzw. die Zuteilungsstrategie), sonst der Zufall; gelost beim ersten Wahlpunkt des
Falls. randomized: die Fälle mit Zufall.
"""
import numpy as np


class RCTPolicy:
    def __init__(self, seed, delta=0.0):
        self.rng = np.random.default_rng(seed)
        self.delta = delta
        self.randomized, self.bank = set(), set()

    # Wird aufgerufen von case_engine._ask -> choose (je Wahlpunkt)
    def choose(self, point, case, options, time, engine):
        if case.case_id not in self.randomized and case.case_id not in self.bank:
            (self.bank if self.rng.random() < self.delta else self.randomized).add(case.case_id)
        if case.case_id in self.bank:
            return None
        return options[int(self.rng.integers(len(options)))]

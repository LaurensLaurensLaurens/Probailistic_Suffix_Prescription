"""
psp_annahme: das PSP entscheidet nach der Angebotsannahme statt nach der Durchlaufzeit (2026-11-12). Ziel wie im Screening
screening/results_accepted: Erreicht der Fall später ein O_Accepted (mindestens ein Angebot angenommen)?

An einem Wahlpunkt zieht das PSP je Option samples Suffixe (estimators/psp_model.py option_suffixes: die Option als
nächstes Event, alle Optionen mit denselben Zufallszahlen) und schätzt je Option die Annahmewahrscheinlichkeit als
Anteil der gezogenen Suffixe, die ein Event TARGET enthalten. Die Policy nimmt die Option mit dem höchsten Anteil, bei
Gleichstand eine zufällige von ihnen. Bewertet wird nur über die gezogenen Suffixe (keine Werte aus dem Log, kein Kopf).

None (wie die Bank), wenn
- das Modell den Wahlpunkt nicht hat,
- der Fall das Ziel schon erreicht hat (TARGET im Prefix; dann hängt es nicht mehr von der Wahl ab, wie im Screening
  ausgeschlossen),
- first_only den Punkt nur bei der ersten Entscheidung im Fall zulässt und es eine spätere ist.

Vorsicht: Endet ein gezogenes Suffix zu früh (EOS ohne Fallende; PSP (c): 39–47 % der Suffixe), zählt es als "nicht
angenommen". Die Anteile liegen dann unter den echten, der Vergleich der Optionen bleibt nur fair, wenn das bei beiden
Optionen gleich oft passiert.

stats: Fragen, davon Ziel schon erreicht; online_evaluation übernimmt sie in die Zeile des Laufs.
"""
import numpy as np

ACTIVITY, LIFECYCLE = "concept:name", "lifecycle:transition"  # wie helpers/datasets.py
TARGET = "O_Accepted"
SAMPLES = 100  # gezogene Suffixe je Option (wie der Test von PSP (c))


def reached(suffix, target=TARGET):
    """Enthält ein gezogenes Suffix (Liste von (Token, Zeit seit Fallbeginn, Abstand)) das Ziel-Event?"""
    return any(str(label).split("|")[0] == target for label, _, _ in suffix)


class PSPAcceptedPolicy:
    def __init__(self, model_path, seed=42, samples=SAMPLES, target=TARGET, first_only=()):
        import torch

        from prescriptive.estimators.psp_model import PSPModel
        self.model = PSPModel.load(model_path)
        self.model.set_threads(1)  # ein Thread je Arbeitsprozess (die Läufe laufen parallel)
        self.model.samples_per_option = int(samples)
        self.target = target
        self.first_only = set(first_only or ())
        torch.manual_seed(seed)  # die gezogenen Suffixe, je Lauf wiederholbar
        self.rng = np.random.default_rng(seed)  # nur für Gleichstände
        self.stats = {"Fragen (Annahme)": 0, "Ziel schon erreicht": 0}

    def values(self, point, events, options, now):
        """Je Option der Anteil der gezogenen Suffixe mit dem Ziel-Event (Array in der Reihenfolge von options)."""
        drawn, _ = self.model.option_suffixes(point, events, options, now, self.model.samples_per_option)
        return np.array([np.mean([reached(suffix, self.target) for suffix in suffixes]) for suffixes in drawn])

    # Wird aufgerufen von case_engine._ask -> choose (je Wahlpunkt)
    def choose(self, point, case, options, time, engine):
        if not self.model.knows(point) or self._repeated(point, case.prefix):
            return None
        self.stats["Fragen (Annahme)"] += 1
        if any(event.get(ACTIVITY) == self.target for event in case.prefix):
            self.stats["Ziel schon erreicht"] += 1
            return None
        values = self.values(point, case.prefix, options, time)
        best = np.flatnonzero(values >= values.max())
        return options[int(best[0] if len(best) == 1 else self.rng.choice(best))]

    def _repeated(self, point, prefix):
        """first_only: steht das Entscheidungs-Event des Punkts (zusammengelegte Punkte a+b: einer der Teile) schon vor dem aktuellen im Prefix?"""
        if not self.first_only or not set(point.name.split("+")) & self.first_only:
            return False
        return sum(1 for event in prefix if event.get(ACTIVITY) == point.activity and event.get(LIFECYCLE, "complete") == point.lifecycle) > 1

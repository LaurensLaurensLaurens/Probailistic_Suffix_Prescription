"""
psp: Policy nach dem PSP-Modell (estimators/psp_model.py), wie die Policies xgb und lstm, nur mit dem PSP als Modell.
Fragt die Engine an einem Wahlpunkt (simulation/case_engine.py set_policy), bekommt die Policy vom Simulator den Prefix
des Falls (case.prefix) und die Optionen. Das PSP hängt je Option das Options-Event an den Prefix (das gewählte Event bzw.
das start-Event mit dem gewählten Mitarbeiter), zieht samples Suffixe (alle Optionen mit denselben Zufallszahlen) und
schätzt den Outcome als Mittel über die Suffixe; die Policy gibt die Option mit dem höchsten zurück, bei Gleichstand eine
zufällige von ihnen. Hat das Modell den Wahlpunkt nicht, None (das gezogene Event bzw. die Zuteilungsstrategie).

Das Modell: models/psp_model.pkl (MODEL_PATH) oder model_path, gespeichert mit PSPModel.save.

first_only (Wahlpunkte, Standard keine): an diesen Punkten entscheidet die Policy nur beim ersten Mal im Fall (das
Entscheidungs-Event steht nur einmal im Prefix), danach None wie die Bank. Gegen die Schleife bei angebot_nachlegen
(2026-11-10): das PSP legte ab dem zweiten Angebot immer öfter nach (Zusammenhang aus den Beobachtungsdaten, kein Effekt),
jedes neue Angebot führt wieder an den Punkt, die Fälle landen bei mehr Angeboten, als der echte Log kennt.
"""
import os

import numpy as np

ACTIVITY, LIFECYCLE = "concept:name", "lifecycle:transition"  # wie helpers/datasets.py

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "psp_model.pkl")
SAMPLES = 30  # gezogene Suffixe je Option (das Ziehen ist der teure Teil der Online-Auswertung)


class PSPPolicy:
    def __init__(self, model_path=MODEL_PATH, seed=42, samples=SAMPLES, value_from=None, first_only=()):
        import torch

        from prescriptive.estimators.psp_model import PSPModel
        self.model = PSPModel.load(model_path)
        self.model.set_threads(1)  # ein Thread je Arbeitsprozess (die Läufe laufen parallel)
        self.model.samples_per_option = samples
        if value_from:  # "suffix" (gezogene Suffixe) oder "head" (Restzeit-Kopf), siehe psp_model.value_source
            self.model.value_from = value_from
            self.model.value_source()
        self.first_only = set(first_only or ())
        torch.manual_seed(seed)  # die gezogenen Suffixe, je Lauf wiederholbar
        self.rng = np.random.default_rng(seed)  # nur für Gleichstände

    # Wird aufgerufen von case_engine._ask -> choose (je Wahlpunkt)
    def choose(self, point, case, options, time, engine):
        if not self.model.knows(point) or self._repeated(point, case.prefix):
            return None
        if len(options) == 1:
            return options[0]
        # Prefix: beim Control Flow mit dem Entscheidungs-Event (O_Created), beim Mitarbeiter bis vor das start
        from prescriptive.estimators.conditional_gaps import SuffixSamplingError
        try:
            base = getattr(self.model, "base", self.model)
            control = (getattr(base, "params", {}) or {}).get("sampling_control")
            if control and len(options) == 2:
                counts = sorted(set([self.model.base.samples_per_option if hasattr(self.model, "base") else self.model.samples_per_option,
                                     int(control["max_samples"])]))
                self.decision_audit = getattr(self, "decision_audit", {"adaptive_calls": 0, "uncertain_fallbacks": 0,
                    "samples_per_option_total": 0, "expanded_calls": 0})
                self.decision_audit["adaptive_calls"] += 1
                for index, count in enumerate(counts):
                    samples = self.model.outcome_samples(point, case.prefix, options, time, n=count)
                    values = samples.mean(axis=1)
                    delta = samples[0]-samples[1]
                    se = delta.std(ddof=1)/np.sqrt(count) if count > 1 else np.inf
                    self.decision_audit["samples_per_option_total"] += count
                    self.decision_audit["expanded_calls"] += int(index > 0)
                    if abs(values[0]-values[1]) > float(control.get("z", 2.))*se:
                        break
                else:
                    self.decision_audit["uncertain_fallbacks"] += 1
                    return None
            else:
                values = self.model.values(point, case.prefix, options, time)
        except SuffixSamplingError:
            self.sampling_fallbacks = getattr(self, "sampling_fallbacks", 0)+1
            return None
        best = np.flatnonzero(values >= values.max())
        return options[int(best[0] if len(best) == 1 else self.rng.choice(best))]

    def _repeated(self, point, prefix):
        """first_only: steht das Entscheidungs-Event des Punkts (zusammengelegte Punkte a+b: einer der Teile) schon vor dem aktuellen im Prefix?"""
        if not self.first_only or not set(point.name.split("+")) & self.first_only:
            return False
        return sum(1 for event in prefix if event.get(ACTIVITY) == point.activity and event.get(LIFECYCLE, "complete") == point.lifecycle) > 1

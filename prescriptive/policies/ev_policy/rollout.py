"""
Rollout mit Causal Forest an späteren Wahlpunkten (2026-11-23, Nutzer: „Setze eine Version auf, die mittels Causal Forest die späteren
Entscheidungen in den Suffixen korrigiert: Rollout-Weg, späteres Entscheiden einsetzen“).

Bisher ziehen die PSP-Netze nach der Option den Rest des Falls so, wie die Bank im Log entschieden hat – auch an späteren Wahlpunkten.
Mit Rollout entscheidet dort ein Causal Forest (estimators/cate_model.py, CATEModel learner "cf"): erreicht ein gezogener Suffix den
Auslöser eines Wahlpunkts (Aktivität|Lifecycle des Punkts, z. B. O_Created|complete bei angebot_nachlegen), wird das nächste Event auf
die Option gesetzt, die der Forest für diesen Teilablauf wählt (Merkmale wie XGBModel/CATEModel: Prefix des Falls + Options-Event +
gezogene Events mit ihren Zeiten). Nur das Event wird erzwungen; seine Zeit und alles danach zieht der Decoder wie sonst (das Netz bleibt
das der aktuellen Option). first_only-Punkte (angebot_nachlegen) nur beim ersten Besuch im Fall (Prefix + gezogener Teil), wie online.

Einschalten: ForestRollout.einschalten(model, forest_pfad, first_only) setzt model.rollout (bei Hüllen auch in allen Netzen);
PSPModel._options_in_encoder gibt dann draw_cases eine force-Funktion (kontext) mit. Ohne model.rollout ändert sich nichts.
"""
from collections import Counter

import numpy as np
import pandas as pd

ACTIVITY, LIFECYCLE, TIMESTAMP = "concept:name", "lifecycle:transition", "time:timestamp"


class ForestRollout:
    def __init__(self, forest_path, points, first_only=()):
        from interventions import option_token
        from prescriptive.estimators.cate_model import CATEModel
        self.forest = CATEModel.load(forest_path)
        self.forest.set_threads(1)
        self.first_only = set(first_only or ())
        self.punkte = {name: p for name, p in points.items() if self.forest.knows(p)}
        self.ausloeser = {f"{p.activity}|{p.lifecycle}": p for p in self.punkte.values()}
        # Spalten der Forest-Werte in der Reihenfolge seiner Optionen (Referenz zuerst)
        self.optionen = {name: [option_token(o) for o in self.forest.options[name]] for name in self.punkte}
        self.eingriffe, self.je_punkt, self.wahlen = 0, Counter(), Counter()

    @classmethod
    def einschalten(cls, model, forest_path, first_only=()):
        ro = cls(forest_path, model.points, first_only)
        for m in [model] + list(dict.fromkeys(getattr(model, "netze", {}).values())):
            m.rollout = ro
        return ro

    @staticmethod
    def ausschalten(model):
        for m in [model] + list(dict.fromkeys(getattr(model, "netze", {}).values())):
            m.__dict__.pop("rollout", None)

    def _prefix(self, events):
        from prescriptive.estimators.xgb_model import Prefix, _seconds
        f = self.forest
        prefix = Prefix(f._attributes(events[0]), f.event_numeric)
        for event, seconds in zip(events, _seconds([e[TIMESTAMP] for e in events])):
            prefix.add(f"{event[ACTIVITY]}|{event.get(LIFECYCLE, 'complete')}", seconds, {name: event.get(name) for name in f.event_numeric})
        return prefix

    def kontext(self, cases, option_token, reached, n):
        """force-Funktion für draw_cases (Zeilen = Fälle × n, Zeile k gehört zu Fall k // n). reached: Zeit seit Fallbeginn am Options-Event
        je Zeile (Sekunden). Rückgabe je Schritt {Zeile: erzwungenes Token} für den nächsten Schritt."""
        from prescriptive.estimators.xgb_model import _seconds
        starts = [float(_seconds([events[0][TIMESTAMP]])[0]) for events in cases]
        prefix_tokens = [[f"{e[ACTIVITY]}|{e.get(LIFECYCLE, 'complete')}" for e in events] for events in cases]
        reached = np.asarray(reached, dtype=float)

        def force(labels, active, suffixes):
            anfragen = {}
            for k in active:
                p = self.ausloeser.get(str(labels[k]))
                if p is None:
                    continue
                c = k // n
                if p.name in self.first_only:
                    bisher = prefix_tokens[c] + [option_token] + [t for t, _, _ in suffixes[k]]
                    if bisher.count(str(labels[k])) > 1:
                        continue
                anfragen.setdefault(p.name, []).append(k)
            out = {}
            for name, zeilen in anfragen.items():
                p = self.punkte[name]
                rows, nows = [], []
                for k in zeilen:
                    c = k // n
                    prefix = self._prefix(cases[c])
                    prefix.add(option_token, starts[c] + reached[k], {})
                    for token, seit, _ in suffixes[k]:
                        prefix.add(str(token), starts[c] + float(seit), {})
                    jetzt = starts[c] + float(suffixes[k][-1][1])
                    rows.append(prefix.row(jetzt))
                    nows.append(jetzt)
                werte = np.column_stack([np.zeros(len(rows)), self.forest.effects(p, rows)])  # Referenz 0, Aktion τ (höher ist besser)
                for k, w in zip(zeilen, werte):
                    wahl = self.optionen[name][int(np.argmax(w))]
                    out[k] = wahl
                    self.je_punkt[name] += 1
                    self.wahlen[wahl] += 1
                self.eingriffe += len(zeilen)
            return out

        return force

"""
konformanz: Policy, die an einem Wahlpunkt die Option mit den wenigsten erwarteten Conformance-Abweichungen wählt (2026-11-20, Nutzer:
„prüfen, ob auch Conformance-Abweichungen verhindert werden können durch das PSP … Zielgröße erstmal separat in eigener Policy in
policies“). Vorgehen wie Mustroph (Conformance Checking Mustroph/src/conformance_checking/conformance.py): gezogene Suffixe des PSP
gegen ein normatives Modell ausrichten (Alignments, pm4py).

Je Option zieht das PSP (estimators/psp_model.py bzw. die T-Learner-Hülle) samples Suffixe mit Event und Zeit je Schritt
(option_suffixes, gemeinsame Zufallszahlen). Jede Spur = Prefix des Falls (case.prefix, bis einschließlich Entscheidungs-Event) +
gezogener Suffix (beginnt mit dem Options-Event; mit_option), abgebildet auf die Aktivitätsebene des Modells (A_* und O_* mit lifecycle complete, W_* mit start;
wie kuenstliche_effekte_20261120/conformance_probe.py). Abweichungen je Spur = Zahl der Log- und Modellzüge des optimalen Alignments
(pm4py-Standardkosten: Abweichung 10.000, stiller Übergang 1 → Kosten // 10.000). Bewertung der Option: Mittel über ihre Suffixe;
gewählt wird die Option mit dem kleinsten Mittel. Unterscheiden sich die Mittel um weniger als toleranz, entscheidet die Bank (None).
Dauer spielt keine Rolle (eigene Zielgröße; eine Kombination mit der Durchlaufzeit wäre ein eigener Schritt).

Modell: process_models/BPIC17_ToBe_angepasst.bpmn (MODELL), gebaut von kuenstliche_effekte_20261120/modell_anpassen.py.
first_only wie PSPPolicy. Gleiche Spuren werden nur einmal ausgerichtet (Zwischenspeicher je Prozess).
"""
from functools import lru_cache
import os

import numpy as np

from prescriptive.policies.ev_policy.psp_policy import ACTIVITY, LIFECYCLE, PSPPolicy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELL = os.path.join(ROOT, "process_models", "BPIC17_ToBe_angepasst.bpmn")
SAMPLES = 100
ABWEICHUNG = 10000  # pm4py: Kosten eines Log- oder Modellzugs


def sichtbar(aktivitaet: str, lifecycle: str) -> bool:
    """Gehört das Event auf die Aktivitätsebene des Modells? A_/O_ complete, W_ start."""
    lc = str(lifecycle).lower()
    return (aktivitaet.startswith(("A_", "O_")) and lc == "complete") or (aktivitaet.startswith("W_") and lc == "start")


def projektion_events(events) -> tuple:
    """Prefix (Liste von Event-dicts wie engine.events) → Aktivitäten auf Modellebene."""
    return tuple(e[ACTIVITY] for e in events if sichtbar(str(e[ACTIVITY]), e.get(LIFECYCLE, "complete")))


def projektion_tokens(tokens) -> tuple:
    """Tokens Aktivität|Lifecycle (gezogene Suffixe) → Aktivitäten auf Modellebene; Sondertokens (ohne |) fallen weg."""
    out = []
    for token in tokens:
        if "|" not in str(token):
            continue
        aktivitaet, lifecycle = str(token).rsplit("|", 1)
        if sichtbar(aktivitaet, lifecycle):
            out.append(aktivitaet)
    return tuple(out)


def mit_option(option_token: str, labels: list) -> list:
    """Spur ab der Option: die gezogenen Suffixe beginnen schon mit dem Options-Event (option_as encoder; geprüft 2026-11-21) – nur
    voranstellen, wenn es fehlt. (Bis 2026-11-21 wurde es immer vorangestellt: Option doppelt, bei Aktions-Optionen eine falsche
    Abweichung je Spur.)"""
    return labels if labels and labels[0] == option_token else [option_token] + list(labels)


@lru_cache(maxsize=4)
def _netz(pfad):
    import pm4py
    return pm4py.convert_to_petri_net(pm4py.read_bpmn(pfad))


@lru_cache(maxsize=200000)
def abweichungen(spur: tuple, pfad: str = MODELL) -> int:
    """Zahl der Log- und Modellzüge des optimalen Alignments der Spur (Aktivitäten) gegen das Modell."""
    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
    from pm4py.objects.log.obj import Event, Trace
    net, im, fm = _netz(pfad)
    trace = Trace([Event({"concept:name": a}) for a in spur])
    ergebnis = alignments.apply_trace(trace, net, im, fm)
    return int(ergebnis["cost"] // ABWEICHUNG)


class KonformanzPolicy(PSPPolicy):
    def __init__(self, model_path, seed=42, samples=SAMPLES, first_only=(), modell=MODELL, toleranz=0.0):
        super().__init__(model_path=model_path, seed=seed, samples=samples, first_only=first_only)
        self.modell, self.toleranz = modell, float(toleranz)
        self.entscheidungen = []  # je Wahl: Punkt, Fall, erwartete Abweichungen je Option, Wahl (für die Auswertung)

    def bewerten(self, point, prefix, options, time, n=None):
        """Erwartete Abweichungen je Option (Array in der Reihenfolge von options) und die Abweichungen je Suffix (Optionen x n)."""
        from interventions import option_token
        drawn, _ = self.model.option_suffixes(point, prefix, options, time, n or self.model.samples_per_option)
        vorne = projektion_events(prefix)
        je_suffix = np.array([[abweichungen(vorne + projektion_tokens(mit_option(option_token(option), [s[0] for s in suffix])), self.modell)
                               for suffix in suffixes] for option, suffixes in zip(options, drawn)], dtype=float)
        return je_suffix.mean(axis=1), je_suffix

    def choose(self, point, case, options, time, engine):
        if not self.model.knows(point) or self._repeated(point, case.prefix):
            return None
        if len(options) == 1:
            return options[0]
        from prescriptive.estimators.conditional_gaps import SuffixSamplingError
        try:
            erwartet, _ = self.bewerten(point, case.prefix, options, time)
        except SuffixSamplingError:
            self.sampling_fallbacks = getattr(self, "sampling_fallbacks", 0) + 1
            return None
        ordnung = np.argsort(erwartet, kind="stable")
        wahl = None if erwartet[ordnung[1]] - erwartet[ordnung[0]] <= self.toleranz else options[int(ordnung[0])]
        self.entscheidungen.append({"point": point.name, "case": case.case_id, "time": time, "wahl": wahl,
                                    **{f"abweichungen {o}": float(v) for o, v in zip(options, erwartet)}})
        return wahl

"""
Ein PSP je Wahlpunkt und Option (2026-11-22, Nutzer: „Basismodell ZZ-TF1 und dann die T-Learner jeweils aufteilen pro Option?
Kalibrierung weglassen?“). Statt zwei Armen (t_learner_policy.py: Aktion / Referenz, jeder über mehrere Wahlpunkte gemischt) bekommt jede
Kombination (Wahlpunkt, Option) ein eigenes Netz, warm vom ZZ-TF1 und nur auf den Proben an diesem Options-Event trainiert
(scripts/train_psp_real.py --only-points --only-points-validation --point <Name> --option <Token>). Damit trifft eine Verschiebung durch
den Suffix-Verlust genau eine Gruppe; die Kalibrierung je (Wahlpunkt, Option) und ihr Druck gegen Unterschiede zwischen Fällen entfallen.

Bewertung wie beim T-Learner: je Option n gezogene Suffixe aus dem Netz dieser Option (gleiche Startsaat für alle Optionen), Mittel des
Outcomes. Die Hülle ist eine Kopie eines der Netze (für knows, draw_cases, …) mit dem Attribut netze {(Wahlpunkt, Token): PSPModel}.

- bauen(paare, ziel): paare = {(Wahlpunkt, Token): Modellpfad}; Hülle bauen und speichern (PSPModel.load-fähig)
- S-Learner je Wahlpunkt (2026-11-22): beide Optionen eines Punkts mit demselben Pfad → ein Netz, option_suffixes_many zieht dann beide
  Optionen in einem Aufruf (wie PSPModel, gekoppelt über die Zufallszahlen)
Aufruf, im Ordner process-simulation-engine:
    .venv/bin/python ../prescriptive/policies/ev_policy/option_learner_policy.py <ziel.pkl> <Wahlpunkt>=<Token>=<modell.pkl> ...
"""
import copy
import os
import sys


def _base():
    from prescriptive.estimators.psp_model import PSPModel
    return PSPModel


class _Hull:
    """Mixin: jede Option an das Netz ihres Wahlpunkts. self.netze: {(Wahlpunkt-Name, Token Aktivität|Lifecycle): PSPModel}."""

    def model_for(self, point, option):
        from interventions import option_token
        schluessel = (point.name, option_token(option))
        if schluessel not in self.netze:
            raise KeyError(f"Option-Learner: kein Netz für {schluessel}")
        return self.netze[schluessel]

    def option_suffixes_many(self, point, cases, options, now, n):
        import torch
        base = _base()
        modelle = [self.model_for(point, option) for option in options]
        if all(m is modelle[0] for m in modelle):  # S-Learner je Wahlpunkt (2026-11-22): ein Netz für alle Optionen, gemeinsam ziehen
            return base.option_suffixes_many(modelle[0], point, cases, options, now, n)
        seed = int(torch.randint(2 ** 31 - 1, (1,)))
        results, elapsed = [], None
        for option in options:
            torch.manual_seed(seed)  # gleiche Startsaat für alle Netze
            drawn, elapsed_now = base.option_suffixes_many(self.model_for(point, option), point, cases, [option], now, n)
            results.append(drawn[0])
            elapsed = elapsed_now if elapsed is None else elapsed
        torch.manual_seed(seed + 1)
        return results, elapsed

    def option_suffixes(self, point, events, options, now, n):
        drawn, elapsed = self.option_suffixes_many(point, [events], options, now, n)
        return [suffixes[0] for suffixes in drawn], elapsed[0]

    def set_threads(self, n):
        import torch
        torch.set_num_threads(n)

    def value_source(self):
        return "suffix"


def _hull_class():
    return type("OptionLearnerPSP", (_Hull, _base()), {"__module__": __name__})


try:  # für pickle: die Klasse muss unter diesem Namen im Modul stehen
    OptionLearnerPSP = _hull_class()
except Exception:
    OptionLearnerPSP = None


def bauen(paare, ziel=None):
    """Hülle aus den Netzen paare {(Wahlpunkt, Token): Pfad}; alle mit gleichen Wahlpunkten und gleichem Optionsmodus."""
    global OptionLearnerPSP
    if OptionLearnerPSP is None:
        OptionLearnerPSP = _hull_class()
    base = _base()
    geladen, netze = {}, {}  # gleicher Pfad → dasselbe Netz (S-Learner je Wahlpunkt: beide Optionen ein Netz)
    for schluessel, pfad in paare.items():
        if str(pfad) not in geladen:
            geladen[str(pfad)] = base.load(str(pfad))
        netze[schluessel] = geladen[str(pfad)]
    erstes = next(iter(netze.values()))
    for schluessel, netz in netze.items():
        assert set(netz.points) == set(erstes.points), f"Option-Learner: verschiedene Wahlpunkte ({schluessel})"
        assert netz.params.get("option_as") == erstes.params.get("option_as"), f"Option-Learner: verschiedene Optionsmodi ({schluessel})"
        assert schluessel[0] in netz.points, f"Option-Learner: {schluessel[0]} ist kein Wahlpunkt des Netzes"
    hull = copy.copy(erstes)
    hull.__class__ = OptionLearnerPSP
    hull.netze = netze
    hull.option_learner = {f"{p}|{t}": str(pfad) for (p, t), pfad in paare.items()}
    if ziel:
        hull.save(str(ziel))
    return hull


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    from prescriptive.policies.ev_policy import option_learner_policy as module
    ziel, rest = sys.argv[1], sys.argv[2:]
    paare = {}
    for eintrag in rest:
        punkt, token, pfad = eintrag.split("=", 2)
        paare[(punkt, token)] = pfad
    module.bauen(paare, ziel)
    print(f"Option-Learner-Hülle gespeichert: {ziel} ({len(paare)} Netze)")

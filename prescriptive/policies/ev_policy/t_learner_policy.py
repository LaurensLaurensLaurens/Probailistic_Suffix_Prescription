"""
T-Learner-PSP (2026-11-17): zwei PSPs statt einem. Das Aktions-Modell ist nur auf Proben trainiert, deren Prefix am Options-Event einer
Aktion endet (annehmen bei Konzept, annehmen nach Vervollständigung, weiteres Angebot, Lead bearbeiten), das Referenz-Modell nur auf
Proben am Options-Event der Referenz (bearbeiten, fortsetzen, senden, zurückziehen). Training: scripts/train_psp_real.py --only-points
--only-points-validation --arm aktion|referenz, warm gestartet vom fertigen ZZ-TF1 (--init-from), siehe data/bpic17_psp_t_learner.sh.

Bewertung wie beim S-Learner (psp_policy.PSPPolicy): je Option n gezogene Suffixe mit Event und Zeit je Schritt, Mittel des Outcomes,
die bessere Option gewinnt; nur zieht jede Option aus ihrem eigenen Modell. Beide Modelle ziehen mit derselben Startsaat (gemeinsame
Zufallszahlen, soweit zwei verschiedene Netze das zulassen).

- ARME: Token (Aktivität|Lifecycle) der beiden Gruppen
- TLearnerPSP: Hülle mit der Schnittstelle von PSPModel (Kopie des Referenz-Modells + Attribut aktion). option_suffixes /
  option_suffixes_many / outcome_samples / values verteilen die Optionen auf die Modelle; alles andere (draw_cases, knows, ...) läuft
  über das Referenz-Modell. Damit funktionieren PSPPolicy, psp_rauschen.py, punkte_alle.py, nachlegen_erklaerung.py unverändert.
  angebote_einfuegen.py und leads_gedaechtnis.py ziehen ohne Option (draw_cases) und messen damit nur das Referenz-Modell.
- bauen(aktion, referenz, ziel): Hülle aus zwei gespeicherten Modellen bauen und speichern (PSPModel.load-fähig)
- TLearnerPolicy: PSPPolicy mit einer gespeicherten Hülle oder zwei Modellpfaden

Aufruf (Hülle bauen), im Ordner process-simulation-engine:
    .venv/bin/python ../prescriptive/policies/ev_policy/t_learner_policy.py <aktion.pkl> <referenz.pkl> <ziel.pkl>
"""
import copy
import os
import sys

ARME = {"aktion": {"A_Accepted|complete", "O_Create Offer|complete", "W_Handle leads|start"},
        "referenz": {"W_Complete application|start", "W_Complete application|resume", "O_Sent (mail and online)|complete",
                     "W_Handle leads|withdraw"}}


def _arm(option):
    from interventions import option_token
    token = option_token(option)
    for arm, tokens in ARME.items():
        if token in tokens:
            return arm
    raise ValueError(f"T-Learner: Option {option} ({token}) gehört zu keiner Gruppe")


def _base():
    from prescriptive.estimators.psp_model import PSPModel
    return PSPModel


class _Hull:
    """Mixin: Verteilung der Optionen auf die beiden Modelle. Die Hülle ist eine Kopie des Referenz-Modells; self.aktion ist das
    Aktions-Modell (ein gewöhnliches PSPModel)."""

    def model_for(self, option):
        return self.aktion if _arm(option) == "aktion" else self

    def option_suffixes_many(self, point, cases, options, now, n):
        import torch
        base = _base()
        seed = int(torch.randint(2 ** 31 - 1, (1,)))
        results, elapsed = [], None
        for option in options:
            model = self.model_for(option)
            torch.manual_seed(seed)  # gleiche Startsaat für beide Modelle
            drawn, elapsed_now = base.option_suffixes_many(model, point, cases, [option], now, n)
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
        return "suffix"  # T-Learner: immer aus gezogenen Suffixen (Restzeit-Kopf nicht unterstützt)


def _hull_class():
    base = _base()
    return type("TLearnerPSP", (_Hull, base), {"__module__": __name__})


# Für pickle: die Klasse muss unter diesem Namen im Modul stehen (wird beim ersten Import gebaut)
try:
    TLearnerPSP = _hull_class()
except Exception:  # ohne add_paths ist psp_model nicht importierbar; dann beim Aufruf von bauen()
    TLearnerPSP = None


def bauen(aktion, referenz, ziel=None):
    """Hülle aus zwei gespeicherten PSPs (gleiche Netzstruktur, gleiche Wahlpunkte, option_as wie beim Training)."""
    global TLearnerPSP
    if TLearnerPSP is None:
        TLearnerPSP = _hull_class()
    base = _base()
    model_a, model_r = base.load(str(aktion)), base.load(str(referenz))
    assert set(model_a.points) == set(model_r.points), "T-Learner: verschiedene Wahlpunkte"
    assert model_a.params.get("option_as") == model_r.params.get("option_as"), "T-Learner: verschiedene Optionsmodi"
    hull = copy.copy(model_r)
    hull.__class__ = TLearnerPSP
    hull.aktion = model_a
    hull.t_learner = {"aktion": str(aktion), "referenz": str(referenz)}
    if ziel:
        hull.save(str(ziel))
    return hull


def _policy_class():
    from prescriptive.policies.ev_policy.psp_policy import PSPPolicy

    class TLearnerPolicy(PSPPolicy):
        """PSPPolicy mit T-Learner-Hülle: model_path (gespeicherte Hülle) oder aktion + referenz (zwei Modellpfade)."""

        def __init__(self, model_path=None, aktion=None, referenz=None, **kwargs):
            if model_path is None:
                import tempfile
                hull_path = os.path.join(tempfile.mkdtemp(), "t_learner.pkl")
                bauen(aktion, referenz, hull_path)
                model_path = hull_path
            super().__init__(model_path=model_path, **kwargs)
            assert hasattr(self.model, "aktion"), "TLearnerPolicy: keine T-Learner-Hülle"

    return TLearnerPolicy


def __getattr__(name):  # TLearnerPolicy erst beim Zugriff bauen (psp_policy importiert torch)
    if name == "TLearnerPolicy":
        return _policy_class()
    raise AttributeError(name)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    # über den Paketnamen bauen, damit pickle die Klasse als prescriptive.policies.ev_policy.t_learner_policy.TLearnerPSP speichert
    from prescriptive.policies.ev_policy import t_learner_policy as module
    built = module.bauen(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"T-Learner-Hülle gespeichert: {sys.argv[3]} (Aktion {sys.argv[1]}, Referenz {sys.argv[2]})")

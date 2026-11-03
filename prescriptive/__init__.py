# Das Log-Profil (prescriptive/profiles.py) muss feststehen, bevor die Module von prescriptive/ ihre Spalten
# festlegen: --profile auf der Kommandozeile setzt deshalb schon hier die Umgebungsvariable (auch für die Arbeitsprozesse).
import os as _os
import sys as _sys

# Der Simulator (next_event_prediction, simulation, resources, ...) liegt neben prescriptive/ in process-simulation-engine/.
# Aufruf der Skripte weiter aus process-simulation-engine/ (relative Pfade wie eventlog/ und models/), mit dem Ordner über
# prescriptive/ im Pfad:  cd process-simulation-engine && PYTHONPATH=.. .venv/bin/python -m prescriptive.main ...
_ENGINE = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "process-simulation-engine")
if _ENGINE not in _sys.path:
    _sys.path.insert(0, _ENGINE)

for _i, _arg in enumerate(_sys.argv):
    if _arg == "--profile" and _i + 1 < len(_sys.argv):
        _os.environ["PRESCRIPTIVE_PROFILE"] = _sys.argv[_i + 1]
    elif _arg.startswith("--profile="):
        _os.environ["PRESCRIPTIVE_PROFILE"] = _arg.split("=", 1)[1]


# Die Schätzer liegen in prescriptive/estimators/, das Outcome in prescriptive/helpers/preprocessing/. Unter den alten
# Namen (prescriptive.psp, prescriptive.xgb_estimator, prescriptive.outcome, prescriptive.event_log, ...) bleiben sie importierbar: gespeicherte
# Schätzer (pickle, z. B. data/xgb_rct/xgb_model.pkl) nennen ihr Modul so, ebenso die Skripte im Archiv data/.
import importlib.abc as _abc
import importlib.util as _util

_MOVED = {f"prescriptive.{_name}": f"prescriptive.estimators.{_name}" for _name in ("psp", "xgb_estimator", "lstm_estimator", "dr_learner")}
_MOVED["prescriptive.outcome"] = "prescriptive.helpers.preprocessing.BPIC2017.outcome"
_MOVED["prescriptive.event_log"] = "prescriptive.enriched_log"
_MOVED["prescriptive.employees"] = "prescriptive.callers"


class _MovedModule(_abc.MetaPathFinder, _abc.Loader):
    def find_spec(self, fullname, path, target=None):
        return _util.spec_from_loader(fullname, self) if fullname in _MOVED else None

    def create_module(self, spec):
        import importlib
        return importlib.import_module(_MOVED[spec.name])

    def exec_module(self, module):
        pass


if not any(isinstance(_finder, _MovedModule) for _finder in _sys.meta_path):
    _sys.meta_path.append(_MovedModule())

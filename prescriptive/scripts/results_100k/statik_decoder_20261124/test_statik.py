"""Prüfungen für static_in_decoder (2026-11-24, Nutzer: „ja“ zum Umbau „statische Merkmale in jeden Decoder-Schritt“).
  null      k3x4ol (schlank, keyed_noise): Ziehungen an 8 Entscheidungszuständen ohne und mit eingehängter _StaticProjection (Gewichte 0)
            müssen bitgenau gleich sein
  wirkt     mit zufälligen Gewichten ≠ 0 müssen sich die Ziehungen ändern (der Weg ist aktiv), und nur bei Netzen mit Projektion
  speichern Hülle mit eingehängter Projektion (Gewichte ≠ 0) per pickle speichern und laden → gleiche Ziehungen wie vor dem Speichern
Zustände und Saaten wie rollout_cf_20261123/test_rollout.py. Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript>
"""
import os
from pathlib import Path
import pickle
import sys
import tempfile
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/"prescriptive/scripts/results_100k/rollout_cf_20261123"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
import test_rollout as tr  # noqa: E402  (setzt Pfade, Katalog, Zustände)


def netze(model):
    return list(dict.fromkeys(m.net for m in getattr(model, "netze", {}).values())) or [model.net]


def main():
    import torch
    from prescriptive.estimators.psp_model import PSPModel, _attach_static, _StaticProjection
    model = PSPModel.load(str(tr.HUELLE))
    model.set_threads(4)
    ohne = tr.ziehen(model)
    for net in netze(model):
        _attach_static(net)
    mit_null = tr.ziehen(model)
    print("null:", "GLEICH (bitgenau)" if ohne == mit_null else "VERSCHIEDEN")
    torch.manual_seed(7)
    for net in netze(model):
        proj = net.decoder.input_proj
        assert isinstance(proj, _StaticProjection)
        with torch.no_grad():
            proj.static_projection.weight.normal_(0, 0.5)
    mit_gewicht = tr.ziehen(model)
    print("wirkt:", "VERSCHIEDEN (gut)" if mit_gewicht != ohne else "GLEICH (schlecht: Weg nicht aktiv)")
    with tempfile.TemporaryDirectory(dir=str(HERE)) as tmp:
        pfad = os.path.join(tmp, "huelle.pkl")
        model.save(pfad)
        geladen = PSPModel.load(pfad)
        geladen.set_threads(4)
        wieder = tr.ziehen(geladen)
    print("speichern:", "GLEICH" if wieder == mit_gewicht else "VERSCHIEDEN")
    ok = ohne == mit_null and mit_gewicht != ohne and wieder == mit_gewicht
    print("OK" if ok else "FEHLER")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

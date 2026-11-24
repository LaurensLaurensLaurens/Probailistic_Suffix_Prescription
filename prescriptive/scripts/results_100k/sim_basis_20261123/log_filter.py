"""Schritt 2 der Simulator-Basis-Pipeline (bpic17_psp_konformanz_sim_basis_teil3.sh), 2026-11-24: 4-facher Log ohne Fälle mit Kategorien
außerhalb des Basisvokabulars. Korrektur gegenüber teil2 (Abbruch 03:51): PSPModel.view liefert CSV-Text (StringIO) → hier mit read_csv
in eine Tabelle (Aktivität bereits als Token Aktivität|Lifecycle, wie im Training). Höchstens 1 % der Fälle dürfen wegfallen.
Aufruf im Ordner process-simulation-engine: CUDA_VISIBLE_DEVICES="" .venv/bin/python <dieses Skript> <Basis.pkl> <Log x4> <Ausgabe.pkl>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import pandas as pd  # noqa: E402
from preprocessing import preprocess  # noqa: E402
from prescriptive.estimators.psp_model import PSPModel  # noqa: E402

base = PSPModel.load(sys.argv[1])
raw = pd.read_pickle(sys.argv[2])
log = preprocess("bpic17", raw, verbose=False).reset_index(drop=True)
view = pd.read_csv(base.view(log), dtype=str, keep_default_na=False)
case = "case:concept:name"
print(f"Kategoriale Spalten im Basisvokabular: {list(base.encoder_decoder.categorical_encoders)}", flush=True)
schlecht = set()
for spalte, enc in base.encoder_decoder.categorical_encoders.items():
    if spalte not in view.columns:
        print(f"  {spalte}: nicht in der Ansicht (übersprungen)")
        continue
    bekannt = {str(v) for v in enc.categories_[0] if not (isinstance(v, float) and v != v)}
    werte = view[spalte].astype(str)
    neu = ~werte.isin(bekannt) & ~werte.isin(["nan", "None", "NaN", ""])
    faelle = set(view.loc[neu, case].astype(str))
    print(f"  {spalte}: {len(bekannt)} bekannte Werte, {len(faelle)} Fälle mit neuen Werten {sorted(set(werte[neu]))[:8]}")
    schlecht |= faelle
anteil = len(schlecht) / raw[case].nunique()
assert anteil < 0.01, f"zu viele Fälle mit unbekannten Kategorien: {len(schlecht)} ({anteil:.1%}) – Filter prüfen"
if len(sys.argv) > 3:
    raw[~raw[case].astype(str).isin(schlecht)].to_pickle(sys.argv[3])
print(f"Log x4 mit Basisvokabular: {len(schlecht)} von {raw[case].nunique():,} Fällen entfernt")

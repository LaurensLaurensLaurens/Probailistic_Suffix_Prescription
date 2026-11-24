"""Schranke vor dem T-Learner auf der β-NLL-Basis (2026-11-18): nur weiter, wenn β-NLL die lange Wartezeit tatsächlich lernt und die
übrigen Abstände nicht verdirbt. Liest die volle Sonde des fertigen β-Modells (sonde_final/sonde.csv).
Kriterien (vorher festgelegt; Ausgangswerte ZZU-TF1: Kopf-Mittel an A_Cancelled 3,6 d, echt 20,6; MAE alle Schritte 0,57 d):
  1. Kopf-Mittel an echten A_Cancelled-Schritten (Teacher Forcing 1) ≥ 10 Tage
  2. MAE des Kopfs über alle Schritte ≤ 0,70 Tage
Exit 0 = weiter, 1 = Halt. Ausgabe eine Zeile für fortschritt.log."""
from pathlib import Path
import sys

import pandas as pd

FOLDER = Path(__file__).resolve().parent/"sonde_final"
try:
    t = pd.read_csv(FOLDER/"sonde.csv")
    get = lambda teil, methode, spalte: float(t[(t.teil == teil) & (t.methode == methode)][spalte].iloc[0])
    cancel = get("A_Cancelled je Schritt", "kopf", "Abstand Mittel")
    real = get("A_Cancelled je Schritt", "abstand", "Abstand Mittel")
    mae = get("alle Schritte", "kopf", "MAE")
    drawn = get("gezogen: seit O_Sent bis A_Cancelled", "abstandssumme", "seit O_Sent Median")
except (FileNotFoundError, IndexError, KeyError) as error:
    print(f"Schranke: Sonde fehlt oder unvollständig ({error}) → Halt")
    sys.exit(1)
ok = cancel >= 10.0 and mae <= 0.70
print(f"Schranke: Kopf an A_Cancelled {cancel:.1f} d (echt {real:.1f}, Ziel ≥ 10), MAE alle Schritte {mae:.2f} d (Ziel ≤ 0,70), "
      f"gezogen seit O_Sent bis Storno Median {drawn:.1f} d → {'weiter' if ok else 'Halt'}")
sys.exit(0 if ok else 1)

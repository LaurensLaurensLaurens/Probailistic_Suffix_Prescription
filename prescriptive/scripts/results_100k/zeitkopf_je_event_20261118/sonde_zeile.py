"""Je Modell einer Sonde (zeitkopf_sonde_20261118/sonde.py, sonde.csv) eine Zeile für fortschritt.log: Kopf-Mittel an echten
A_Cancelled-Schritten (Teacher Forcing 1) gegen echt, MAE dort und über alle Schritte, Summen O_Sent → Storno / → Validierung (Median),
σ des Kopfs an A_Cancelled, gezogene Tage seit O_Sent bis A_Cancelled (Abstandssumme, wie die Policy rechnet).
Aufruf: python sonde_zeile.py <Ordner mit sonde.csv>"""
import sys
from pathlib import Path

import pandas as pd

t = pd.read_csv(Path(sys.argv[1])/"sonde.csv")
for label, part in t.groupby("modell", sort=False):
    def get(teil, methode, spalte):
        rows = part[(part.teil == teil) & (part.methode == methode)][spalte]
        return float(rows.iloc[0]) if len(rows) else float("nan")
    print(f"Sonde {label}: Kopf an A_Cancelled {get('A_Cancelled je Schritt', 'kopf', 'Abstand Mittel'):.1f} d "
          f"(echt {get('A_Cancelled je Schritt', 'abstand', 'Abstand Mittel'):.1f}), MAE {get('A_Cancelled je Schritt', 'kopf', 'MAE'):.1f}; "
          f"Summe O_Sent→Storno Median Kopf {get('Summe O_Sent → storno (1 O_Sent)', 'kopf', 'seit O_Sent Median'):.1f} "
          f"(echt {get('Summe O_Sent → storno (1 O_Sent)', 'abstand', 'seit O_Sent Median'):.1f}); "
          f"O_Sent→Validierung Kopf {get('Summe O_Sent → validierung (1 O_Sent)', 'kopf', 'seit O_Sent Median'):.1f}/"
          f"{get('Summe O_Sent → validierung (≥ 2 O_Sent)', 'kopf', 'seit O_Sent Median'):.1f} "
          f"(echt {get('Summe O_Sent → validierung (1 O_Sent)', 'abstand', 'seit O_Sent Median'):.1f}/"
          f"{get('Summe O_Sent → validierung (≥ 2 O_Sent)', 'abstand', 'seit O_Sent Median'):.1f}); "
          f"σ Kopf A_Cancelled {get('σ des Abstandskopfs (Tage, Median)', 'kopf', 'A_Cancelled'):.1f} d; "
          f"gezogen seit O_Sent bis Storno Median {get('gezogen: seit O_Sent bis A_Cancelled', 'abstandssumme', 'seit O_Sent Median'):.1f} d, "
          f"Anteil 29–32 d {get('gezogen: seit O_Sent bis A_Cancelled', 'abstandssumme', 'Anteil 29–32 d'):.2f}; "
          f"MAE alle Schritte {get('alle Schritte', 'kopf', 'MAE'):.2f} d")

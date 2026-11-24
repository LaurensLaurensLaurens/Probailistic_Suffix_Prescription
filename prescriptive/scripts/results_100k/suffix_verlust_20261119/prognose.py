"""Erwarteter Online-Rückstand zur Regel aus der Wirkungsdiagnose (2026-11-19), wie im Befund 10:45 (Φ-Formel):
P(falsch) je Gruppe = Φ(−Vorteil / √(Rauschen bei 300² + rauschfreie Streuung²)), Bedingung event (keyed_noise).
Rückstand je Fall = Σ Punkte Wahlen je Fall × P(falsch) × wahrer Nachteil (abweichungen_regel.WAHR); Wahlen je Fall aus den
Online-Läufen von tlcrn_fest300 (12 Seeds, Zweierwahlen, nachlegen nur 1. Besuch).
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript>
"""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"modellvergleich_20261116"))
import abweichungen_regel as ab  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

GRUPPE = {"annehmen_bei_konzept": "annehmen_bei_konzept", "annehmen_nach_vervollstaendigung": "annehmen_nach_vervollstaendigung",
          "angebot_nachlegen": "nachlegen 1. Besuch", "handle_leads": "handle_leads"}


def main():
    counts, cases = pd.Series(0.0, index=ab.POINTS), 0
    for seed in range(12):
        folder = ab.fz.folder("tlcrn_fest300", seed)
        if folder is None:
            continue
        c = ab.choices(folder)
        counts += c.point.value_counts().reindex(ab.POINTS, fill_value=0)
        cases += 200
    per_case = counts / cases
    folder = HERE/os.environ.get("WIRKUNG_OUT", "wirkung")
    table = pd.read_csv(folder/"wirkung.csv")
    table = table[table.Bedingung == "event"]
    rows = []
    for model, part in table.groupby("Modell", sort=False):
        part = part.set_index("Gruppe")
        total = 0.0
        row = {"Modell": model}
        for point in ab.POINTS:
            g = part.loc[GRUPPE[point]]
            wrong = norm.cdf(-g["Vorteil Regel-Option (d)"] / np.hypot(g["Rauschen bei 300 (d)"], g["rauschfreie Streuung zw. Zuständen (d)"]))
            row[f"falsch {ab.KURZ[point]}"] = f"{100 * wrong:.1f} %"
            total += per_case[point] * wrong * ab.WAHR[point]
        row["erwarteter Rückstand zur Regel (d)"] = round(total, 3)
        rows.append(row)
    out = pd.DataFrame(rows)
    text = (f"Wahlen je Fall (tlcrn online): " + ", ".join(f"{ab.KURZ[p]} {per_case[p]:.3f}" for p in ab.POINTS) + "\n\n"
            + ab.markdown(out, index=False))
    (folder/"prognose.md").write_text("# Erwarteter Rückstand zur Regel (Φ-Formel, Bedingung event)\n\n" + text + "\n")
    print(text)


if __name__ == "__main__":
    main()

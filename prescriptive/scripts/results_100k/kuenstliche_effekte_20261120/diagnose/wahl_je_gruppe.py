"""Online-Wahlen der Konformanz-Arme je Wahlpunkt und Teilgruppe (2026-11-23): Anteil „andere Option“ (nicht Regel), Anteil Bank (None,
Gleichstand), Wahrheit andere − Regel (wahrheit_effekte_v3.csv) und der erwartete Verlust gegenüber dem Orakel je Entscheidung
(|Wahrheit| bei falscher Wahl). Teilgruppen wie online_konformanz.Orakel.gruppe (Fallattribute aus events.pkl, Dauer seit dem ersten Event).
Nur erste Besuche je (Fall, Punkt), wie die Wahrheit. Ausgabe wahl_je_gruppe.md / .csv hier.
Aufruf im Ordner process-simulation-engine: .venv/bin/python <dieses Skript> [Arme, Komma]
"""
from pathlib import Path
import sys
import warnings

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ARME = (sys.argv[1] if len(sys.argv) > 1 else "konformanz_x4,konformanz_ol").split(",")
del sys.argv[1:]
warnings.filterwarnings("ignore")
sys.path.insert(0, str(PARENT.parents[3]))
sys.path.insert(0, str(PARENT))
import online_konformanz as ok  # noqa: E402
import pandas as pd  # noqa: E402
from simulation.case_engine_lifecycle_aware import to_seconds  # noqa: E402

wahr = pd.read_csv(PARENT/"wahrheit_effekte_v3.csv").set_index(["Punkt", "Gruppe"])["n Abweichungen andere−Regel"]
orakel = ok.Orakel()
zeilen = []
for arm in ARME:
    for ordner in sorted(ok.OUT.glob(f"{arm}_seed*")):
        if not (ordner/"entscheidungen.csv").exists():
            continue
        e = pd.read_csv(ordner/"entscheidungen.csv")
        ev = pd.read_pickle(ordner/"events.pkl") if (ordner/"events.pkl").exists() else None
        # events.pkl enthält nur die bewerteten Fälle; Attribute und Start daher aus allen Events des Laufs, soweit vorhanden
        attrs = ev.groupby("case:concept:name").agg(**{"case:RequestedAmount": ("case:RequestedAmount", "first"), "case:LoanGoal": ("case:LoanGoal", "first"),
                                                     "case:ApplicationType": ("case:ApplicationType", "first"), "start": ("time:timestamp", "min")})
        e = e.drop_duplicates(["case", "point"]).merge(attrs, left_on="case", right_index=True, how="inner")
        for _, z in e.iterrows():
            regel, andere, _ = orakel.PUNKTE[z.point]
            dauer = (z.time - to_seconds(z.start)) / 86400
            g = orakel.gruppe({"case:RequestedAmount": z["case:RequestedAmount"], "case:LoanGoal": z["case:LoanGoal"],
                               "case:ApplicationType": z["case:ApplicationType"], "dauer_tage": dauer}, z.point)
            wahl = z.wahl if isinstance(z.wahl, str) else None
            zeilen.append({"arm": arm, "punkt": z.point, "gruppe": g, "andere": wahl == andere, "bank": wahl is None,
                           "wahrheit": wahr.get((z.point, g))})
d = pd.DataFrame(zeilen)
d["falsch"] = ((d.wahrheit < 0) & ~d.andere & ~d.bank) | ((d.wahrheit > 0) & d.andere)
d["verlust"] = d.falsch * d.wahrheit.abs()
out = d.groupby(["punkt", "gruppe", "arm"]).agg(n=("andere", "size"), wahrheit=("wahrheit", "first"), anteil_andere=("andere", "mean"),
                                                 anteil_bank=("bank", "mean"), falsch=("falsch", "mean"), verlust_je_entscheidung=("verlust", "mean")).round(3)
summe = d.groupby("arm").agg(entscheidungen=("andere", "size"), falsch=("falsch", "mean"), verlust_summe=("verlust", "sum")).round(3)
out.to_csv(HERE/"wahl_je_gruppe.csv")
with pd.option_context("display.width", 250, "display.max_colwidth", 70):
    text = f"```\n{out.to_string()}\n```\n\n```\n{summe.to_string()}\n```\n"
(HERE/"wahl_je_gruppe.md").write_text("# Online-Wahlen je Teilgruppe (bewertete Fälle, erste Besuche; 2026-11-23)\n\n" + text)
print(text)

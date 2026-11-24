"""Bild EXTRAZWISCHENERGEBNISSE/2017/aipw_richtung_bpic17.png (Stand 2026-11-17): welches PSP-Modell an welchem Wahlpunkt wie oft in
AIPW-Richtung entschieden hat und mit welcher Effektschätzung. Quelle register.csv (register_erstellen.py), Messung psp_rauschen.py:
200 echte Testentscheidungen je Wahlpunkt, 100 Ziehungen je Option, gemeinsame Zufallszahlen.
Farbe = Anteil der Entscheidungen mit Vorzeichen wie AIPW (blau = AIPW-Richtung, rot = dagegen, grau = 50 % = Zufall),
Text = Anteil und mittlere Effektschätzung des Modells (Vorteil der Aktion gegenüber der Referenz in Tagen, Konvention wie AIPW).
Kopfzeilen: AIPW auf dem echten Log (Screening, 95-%-KI) und Simulator-Wahrheit (punkte_alle.py, Vorteil der Regel-Option).
Aufruf: <venv>/python aipw_richtung_bild.py   (nach register_erstellen.py; wird dort am Ende aufgerufen)
"""
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
P = OUT.parents[2]
ZIEL = P.parent/"EXTRAZWISCHENERGEBNISSE/2017/aipw_richtung_bpic17.png"

PUNKTE = [("konzept", "Annehmen bei Konzept", "annehmen statt vervollst."),
          ("nach_v", "Annehmen nach Vervollst.", "annehmen statt fortsetzen"),
          ("nachlegen", "Angebot nachlegen", "weiteres Angebot statt senden"),
          ("leads", "Handle leads", "bearbeiten statt zurückziehen")]
# Gruppen (Reihenfolge im Bild) und Erkennung am Register-Label; innerhalb einer Gruppe nach Trainingsdatum
GRUPPEN = [("Erste Läufe (bis 09.11.)", lambda l: l in ("10k neue Punkte", "s40", "(a)", "(b)")),
           ("Stufe 3: Decoder 60, freies Training", lambda l: l in ("(c)", "(d) 1 ms", "D", "OE", "(e)")),
           ("Teacher Forcing 1 durchgehend", lambda l: l.endswith("-TF1") or l.startswith("OE-TF1")),
           ("ZZ: Zähler + Zeit je Event, TF fällt bis Ep. 8 auf 0 (Checkpoints)", lambda l: l.startswith("ZZ ep") and "Masken" not in l
            or l == "ZZ final"),
           ("ZZ-TF1: wie ZZ, Teacher Forcing 1 (Checkpoints)", lambda l: l.startswith("ZZ-TF1 ") and "Gumbel" not in l),
           ("ZZ-TF0,5: ZZ-TF1 fertig, 6 Ep. nachtrainiert mit Teacher-Forcing-Boden 0,5", lambda l: l.startswith("ZZ-TF0,5")),
           ("ZZ-TF1-WP: wie ZZ-TF1, nur Wahlpunkt-Proben (30 Ep., abgebrochen bei 18; WP150: 150 Ep.)", lambda l: l.startswith("ZZ-TF1-WP")),
           ("ZZU-TF1: ZZ-TF1 + Uhren (Checkpoints)", lambda l: l.startswith("ZZU-TF1")),
           ("β-NLL: ZZU-TF1 fertig, 5 Ep. nachtrainiert mit β-NLL (β 0,5) der Zeitköpfe", lambda l: l.startswith("β-NLL")),
           ("Beschränktes / gekoppeltes Ziehen (Übergangsmasken, Gumbel-Kopplung)", lambda l: "Masken" in l or "Gumbel" in l),
           ("T-Learner + Zeitkopf je Event (nur Zeitköpfe nachtrainiert)", lambda l: l.startswith("T-Learner Kopf je Event")),
           ("T-Learner: je Option ein eigenes PSP (warm vom ZZ-TF1)", lambda l: l.startswith("T-Learner") and "RCT" not in l),
           ("Auf Simulator-RCT-Daten trainiert (Zirkelschluss: Trainingsdaten aus der Bewertungsumgebung)", lambda l: "RCT" in l),
           ("Piloten (3.000 Fälle, 6 Epochen)", lambda l: l.startswith("Pilot"))]
NAMEN = {"10k neue Punkte": "10k Original, Decoder 5", "s40": "s40: Decoder 40", "(a)": "(a): Decoder 60, Log-Abstände",
         "(b)": "(b): wie (a) + Restzeit-Kopf", "(c)": "(c): Decoder 60, normale Abstände", "(d) 1 ms": "(d): wie (c), Zeitklassen",
         "D": "D: (d), Option im Encoder", "OE": "OE: Token mit Zeitklasse", "(e)": "(e): Zeitklasse je Event",
         "(c)-TF1": "(c)-TF1", "D-TF1": "D-TF1", "OE-TF1 Ep. 1": "OE-TF1 Ep. 1 (bester Val.-Verlust)", "OE-TF1 Ep. 26": "OE-TF1 Ep. 26",
         "ZZ ep3": "ZZ Ep. 3 (TF 0,73)", "ZZ ep4": "ZZ Ep. 4 (TF 0,60)", "ZZ ep5": "ZZ Ep. 5 (TF 0,47)", "ZZ ep6": "ZZ Ep. 6 (TF 0,33)",
         "ZZ ep7": "ZZ Ep. 7 (TF 0,20)", "ZZ ep9": "ZZ Ep. 9 (TF 0, 1. freie)", "ZZ ep11": "ZZ Ep. 11 (TF 0)",
         "ZZ ep13": "ZZ Ep. 13 (TF 0)", "ZZ final": "ZZ Ep. 15, fertig", "(c) Masken": "(c) mit Masken",
         "(c) Masken+Gumbel": "(c) mit Masken + Gumbel", "ZZ ep9 Masken+Gumbel": "ZZ Ep. 9 mit Masken + Gumbel",
         "ZZ-TF1 ep12 Gumbel": "ZZ-TF1 Ep. 12, nur Gumbel-Kopplung"}

# Farben: Referenzpalette (dataviz) – divergierend blau ↔ rot, neutrales Grau in der Mitte; Tinte/Flächen
BLAU, ROT, GRAU = "#256abf", "#e34948", "#f0efec"
TINTE, TINTE2, GEDAEMPFT, LINIE, FLAECHE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
KARTE = LinearSegmentedColormap.from_list("aipw", [(0, ROT), (0.25, "#f2a3a2"), (0.5, GRAU), (0.75, "#9ec5f4"), (1, BLAU)])


def tage(value, digits=1):
    if pd.isna(value):
        return "–"
    text = f"{abs(value):.{digits}f}".replace(".", ",")
    return ("−" if value < -0.05 else "+" if value > 0.05 else "±") + text


def name(label):
    if label in NAMEN:
        return NAMEN[label]
    if label.startswith(("ZZ-TF1 ", "ZZU-TF1 ", "ZZ-TF1-WP ", "ZZ-TF1-WP150 ")):
        model, stand = label.split(" ", 1)
        return f"{model} " + ((f"Ep. {dict(ZZ_TF1_WP=30, ZZ_TF1_WP150=150).get(model.replace('-', '_'), 15)}, fertig") if stand == "final" else stand.replace("ep", "Ep. "))
    if label.startswith("Pilot "):
        parts = label.removeprefix("Pilot ").split("_")
        art = {"k": "Zeitklassen", "klassen": "Zeitklassen", "t": "Token+Zeitklasse", "token": "Token+Zeitklasse", "n": "normale Abst.",
               "normal": "normale Abst."}
        text = [art[x] for x in parts if x in art]
        text += [f"TF {x[2:3]},{x[3:]}" if len(x) > 3 else f"TF {x[2:]}" for x in parts if x.startswith("tf")]
        text += [{"od": "Option im Decoder", "oe": "Option im Encoder", "kx": "+ Kontext"}[x] for x in parts if x in ("od", "oe", "kx")]
        return "Pilot: " + ", ".join(text)
    return label


def reihenfolge(label):
    """Checkpoints nach Epoche (final zuletzt), sonst 0 (dann Trainingsdatum)."""
    import re
    if label.endswith(" final"):
        return 99
    hit = re.search(r"(?:ep|Ep\. )(\d+)", label)
    return int(hit.group(1)) if hit else 0


def zeilen(register):
    register = register[register.label != "Wahrheit/Referenz"].copy()
    register["datum"] = pd.to_datetime(register.modell_datum, errors="coerce")
    used, out = set(), []
    for title, test in GRUPPEN:
        part = register[[test(l) and l not in used for l in register.label]]
        part = part[part[[f"aipw_vorzeichen_{k}" for k, _, _ in PUNKTE]].notna().any(axis=1)]
        if part.empty:
            continue
        used.update(part.label)
        out.append(("gruppe", title, None))
        part = part.assign(_ep=[reihenfolge(l) for l in part.label], _pilot=[name(l) for l in part.label])
        keys = ["_pilot"] if title.startswith("Piloten") else ["_ep", "datum"] if "Checkpoints" in title else ["datum", "_ep"]
        for r in part.sort_values(keys, kind="stable").itertuples():
            out.append(("modell", r.label, r))
    rest = register[~register.label.isin(used) & register[[f"aipw_vorzeichen_{k}" for k, _, _ in PUNKTE]].notna().any(axis=1)]
    if len(rest):
        out.append(("gruppe", "Weitere", None))
        out.extend(("modell", r.label, r) for r in rest.itertuples())
    return out


def main():
    register = pd.read_csv(OUT/"register.csv")
    wahr = register.set_index("label").loc["Wahrheit/Referenz"]
    rows = zeilen(register)
    n_rows = 2 + len(rows)
    height = 0.36 * n_rows + 3.9
    fig = plt.figure(figsize=(15.5, height), dpi=150, facecolor=FLAECHE)
    left, cell_w, top = 0.27, 0.145, 1 - 1.95 / height
    x5, w5, se_max = left + 4 * cell_w + 0.008, 0.12, 4.5  # Spalte Rauschen; Balkenlänge ∝ SE bis se_max Tage
    row_h = 0.36 / height
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.text(0.02, 1 - 0.32 / height, "Wann hat welches PSP-Modell in AIPW-Richtung entschieden – und mit welcher Effektschätzung?",
             fontsize=15, fontweight="bold", color=TINTE, va="top")
    fig.text(0.02, 1 - 0.66 / height, "Farbe: Anteil der 200 echten Testentscheidungen je Wahlpunkt mit Vorzeichen wie AIPW (100 Ziehungen je "
             "Option).\nText: Anteil · mittlere Effektschätzung des Modells in Tagen (Vorteil der Aktion gegenüber der Referenz, "
             "Vorzeichen-Konvention wie AIPW) ± Rauschen einer Einzelentscheidung\n(gepaarter Standardfehler der Optionsdifferenz bei "
             "100 Ziehungen, Median über die 200 Entscheidungen). Effekt deutlich kleiner als ± = Entscheidung im Einzelfall vom Zufall bestimmt.",
             fontsize=9.5, color=TINTE2, va="top", linespacing=1.4)
    for j, (_, title, action) in enumerate(PUNKTE):
        x = left + j * cell_w + cell_w / 2
        ax.text(x, top + 0.75 * row_h, title, ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=TINTE)
        ax.text(x, top + 0.15 * row_h, f"Aktion: {action}", ha="center", va="bottom", fontsize=7.8, color=TINTE2)
    ax.text(x5 + w5 / 2, top + 0.75 * row_h, "Rauschen je Entscheidung", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=TINTE)
    ax.text(x5 + w5 / 2, top + 0.15 * row_h, "SE Mittel · Kopplung ρ · |z| > 2", ha="center", va="bottom", fontsize=8, color=TINTE2)
    y = top
    # Referenzzeilen
    for label, texts in (
            ("AIPW auf dem echten Log (95-%-KI)",
             [f"{tage(wahr[f'aipw_effekt_{k}'])} d  [{tage(wahr[f'aipw_ki_unten_{k}'])}; {tage(wahr[f'aipw_ki_oben_{k}'])}]" for k, _, _ in PUNKTE]),
            ("Simulator-Wahrheit (Vorteil der Aktion)",
             [f"{tage(np.sign(wahr['aipw_effekt_konzept']) * wahr['sim_vorteil_konzept'])} d",
              f"{tage(np.sign(wahr['aipw_effekt_nach_v']) * wahr['sim_vorteil_nach_v'])} d",
              f"1. Besuch {tage(np.sign(wahr['aipw_effekt_nachlegen']) * wahr['sim_vorteil_nachlegen1'])} d · später "
              f"{tage(np.sign(wahr['aipw_effekt_nachlegen']) * wahr['sim_vorteil_nachlegen2'])} d",
              f"{tage(np.sign(wahr['aipw_effekt_leads']) * wahr['sim_vorteil_leads'])} d"])):
        y -= row_h
        ax.text(0.02, y + row_h / 2, label, va="center", fontsize=9.5, fontweight="bold", color=TINTE)
        for j, text in enumerate(texts):
            ax.add_patch(Rectangle((left + j * cell_w + 0.002, y + 0.06 * row_h), cell_w - 0.004, 0.88 * row_h, facecolor="#ffffff",
                                   edgecolor=GEDAEMPFT, linewidth=0.8))
            ax.text(left + j * cell_w + cell_w / 2, y + row_h / 2, text, ha="center", va="center", fontsize=8.6, color=TINTE)
    data = []
    for kind, label, r in rows:
        y -= row_h
        if kind == "gruppe":
            ax.text(0.02, y + 0.3 * row_h, label, va="center", fontsize=9.5, fontweight="bold", color=TINTE2)
            ax.plot([0.02, x5 + w5], [y + 0.02 * row_h] * 2, color=LINIE, linewidth=0.8)
            continue
        datum = pd.to_datetime(r.modell_datum, errors="coerce")
        ax.text(0.03, y + row_h / 2, name(label), va="center", fontsize=9, color=TINTE)
        if pd.notna(datum):
            ax.text(left - 0.008, y + row_h / 2, f"{datum:%d.%m.}", va="center", ha="right", fontsize=8, color=GEDAEMPFT)
        entry = {"modell": label, "gruppe": next(t for t, test in GRUPPEN if test(label)) if any(test(label) for _, test in GRUPPEN) else "Weitere",
                 "trainingsdatum": r.modell_datum}
        for j, (k, _, _) in enumerate(PUNKTE):
            share, diff = getattr(r, f"aipw_vorzeichen_{k}"), getattr(r, f"optionsdiff_{k}")
            se = getattr(r, f"se_gepaart_{k}", np.nan)
            entry[f"anteil_aipw_{k}"], entry[f"effekt_tage_{k}"], entry[f"se_einzelentscheidung_tage_{k}"] = share, diff, se
            x0 = left + j * cell_w
            if pd.isna(share):
                ax.text(x0 + cell_w / 2, y + row_h / 2, "–", ha="center", va="center", fontsize=9, color=GEDAEMPFT)
                continue
            ax.add_patch(Rectangle((x0 + 0.002, y + 0.06 * row_h), cell_w - 0.004, 0.88 * row_h, facecolor=KARTE(float(share)),
                                   edgecolor=FLAECHE, linewidth=1.5))
            ink = "#ffffff" if abs(share - 0.5) > 0.3 else TINTE
            noise = "" if pd.isna(se) else f" ± {f'{se:.1f}'.replace('.', ',')}"
            ax.text(x0 + cell_w / 2, y + row_h / 2, f"{100 * share:.0f} %  ·  {tage(diff)}{noise} d", ha="center", va="center",
                    fontsize=9, color=ink, fontweight="bold" if abs(share - 0.5) >= 0.25 else "normal")
        se_mean = r.se_gepaart if hasattr(r, "se_gepaart") else np.nan
        rho = getattr(r, "korrelation_optionen", np.nan)
        sure = np.nanmean([getattr(r, f"anteil_z_gt2_{k}", np.nan) for k, _, _ in PUNKTE])
        entry.update(se_mittel_tage=se_mean, korrelation_optionen=rho, anteil_z_gt2_mittel=sure)
        if pd.notna(se_mean):
            ax.add_patch(Rectangle((x5, y + 0.14 * row_h), w5 * min(se_mean, se_max) / se_max, 0.72 * row_h, facecolor="#e4e2da",
                                   edgecolor="none"))
            text = f"± {f'{se_mean:.1f}'.replace('.', ',')} d"
            if pd.notna(rho):
                text += f"  ·  ρ {f'{rho:.2f}'.replace('.', ',')}"
            if pd.notna(sure):
                text += f"  ·  {100 * sure:.0f} %"
            ax.text(x5 + 0.004, y + row_h / 2, text, va="center", fontsize=8.6, color=TINTE)
        data.append(entry)
    # Farbskala
    y -= 1.1 * row_h
    bar = fig.add_axes([left, y, 4 * cell_w, 0.28 * row_h])
    bar.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=KARTE, extent=(0, 100, 0, 1))
    bar.set_yticks([]); bar.set_xticks([0, 25, 50, 75, 100])
    labels = bar.set_xticklabels(["0 %\nimmer gegen AIPW", "25 %", "50 %\nZufall", "75 %", "100 %\nimmer AIPW-Richtung"], fontsize=8,
                                 color=TINTE2)
    labels[0].set_ha("left"); labels[-1].set_ha("right")
    for side in bar.spines.values():
        side.set_visible(False)
    bar.tick_params(length=0)
    note = ("AIPW: Screening auf dem echten Log (6.11.), Effekt der Aktion mit 95-%-KI; die feste Regel wählt an jedem Punkt die AIPW-Richtung "
            "(annehmen, annehmen, senden, zurückziehen). Simulator-Wahrheit: 60 Simulatorzustände je Gruppe, beide Optionen ausgeführt, "
            "Bank danach. Effektschätzung des Modells: Mittel über die 200 Entscheidungen der Differenz Aktion minus Referenz der gezogenen "
            "Durchlaufzeiten (positiv = Aktion schneller). Gleiches Vorzeichen wie AIPW und ähnliche Größe = Modell trifft den Effekt; "
            "Anteil um 50 % = Entscheidung praktisch zufällig. Rauschen: ± = gepaarter Standardfehler der Differenz einer Einzelentscheidung "
            "bei 100 Ziehungen je Option (gemeinsame Zufallszahlen), je Punkt im Feld, Mittel über die Punkte in der rechten Spalte "
            f"(Balken bis {se_max:.1f} d); ρ = Korrelation der gezogenen Durchlaufzeiten beider Optionen (Kopplung, höher = weniger "
            "Rauschen); |z| > 2 = Anteil der Entscheidungen, deren Differenz mehr als 2 SE beträgt (sicher entschieden). Datum = Trainingsende. "
            f"Quelle: prescriptive/scripts/results_100k/psp_register/register.csv. Stand {datetime.date.today():%d.%m.%Y}.")
    fig.text(0.02, y - 1.25 * row_h, note, fontsize=8, color=TINTE2, va="top", wrap=True)
    fig.savefig(ZIEL, facecolor=FLAECHE)
    plt.close(fig)
    pd.DataFrame(data).to_csv(ZIEL.with_suffix(".csv"), index=False)
    print("geschrieben:", ZIEL)


if __name__ == "__main__":
    main()

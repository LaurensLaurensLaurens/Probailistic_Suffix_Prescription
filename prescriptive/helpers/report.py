"""
Ausgabe eines Laufs von scripts/main.py nach --out (Standard prescriptive/scripts/results/<Datum_Uhrzeit>/): je Tabelle eine CSV
und ein Bild (PNG), zu den zentralen Ergebnissen Diagramme, dazu bericht.html mit allem in der Reihenfolge des Laufs
(Tabellen als HTML, Bilder eingebunden, auch die Diagramme der Simulator-KPIs aus ihren Unterordnern). Der Bericht wird
nach jedem Eintrag neu geschrieben, ein abgebrochener Lauf hat also alles bis dahin.

    report = Report(out)
    report.table("bpic17", "online", table, "Online-Auswertung")
    report.figure("bpic17", "online_uplift", uplift_chart(table, runs), "Uplift gegen die Bank")
"""
import glob
import html
import os
import re
import time

import numpy as np
import pandas as pd

# Farben (Referenzpalette des dataviz-Skills, hell): Fläche, Schrift, Raster; Modelle in Slot 1 (blau), Vergleichs-Policies grau
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
MODEL, BASELINE = "#2a78d6", "#a3a29d"
# Policies, die keine Modelle sind (Vergleich): in Grau
BASELINES = {"bank", "Bank", "random_resource", "Zufall", "DR-Learner (echter Log)"}
MAX_IMAGE_CELLS = 600  # größere Tabellen nur als CSV und im Bericht, nicht als Bild
# Name und Einheit des Outcomes in Beschriftungen, Tabellen und CSVs (set_outcome; Standard: der Gewinn von BPIC17)
OUTCOME = {"name": "Gewinn", "unit": "€"}
OUTCOME_LABELS = {"profit": ("Gewinn", "€"), "negative_cycle_days": ("−Durchlaufzeit", "Tage"), "negative_days_until": ("−Dauer", "Tage"),
                  "negative_repair_cost": ("−Kosten", "€")}


# Wird aufgerufen von scripts/main.py (je Lauf, nach dem Outcome des Profils)
def set_outcome(outcome_type):
    OUTCOME["name"], OUTCOME["unit"] = OUTCOME_LABELS.get(outcome_type, ("Outcome", ""))


# "Gewinn" in Zeilen- und Spaltennamen durch den Namen des gewählten Outcomes ersetzen
def _relabel(table):
    if OUTCOME["name"] == "Gewinn":
        return table
    def rename(label):
        return label.replace("Gewinn", OUTCOME["name"]) if isinstance(label, str) else label
    return table.rename(index=rename, columns=rename)


def slug(text):
    return re.sub(r"[^\w.-]+", "_", text)


class Report:
    def __init__(self, out, title="Ergebnisse"):
        self.out, self.title = out, title
        self.entries = []  # (Art, Datensatz, Titel, Inhalt)
        self.started = time.strftime("%Y-%m-%d %H:%M")

    def table(self, dataset, name, table, title=None, sub=None, hide=(), transpose=None):
        """CSV und, wenn nicht zu groß, ein Bild der Tabelle; im Bericht als HTML-Tabelle. sub: Unterordner von out;
        hide: Spalten, die im Bild und im Bericht fehlen (die CSV hat alle); transpose: das Bild gedreht (None: wenn breit)."""
        folder = os.path.join(self.out, sub) if sub else self.out
        os.makedirs(folder, exist_ok=True)
        base = os.path.join(folder, f"{dataset}_{slug(name)}")
        table = _relabel(table)
        table.to_csv(base + ".csv")
        table = table.drop(columns=[column for column in hide if column in table.columns])
        image = None
        if table.size and table.size <= MAX_IMAGE_CELLS:
            image = base + ".png"
            _table_image(table, title or name, image, transpose)
        self.entries.append(("table", dataset, title or name, (table, image)))
        self.write()

    def figure(self, dataset, name, figure, title, sub=None):
        """Ein Diagramm (matplotlib-Figure) als PNG; sub: Unterordner von out."""
        import matplotlib.pyplot as plt
        if figure is None:
            return
        folder = os.path.join(self.out, sub) if sub else self.out
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{dataset}_{slug(name)}.png")
        figure.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(figure)
        self.entries.append(("figure", dataset, title, path))
        self.write()

    def folder(self, dataset, folder, title):
        """Alle Bilder eines Unterordners (z. B. die Diagramme der Simulator-KPIs) in den Bericht."""
        images = sorted(glob.glob(os.path.join(folder, "*.png")), key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(p))])
        if images:
            self.entries.append(("gallery", dataset, title, images))
            self.write()

    def heading(self, text):
        self.entries.append(("heading", None, text, None))

    def write(self):
        parts = [f"<!doctype html><html lang='de'><head><meta charset='utf-8'><title>{html.escape(self.title)}</title><style>{_CSS}</style></head><body>",
                 f"<h1>{html.escape(self.title)}</h1><p class='meta'>Lauf vom {self.started}, Ordner {html.escape(self.out)}</p>"]
        for kind, dataset, title, content in self.entries:
            label = f"{dataset}: {title}" if dataset else title
            if kind == "heading":
                parts.append(f"<h2>{html.escape(title)}</h2>")
            elif kind == "table":
                table, _ = content
                parts.append(f"<h3>{html.escape(label)}</h3><div class='scroll'>{_html_table(table)}</div>")
            elif kind == "figure":
                parts.append(f"<h3>{html.escape(label)}</h3><img src='{html.escape(os.path.relpath(content, self.out))}' alt='{html.escape(label)}'>")
            elif kind == "gallery":
                images = "".join(f"<figure><img src='{html.escape(os.path.relpath(p, self.out))}' alt=''><figcaption>{html.escape(os.path.basename(p))}</figcaption></figure>" for p in content)
                parts.append(f"<h3>{html.escape(label)}</h3><div class='gallery'>{images}</div>")
        parts.append("</body></html>")
        with open(os.path.join(self.out, "bericht.html"), "w", encoding="utf-8") as f:
            f.write("".join(parts))


_CSS = f"""
body {{ background: {SURFACE}; color: {INK}; font: 14px/1.45 system-ui, sans-serif; margin: 24px auto; max-width: 1200px; padding: 0 16px; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 18px; margin-top: 36px; border-bottom: 1px solid {GRID}; }} h3 {{ font-size: 15px; margin-top: 24px; }}
.meta {{ color: {INK_2}; }} .scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-size: 12.5px; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 3px 8px; border-bottom: 1px solid {GRID}; text-align: right; white-space: nowrap; }}
th {{ color: {INK_2}; font-weight: 600; }} td:first-child, th:first-child {{ text-align: left; }}
img {{ max-width: 100%; }} .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }}
figure {{ margin: 0; }} figcaption {{ color: {INK_2}; font-size: 12px; }}
"""


# Zahlen in deutscher Schreibweise: ganze Zahlen und ab 1.000 ohne Nachkommastellen, ab 10 mit einer, darunter mit drei
def _format(value):
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return f"{value:,}".replace(",", ".")
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        digits = 0 if abs(value) >= 1000 or value == int(value) else 1 if abs(value) >= 10 else 3
        return f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return str(value)


def _html_table(table):
    return table.to_html(na_rep="", float_format=lambda v: _format(v), border=0)


# Die Tabelle als Bild; breite Tabellen mit wenigen Zeilen quer (Spalten werden Zeilen), damit sie lesbar bleiben
def _table_image(table, title, path, transpose=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if transpose or transpose is None and table.shape[1] > 8 and table.shape[0] < table.shape[1]:
        table = table.T
    index = [" / ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in table.index]
    columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in table.columns]
    cells = [[_format(v) for v in row] for row in table.itertuples(index=False)]
    # Kleine Figur: bbox_inches="tight" schneidet beim Speichern auf Tabelle und Titel zu
    figure, axis = plt.subplots(figsize=(1, 0.3))
    figure.patch.set_facecolor(SURFACE)
    axis.axis("off")
    grid = axis.table(cellText=cells, rowLabels=index, colLabels=columns, loc="upper left", cellLoc="right", rowLoc="left")
    grid.auto_set_font_size(False)
    grid.set_fontsize(8.5)
    for (row, column), cell in grid.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        cell.set_facecolor(SURFACE)
        cell.get_text().set_color(INK if row > 0 and column >= 0 else INK_2)
        if row == 0:
            cell.get_text().set_fontweight("bold")
    grid.auto_set_column_width(list(range(-1, len(columns))))
    axis.set_title(title, loc="left", color=INK, fontsize=10.5, pad=4)
    figure.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


# ------------------------------------------------------------------ Diagramme

def _axes(height):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, height))
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=INK_2, labelsize=9, length=0)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    return figure, axis


# Waagerechte Balken mit 95-%-KI je Policy gegen eine Nulllinie (die Bank); points: einzelne Werte je Policy (z. B. je
# Wiederholung) als kleine Punkte
def _interval_bars(values, low, high, unit, title, reference="Bank", points=None, highlight=None):
    order = list(values.sort_values().index)
    figure, axis = _axes(0.9 + 0.45 * len(order))
    y = np.arange(len(order))
    # highlight: diese blau, alle übrigen grau (sonst: Vergleichs-Policies grau)
    colors = [MODEL if name in highlight else BASELINE for name in order] if highlight is not None else [BASELINE if name in BASELINES else MODEL for name in order]
    axis.barh(y, values[order], height=0.5, color=colors, zorder=2)
    axis.errorbar(values[order], y, xerr=[values[order] - low[order], high[order] - values[order]], fmt="none", ecolor=INK_2, elinewidth=1.4, capsize=3, zorder=3)
    if points is not None:
        for row, name in enumerate(order):
            if name in points:
                axis.scatter(points[name], np.full(len(points[name]), row), s=12, color=INK, alpha=0.55, zorder=4, linewidths=0)
    axis.axvline(0, color=INK_2, linewidth=1)
    for row, name in enumerate(order):
        # rechts neben dem KI und allen Punkten, damit nichts überdeckt wird
        right = max([high[name], values[name]] + list(points.get(name, []) if points else []))
        axis.text(right, row, f"  {values[name]:+.1f}{unit}", va="center", fontsize=8.5, color=INK)
    axis.set_yticks(y, order)
    axis.margins(x=0.12)
    axis.set_xlabel(f"gegenüber {reference} ({unit.strip()}; 0 = {reference}), Balken: Mittel, Linie: 95-%-KI", color=INK_2, fontsize=9)
    axis.set_title(title, loc="left", color=INK, fontsize=11)
    return figure


def uplift_chart(table, runs=None):
    """Online-Auswertung (joint.compare): Uplift gegen die Bank in % mit KI, Punkte je Wiederholung."""
    table = table.drop(index="bank", errors="ignore")
    if table.empty:
        return None
    points = runs.groupby("Policy")["Uplift ggü. Bank (%)"].apply(list).to_dict() if runs is not None and "Uplift ggü. Bank (%)" in runs else None
    return _interval_bars(table["Uplift ggü. Bank (%)"], table["Uplift 95-%-KI unten"], table["Uplift 95-%-KI oben"], " %",
                          f"Online: {OUTCOME['name']} gegenüber der Bank (Punkte: einzelne Wiederholungen)", points=points)


def offline_chart(table):
    """Offline-Auswertung (model_evaluation.evaluate_models): Gewinn je Fall gegenüber der Bank in € mit KI."""
    table = table.drop(index="Bank", errors="ignore")
    if table.empty:
        return None
    return _interval_bars(table["ggü. Bank"], table["Bank KI unten"], table["Bank KI oben"], f" {OUTCOME['unit']}",
                          f"Offline (AIPW, echter Log): {OUTCOME['name']} je Fall gegenüber der Bank")


def screening_chart(table, title):
    """Screening: Gewinn der Policy je Fall gegenüber der Bank mit KI; blau die Kandidaten, die das Screening besteht."""
    table = _relabel(table)
    column = f"{OUTCOME['name']} ggü. Bank"
    table = table.dropna(subset=[column])
    if table.empty:
        return None
    passed = set(table.index[table["Ergebnis"].astype(str).str.startswith("ja")])
    figure = _interval_bars(table[column], table["Bank KI unten"], table["Bank KI oben"], f" {OUTCOME['unit']}", title, highlight=passed)
    figure.axes[0].set_xlabel(f"{OUTCOME['name']} je Fall gegenüber der Bank ({OUTCOME['unit']}; 0 = Bank), Linie: 95-%-KI; blau: besteht das Screening",
                              color=INK_2, fontsize=9)
    return figure


def check_chart(table):
    """Simulator gegen echt: Abweichung des Simulierten vom echten Log je Kennzahl in %."""
    real, simulated = table.iloc[:, 0].astype(float), table.iloc[:, 1:].astype(float)
    rows = [name for name in table.index if name != "Fälle" and real[name] not in (0, np.nan) and np.isfinite(real[name])]
    if not rows:
        return None
    deviation = (simulated.loc[rows].iloc[:, 0] / real[rows] - 1) * 100
    figure, axis = _axes(0.9 + 0.4 * len(rows))
    y = np.arange(len(rows))
    axis.barh(y, deviation, height=0.5, color=MODEL, zorder=2)
    axis.axvline(0, color=INK_2, linewidth=1)
    for row, name in enumerate(rows):
        axis.text(deviation[name], row, f"  {deviation[name]:+.1f} %" if deviation[name] >= 0 else f"{deviation[name]:+.1f} %  ",
                  va="center", ha="left" if deviation[name] >= 0 else "right", fontsize=8.5, color=INK)
    axis.set_yticks(y, rows)
    axis.invert_yaxis()
    axis.margins(x=0.22)
    axis.set_xlabel("Abweichung simuliert gegen echt (%)", color=INK_2, fontsize=9)
    axis.set_title(f"Simulator gegen echtes Log: {table.columns[1]}", loc="left", color=INK, fontsize=11)
    return figure


def portfolio_chart(portfolio):
    """Lagebild: erwartete gegen tatsächliche Abschlüsse in 7 Tagen je Beobachtung."""
    columns = [c for c in ("erwartet in 7 T", "tatsächlich in 7 T") if c in portfolio]
    if len(columns) < 2 or portfolio.empty:
        return None
    figure, axis = _axes(3.6)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    for column, color in zip(columns, (MODEL, "#eb6834")):
        axis.plot(portfolio.index, portfolio[column], color=color, linewidth=2, label=column)
        axis.text(portfolio.index[-1], portfolio[column].iloc[-1], f"  {column}", color=INK, fontsize=8.5, va="center")
    axis.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    axis.set_ylabel("Fälle", color=INK_2, fontsize=9)
    axis.set_title("Lagebild: erwartete und tatsächliche Abschlüsse in 7 Tagen", loc="left", color=INK, fontsize=11)
    figure.autofmt_xdate()
    return figure


# ------------------------------------------------------------------ Nachträglich

# Bilder, Diagramme und bericht.html für einen Lauf, der nur CSVs hat (vor helpers/report.py), aus seinen CSVs
# Aufruf: python helpers/report.py <Ordner> [<Ordner> ...]
def rebuild(folder):
    report = Report(folder, f"Lauf {os.path.basename(folder.rstrip('/'))} (nachträglich aus den CSVs)")
    charts = {"simulator_check": ("simulator_check_diagramm", check_chart, "Simulator gegen echt: Abweichung je Kennzahl"),
              "screening_control_flow": ("screening_control_flow_diagramm", lambda t: screening_chart(t, "Screening: Control-Flow-Kandidaten"), "Control-Flow-Kandidaten"),
              "screening_variablen": ("screening_variablen_diagramm", lambda t: screening_chart(t, "Screening: Variablen-Kandidaten"), "Variablen-Kandidaten"),
              "offline": ("offline_diagramm", offline_chart, "Offline-Auswertung: gegenüber der Bank"),
              "lagebild_abschluesse": ("lagebild_diagramm", portfolio_chart, "Lagebild: Abschlüsse erwartet gegen tatsächlich")}
    for path in sorted(glob.glob(os.path.join(folder, "*.csv")), key=os.path.getmtime):
        dataset, _, name = os.path.basename(path)[:-4].partition("_")
        table = pd.read_csv(path, index_col=[0, 1], header=[0, 1]) if name == "kennzahlen" else pd.read_csv(path, index_col=0)
        if name in charts:
            chart_name, function, title = charts[name]
            try:
                report.figure(dataset, chart_name, function(table), title)
            except Exception as error:  # ein Diagramm, das zu dieser Tabelle nicht passt: nur die Tabelle
                print(f"  {chart_name}: {type(error).__name__}: {error}")
        if name == "online":
            runs = os.path.join(folder, f"{dataset}_online_laeufe.csv")
            report.figure(dataset, "online_diagramm", uplift_chart(table, pd.read_csv(runs, index_col=0) if os.path.exists(runs) else None),
                          "Online-Auswertung: Uplift gegen die Bank")
        report.table(dataset, name, table, name.replace("_", " "))
        if name == "simulator_kpis":
            report.folder(dataset, os.path.join(folder, f"{dataset}_simulator_kpis"), "Diagramme der Simulator-KPIs")
    return report


# ------------------------------------------------------------------ Schätzer: vorhergesagter gegen tatsächlichen Gewinn

# Je Wahlpunkt eine Zeile, je Datengrundlage eine Spalte. values: {Daten: {Wahlpunkt: (tatsächlich, geschätzt)}}
def _panels(values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    points = list(dict.fromkeys(point for by_point in values.values() for point in by_point))
    figure, axes = plt.subplots(len(points), len(values), figsize=(4.2 * len(values), 3.6 * len(points)), squeeze=False)
    figure.patch.set_facecolor(SURFACE)
    for column, (source, by_point) in enumerate(values.items()):
        for row, point in enumerate(points):
            axis = axes[row][column]
            axis.set_facecolor(SURFACE)
            axis.grid(color=GRID, linewidth=0.8)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            axis.tick_params(colors=INK_2, labelsize=8)
            yield figure, axis, source, point, by_point.get(point)


def calibration_chart(values, title):
    """values: {Daten: {Wahlpunkt: (tatsächlich, vorhergesagt)}} (training.evaluate_estimators); je Zehntel der Vorhersage
    das Mittel der Vorhersage gegen das Mittel des tatsächlichen Gewinns, gestrichelt: perfekt kalibriert."""
    figure = None
    for figure, axis, source, label, pair in _panels(values):
        if pair is None or len(pair[0]) < 10:
            axis.set_visible(False)
            continue
        actual, predicted = pair
        bins = pd.qcut(predicted, 10, labels=False, duplicates="drop")
        means = pd.DataFrame({"vorhergesagt": predicted, "tatsächlich": actual, "bin": bins}).groupby("bin").mean()
        low, high = min(means.min()), max(means.max())
        axis.plot([low, high], [low, high], color=INK_2, linewidth=1, linestyle="--")
        axis.plot(means["vorhergesagt"], means["tatsächlich"], color=MODEL, linewidth=2, marker="o", markersize=5)
        count = f"{len(actual):,}".replace(",", ".")
        axis.set_title(f"{label}, {source} ({count} Proben)", loc="left", color=INK, fontsize=9.5)
        axis.set_xlabel(f"vorhergesagt: {OUTCOME['name']} ({OUTCOME['unit']}, Mittel je Zehntel)", color=INK_2, fontsize=8.5)
        axis.set_ylabel(f"tatsächlich: {OUTCOME['name']} ({OUTCOME['unit']})", color=INK_2, fontsize=8.5)
    if figure is not None:
        figure.suptitle(title + " (gestrichelt: perfekt kalibriert)", x=0.01, ha="left", color=INK, fontsize=11)
        figure.tight_layout()
    return figure


def scatter_chart(values, title):
    """Wie calibration_chart, aber jede Probe: Dichte von vorhergesagt gegen tatsächlich (1. bis 99. Perzentil)."""
    figure = None
    for figure, axis, source, label, pair in _panels(values):
        if pair is None or len(pair[0]) < 10:
            axis.set_visible(False)
            continue
        actual, predicted = pair
        low, high = np.percentile(np.concatenate([actual, predicted]), [1, 99])
        axis.hexbin(predicted, actual, gridsize=40, extent=(low, high, low, high), cmap="Blues", mincnt=1, linewidths=0)
        axis.plot([low, high], [low, high], color=INK_2, linewidth=1, linestyle="--")
        axis.set_title(f"{label}, {source}", loc="left", color=INK, fontsize=9.5)
        axis.set_xlabel(f"vorhergesagt: {OUTCOME['name']} ({OUTCOME['unit']})", color=INK_2, fontsize=8.5)
        axis.set_ylabel(f"tatsächlich: {OUTCOME['name']} ({OUTCOME['unit']})", color=INK_2, fontsize=8.5)
    if figure is not None:
        figure.suptitle(title + " (Dichte je Feld)", x=0.01, ha="left", color=INK, fontsize=11)
        figure.tight_layout()
    return figure


if __name__ == "__main__":
    import sys
    for folder in sys.argv[1:]:
        print(folder, flush=True)
        rebuild(folder)

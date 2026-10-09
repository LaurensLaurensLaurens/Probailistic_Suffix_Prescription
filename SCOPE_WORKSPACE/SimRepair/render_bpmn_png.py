# Zeichnet simrepair_process.bpmn als PNG (liest Positionen und Texte direkt aus der BPMN-Datei).
#
# Aufruf aus SCOPE_WORKSPACE:
#   MPLCONFIGDIR="$PWD/.tmp/mpl" .venv/bin/python SimRepair/render_bpmn_png.py
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, FancyArrowPatch

FOLDER = Path(__file__).resolve().parent
BPMN_FILE = FOLDER / "simrepair_process.bpmn"
PNG_FILE = FOLDER / "simrepair_process.png"

NS = {"bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL", "di": "http://www.omg.org/spec/BPMN/20100524/DI",
      "dc": "http://www.omg.org/spec/DD/20100524/DC", "ddi": "http://www.omg.org/spec/DD/20100524/DI"}
INK, MUTED, DECISION = "#1f2933", "#52606d", "#b45309"


def tag(el):
    return el.tag.split("}")[1]


def wrap_name(name, width=15):
    # Aktivitätsnamen an den Unterstrichen umbrechen
    lines, current = [], ""
    for part in name.replace("_", "_​").split("​"):
        if current and len(current) + len(part) > width:
            lines.append(current)
            current = part
        else:
            current += part
    return "\n".join(lines + [current])


def bounds(el):
    return tuple(float(el.get(k)) for k in ("x", "y", "width", "height"))


root = ET.parse(BPMN_FILE).getroot()
elements = {el.get("id"): el for el in root.find("bpmn:process", NS)}
shapes, edges = {}, {}
for shape in root.iter("{%s}BPMNShape" % NS["di"]):
    label = shape.find("di:BPMNLabel/dc:Bounds", NS)
    shapes[shape.get("bpmnElement")] = (bounds(shape.find("dc:Bounds", NS)), bounds(label) if label is not None else None)
for edge in root.iter("{%s}BPMNEdge" % NS["di"]):
    edges[edge.get("bpmnElement")] = [(float(w.get("x")), float(w.get("y"))) for w in edge.findall("ddi:waypoint", NS)]

# 100 BPMN-Einheiten = 1 Zoll
xs = [v for (x, y, w, h), _ in shapes.values() for v in (x, x + w)]
ys = [v for (x, y, w, h), _ in shapes.values() for v in (y, y + h)]
x0, x1, y0, y1 = min(xs) - 40, max(xs) + 40, min(ys) - 70, max(ys) + 40
fig = plt.figure(figsize=((x1 - x0) / 100, (y1 - y0) / 100), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(x0, x1)
ax.set_ylim(y1, y0)
ax.axis("off")

for eid, ((x, y, w, h), label) in shapes.items():
    el = elements[eid]
    kind, name = tag(el), el.get("name", "")
    if kind == "task":
        ax.add_patch(FancyBboxPatch((x + 6, y + 6), w - 12, h - 12, boxstyle="round,pad=6,rounding_size=10", fc="#f0f4f8", ec=INK, lw=1.0, zorder=3))
        ax.text(x + w / 2, y + h / 2, wrap_name(name), ha="center", va="center", fontsize=6.5, color=INK, zorder=4, family="DejaVu Sans Mono")
    elif kind == "exclusiveGateway":
        # Beschriftete Gateways sind Entscheidungen, unbeschriftete führen Wege zusammen
        cx, cy = x + w / 2, y + h / 2
        color = DECISION if name else INK
        ax.add_patch(Polygon([(cx, y), (x + w, cy), (cx, y + h), (x, cy)], closed=True, fc="#fff7ed" if name else "white", ec=color, lw=1.0, zorder=3))
        ax.plot([cx - 8, cx + 8], [cy - 8, cy + 8], color=color, lw=1.3, zorder=4)
        ax.plot([cx - 8, cx + 8], [cy + 8, cy - 8], color=color, lw=1.3, zorder=4)
    elif kind in ("startEvent", "endEvent"):
        ax.add_patch(Circle((x + w / 2, y + h / 2), w / 2, fc="white", ec=INK, lw=2.6 if kind == "endEvent" else 1.0, zorder=3))
    elif kind == "textAnnotation":
        ax.plot([x + 8, x, x, x + 8], [y, y, y + h, y + h], color=MUTED, lw=0.8, zorder=2)
        text = el.find("bpmn:text", NS).text
        ax.text(x + 6, y + 4, textwrap.fill(text, width=int(w / 4.3)), ha="left", va="top", fontsize=5.6, color=MUTED, zorder=4, linespacing=1.25)
    if label is not None and name:
        lx, ly, lw_, lh = label
        ax.text(lx + lw_ / 2, ly + lh / 2, name, ha="center", va="center", fontsize=5.8, color=DECISION if kind == "exclusiveGateway" else INK, zorder=5, linespacing=1.2)

for eid, wps in edges.items():
    el = elements[eid]
    if tag(el) == "association":
        ax.plot([p[0] for p in wps], [p[1] for p in wps], color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        continue
    ax.plot([p[0] for p in wps[:-1]], [p[1] for p in wps[:-1]], color=INK, lw=0.9, zorder=2, solid_joinstyle="round")
    ax.add_patch(FancyArrowPatch(wps[-2], wps[-1], arrowstyle="-|>", mutation_scale=7, color=INK, lw=0.9, shrinkA=0, shrinkB=0, zorder=2))
    name = el.get("name")
    if name:
        # Beschriftung nahe am Gateway: erstes Segment mit genug Platz
        for (sx, sy), (ex, ey) in zip(wps, wps[1:]):
            length = abs(ex - sx) + abs(ey - sy)
            if length >= 30:
                break
        if sy == ey:
            right = ex > sx
            ax.text(sx + (8 if right else -8), sy - 4, name, ha="left" if right else "right", va="bottom", fontsize=5.6, color=INK, zorder=5)
        else:
            down = ey > sy
            ax.text(sx + 5, sy + (1 if down else -1) * min(length / 2, 40), name, ha="left", va="center", fontsize=5.6, color=INK, zorder=5)

ax.text(x0 + 30, y0 + 26, "SimRepair: Reparaturprozess (aktueller Stand im Code)", fontsize=10, color=INK, weight="bold", va="center")
ax.text(x0 + 30, y0 + 48, "Orange Gateways sind Entscheidungen, Gateways ohne Beschriftung führen Wege zusammen. "
        "Aufgaben tragen die Aktivitätsnamen des Event-Logs.", fontsize=6, color=MUTED, va="center")
fig.savefig(PNG_FILE, dpi=200, facecolor="white")
print("gespeichert:", PNG_FILE)

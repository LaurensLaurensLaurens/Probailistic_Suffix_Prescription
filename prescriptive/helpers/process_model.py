"""
Prozessmodell je Datensatz (scripts/main.py): fehlt process_models/<DATENSATZ>.bpmn, entsteht es aus dem vorverarbeiteten Log
und wird als .bpmn und .png abgelegt. Vorhandene Modelle bleiben unverändert.

Verfahren: je Aktivität eine Task (mit mehreren Lifecycles im Log Aktivität|Lifecycle, wie in der Pipeline), XOR an jeder
Verzweigung. Es bleiben die Kanten mit einem Anteil von mindestens TAU am Ausgang ihrer Quelle, dazu je Task die häufigste
Kante, damit sie erreichbar ist und enden kann. Der Inductive Miner taugt hier nicht für alle Logs: bei BPIC15 (500 Aktivitäten,
fast jeder Fall ein eigener Verlauf) läuft er nicht durch, auf den Übergängen wird er zum Flower-Modell (Fitness 1, jede
Reihenfolge erlaubt). BPIC15 steht auf Phasenebene (bpic15_phases in preprocessing/BPIC2015/steps.py), auf den 500 Codes gibt
das Verfahren 1741 Kanten und kein lesbares Modell. Mit diesem Verfahren BPIC15: Fitness 0,98, Precision 0,97 (Token Replay
auf der Phasenebene, alle Fälle); BPIC17: Fitness 0,98, Precision 0,63.

    from helpers.process_model import process_model
    path = process_model("bpic13", log)
"""
import os
from collections import defaultdict, deque
from xml.sax.saxutils import escape

import numpy as np

from .datasets import ACTIVITY, CASE, LIFECYCLE, PROJECT, TIMESTAMP

FOLDER = os.path.join(PROJECT, "process_models")
TAU = 0.05  # Mindestanteil einer Kante am Ausgang ihrer Quelle
START, END = "__start__", "__end__"


# Wie oft auf eine Task die nächste folgt, dazu Beginn und Ende der Fälle
def _edges(log):
    log = log.sort_values([CASE, TIMESTAMP], kind="stable")
    labels = log[ACTIVITY].astype(str)
    if LIFECYCLE in log and log[LIFECYCLE].nunique() > 1:
        labels = labels + "|" + log[LIFECYCLE].astype(str)
    tasks, cases = labels.to_numpy(), log[CASE].to_numpy()
    new_case = cases[1:] != cases[:-1]
    count = defaultdict(int)
    for a in tasks[np.r_[True, new_case]]:
        count[(START, a)] += 1
    for a in tasks[np.r_[new_case, True]]:
        count[(a, END)] += 1
    for a, b, same in zip(tasks[:-1], tasks[1:], ~new_case):
        if same:
            count[(a, b)] += 1
    return count


# Von allem, was von start aus erreichbar ist (next_of: Nachfolger je Knoten)
def _reachable(start, next_of):
    seen, queue = {start}, deque([start])
    while queue:
        for node in next_of[queue.popleft()]:
            if node not in seen:
                seen.add(node)
                queue.append(node)
    return seen


# Die Kanten mit mindestens TAU; solange eine Task nicht erreichbar ist oder nicht enden kann, ihre häufigste passende Kante dazu
def _choose(count):
    total = defaultdict(int)
    for (a, _), n in count.items():
        total[a] += n
    keep = {edge for edge, n in count.items() if n >= TAU * total[edge[0]]}
    nodes = {node for edge in keep for node in edge} - {START, END}
    while True:
        forward, backward = defaultdict(list), defaultdict(list)
        for a, b in keep:
            forward[a].append(b)
            backward[b].append(a)
        from_start, to_end = _reachable(START, forward), _reachable(END, backward)
        bad = [node for node in nodes if node not in from_start or node not in to_end]
        if not bad:
            return keep
        added = False
        for node in bad:
            options = {edge: n for edge, n in count.items() if edge[0] == node and edge[1] in to_end} if node not in to_end else \
                {edge: n for edge, n in count.items() if edge[1] == node and edge[0] in from_start}
            if options:
                keep.add(max(options, key=options.get))
                added = True
        if not added:  # nichts mehr anzubinden: was übrig ist, fällt weg
            nodes -= set(bad)
            keep = {edge for edge in keep if not set(edge) & set(bad)}


# Die Kanten als BPMN: je Task ein Knoten, XOR-Gateways dort, wo mehrere Kanten aus- oder eingehen
def _write_bpmn(keep, path):
    nodes = sorted({node for edge in keep for node in edge} - {START, END})
    ids = {node: f"t{k}" for k, node in enumerate(nodes)}
    ids[START], ids[END] = "start", "end"
    out, into = defaultdict(int), defaultdict(int)
    for a, b in keep:
        out[a] += 1
        into[b] += 1
    split, join = (lambda node: "s_" + ids[node]), (lambda node: "j_" + ids[node])
    kinds, flows = {"start": "startEvent", "end": "endEvent", **{ids[node]: "task" for node in nodes}}, []
    for node in nodes + [END]:
        if into[node] > 1:
            kinds[join(node)] = "exclusiveGateway"
            flows.append((join(node), ids[node]))
    for node in nodes + [START]:
        if out[node] > 1:
            kinds[split(node)] = "exclusiveGateway"
            flows.append((ids[node], split(node)))
    for a, b in sorted(keep):
        flows.append((split(a) if out[a] > 1 else ids[a], join(b) if into[b] > 1 else ids[b]))
    names = {ids[node]: node for node in nodes}
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for k, (source, target) in enumerate(flows):
        outgoing[source].append(f"f{k}")
        incoming[target].append(f"f{k}")
    body = []
    for node, kind in kinds.items():
        refs = "".join(f"<incoming>{flow}</incoming>" for flow in incoming[node]) + "".join(f"<outgoing>{flow}</outgoing>" for flow in outgoing[node])
        name = f' name="{escape(names[node], {chr(34): "&quot;"})}"' if node in names else ""
        direction = (' gatewayDirection="Diverging"' if len(outgoing[node]) > 1 else ' gatewayDirection="Converging"') if kind == "exclusiveGateway" else ""
        body.append(f'<{kind} id="{node}"{name}{direction}>{refs}</{kind}>')
    body += [f'<sequenceFlow id="f{k}" sourceRef="{source}" targetRef="{target}"/>' for k, (source, target) in enumerate(flows)]
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?><definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" id="d" targetNamespace="http://process-model">'
                '<process id="p" isExecutable="false">' + "".join(body) + "</process></definitions>")
    return len(nodes), len(keep)


# Das Modell als Bild
def _save_picture(path):
    import pm4py
    from pm4py.visualization.bpmn import visualizer
    visualizer.save(visualizer.apply(pm4py.read_bpmn(path), parameters={"format": "png"}), path[:-len(".bpmn")] + ".png")


def process_model(dataset, log, folder=FOLDER):
    """Pfad des Modells von dataset; fehlt es, entsteht es aus log (vorverarbeitet) und wird gespeichert."""
    path = os.path.join(folder, dataset.upper() + ".bpmn")
    if os.path.exists(path):
        print(f"{dataset}: Prozessmodell vorhanden ({path})", flush=True)
        return path
    os.makedirs(folder, exist_ok=True)
    if dataset == "bpic15":
        from preprocessing.BPIC2015.steps import bpic15_phases
        log = bpic15_phases(log)
    tasks, edges = _write_bpmn(_choose(_edges(log)), path)
    _save_picture(path)
    print(f"{dataset}: Prozessmodell erstellt ({path}), {tasks} Tasks, {edges} Kanten", flush=True)
    return path

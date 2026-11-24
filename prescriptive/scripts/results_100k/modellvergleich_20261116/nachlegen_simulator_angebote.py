"""Woher kommen die vielen späten Zusatzangebote im Simulator? (2026-11-19, Nutzer: „ja, mach Punkt 1“)

Anlass nachlegen_verlaeufe.py: das erzwungene O_Create Offer direkt nach der Entscheidung kommt im Simulator wie im Log sofort;
die Mehrzeit des Simulators nach einem Zusatzangebot steckt in späten neuen Angeboten während des Nachfassens (vor allem
W_Call after offers|suspend → O_Create Offer), die der Simulator auch nach „senden“ öfter erzeugt als der echte Log.

Lauf: korrigierter Simulator, 3.000 Ankünfte je Seed, gewertet Fälle 500–2.499, erster Besuch an angebot_nachlegen 50/50 gelost
(wie nachlegen_erklaerung.py), alles Weitere entscheidet die Bank. Zusätzlich wird festgehalten, welche Option das Simulator-
Modell an dieser Stelle selbst gezogen hätte (natürliche Wahl; EventTimeModel.draw umhüllt). Zwei Zuteilungen:
  random   Katalog d_oe_zustand_20261115/catalog.yaml (wie Aufbau F und nachlegen_erklaerung)
  learned  Kopie davon mit zuteilung learned, berechtigung erfahrung (wie der aktuelle Arbeitskatalog)
Damit lässt sich trennen: (a) unerzwungene Zellen (natürlich = gewählt) gegen den echten Log – bildet der Simulator die beobachteten
Verläufe nach? (b) erzwungene gegen unerzwungene Zellen – woher kommt der randomisierte Effekt? (c) Zuteilung random gegen learned.
Ausgabe nachlegen_simulator_angebote/ (faelle.pkl, Auswertung nachlegen_simulator_angebote.md).
Aufruf im Ordner process-simulation-engine: .venv/bin/python -u <dieses Skript> [laufen|auswerten]
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
OUT = HERE/"nachlegen_simulator_angebote"
SEEDS, ARRIVALS, WINDOW = tuple(range(10)), 3000, (500, 2500)
CASE, ACT, LIFE, TIME, RES = "case:concept:name", "concept:name", "lifecycle:transition", "time:timestamp", "org:resource"
VAL = "W_Validate application|start"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def catalog(allocation):
    if allocation == "random":
        return SOURCE/"catalog.yaml"
    path = OUT/"catalog_learned.yaml"
    if not path.exists():
        text = (SOURCE/"catalog.yaml").read_text()
        head, rest = text.split("\nbpic17:\n", 1)
        block, tail = rest.split("\nbpic19:\n", 1)
        assert block.count("    zuteilung: random\n") == 1
        block = block.replace("    zuteilung: random\n", "    zuteilung: learned\n    berechtigung: erfahrung\n")
        path.write_text("# Kopie von d_oe_zustand_20261115/catalog.yaml, nur bpic17 zuteilung learned + berechtigung erfahrung "
                        "(nachlegen_simulator_angebote.py, 2026-11-19)\n" + head + "\nbpic17:\n" + block + "\nbpic19:\n" + tail)
    return path


def simulate(job):
    allocation, seed = job
    os.environ["PSP_CATALOG"] = str(catalog(allocation))
    sys.path.insert(0, str(ROOT))
    from prescriptive.helpers.datasets import add_paths, load_dataset
    add_paths()
    import interventions
    interventions.CANDIDATES_FILE = os.environ["PSP_CATALOG"]
    assert interventions.run_allocation("bpic17") == allocation, interventions.run_allocation("bpic17")
    from next_event_prediction import event_time_model as etm
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator

    last = {}
    original = etm.EventTimeModel.draw

    def draw(logits, rng, min_probability=0.0):
        out = original(logits, rng, min_probability)
        last["value"] = out
        return out
    etm.EventTimeModel.draw = staticmethod(draw)

    class FirstVisitRandom:
        def __init__(self):
            self.rng = np.random.default_rng(1000+seed)
            self.drawn, self.natural = {}, {}

        def choose(self, point, case, options, time, engine):
            if point.name != "angebot_nachlegen" or len(options) != 2:
                return None
            visits = sum(1 for e in case.prefix if str(e.get(ACT)).strip() == "O_Created")
            if visits != 1 or case.case_id in self.drawn:
                return None
            self.natural[case.case_id] = engine._tokens[int(last["value"][0])][0]
            choice = options[int(self.rng.integers(2))]
            self.drawn[case.case_id] = choice
            return choice

    raw, _, _ = load_dataset("bpic17")
    sim = Simulator("bpic17", preprocess("bpic17", raw))
    policy = FirstVisitRandom()
    events, _ = sim.run(ARRIVALS, policy=policy, window=WINDOW, seed=seed)
    numbers = events[CASE].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    events = events.loc[numbers.ge(WINDOW[0]) & numbers.lt(WINDOW[1])].sort_values([CASE, TIME], kind="stable")
    rows = []
    for case, part in events.groupby(CASE, sort=False):
        tokens = (part[ACT].astype(str).str.strip()+"|"+part[LIFE].astype(str)).tolist()
        days = part[TIME].astype("datetime64[ms]").astype("int64").to_numpy()/86400000
        rows.append({"zuteilung": allocation, "seed": seed, "case": case, "natuerlich": policy.natural.get(case),
                     "gewaehlt": policy.drawn.get(case), "tokens": tokens, "gaps": list(np.diff(days, prepend=days[0])),
                     "resources": part[RES].astype(str).tolist()})
    print(f"{allocation} seed {seed}: {len(rows)} Fälle, {len(policy.drawn)} Entscheidungen", flush=True)
    return pd.DataFrame(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog("learned")
    jobs = [(a, s) for a in ("random", "learned") for s in SEEDS]
    with ProcessPoolExecutor(len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        frame = pd.concat(list(pool.map(simulate, jobs)), ignore_index=True)
    frame.to_pickle(OUT/"faelle.pkl")
    print("Simulation fertig", flush=True)


def opt(name):
    return None if name is None or (isinstance(name, float) and np.isnan(name)) else ("extra" if str(name).startswith("O_Create") else "senden")


def late(tokens, j):
    """Spätes neues Angebot: O_Create Offer nicht direkt an der Entscheidung und nicht im selben Erstellungsschub."""
    return tokens[j].startswith("O_Create Offer") and j > 0 and tokens[j-1] != "O_Created|complete" and not tokens[j-1].startswith("A_Complete")


def measures(tokens, gaps):
    """tokens/gaps ab dem Event nach der Entscheidung (gaps[k] = Abstand vor tokens[k], Tage)."""
    stop = next((j for j, t in enumerate(tokens) if t == VAL), None)
    end = len(tokens) if stop is None else stop + 1
    t, g = tokens[:end], gaps[:end]
    lates = [j for j in range(len(t)) if late(t, j)]
    return {"bis Validierung/Ende (d)": float(np.sum(g)), "Validierung erreicht": stop is not None,
            "späte Angebote": len(lates), "Zeit vor späten Angeboten (d)": float(sum(g[j] for j in lates)),
            "O_Sent": sum(x.startswith("O_Sent") for x in t), "Nachfass-suspend": sum(x == "W_Call after offers|suspend" for x in t),
            "Storno (A_Cancelled)": "A_Cancelled|complete" in tokens, "Restzeit bis Fallende (d)": float(np.sum(gaps))}


def real_cases():
    d = pd.read_pickle(ROOT/"prescriptive/data/cache/bpic17_raw.pkl")
    d[ACT] = d[ACT].str.strip()
    d = d.sort_values([CASE, TIME], kind="stable")
    d["tok"] = d[ACT] + "|" + d[LIFE]
    rows = []
    for case, g in d.groupby(CASE, sort=False):
        days = g[TIME].astype("int64").to_numpy() / 86400e9
        names = g[ACT].tolist()
        natural = None
        if "O_Created" in names:
            i = names.index("O_Created")
            if i + 1 < len(names) and names[i + 1] in ("O_Sent (mail and online)", "O_Create Offer"):
                natural = names[i + 1]
        rows.append({"zuteilung": "echt", "seed": -1, "case": case, "natuerlich": natural, "gewaehlt": natural,
                     "tokens": g.tok.tolist(), "gaps": list(np.diff(days, prepend=days[0]))})
    return pd.DataFrame(rows)


def segment_frame(frame):
    rows = []
    for r in frame.itertuples():
        nat, cho = opt(r.natuerlich), opt(r.gewaehlt)
        if cho is None:
            continue
        i = r.tokens.index("O_Created|complete")
        rows.append({"zuteilung": r.zuteilung, "seed": r.seed, "natuerlich": nat, "gewaehlt": cho,
                     **measures(r.tokens[i+1:], list(r.gaps[i+1:]))})
    return pd.DataFrame(rows)


def contrast(a, b, column):
    x, y = a[column].astype(float), b[column].astype(float)
    return x.mean() - y.mean(), np.sqrt(x.var()/len(x) + y.var()/len(y))


def transitions(frame):
    """Ganze Fälle ohne Eingriff (keine Entscheidung oder natürlich = gewählt): was folgt auf W_Call after offers|suspend?"""
    out = []
    for source, part in frame.groupby("zuteilung"):
        part = part[[opt(n) == opt(c) for n, c in zip(part.natuerlich, part.gewaehlt)]]
        nxt, gap, lates, cases, by_offers = {}, {}, 0, len(part), {}
        for r in part.itertuples():
            t = r.tokens
            first = next((j for j, x in enumerate(t) if x.startswith("O_Sent")), None)
            if first is not None:  # neue Angebote nach dem ersten O_Sent (nicht im selben Erstellungsschub)
                lates += sum(1 for j in range(first + 1, len(t)) if t[j].startswith("O_Create Offer") and t[j-1] != "O_Created|complete")
            created = 0
            for j in range(len(t) - 1):
                created += t[j] == "O_Created|complete"
                if t[j] == "W_Call after offers|suspend":
                    nxt[t[j+1]] = nxt.get(t[j+1], 0) + 1
                    gap[t[j+1]] = gap.get(t[j+1], 0.0) + r.gaps[j+1]
                    k = by_offers.setdefault(min(created, 3), [0, 0])
                    k[0] += t[j+1] == "O_Create Offer|complete"
                    k[1] += 1
        total = sum(nxt.values())
        for token, count in nxt.items():
            out.append({"quelle": source, "folgt": token, "anteil": count/total, "abstand_d": gap[token]/count,
                        "suspend_je_fall": total/cases, "neue_angebote_nach_1_sent_je_fall": lates/cases, "faelle": cases,
                        **{f"neues_angebot_nach_suspend_bei_{k}_angeboten": v[0]/v[1] for k, v in sorted(by_offers.items()) if k >= 1}})
    return pd.DataFrame(out)


def evaluate():
    sim = pd.read_pickle(OUT/"faelle.pkl")
    real = real_cases()
    seg = pd.concat([segment_frame(sim), segment_frame(real)], ignore_index=True)
    seg.to_pickle(OUT/"segmente.pkl")
    cols = [c for c in seg.columns if c not in ("zuteilung", "seed", "natuerlich", "gewaehlt")]
    level = seg.groupby(["zuteilung", "natuerlich", "gewaehlt"])[cols].mean()
    level["n"] = seg.groupby(["zuteilung", "natuerlich", "gewaehlt"]).size()
    level.to_csv(OUT/"zellen.csv")
    rows = []
    for source, part in seg.groupby("zuteilung"):
        cells = {k: v for k, v in part.groupby(["natuerlich", "gewaehlt"])}
        defs = {"beobachtet: unerzwungen extra − unerzwungen senden": (cells.get(("extra", "extra")), cells.get(("senden", "senden")))}
        if source != "echt":
            defs.update({"randomisiert gesamt: extra − senden": (part[part.gewaehlt == "extra"], part[part.gewaehlt == "senden"]),
                         "bei natürlich senden: erzwungen extra − senden": (cells[("senden", "extra")], cells[("senden", "senden")]),
                         "bei natürlich extra: extra − erzwungen senden": (cells[("extra", "extra")], cells[("extra", "senden")])})
        for name, (a, b) in defs.items():
            row = {"quelle": source, "vergleich": name, "n": f"{len(a)} / {len(b)}"}
            for column in ("Restzeit bis Fallende (d)", "bis Validierung/Ende (d)", "späte Angebote", "Zeit vor späten Angeboten (d)",
                           "Storno (A_Cancelled)", "O_Sent"):
                value, se = contrast(a, b, column)
                row[column] = f"{value:+.2f} ± {se:.2f}" if "(d)" in column else f"{value:+.3f} ± {se:.3f}"
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT/"kontraste.csv", index=False)
    trans = transitions(pd.concat([sim, real], ignore_index=True))
    trans.to_csv(OUT/"uebergaenge_suspend.csv", index=False)
    top = trans[trans.folgt.isin(trans.groupby("folgt").anteil.mean().sort_values(ascending=False).index[:6])]
    share = top.pivot_table(index="folgt", columns="quelle", values="anteil").round(3)
    wait = top.pivot_table(index="folgt", columns="quelle", values="abstand_d").round(2)
    per_case = trans.groupby("quelle")[[c for c in trans.columns if c.endswith("je_fall") or c.startswith("neues_") or c == "faelle"]].first().T.round(3)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        lines = ["# Späte Zusatzangebote im Simulator: Zuteilung, natürliche Wahl, echter Log (2026-11-19)", "",
                 "Erster Besuch angebot_nachlegen 50/50 gelost, natürliche Wahl des Simulator-Modells festgehalten; 10 Seeds × 2.000 "
                 "gewertete Fälle je Zuteilung. Abschnitt ab der Entscheidung bis W_Validate application|start (sonst Fallende); "
                 "Restzeit bis Fallende. Echt = beobachtet (natürlich = gewählt).", "",
                 "## Zellen (Mittel je Fall)", "", "```", level.round(3).to_string(), "```", "",
                 "## Kontraste (± Standardfehler)", "", "```", table.set_index(["quelle", "vergleich"]).to_string(), "```", "",
                 "## Ganze Fälle ohne Eingriff: was folgt auf W_Call after offers|suspend", "", "Anteil:", "```", share.to_string(), "```",
                 "Abstand davor (Tage):", "```", wait.to_string(), "```", "Je Fall:", "```", per_case.to_string(), "```", ""]
    (OUT/"nachlegen_simulator_angebote.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "laufen":
        run()
    if len(sys.argv) < 2 or sys.argv[1] in ("laufen", "auswerten"):
        evaluate()
    print("FERTIG", flush=True)

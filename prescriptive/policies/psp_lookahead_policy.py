"""
psp_lookahead: Mitarbeiterwahl nach dem PSP (estimators/psp_model.py) mit Blick auf die anderen laufenden Fälle, für das
Outcome des Laufs (Durchlaufzeit oder Kosten). Ersetzt die Vorausschau vom 29.10.2026 (Überhang über τ, Schaden anderer
Fälle nur aufaddiert, kein Warten). Fragt die Engine an einem Mitarbeiter-Wahlpunkt (set_policy), bekommt die Policy den
Prefix des Falls A (case.prefix), die Optionen (die jetzt freien, kompetenten Mitarbeiter) und über engine.other_cases die
anderen laufenden Fälle: ihren Prefix, ob sie den Arbeitsschritt gerade offen haben und ob sie gerade darauf warten.

Die Entscheidung, alles aus gezogenen Suffixen. Zeiten aus den gezogenen Abständen, nicht aus der gezogenen Zeit seit
Fallbeginn (die staucht die Dauern, siehe prescriptive/costs/repair/outcome.py):
1. Optionen von A: je freiem Mitarbeiter r samples Suffixe (alle Optionen mit denselben Zufallszahlen), V_A(r) = Mittel des
   Outcomes. Bindung von r: bis zum nächsten Event des Arbeitsschritts im Suffix (complete, suspend); T_r = jetzt + Quantil
   quantile der Bindung.
2. Warten als Option: A bekommt den Mitarbeiter, der als nächster frei wird (je laufendem Fall mit dem Arbeitsschritt offen
   dessen Ende aus seinen Suffixen, Median; die WAIT_CANDIDATES frühesten), mit Beginn dann: V_A(warten) = das beste davon.
   Kein belegter Mitarbeiter, wait=False oder A wartet schon länger als max_wait_days: kein Warten.
3. Rangfolge der Optionen und des Wartens nach V_A. Von oben: Warten an der Reihe: A wartet (interventions.WAIT, die Engine
   fragt wieder, wenn sich etwas ändert). Ein Mitarbeiter r an der Reihe:
   a. Wer könnte r bis T_r brauchen: je anderem laufenden Fall j ohne offenen Arbeitsschritt arrival_samples Suffixe bis zum
      Arbeitsschritt (höchstens HORIZON_DAYS, ARRIVAL_STEPS Events; zwischengespeichert, bis j ein neues Event hat);
      P_j = Anteil der noch möglichen Ziehungen, in denen j bis T_r ankommt (wartet j gerade: 1). Kandidaten: die top_k mit
      dem größten P_j, mindestens threshold.
   b. Wäre r bei einem Kandidaten besser aufgehoben: sein wahrscheinlichster Verlauf bis zur Ankunft (die häufigste Folge
      von Events unter seinen Ziehungen mit Ankunft bis T_r, je Event der Median der aufsummierten Abstände, Mitarbeiter
      der seines letzten Events) an seinen Prefix gehängt, dort jede Option mit harm_samples Suffixen. Gewinn_j(r) =
      V_j(r) - das Bessere aus: beste übrige Option jetzt, r nach A (Beginn T_r).
   c. Gibt A r ab, verliert A V_A(r) - V_A(nächste in der Rangfolge; nach der letzten: unendlich, A muss nehmen). Gewinnt ein
      Kandidat im Erwartungswert P_j · Gewinn_j(r) mehr, bleibt r für ihn frei und die nächste Stelle der Rangfolge ist dran;
      sonst bekommt A r.
An Control-Flow-Punkten None (das gezogene Event). may_wait: repair_workshop fragt die Policy auch bei nur einem freien Mitarbeiter.

stats: je Lauf Fragen, Kandidaten je Frage und wie oft die Policy wartet, weil es für A selbst besser ist, wartet, weil alle
freien Mitarbeiter bei anderen besser aufgehoben sind, oder einen anderen als den besten eigenen Mitarbeiter nimmt, weil der
bei einem anderen besser aufgehoben ist (online_evaluation übernimmt es in die Zeile des Laufs).

Das Modell: models/psp_model.pkl (MODEL_PATH) oder model_path (helpers/training.model_path), gespeichert mit PSPModel.save.
"""
import os
from collections import Counter

import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "psp_model.pkl")
SAMPLES, ARRIVAL_SAMPLES, HARM_SAMPLES = 30, 20, 20  # gezogene Suffixe: je Option von A, je anderem Fall, je Option eines Kandidaten
QUANTILE = 0.8  # T_r: so viele der Ziehungen sind bis dahin mit r fertig
THRESHOLD = 0.2  # ab dieser Wahrscheinlichkeit, bis T_r anzukommen, ist ein anderer Fall Kandidat
TOP_K = 5  # höchstens so viele Kandidaten je Mitarbeiter
MAX_WAIT_DAYS = 2.0  # wartet A schon so lange (seit seinem letzten Event), wartet er nicht weiter
HORIZON_DAYS = 7.0  # so weit voraus werden die anderen Fälle gezogen
ARRIVAL_STEPS = 25  # höchstens so viele Events je Ziehung eines anderen Falls
WAIT_CANDIDATES = 3  # beim Warten die so vielen am frühesten frei werdenden Mitarbeiter
BATCH = 256  # so viele andere Fälle je Batch
DAY = 86400.0
CASE, ACTIVITY, LIFECYCLE, RESOURCE, TIMESTAMP = "case:concept:name", "concept:name", "lifecycle:transition", "org:resource", "time:timestamp"


def _seconds(timestamp):
    return (pd.Timestamp(timestamp) - pd.Timestamp("1970-01-01")).total_seconds()


def _until(suffix, activity):
    """Bis zum ersten Event von activity im Suffix: aufsummierte Abstände (Sekunden) und die Events davor als (Token, Abstände bis dahin); ohne: inf."""
    total, before = 0.0, []
    for label, _, gap in suffix:
        total += gap
        if label.split("|")[0] == activity:
            return total, before
        before.append((label, total))
    return np.inf, before


class PSPLookaheadPolicy:
    def __init__(self, model_path=MODEL_PATH, seed=42, samples=SAMPLES, arrival_samples=ARRIVAL_SAMPLES, harm_samples=HARM_SAMPLES,
                 quantile=QUANTILE, threshold=THRESHOLD, top_k=TOP_K, wait=True, max_wait_days=MAX_WAIT_DAYS):
        import torch

        from prescriptive.estimators.psp_model import PSPModel
        self.model = PSPModel.load(model_path)
        self.model.set_threads(1)  # ein Thread je Arbeitsprozess (die Läufe laufen parallel)
        torch.manual_seed(seed)  # die gezogenen Suffixe, je Lauf wiederholbar
        self.rng = np.random.default_rng(seed)  # gleiche Zufallszahlen für Vergleiche, Gleichstände
        self.samples, self.arrival_samples, self.harm_samples = samples, arrival_samples, harm_samples
        self.quantile, self.threshold, self.top_k = quantile, threshold, top_k
        self.may_wait, self.max_wait = wait, max_wait_days * DAY
        self._drawn = {}  # (Fall, Arbeitsschritt) -> (Zahl der Events, gültig bis, [(Zeitpunkt in Sekunden seit 1970, Events davor)])
        self.stats = {"Fragen (Vorausschau)": 0, "Kandidaten je Frage": 0.0, "Warten, selbst besser": 0, "Warten, für andere": 0,
                      "anderer Mitarbeiter, für andere": 0}

    # Wird aufgerufen von case_engine._ask bzw. repair_engine._ask -> choose (je Wahlpunkt)
    def choose(self, point, case, options, time, engine):
        import torch

        from interventions import WAIT
        if point.kind != "employee" or not options or not self.model.knows(point):
            return None
        events, options = case.prefix, list(options)
        others = engine.other_cases(case.case_id, point.activity)
        seed = int(self.rng.integers(2 ** 31 - 1))
        torch.manual_seed(seed)
        values, binding = self._own(point, events, options, time)
        waited = time - _seconds(events[-1][TIMESTAMP])
        wait_value = self._wait_value(point, events, [o for o in others if o["open"]], time, seed) if self.may_wait and waited < self.max_wait else None
        # Rangfolge: die Optionen (Index) und das Warten (None)
        ranking = sorted(range(len(options)), key=lambda k: -values[k])
        if wait_value is not None:
            ranking = sorted(ranking + [None], key=lambda k: -(wait_value if k is None else values[k]))
        arriving = [o for o in others if not o["open"]]
        arrivals = self._times(point, arriving, time)
        gains, seen, choice = {}, set(), None
        for place, k in enumerate(ranking):
            if k is None:
                choice = WAIT
                break
            following = ranking[place + 1] if place + 1 < len(ranking) else None
            if place + 1 == len(ranking):
                choice = options[k]  # die letzte Stelle: A muss nehmen
                break
            loss = values[k] - (wait_value if following is None else values[following])
            release = time + float(np.quantile(binding[k], self.quantile))
            p = np.array([1.0 if other["waiting"] else self._probability(draws, time, release) for other, draws in zip(arriving, arrivals)])
            at_risk = [j for j in np.argsort(-p, kind="stable")[:self.top_k] if p[j] >= self.threshold]
            seen.update(at_risk)
            best = 0.0
            for j in at_risk:
                gain = self._gain(point, arriving[j], arrivals[j], options, k, time, release, gains)
                best = max(best, p[j] * gain)
            if best <= loss:
                choice = options[k]
                break
        else:
            choice = options[ranking[-1]]
        self.stats["Fragen (Vorausschau)"] += 1
        self.stats["Kandidaten je Frage"] += (len(seen) - self.stats["Kandidaten je Frage"]) / self.stats["Fragen (Vorausschau)"]
        if choice == WAIT:
            self.stats["Warten, selbst besser" if ranking[0] is None else "Warten, für andere"] += 1
        elif choice != options[ranking[0]]:
            self.stats["anderer Mitarbeiter, für andere"] += 1
        return choice

    def _own(self, point, events, options, now):
        """Für A je Option der Erwartungswert des Outcomes und je Ziehung die Bindung des Mitarbeiters in Sekunden."""
        model = self.model
        drawn, elapsed = model.option_suffixes(point, events, options, now, self.samples)
        costs, kind = model._option_costs(point, events, options, now), model._outcome_kind()
        values = np.array([np.mean([model.outcome(elapsed, s, kind, c) for s in suffixes]) for suffixes, c in zip(drawn, costs)])
        binding = [np.minimum([_until(s, point.activity)[0] for s in suffixes], HORIZON_DAYS * DAY) for suffixes in drawn]
        return values, binding

    def _value_at(self, point, events, option, at, seed, n):
        """Erwartungswert des Outcomes, wenn der Fall option erst zum Zeitpunkt at bekommt (gleiche Zufallszahlen über seed)."""
        import torch
        torch.manual_seed(seed)
        return float(self.model.outcome_samples(point, events, [option], at, n).mean())

    def _wait_value(self, point, events, busy, now, seed):
        """V_A(warten): der beste der WAIT_CANDIDATES am frühesten frei werdenden Mitarbeiter, mit Beginn dann; ohne belegte None."""
        if not busy:
            return None
        releases = self._times(point, busy, now)
        ends = []
        for other, draws in zip(busy, releases):
            holder = next((e.get(RESOURCE) for e in reversed(other["prefix"]) if e[ACTIVITY] == point.activity and e.get(LIFECYCLE) in ("start", "resume")), None)
            valid = [at for at, _ in draws if now <= at < np.inf]  # inf: kein Ende im Horizont
            if holder is not None and valid:
                ends.append((float(np.median(valid)), holder))
        if not ends:
            return None
        return max(self._value_at(point, events, holder, at, seed, self.samples) for at, holder in sorted(ends)[:WAIT_CANDIDATES])

    @staticmethod
    def _probability(draws, now, until):
        """P(Ankunft bis until | noch nicht angekommen): unter den Ziehungen mit Ankunft ab jetzt (inf: keine im Horizont)."""
        times = np.array([at for at, _ in draws])
        alive = times[times >= now]
        return float(np.mean(alive <= until)) if len(alive) else 0.0

    def _gain(self, point, other, draws, options, k, now, release, cache):
        """Gewinn_j(r) für den Kandidaten j (other, seine Ziehungen draws) und r = options[k], siehe oben (3b)."""
        key = other["case"]
        if key not in cache:
            # Sein wahrscheinlichster Verlauf bis zur Ankunft, dort alle Optionen (gleiche Zufallszahlen)
            events, arrival = self._likely_prefix(other, draws, now, release)
            seed = int(self.rng.integers(2 ** 31 - 1))
            import torch
            torch.manual_seed(seed)
            values = self.model.outcome_samples(point, events, options, arrival, self.harm_samples).mean(axis=1)
            cache[key] = (events, arrival, seed, values, {})
        events, arrival, seed, values, after = cache[key]
        others = np.delete(values, k)
        if k not in after:
            # r erst nach A: Beginn T_r (oder bei der Ankunft, wenn r dann schon wieder frei ist)
            after[k] = self._value_at(point, events, options[k], max(release, arrival), seed, self.harm_samples)
        alternative = max(others.max() if len(others) else -np.inf, after[k])
        return max(values[k] - alternative, 0.0)

    def _likely_prefix(self, other, draws, now, until):
        """Der Prefix von j bis zu seiner Ankunft: die häufigste gezogene Folge mit Ankunft bis until, Zeiten der Median; Ankunftszeit."""
        if other["waiting"]:
            return list(other["prefix"]), now
        hits = [(at, before) for at, before in draws if now <= at <= until]
        if not hits:
            return list(other["prefix"]), now
        common = Counter(tuple(label for label, _ in before) for _, before in hits).most_common(1)[0][0]
        chosen = [(at, before) for at, before in hits if tuple(label for label, _ in before) == common]
        last = other["prefix"][-1]
        start = _seconds(last[TIMESTAMP])
        static = {name: value for name, value in last.items() if str(name).startswith("case:")}
        events = list(other["prefix"])
        for position, label in enumerate(common):
            activity, _, lifecycle = label.partition("|")
            offset = float(np.median([before[position][1] for _, before in chosen]))
            events.append({**static, CASE: last[CASE], ACTIVITY: activity, LIFECYCLE: lifecycle or "complete", RESOURCE: last.get(RESOURCE),
                           TIMESTAMP: pd.Timestamp(start + offset, unit="s")})
        arrival = float(np.median([at for at, _ in chosen]))
        return events, max(arrival, _seconds(events[-1][TIMESTAMP]))

    def _times(self, point, cases, now):
        """
        Je Fall seine Ziehungen bis zum nächsten Event des Arbeitsschritts: (Zeitpunkt in Sekunden seit 1970, inf: keins im
        Horizont; Events davor). Für Fälle ohne offenen Arbeitsschritt die Ankunft dort, für Fälle mit offenem dessen Ende.
        Neu gezogen wird nur für Fälle mit neuen Events oder abgelaufenem Horizont.
        """
        running = {other["case"] for other in cases}
        todo = []
        for other in cases:
            cached = self._drawn.get((other["case"], point.activity))
            if cached is None or cached[0] != len(other["prefix"]) or cached[1] < now:
                todo.append(other)
        for begin in range(0, len(todo), BATCH):
            self._draw(point, todo[begin:begin + BATCH], now)
        # Fälle, die nicht mehr laufen, fallen aus dem Zwischenspeicher, wenn er groß wird
        if len(self._drawn) > 4 * BATCH:
            self._drawn = {key: value for key, value in self._drawn.items() if key[0] in running}
        return [self._drawn[(other["case"], point.activity)][2] for other in cases]

    def _draw(self, point, cases, now):
        encoded, until, lasts = [], [], []
        for other in cases:
            prefix, static, mask, length, elapsed = self.model._encode(other["prefix"])
            last = _seconds(other["prefix"][-1][TIMESTAMP])
            encoded.append((prefix, static, mask, length))
            until.append(elapsed + (now - last) + HORIZON_DAYS * DAY)
            lasts.append(last)
        drawn = self.model.draw_cases(encoded, self.arrival_samples, until=until, stop=point.activity, max_steps=ARRIVAL_STEPS)
        for other, last, suffixes in zip(cases, lasts, drawn):
            draws = []
            for suffix in suffixes:
                offset, before = _until(suffix, point.activity)
                draws.append((last + offset, before))
            self._drawn[(other["case"], point.activity)] = (len(other["prefix"]), now + HORIZON_DAYS * DAY, draws)

"""
Kennzahlen und Ergebnis je Kandidat, für Control Flow (zwei Optionen: keine Aktion, Aktion) und Variablen (zwei oder
mehr Optionen) mit demselben Verfahren. Eine Offline-Auswertung auf dem beobachteten Log (Off-Policy-Evaluation mit
AIPW), ohne Simulator.

Die drei Kennzahlen:
- Policy-Gewinn mit 95-%-Intervall: Wert der Policy "die Option mit dem höchsten geschätzten Wert" (helpers.screening.
  causal.option_scores; bei zwei Optionen der DR-Learner) minus Wert der besten festen Option, beides mit AIPW. Lohnt
  sich eine Entscheidung je Fall?
- Gewinn ggü. Bank mit 95-%-Intervall: Wert der Policy (AIPW) minus Wert der Bank, je Fall gepaart; der Wert der Bank
  ist das beobachtete Outcome (ihre Wahl steht im Log). Ist die Policy besser als die bisherige Praxis?
- Überlappung: Anteil der Fälle, bei denen jede Option eine Propensity über 0,1 / K hat (K = 2: zwischen 0,05 und 0,95).
  Lässt sich der Effekt aus dem Log schätzen? Deckt auch "Option fast immer oder fast nie" ab.
- Placebo d: derselbe Schätzer mit einem Outcome vor der Entscheidung (log1p der Stunden seit Fallbeginn, aus X
  genommen), standardisiert; bei K = 2 der Effekt der Aktion, sonst die größte Differenz einer Option zur häufigsten.
  Erklären die Merkmale die Wahl der Option?
Ergebnis "ja", wenn Überlappung >= 0,5, das Intervall des Policy-Gewinns über 0 und |Placebo d| < 0,1. Bei
Mitarbeiter-Kandidaten mit Pool zählt statt des Policy-Gewinns der Gewinn ggü. Bank: "immer Mitarbeiter X" ist dort kein
fairer Vergleich (X ist oft nicht frei, sein Wert in diesen Fällen nur geschätzt).

Mitarbeiter-Kandidaten mit Pool (candidate.pool, helpers.screening.candidates: kompetent, im Dienst, frei): je Fall
wird nur unter den Mitarbeitern seines Pools gewählt. Optionen sind die Pool-Mitglieder mit mindestens min_cases Fällen
(Standard MIN_RESOURCE_CASES = 1: jeder, der die Tätigkeit im Log selbst ausgeführt hat, mit eigener Schätzung; wer sie
nie ausgeführt hat, ist keine Option, über ihn gibt es keine Daten); mit min_cases > 1 alle übrigen zusammen als
"other" (im Pool eines Falls, wenn dort ein übriger Mitarbeiter ist). Die Propensity gilt je Fall nur über seinen Pool
(auf 1 normiert), Überlappung heißt: jedes Mitglied seines Pools hat eine Propensity über 0,1 / Poolgröße, gezählt
über die Fälle mit Wahl (mindestens zwei Optionen im Pool; mit einer gibt es nichts zu entscheiden); die Policy wählt
je Fall das Mitglied mit dem höchsten geschätzten Wert. Bei Mitarbeitern mit wenigen Fällen ist dieser Wert kaum mehr
als deren wenige Outcomes; wie belastbar er ist, zeigt die Tabelle der Mitarbeiter.

Je Mitarbeiter-Kandidat zusätzlich die Tabelle aller Mitarbeiter, die die Tätigkeit ausgeführt haben
(employee_propensities: ein Modell über alle Mitarbeiter, je Fall auf seinen Pool beschränkt): Fälle, Anteil im Pool,
mittlere Propensity in den Fällen, in denen er im Pool ist, und in seinen eigenen Fällen, Anteil über der Schwelle der
Überlappung; table.attrs["details"], im Notebook aufklappbar mit show_screening, in den Ergebnissen unter
prescription/screening/details.

detailed=True gibt dazu alle Kennzahlen aus: Fälle, Optionen, Wert je Option, was die Policy wählt, Verzögerung bis zur
Aktion, Anteil Systemnutzer; bei zwei Optionen auch mittlerer Effekt, BLP-Test, Anteil mit anderem Vorzeichen, E-Wert.
Unbeobachtete Störfaktoren können alle Schätzungen verzerren.
"""
import numpy as np
import pandas as pd

from .causal import aipw, blp, e_value, interval, nuisances, option_scores, propensity
from .candidates import build_candidates

MIN_GROUP = 100  # mindestens so viele Fälle je Option
MAX_OPTIONS = 20  # höchstens so viele Optionen (die häufigsten); Fälle mit anderen fallen weg
MIN_RESOURCE_CASES = 1  # mit Pool: eigene Option für jeden Mitarbeiter mit so vielen Fällen
MIN_OVERLAP, MAX_PLACEBO = 0.5, 0.1
OTHER = "other"  # mit Pool und min_cases > 1: alle Mitarbeiter ohne eigene Option zusammen als eine Option
SUMMARY = ["Policy-Gewinn", "KI unten", "KI oben", "Gewinn ggü. Bank", "Bank KI unten", "Bank KI oben", "Überlappung", "Placebo d", "Ergebnis"]

# Wird aufgerufen von Screening in main.py
def screen_control_flow(dataset, log, folds=5, seed=0, detailed=False, min_overlap=MIN_OVERLAP):
    return screen(build_candidates(dataset, log, "control_flow", seed), folds, seed, detailed, min_overlap=min_overlap)


# pool: Pool der Mitarbeiter-Kandidaten (helpers.screening.candidates.POOLS, Standard kompetent; None: ohne Pool).
# Eine eigene Option bekommt jeder Mitarbeiter mit mindestens min_cases Fällen (Standard 1); mit min_cases > 1 sind alle
# übrigen zusammen other. min_overlap: Schwelle des Ergebnisses
def screen_variables(dataset, log, folds=5, seed=0, detailed=False, pool="kompetent", min_cases=MIN_RESOURCE_CASES, min_overlap=MIN_OVERLAP):
    return screen(build_candidates(dataset, log, "variable", seed, pool), folds, seed, detailed, show_options=True,
                  min_cases=min_cases, min_overlap=min_overlap)


# Eine Zeile je Kandidat; table.attrs["details"]: je Mitarbeiter-Kandidat die Tabelle seiner Mitarbeiter
def screen(data, folds=5, seed=0, detailed=False, show_options=False, min_cases=MIN_RESOURCE_CASES, min_overlap=MIN_OVERLAP):
    rows, details = [], {}
    for name, candidate in data.items():
        print(f"{name} ...", flush=True)
        # Candidate kommt von build_candidates
        if candidate.kind == "resource" and not candidate.skip and len(candidate.X):
            # Die Mitarbeiter einzeln: Fälle und Propensity je Mitarbeiter
            details[name] = employee_propensities(candidate, folds, seed)
        row = evaluate(candidate, folds, seed, min_cases, min_overlap) if not candidate.skip else {"Ergebnis": f"nicht testbar: {candidate.skip}"}
        rows.append({"Kandidat": name, **row})
    table = pd.DataFrame(rows).set_index("Kandidat")
    if not detailed:
        table = table.reindex(columns=SUMMARY + (["Optionen", "Pool", "Poolgröße Median", "Fälle mit Wahl"] if show_options else []))
    table.attrs["details"] = details
    return table


# Mitarbeiter-Kandidat mit Pool: alle außer den eigenen Optionen own (Standard: mindestens min_cases Fälle) werden zu
# OTHER, auch in den Pools
def grouped(candidate, min_cases, own=None):
    counts = candidate.T.value_counts()
    members = frozenset().union(*set(candidate.pool)) if len(candidate.pool) else frozenset()
    own = set(own) if own is not None else {o for o in counts.index if counts[o] >= min_cases}
    frequent = [o for o in counts.index if o in own and o in members]
    known = set(frequent)
    # other: wer die Tätigkeit im Log ausgeführt hat, aber keine eigene Option ist (wer sie nie ausgeführt hat, ist keine
    # Option: über ihn gibt es keine Daten)
    performers = set(counts.index)
    T = candidate.T.where(candidate.T.isin(known), OTHER)
    pools = {pool: frozenset(pool & known) | ({OTHER} if (pool & performers) - known else frozenset()) for pool in set(candidate.pool)}
    rare = len(performers - known)
    return T, candidate.pool.map(pools), frequent, rare, members


# Alle Kennzahlen eines Kandidaten
# own: mit Pool die Mitarbeiter mit eigener Option (Standard: mindestens min_cases Fälle)
def evaluate(candidate, folds=5, seed=0, min_cases=MIN_RESOURCE_CASES, min_overlap=MIN_OVERLAP, own=None):
    binary = pd.api.types.is_numeric_dtype(candidate.T)
    pooled = candidate.pool is not None
    T_all, pools = candidate.T, candidate.pool
    # Control Flow: 0 = keine Aktion, 1 = Aktion; Variablen: die häufigsten Optionen; mit Pool: die eigenen Optionen und other
    if pooled:
        T_all, pools, frequent, rare, members = grouped(candidate, min_cases, own)
        counts = T_all.value_counts()
        options = frequent + ([OTHER] if counts.get(OTHER, 0) >= min_cases else [])
        least = min_cases
    else:
        counts = T_all.value_counts()
        options = [0, 1] if binary else list(counts[counts >= MIN_GROUP].index[:MAX_OPTIONS])
        least = MIN_GROUP
    keep = T_all.isin(options).to_numpy()
    X, T, Y = candidate.X[keep], T_all[keep], candidate.Y[keep].to_numpy()
    row = {"Entscheidungsfälle": len(X), "Optionen": ", ".join(f"{o} ({counts.get(o, 0)})" for o in options),
           "Verzögerung Median (h)": float(candidate.lag_hours.median()) if len(candidate.lag_hours) else np.nan,
           "Anteil System": float(candidate.system.mean()) if len(candidate.system) else 0.0}
    if pooled:
        outside = candidate.pool_outside
        row["Optionen"] = f"{len(frequent)} eigene (mindestens {min_cases} {'Fall' if min_cases == 1 else 'Fälle'})" + (
            f" + other ({rare} Mitarbeiter)" if OTHER in options else "")
        row["Pool"] = f"{candidate.pool_name}: {len(members)} Mitarbeiter" + (f", gewählter in {100 * outside:.0f} % der Fälle laut Regel außerhalb" if outside >= 0.005 else "")
    if len(options) < 2 or min(counts.get(o, 0) for o in options) < least:
        return {**row, "Ergebnis": f"nein: zu wenige Fälle (unter {least} je Option)"}
    codes = T.map({o: i for i, o in enumerate(options)}).to_numpy(dtype=int)
    # Je Fall: welche Optionen sein Pool erlaubt (ohne Pool: alle)
    allowed = pool_mask(pools[keep], options) if pooled else np.ones((len(X), len(options)), dtype=bool)
    e, mu = nuisances(X, codes, Y, len(options), folds, seed)
    e = within_pool(e, allowed)
    psi = aipw(codes, Y, e, mu)
    scores = option_scores(X, psi, folds, seed)
    policy = np.where(allowed, scores, -np.inf).argmax(axis=1)
    best = int(psi.mean(axis=0).argmax())
    gain = interval(psi[np.arange(len(X)), policy] - psi[:, best])
    # gegenüber der Bank: ihr Wert ist das beobachtete Outcome
    bank = interval(psi[np.arange(len(X)), policy] - Y)
    names = ["nie", "immer"] if binary else options
    size = allowed.sum(axis=1)
    # Überlappung über die Fälle mit Wahl (ohne Pool: alle)
    choice = size >= 2
    overlap = (~allowed | (e > (0.1 / size)[:, None])).all(axis=1)
    if pooled:
        row.update({"Fälle mit Wahl": float(choice.mean()), "Poolgröße Median": float(np.median(size))})
    row.update({
        "Policy-Gewinn": gain[0], "KI unten": gain[1], "KI oben": gain[2],
        "Gewinn ggü. Bank": bank[0], "Bank KI unten": bank[1], "Bank KI oben": bank[2],
        "Überlappung": float(overlap[choice].mean()) if choice.any() else float("nan"),
        "Placebo d": placebo(X, codes, len(options), folds, seed, allowed),
        "beste feste Regel": names[best],
        "Wert je Option": ", ".join(f"{n}: {v:.3g}" for n, v in zip(names, psi.mean(axis=0))),
        "Policy wählt": ", ".join(f"{n}: {s:.2f}" for n, s in zip(names, np.bincount(policy, minlength=len(options)) / len(X))),
    })
    if binary:
        gamma, tau = psi[:, 1] - psi[:, 0], scores[:, 1]
        effect = interval(gamma)
        beta, _, beta_p = blp(gamma, tau)
        row.update({"Anteil Aktion": float(codes.mean()), "Effekt (AIPW)": effect[0], "Effekt KI": f"[{effect[1]:.3g}; {effect[2]:.3g}]",
                    "BLP β": beta, "BLP p": beta_p, "anderes Vorzeichen": float(np.mean(np.sign(tau) != np.sign(effect[0]))),
                    "E-Wert": e_value(effect[0], float(np.std(Y)))})
    row["Ergebnis"] = verdict(row, min_overlap, against_bank=pooled)
    return row


# Je Mitarbeiter, der die Tätigkeit ausgeführt hat: Fälle (Daten für sein Outcome-Modell), wie oft im Pool (wie oft er
# wählbar ist), seine mittlere Propensity in den Fällen, in denen er im Pool ist, und in seinen eigenen Fällen, und wie
# oft er im Pool über der Schwelle der Überlappung liegt (0,1 / Poolgröße: es gibt vergleichbare Fälle, eine Empfehlung
# von ihm ist gestützt). Ein Modell über alle Mitarbeiter (propensity), je Fall auf den Pool beschränkt (ohne Pool: alle
# Mitarbeiter).
def employee_propensities(candidate, folds=5, seed=0):
    counts = candidate.T.value_counts()
    employees = list(counts.index)
    codes = candidate.T.map({o: i for i, o in enumerate(employees)}).to_numpy(dtype=int)
    allowed = pool_mask(candidate.pool, employees) if candidate.pool is not None else np.ones((len(codes), len(employees)), dtype=bool)
    e = within_pool(propensity(candidate.X, codes, len(employees), folds, seed), allowed)
    size = allowed.sum(axis=1)
    above = e > (0.1 / np.maximum(size, 1))[:, None]
    rows = []
    for j, employee in enumerate(employees):
        inside, own = allowed[:, j], codes == j
        rows.append({"Mitarbeiter": employee, "Fälle": int(own.sum()), "im Pool (Anteil der Fälle)": float(inside.mean()),
                     "Propensity im Pool (Mittel)": float(e[inside, j].mean()) if inside.any() else np.nan,
                     "Propensity in seinen Fällen (Mittel)": float(e[own, j].mean()) if own.any() else np.nan,
                     "über der Schwelle, wenn im Pool": float(above[inside, j].mean()) if inside.any() else np.nan})
    return pd.DataFrame(rows).set_index("Mitarbeiter")


# Im Notebook: die Ergebnistabelle; bei Mitarbeiter-Kandidaten klappt ein Klick auf den Namen die Tabelle ihrer
# Mitarbeiter auf (table.attrs["details"]: Fälle, Anteil im Pool, Propensity, Anteil über der Schwelle)
def show_screening(table, digits=3):
    import html
    from IPython.display import HTML
    details = table.attrs.get("details", {})
    shown = table.round(digits).astype(object).where(table.notna(), "")
    shown = shown.apply(lambda column: column.map(lambda value: html.escape(str(value))))

    def label(name):
        if name not in details:
            return html.escape(str(name))
        inner = details[name].round(4).to_html()
        return (f'<details><summary style="cursor:pointer">{html.escape(str(name))}</summary>'
                f'<div style="margin:6px 0 10px 0">{inner}</div></details>')
    shown.index = [label(name) for name in shown.index]
    shown.index.name = "Kandidat"
    return HTML(shown.to_html(escape=False))


# Je Fall und Option: erlaubt der Pool des Falls die Option? (Pools sind oft für viele Fälle dieselben)
def pool_mask(pools, options):
    rows = {pool: np.array([option in pool for option in options]) for pool in set(pools)}
    return np.stack([rows[pool] for pool in pools])


# Propensity je Fall nur über die erlaubten Optionen, auf 1 normiert
def within_pool(e, allowed):
    if allowed.all():
        return e
    e = np.where(allowed, e, 0.0)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


# Placebo d: Effekt auf log1p(Stunden seit Fallbeginn), standardisiert; nan ohne diese Zeit (Entscheidung zu Fallbeginn)
def placebo(X, codes, n_options, folds, seed, allowed=None):
    if "elapsed_h" not in X or not X["elapsed_h"].std() > 0:
        return np.nan
    Y = np.log1p(np.clip(X["elapsed_h"].to_numpy(dtype=float), 0, None))
    e, mu = nuisances(X.drop(columns="elapsed_h"), codes, Y, n_options, folds, seed)
    psi = aipw(codes, Y, within_pool(e, allowed) if allowed is not None else e, mu)
    if n_options == 2:
        return interval(psi[:, 1] - psi[:, 0])[0] / float(np.std(Y))
    reference = int(np.bincount(codes).argmax())
    return float(max(abs(interval(psi[:, a] - psi[:, reference])[0]) for a in range(n_options) if a != reference) / np.std(Y))


# against_bank: der Gewinn ggü. Bank entscheidet statt des Policy-Gewinns (Mitarbeiter-Kandidaten mit Pool)
def verdict(row, min_overlap=MIN_OVERLAP, against_bank=False):
    if not np.isfinite(row["Überlappung"]):
        return "nein: keine Wahl im Pool"
    if row["Überlappung"] < min_overlap:
        return "nein: kaum Überlappung"
    if against_bank and row["Bank KI unten"] <= 0:
        return "nein: kein Gewinn gegenüber der Bank"
    if not against_bank and row["KI unten"] <= 0:
        return "nein: kein Policy-Gewinn gegenüber fester Regel"
    if np.isfinite(row["Placebo d"]) and abs(row["Placebo d"]) >= MAX_PLACEBO:
        return "unsicher: Placebo auffällig"
    return "ja"

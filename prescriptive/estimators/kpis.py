"""
Kennzahlen der Schätzer (XGBEstimator.evaluate, LSTMEstimator.evaluate, psp.evaluate): wie gut sie auf Fällen
vorhersagen, die sie im Training nicht gesehen haben. Für alle Schätzer gleich berechnet, damit ihre Spalten vergleichbar
sind (helpers/training.py, evaluate_estimators).

- Gewinn (je Stelle: Entscheidungspunkt, Anruf-Beginn): die Schätzung für die tatsächlich gewählte Option gegen den
  tatsächlichen Gewinn des Falls: MAE, RMSE, R², Bias, Spearman (Rangfolge der Fälle)
- Effekt (Entscheidungspunkt, nur randomisierte Fälle des RCT): Die Aktion ist zufällig, der beobachtete Effekt ist der
  Unterschied der Mittel. Der Wert der Policy des Schätzers (je Fall die Aktion mit dem höchsten geschätzten Gewinn) ist
  das Mittel der Fälle, deren zufällige Aktion mit ihr übereinstimmt (selbstnormiertes IPW, wie
  evaluation/offline_evaluation.py), mit 95-%-Intervall (Bootstrap) gegen die beste feste Aktion. Ranking: der
  beobachtete Effekt in der Hälfte mit dem höheren geschätzten Effekt minus der in der anderen Hälfte (> 0: der
  Schätzer erkennt, wo die Aktion mehr bringt).

- Suffix (nur PSP): die Kennzahlen des PSP-Papers (probabilistic_suffix_prediction_U-ED-LSTM/src/evaluation_metrics/
  metrics.py, gewählt wie in src/notebooks/evaluation_metric_notebooks/version_2/BPIC17/BPIC17_eval.ipynb), je für den
  wahrscheinlichsten Suffix (ohne Dropout, je Schritt die wahrscheinlichste Aktivität) und die gezogenen Suffixe:
  Suffixlänge MAE (SuffixCountMAE), DLS (NormalizedDamerauLevenshteinDistanceMeanVar), Restzeit MAE aus dem letzten
  case_elapsed_time (LastValueMean2VarMAE: Median der Fehler der gezogenen Suffixe) und aus der Summe der
  event_elapsed_time (SumValueMeanMAE), PIT (LastValuePIT, SumValuesPIT) und die Abdeckung der 50-, 75- und
  95-%-Intervalle (SuffixCountInterval, LastValueInterval, SumValuesInterval). Die Pakete des Paper-Codes
  (pyxdameraulevenshtein, CRPS) fehlen in der Umgebung des Simulators, die Formeln stehen deshalb hier (suffix_case).
  ACHTUNG: Das Paper bewertet alle Prefixe der Testfälle (jede Länge), hier ist es je Fall nur ein Prefix, der am
  Entscheidungspunkt (dort, wo das PSP entscheidet). Die Werte sind deshalb nicht direkt mit dem Paper vergleichbar.

Jede evaluate-Funktion liefert eine Series mit Index (Bereich, Kennzahl), table baut sie aus {Bereich: {Kennzahl: Wert}}.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

PROFIT = "Gewinn am Entscheidungspunkt"
EFFECT = "Effekt am Entscheidungspunkt (RCT)"
CALLER = "Gewinn am Anruf-Beginn"
SUFFIX = "Suffix wie im PSP-Paper (nur Prefix am Entscheidungspunkt, nicht alle Prefixe)"
SUFFIX_NOTE = ("Suffix-Kennzahlen wie im PSP-Paper, aber nur für den Prefix am Entscheidungspunkt je Fall; das Paper bewertet "
               "alle Prefixe der Testfälle. Nicht direkt mit den Werten des Papers vergleichbar.")
INTERVALS = (0.5, 0.75, 0.95)
DAY = 24 * 3600


def table(sections):
    return pd.Series({(section, name): float(value) for section, values in sections.items() for name, value in values.items()}, dtype=float)


# Schätzung gegen tatsächlich; Fälle ohne Schätzung (NaN) zählen nicht
def regression(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) < 2:
        return {"Proben": len(actual)}
    error = predicted - actual
    spread = ((actual - actual.mean()) ** 2).sum()
    return {"Proben": len(actual), "MAE": np.abs(error).mean(), "RMSE": np.sqrt((error ** 2).mean()),
            "R²": 1 - (error ** 2).sum() / spread if spread > 0 else np.nan, "Bias (Schätzung − tatsächlich)": error.mean(),
            "Spearman": pd.Series(predicted).corr(pd.Series(actual), method="spearman")}


# action: zufällige Aktion je Fall, outcome: Gewinn, q: geschätzter Gewinn je Fall und Aktion [n, k], names: die Aktionen
def effect(action, outcome, q, names, n_boot=500, seed=0):
    action, outcome, q = np.asarray(action, dtype=int), np.asarray(outcome, dtype=float), np.asarray(q, dtype=float)
    valid = np.isfinite(q).all(axis=1)
    action, outcome, q = action[valid], outcome[valid], q[valid]
    if len(action) < 2:
        return {"Proben": len(action)}
    chosen = q.argmax(axis=1)

    def value(choice, index):
        match = action[index] == choice[index]
        return outcome[index][match].mean() if match.any() else np.nan

    def difference(index, lower, upper):
        a, y = action[index], outcome[index]
        return y[a == upper].mean() - y[a == lower].mean() if (a == upper).any() and (a == lower).any() else np.nan

    everything = np.arange(len(action))
    constant = {k: np.full(len(action), k) for k in range(len(names))}
    best = max(constant, key=lambda k: value(constant[k], everything))
    rows = {"Proben": len(action)}
    if len(names) == 2:
        estimated = q[:, 1] - q[:, 0]
        high = estimated > np.median(estimated)
        rows[f"Effekt beobachtet ({names[1]} − {names[0]})"] = difference(everything, 0, 1)
        rows["Effekt geschätzt"] = estimated.mean()
        rows["Ranking: Effekt obere − untere Hälfte"] = difference(everything[high], 0, 1) - difference(everything[~high], 0, 1)
    rows["Wert der Policy (IPW)"] = value(chosen, everything)
    for k, name in enumerate(names):
        rows[f"Wert immer {name}"] = value(constant[k], everything)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        index = rng.integers(len(action), size=len(action))
        boot.append(value(chosen, index) - value(constant[best], index))
    rows[f"Policy − immer {names[best]}"] = rows["Wert der Policy (IPW)"] - rows[f"Wert immer {names[best]}"]
    rows["  95-%-KI unten"], rows["  95-%-KI oben"] = np.nanquantile(boot, 0.025), np.nanquantile(boot, 0.975)
    for k, name in enumerate(names):
        rows[f"Anteil {name} gewählt"] = (chosen == k).mean()
    return rows


# Fälle, deren Aktion am Entscheidungspunkt zufällig war (Spalte rct der RCT-Daten; ältere RCT-Daten ohne sie: alle)
def randomized_cases(log, case_column):
    decisions = log[log["stage"] == 0]
    if "rct" in decisions.columns:
        decisions = decisions[decisions["rct"].astype(bool)]
    return set(decisions[case_column])


# Ähnlichkeit zweier Folgen von Aktivitäten (1 = gleich), normierte Damerau-Levenshtein-Distanz (Optimal String
# Alignment) wie NormalizedDamerauLevenshteinDistance im PSP (probabilistic_suffix_prediction_U-ED-LSTM)
def dls(a, b):
    a, b = tuple(a), tuple(b)
    longest = max(len(a), len(b))
    return 1.0 if longest == 0 else 1 - _distance(a, b) / longest


@lru_cache(maxsize=100_000)
def _distance(a, b):
    before, previous = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                current[j] = min(current[j], before[j - 2] + 1)
        before, previous = previous, current
    return previous[len(b)]


# Werte eines Prefix wie die Metriken des Papers (siehe oben). Suffixe: Listen von Events {"activity", "case_elapsed_time"
# (Sekunden seit Fallbeginn), "event_elapsed_time" (Sekunden seit dem vorigen Event)}; true: der tatsächliche, mean: der
# wahrscheinlichste, sampled: die gezogenen. prefix_seconds: case_elapsed_time am letzten Prefix-Event (für leere Suffixe).
# Wird aufgerufen von psp._evaluation_case -> suffix_case (je Fall)
def suffix_case(prefix_seconds, true, mean, sampled):
    def last(suffix):
        return (suffix[-1]["case_elapsed_time"] if suffix else prefix_seconds) / DAY

    def total(suffix):
        return sum(event["event_elapsed_time"] for event in suffix) / DAY

    def activities(suffix):
        return [event["activity"] for event in suffix]

    lengths, lasts, sums = (np.array([f(s) for s in sampled], dtype=float) for f in (len, last, total))
    row = {"length_mean": abs(len(true) - len(mean)), "length_prob": abs(len(true) - lengths.mean()),
           "dls_mean": dls(activities(mean), activities(true)), "dls_prob": float(np.mean([dls(activities(s), activities(true)) for s in sampled])),
           "last_mean": abs(last(mean) - last(true)), "last_prob": float(np.median(np.abs(lasts - last(true)))),
           "sum_mean": abs(total(mean) - total(true)), "sum_prob": abs(sums.mean() - total(true)),
           "pit_last": float(np.mean(lasts <= last(true))), "pit_sum": float(np.mean(sums <= total(true)))}
    for name, values, actual in (("length", lengths, len(true)), ("last", lasts, last(true)), ("sum", sums, total(true))):
        for share in INTERVALS:
            low, high = np.percentile(values, 50 - 50 * share), np.percentile(values, 50 + 50 * share)
            row[f"in_{name}_{int(100 * share)}"] = float(low <= actual <= high)
    return row


# Kalibrierung aus den PIT-Werten: bei guter Kalibrierung gleichverteilt (Mittel 0,5, KS-Abstand 0)
def _ks_uniform(values):
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    return float(max(np.max(np.arange(1, n + 1) / n - values), np.max(values - np.arange(n) / n))) if n else np.nan


# Mittel der Werte je Fall (suffix_case) über die Fälle, wie die Tabelle im BPIC17-Notebook des Papers
# Wird aufgerufen von psp._evaluation_table -> suffix_table
def suffix_table(rows):
    rows = rows.dropna(subset=["dls_prob"]) if "dls_prob" in rows else rows.iloc[0:0]
    if rows.empty:
        return {}
    table = {"Fälle (je ein Prefix, am Entscheidungspunkt)": len(rows),
             "Suffixlänge MAE (Events), wahrscheinlichster Suffix": rows["length_mean"].mean(),
             "Suffixlänge MAE (Events), gezogene Suffixe": rows["length_prob"].mean(),
             "DLS, wahrscheinlichster Suffix": rows["dls_mean"].mean(), "DLS, gezogene Suffixe": rows["dls_prob"].mean(),
             "Restzeit MAE (Tage, letzter case_elapsed_time), wahrscheinlichster": rows["last_mean"].mean(),
             "Restzeit MAE (Tage, letzter case_elapsed_time), gezogene (Median)": rows["last_prob"].mean(),
             "Restzeit MAE (Tage, Summe event_elapsed_time), wahrscheinlichster": rows["sum_mean"].mean(),
             "Restzeit MAE (Tage, Summe event_elapsed_time), gezogene": rows["sum_prob"].mean(),
             "PIT Restzeit (letzter Wert): Mittel, ideal 0,5": rows["pit_last"].mean(),
             "PIT Restzeit (letzter Wert): KS-Abstand, ideal 0": _ks_uniform(rows["pit_last"]),
             "PIT Summe event_elapsed_time: Mittel, ideal 0,5": rows["pit_sum"].mean(),
             "PIT Summe event_elapsed_time: KS-Abstand, ideal 0": _ks_uniform(rows["pit_sum"])}
    for name, label in (("length", "Suffixlänge"), ("last", "Restzeit (letzter Wert)"), ("sum", "Summe event_elapsed_time")):
        for share in INTERVALS:
            table[f"Abdeckung {int(100 * share)}-%-Intervall {label}"] = rows[f"in_{name}_{int(100 * share)}"].mean()
    return table

"""AIPW contrasts for every catalog point, with explicit multi-arm comparisons.

Observational data use estimated propensities (clipped at .01/.99).
RCT data use recorded randomization probabilities without clipping. Staff effects
compare a named employee with a uniform choice from the actual available pool.
Each case contributes its first recorded decision at the point. RCT folds and
standard errors are grouped by simulation run to respect shared resources.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from helpers.datasets import CASE, dataset_info
from .candidates import CandidateData, candidate, prepared_log, ResourcePool, with_allocation
from .causal import XGB
from .events import prefix_until
from .features import features


def estimate_interval(values, groups=None):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan, np.nan, np.nan, np.nan
    mean = float(values.mean())
    if groups is None:
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        critical = float(student_t.ppf(.975, len(values) - 1))
    else:
        groups = np.asarray(groups)
        unique = np.unique(groups)
        if len(unique) < 2:
            return mean, np.nan, np.nan, np.nan
        sums = np.array([(values[groups == group] - mean).sum() for group in unique])
        se = float(np.sqrt(len(unique) / (len(unique) - 1) * np.square(sums).sum()) / len(values))
        critical = float(student_t.ppf(.975, len(unique) - 1))
    return mean, mean - critical * se, mean + critical * se, se


def crossfit_folds(n, folds, seed, groups=None):
    rng = np.random.default_rng(seed)
    units = np.arange(n) if groups is None else np.unique(groups)
    if len(units) < 2:
        raise ValueError("Cross-fitting requires at least two independent units")
    assignments = np.arange(len(units)) % min(folds, len(units))
    rng.shuffle(assignments)
    if groups is None:
        return assignments
    lookup = dict(zip(units, assignments))
    return np.array([lookup[group] for group in groups])


class _ConstantOutcome:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value)


def fit_nuisances(X, codes, Y, n_options, fold, known_e=None, threads=2, seed=0, *, binary_outcome=False):
    from xgboost import XGBClassifier, XGBRegressor
    params = {**XGB, "n_jobs": threads, "random_state": seed}
    e = np.array(known_e, dtype=float, copy=True) if known_e is not None else np.zeros((len(X), n_options))
    mu = np.zeros((len(X), n_options))
    outcome_params = {**params, **({"objective": "binary:logistic", "base_score": .5} if binary_outcome else {})}

    def outcome_model(frame, target):
        if binary_outcome and np.unique(target).size == 1:
            return _ConstantOutcome(target[0])
        return XGBRegressor(**outcome_params).fit(frame, target)

    for k in np.unique(fold):
        train, test = fold != k, fold == k
        if known_e is None:
            present = np.unique(codes[train])
            if len(present) == 1:
                e[np.ix_(test, present)] = 1.0
            else:
                objective = {"objective": "multi:softprob", "num_class": len(present)} if len(present) > 2 else {}
                model = XGBClassifier(**params, **objective)
                model.fit(X[train], np.searchsorted(present, codes[train]))
                e[np.ix_(test, present)] = model.predict_proba(X[test])
        if n_options >= 6:
            categories = list(range(n_options))
            frame = X[train].copy()
            frame["_option"] = pd.Categorical(codes[train], categories=categories)
            model = outcome_model(frame, Y[train])
            part = X[test].copy()
            for option in categories:
                part["_option"] = pd.Categorical(np.full(test.sum(), option), categories=categories)
                mu[test, option] = model.predict(part)
        else:
            fallback = None
            for option in range(n_options):
                own = train & (codes == option)
                if own.any():
                    model = outcome_model(X[own], Y[own])
                else:
                    if fallback is None:
                        fallback = outcome_model(X[train], Y[train])
                    model = fallback
                mu[test, option] = model.predict(X[test])
    return e, mu


def aipw_scores(codes, Y, e, mu, *, randomized):
    if e.shape != mu.shape or not np.isfinite(e).all() or (e < 0).any():
        raise ValueError("Invalid propensity matrix")
    if not np.allclose(e.sum(axis=1), 1, atol=1e-7):
        raise ValueError("Propensities must sum to one within each available action set")
    if (e[np.arange(len(codes)), codes] <= 0).any():
        raise ValueError("Observed choice has zero randomization probability")
    observed = codes[:, None] == np.arange(mu.shape[1])[None, :]
    divisor = np.where(e > 0, e, 1) if randomized else np.clip(e, .01, .99)
    return mu + observed * (Y[:, None] - mu) / divisor


def rct_candidate(name, test, log, records):
    kind = test.get("kind", "action")
    if kind == "action":
        rows = records[records["point"].map(lambda point: name in point.split("+"))].copy()
        rows = rows[rows["choice"].isin(test["action"])]
        options = [0, 1]
        rows["choice"] = rows["choice"].map(dict(zip(test["action"], options)))
        # A merged CF point is uniform over its union of actions. Conditional on
        # either action in this contrast, each has probability 1/2.
        rows["options"] = [(0, 1)] * len(rows)
    else:
        rows = records[records["point"] == name].copy()
        options = list(test.get("labels", ()))
    rows = rows.sort_values(["time"], kind="stable").groupby("case").head(1).set_index("case")
    pos = rows["event_pos"].astype(int)
    pos.index.name = CASE
    prefix = prefix_until(log, pos)
    X = features(log, prefix, pos, dataset_info("bpic17")["screening"])
    X = X.drop(columns=[col for col in test.get("exclude_features", ()) if col in X])
    T = rows["choice"].reindex(X.index)
    data = CandidateData(name, X, T, log.groupby(CASE)["outcome"].first().reindex(X.index),
                         pd.Series(dtype=float), pd.Series(dtype=bool), kind=kind, decision_pos=pos)
    if kind == "resource":
        data.pool = rows["options"].map(frozenset).reindex(X.index)
        data.pool_name = "recorded available employees"
        data.X = with_allocation(log, data, dataset_info("bpic17")["screening"])
        options = sorted(set().union(*data.pool)) if len(data.pool) else []
    groups = rows["simulation_run"].reindex(X.index).to_numpy()
    return data, options, groups


def evaluate_point(name, test, log, records=None, *, seed=42, folds=5, threads=2, outcome_event=None):
    randomized = records is not None
    log = prepared_log(log)
    kind = test.get("kind", "action")
    groups = None
    if randomized:
        data, options, groups = rct_candidate(name, test, log, records)
    else:
        spec = dataset_info("bpic17")["screening"]
        pools = ResourcePool("bpic17", log, "frei") if kind == "resource" else None
        data = candidate(name, log, log.groupby(CASE)["outcome"].first(), test, spec, seed, pools, "frei")
        options = [0, 1] if kind == "action" else list(test.get("labels", sorted(data.T.unique())))
        if kind == "resource":
            # A worker who never performed this activity in the observational
            # log has no support. Do not put model-only extrapolations into the
            # uniform reference pool.
            options = sorted(data.T.unique())
    original_n = len(data.X)
    if outcome_event is not None:
        from .acceptance import prospective_outcome
        data, groups = prospective_outcome(data, log, outcome_event, groups)
        if kind == "resource" and not randomized:
            options = sorted(data.T.unique())
    summary = {"Interventionspunkt": name, "Art": "Control Flow" if kind == "action" else "Mitarbeiter" if kind == "resource" else "Variable",
               "Daten": "RCT" if randomized else "Echtes Log", "Entscheidungsfälle": len(data.X), "Optionen": len(options),
               "Propensity": "bekannte Randomisierung" if randomized else "XGBoost, Cross-Fitting, Clip [0.01; 0.99]",
               "Vergleich": "", "Effekt (AIPW)": np.nan, "KI unten": np.nan, "KI oben": np.nan}
    if outcome_event is not None:
        summary.update({"Zielereignis": outcome_event, "Entscheidungsfälle vor Ausschluss": original_n,
                        "Ziel bereits erreicht: ausgeschlossen": original_n - len(data.X),
                        "Ziel später erreicht": int(data.Y.sum()),
                        "Beobachtete Zielquote": float(data.Y.mean()) if len(data.Y) else np.nan})
    counts = data.T.value_counts()
    observed_options = [o for o in options if counts.get(o, 0) > 0]
    summary["Beobachtete Optionen"] = len(observed_options)
    summary["Minimum Fälle je beobachteter Option"] = int(counts.min()) if len(counts) else 0
    if len(data.X) < 2 or len(observed_options) < 2:
        return {**summary, "Status": "nicht schätzbar: weniger als zwei beobachtete Optionen"}, pd.DataFrame()
    X, Y = data.X.copy(), data.Y.to_numpy(dtype=float)
    for col in X.select_dtypes(include="number"):
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    codes = data.T.map({o: j for j, o in enumerate(options)}).to_numpy(dtype=int)
    if data.pool is None:
        allowed = np.ones((len(X), len(options)), dtype=bool)
    else:
        allowed = np.array([[o in pool for o in options] for pool in data.pool])
    size = allowed.sum(axis=1)
    if not (size > 0).all() or not allowed[np.arange(len(X)), codes].all():
        raise ValueError(f"{name}: chosen arm outside its available pool")
    known_e = allowed / size[:, None] if randomized else None
    fold = crossfit_folds(len(X), folds, seed, groups)
    e, mu = fit_nuisances(X, codes, Y, len(options), fold, known_e, threads, seed,
                          binary_outcome=outcome_event is not None)
    if not randomized:
        e = np.where(allowed, e, 0)
        denominator = e.sum(axis=1, keepdims=True)
        # An unseen employee may be the sole available arm in a held-out row.
        # That deterministic choice has propensity one. Other unseen arms are
        # flagged through overlap and require clipping in observational AIPW.
        e = np.divide(e, denominator, out=allowed / size[:, None], where=denominator > 0)
        if (e[np.arange(len(codes)), codes] <= 0).any():
            e = np.where(allowed, np.maximum(e, 1e-12), 0)
            e /= e.sum(axis=1, keepdims=True)
    psi = aipw_scores(codes, Y, e, mu, randomized=randomized)
    choice = size >= 2
    overlap = (~allowed | (e > (.1 / size)[:, None])).all(axis=1)
    summary["Überlappung"] = float(overlap[choice].mean()) if choice.any() else np.nan
    summary["Fälle mit Wahl"] = int(choice.sum())
    rows = []
    if kind == "resource":
        base = np.where(allowed, psi, 0).sum(axis=1) / size
        reference = "zufälliger verfügbarer Mitarbeiter"
        contrasts = [(j, allowed[:, j] & choice, psi[:, j] - base) for j in range(len(options))]
        representative = counts.index[0]  # selected by frequency only, never by outcome/effect
    else:
        reference = options[0]
        contrasts = [(j, allowed[:, j] & allowed[:, 0], psi[:, j] - psi[:, 0]) for j in range(1, len(options))]
        representative = options[-1]  # prespecified catalog order
    for j, eligible, values in contrasts:
        own = eligible & (codes == j)
        mean, low, high, se = estimate_interval(values[eligible], groups[eligible] if randomized else None)
        observed_count = int(own.sum())
        if observed_count == 0 or (kind != "resource" and not (codes[eligible] == 0).any()):
            mean = low = high = se = np.nan
        row = {"Interventionspunkt": name, "Option": options[j], "Referenz": reference,
               "Vergleichsfälle": int(eligible.sum()), "Gewählt in Vergleichsfällen": observed_count,
               "Effekt (AIPW)": mean, "KI unten": low, "KI oben": high, "Std.-Fehler": se,
               "Überlappung": float(overlap[eligible].mean()) if eligible.any() else np.nan,
               "RCT-Läufe im Vergleich": int(len(np.unique(groups[eligible]))) if randomized else np.nan,
               "Status": "geschätzt" if np.isfinite(low) else "nicht schätzbar: keine Wahl/keine replizierte Beobachtung"}
        if observed_count < folds:
            row["Status"] = "sehr selten: weniger Beobachtungen als Folds" if observed_count else row["Status"]
        rows.append(row)
        if options[j] == representative:
            summary.update({"Vergleich": f"{options[j]} vs. {reference}", "Effekt (AIPW)": mean, "KI unten": low,
                            "KI oben": high, "Status": row["Status"]})
    summary.setdefault("Status", "nicht schätzbar: kein unterstützter Vergleich")
    if kind == "lag_bins" and randomized:
        summary["Vergleich"] += " (zugewiesene Frist, ITT)"
    if summary.get("Überlappung", 1) < .5 and summary["Status"] == "geschätzt":
        summary["Status"] += "; geringe Überlappung"
    return summary, pd.DataFrame(rows)

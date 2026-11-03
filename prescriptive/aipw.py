"""
Doppelt robuste Offline-Bewertung (AIPW) der gespeicherten Offline-Ergebnisse (eval/offline*.pkl) auf denselben
gehaltenen RCT-Fällen.

Bisher (offline_evaluation.py): selbstnormiertes IPW, es zählen nur die Fälle, deren zufällige Aktion zur Wahl der Policy passt
(etwa die Hälfte). AIPW nutzt alle Fälle: Beitrag eines Falls zum Wert der Policy π

    μ_π(x)(x) + 1{T = π(x)} / e · (Y − μ_T(x))

mit der bekannten RCT-Propensity e = 0,5 und einem Outcome-Modell μ_a(x) (geschätzter Gewinn bei Aktion a). Weil e
bekannt ist, bleibt der Wert für jedes μ unverzerrt; ein gutes μ macht ihn nur genauer (engere Intervalle). μ ist ein
XGBoost-Modell (Merkmale wie xgb_estimator), per Cross-Fitting auf den Testfällen selbst geschätzt (5 Teile: auf 4
angepasst, auf dem fünften vorhergesagt), also unabhängig von allen bewerteten Schätzern.

Unterschiede zweier Policies: Mittel der Differenz der Beiträge je Fall, 95-%-Intervall aus der Standardabweichung
(die Beiträge sind je Fall unabhängig).

Aufruf (aus process-simulation-engine/): python -m prescriptive.aipw
"""
import argparse
import os

import numpy as np
import pandas as pd

from prescriptive.enriched_log import CASE, EVENT, with_current_costs
from prescriptive.main import DATA_FOLDER, load_data, save_data, test_cases
from prescriptive.estimators.xgb_estimator import XGB_PARAMS, attribute_categories, training_frames

# Gespeicherte Offline-Ergebnisse: Name -> Datei in eval/
RUNS = {
    "PSP erstes (100 Suffixe)": "offline", "PSP 500 Suffixe": "offline_s500", "PSP 1000 Suffixe": "offline_s1000",
    "PSP EOS-Regel": "offline_eosmask", "PSP normale Kreuzentropie": "offline_ce", "PSP 2 Schichten": "offline_ce_l2",
    "PSP Fallattribute im Decoder": "offline_ce_l2_static", "PSP Verzweigungen x10": "offline_ce_l2_bw10",
    "PSP Paper-Einstellungen": "offline_paper", "PSP 3-fache Daten": "offline_rct3x",
    "PSP Übergänge beschränkt": "offline_dfg", "PSP Fallattribute + Verzweigungen x10": "offline_ce_l2_static_bw10",
    "PSP Fallattribute + Verzweigungen, Übergänge": "offline_ce_l2_static_bw10_dfg",
    "PSP erstes, Decoder-Maske je Suffix": "offline_var", "PSP Fallattribute + Verzweigungen, Übergänge, Decoder-Maske je Suffix": "offline_ce_l2_static_bw10_dfg_var",
    "XGBoost": "offline_xgb", "XGBoost 3-fache Daten": "offline_rct3x_xgb", "LSTM": "offline_lstm", "LSTM 3-fache Daten": "offline_rct3x_lstm",
}


# μ_0(x), μ_1(x) je Fall per Cross-Fitting auf den Testfällen (Merkmale am Entscheidungspunkt wie bei XGBoost)
# Wird aufgerufen von __main__ -> cross_fitted_outcomes
def cross_fitted_outcomes(log, cases, folds=5, seed=0):
    from xgboost import XGBRegressor
    part = log[log[CASE].isin(set(cases))]
    activities = sorted(log[EVENT].dropna().astype(str).unique())
    goals, types, *extra = attribute_categories(log)
    (X, y), _ = training_frames(part, activities, goals, types, [], extra)
    X, y = X.loc[cases], y.loc[cases].to_numpy()
    fold = np.random.default_rng(seed).integers(folds, size=len(X))
    mu = np.zeros((len(X), 2))
    for k in range(folds):
        params = {**XGB_PARAMS, "early_stopping_rounds": None, "n_estimators": 300, "n_jobs": 8}
        model = XGBRegressor(**params, random_state=seed + k)
        model.fit(X[fold != k], y[fold != k], verbose=False)
        for a in (0, 1):
            mu[fold == k, a] = model.predict(X[fold == k].assign(action=a))
    return mu


# Beiträge je Fall zum AIPW-Wert einer Wahl (choice: 0/1 je Fall)
def contributions(choice, action, outcome, mu, propensity=0.5):
    chosen_mu = mu[np.arange(len(choice)), choice]
    observed_mu = mu[np.arange(len(choice)), action]
    return chosen_mu + (action == choice) / propensity * (outcome - observed_mu)


def interval(values):
    mean, half = values.mean(), 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return mean, mean - half, mean + half


# Selbstnormiertes IPW (wie offline_evaluation.py) mit Bootstrap-Intervall, zum Vergleich der Intervallbreite
def snipw_interval(choice, other, action, outcome, n_boot=1000, seed=0):
    def value(c, i):
        return outcome[i][action[i] == c[i]].mean()
    rng = np.random.default_rng(seed)
    full = np.arange(len(action))
    boot = [value(choice, i) - value(other, i) for i in (rng.integers(len(action), size=len(action)) for _ in range(n_boot))]
    return value(choice, full) - value(other, full), np.quantile(boot, 0.025), np.quantile(boot, 0.975)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doubly robust (AIPW) offline evaluation of all saved offline results")
    parser.add_argument("--references", type=str, default="immer call,PSP erstes (100 Suffixe),XGBoost", help="Policies to compare against")
    args = parser.parse_args()
    rows = {}
    for name, file in RUNS.items():
        path = os.path.join(DATA_FOLDER, "eval", file)
        if os.path.exists(path + ".pkl"):
            rows[name] = load_data(path)["rows"].dropna(subset=["q0", "q1"]).set_index("case_id")
    cases = sorted(set.intersection(*[set(r.index) for r in rows.values()]))
    base = rows["PSP erstes (100 Suffixe)"].loc[cases]
    action, outcome = base["action"].to_numpy().astype(int), base["outcome"].to_numpy()
    log = with_current_costs(load_data(os.path.join(DATA_FOLDER, "rct", "log")))
    assert set(cases) <= test_cases(os.path.join(DATA_FOLDER, "psp_rct"))
    mu = cross_fitted_outcomes(log, cases)
    print(f"Fälle: {len(cases)}; Outcome-Modell (Cross-Fitting): R² bei der beobachteten Aktion "
          f"{1 - np.mean((outcome - mu[np.arange(len(cases)), action]) ** 2) / outcome.var():.3f}", flush=True)

    choices = {"immer call": np.ones(len(cases), int), "immer no_call": np.zeros(len(cases), int)}
    choices.update({name: (r.loc[cases, "q1"] > r.loc[cases, "q0"]).astype(int).to_numpy() for name, r in rows.items()})
    contrib = {name: contributions(choice, action, outcome, mu) for name, choice in choices.items()}
    references = [r.strip() for r in args.references.split(",") if r.strip() in choices]
    table = []
    for name, c in contrib.items():
        row = {"Policy": name, "call": choices[name].mean()}
        row["Gewinn je Fall"], row["KI unten"], row["KI oben"] = interval(c)
        for ref in references:
            if ref != name:
                d = interval(c - contrib[ref])
                row[f"gegen {ref}"] = f"{d[0]:+.1f} [{d[1]:+.0f}; {d[2]:+.0f}]"
        table.append(row)
    table = pd.DataFrame(table).set_index("Policy")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 30)
    print(table.round(1).to_string())
    # Intervallbreite AIPW gegen bisheriges IPW für einen Vergleich
    for name in ("PSP 1000 Suffixe", "LSTM"):
        if name in choices:
            s = snipw_interval(choices[name], choices["XGBoost"], action, outcome)
            a = interval(contrib[name] - contrib["XGBoost"])
            print(f"{name} − XGBoost: IPW {s[0]:+.1f} [{s[1]:+.0f}; {s[2]:+.0f}] (Breite {s[2] - s[1]:.0f}),  AIPW {a[0]:+.1f} [{a[1]:+.0f}; {a[2]:+.0f}] (Breite {a[2] - a[1]:.0f})")
    save_data({"table": table, "mu": mu, "cases": cases}, os.path.join(DATA_FOLDER, "eval", "aipw"))

"""
Kausale Schätzer für das Screening, für K >= 2 Optionen (Control Flow: K = 2), alle mit XGBoost und Cross-Fitting
(jeder Fall wird von Modellen geschätzt, die ihn nicht gesehen haben).

- nuisances: Propensity e[:, a] = P(T = a | x) (propensity) und Outcome-Modelle μ[:, a] (je Option ein Modell)
- aipw: doppelt robuste Beiträge ψ[:, a] je Fall ("immer Option a")
- option_scores: geschätzter Wert je Fall und Option; die Policy wählt die Option mit dem höchsten. Bei K = 2 der
  DR-Learner: τ(x) aus Γ = ψ[:, 1] − ψ[:, 0], Werte (0, τ(x))
- blp: Test, ob τ(x) Fälle mit großem und kleinem Effekt trennt (K = 2)
- interval, e_value: Mittel mit 95-%-Intervall, E-Wert
"""
import numpy as np

XGB = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "min_child_weight": 20, "tree_method": "hist",
       "enable_categorical": True, "n_jobs": 8}


def folds_of(n, folds, seed):
    return np.random.default_rng(seed).integers(folds, size=n)


# T: Optionen als Zahlen 0..K-1. Auch für Optionen mit sehr wenigen Fällen (Mitarbeiter mit einem Fall): fehlt eine
# Option in einem Trainingsteil, hat sie dort die Propensity 0 und als Outcome-Modell eines über alle Fälle des Teils
def nuisances(X, T, Y, n_options, folds=5, seed=0):
    from xgboost import XGBRegressor
    fold = folds_of(len(X), folds, seed)
    e, mu = propensity(X, T, n_options, folds, seed), np.zeros((len(X), n_options))
    for k in range(folds):
        train, test = fold != k, fold == k
        overall = None
        for option in range(n_options):
            rows = train & (T == option)
            if rows.any():
                mu[test, option] = XGBRegressor(**XGB, random_state=seed).fit(X[rows], Y[rows]).predict(X[test])
                continue
            if overall is None:
                overall = XGBRegressor(**XGB, random_state=seed).fit(X[train], Y[train])
            mu[test, option] = overall.predict(X[test])
    return e, mu


# Propensity P(T = a | x) mit Cross-Fitting; fehlt eine Option in einem Trainingsteil, bekommt sie dort die
# Wahrscheinlichkeit 0
def propensity(X, codes, n_options, folds=5, seed=0):
    from xgboost import XGBClassifier
    fold = folds_of(len(X), folds, seed)
    e = np.zeros((len(X), n_options))
    for k in range(folds):
        train, test = fold != k, fold == k
        present = np.unique(codes[train])
        if len(present) < 2 or not test.any():
            e[np.ix_(test, present)] = 1.0 / max(len(present), 1)
            continue
        local = np.searchsorted(present, codes[train])
        objective = {"objective": "multi:softprob", "num_class": len(present)} if len(present) > 2 else {}
        model = XGBClassifier(**XGB, **objective, random_state=seed).fit(X[train], local)
        e[np.ix_(test, present)] = model.predict_proba(X[test])
    return e


# ψ[:, a] = μ[:, a] + 1{T = a} (Y − μ[:, a]) / e[:, a]; e auf [0,01; 0,99] begrenzt
def aipw(T, Y, e, mu):
    observed = T[:, None] == np.arange(mu.shape[1])[None, :]
    return mu + observed * (Y[:, None] - mu) / np.clip(e, 0.01, 0.99)


def option_scores(X, psi, folds=5, seed=0):
    from xgboost import XGBRegressor
    fold = folds_of(len(X), folds, seed + 1)
    # Bei zwei Optionen eine Regression auf die Differenz (DR-Learner), sonst eine je Option
    targets = [psi[:, 1] - psi[:, 0]] if psi.shape[1] == 2 else [psi[:, a] for a in range(psi.shape[1])]
    fitted = np.zeros((len(X), len(targets)))
    for k in range(folds):
        for j, target in enumerate(targets):
            model = XGBRegressor(**{**XGB, "min_child_weight": 50}, random_state=seed)
            fitted[fold == k, j] = model.fit(X[fold != k], target[fold != k]).predict(X[fold == k])
    return np.column_stack([np.zeros(len(X)), fitted[:, 0]]) if psi.shape[1] == 2 else fitted


# Γ = a + β (τ − Mittel τ): Steigung β, robuster Standardfehler (HC1), zweiseitiger p-Wert
def blp(gamma, tau):
    from scipy import stats
    Z = np.column_stack([np.ones(len(tau)), tau - tau.mean()])
    coef, *_ = np.linalg.lstsq(Z, gamma, rcond=None)
    residual = gamma - Z @ coef
    bread = np.linalg.inv(Z.T @ Z)
    meat = (Z * residual[:, None] ** 2).T @ Z * len(tau) / (len(tau) - 2)
    se = float(np.sqrt((bread @ meat @ bread)[1, 1]))
    return float(coef[1]), se, float(2 * stats.norm.sf(abs(coef[1] / se)))


# Mittel mit 95-%-Intervall (mean, unten, oben)
def interval(values):
    mean, half = float(np.mean(values)), 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean, mean - half, mean + half


# E-Wert über die standardisierte Differenz d (RR ≈ exp(0,91 d)): so stark müsste ein unbeobachteter Störfaktor mit
# Aktion und Outcome zusammenhängen, um den Effekt zu erklären
def e_value(effect, sd):
    rr = float(np.exp(0.91 * abs(effect) / sd)) if sd > 0 else 1.0
    return rr + np.sqrt(rr * (rr - 1))

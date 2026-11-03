from __future__ import annotations

import numpy as np
import pandas as pd
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .base import AttributePredictorBase
from .utils import to_case_level, resolve_col


class AcceptedPredictor(AttributePredictorBase):
    name = "Accepted"

    def fit(self, df: pd.DataFrame, accepted_col: str = "Accepted") -> "AcceptedPredictor":
        self.rng = np.random.default_rng(self.seed)

        acc_col = resolve_col(df, accepted_col)
        cols = [acc_col, "MonthlyCost", "CreditScore"]
        case_tbl = to_case_level(df, cols).dropna()

        if len(case_tbl) == 0:
            raise ValueError("Keine gültigen Daten für Accepted Predictor gefunden.")

        # Verwende logistische Regression falls verfügbar, sonst verbesserte lineare Formel
        use_lr = False
        if HAS_SKLEARN and len(case_tbl) > 10:
            try:
                X = case_tbl[["CreditScore", "MonthlyCost"]].values
                y = case_tbl[acc_col].values.astype(int)
                
                # Standardisiere Features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Trainiere logistische Regression
                lr_model = LogisticRegression(random_state=self.seed, max_iter=1000)
                lr_model.fit(X_scaled, y)
                
                # Speichere auch base_rate als Fallback
                base_rate = float(case_tbl[acc_col].mean())
                self.model = {
                    "base_rate": base_rate,
                    "use_lr": True,
                    "lr_model": lr_model,    # serialisierbar über pickle
                    "scaler": scaler,        # serialisierbar über pickle
                    "credit_score_mean": float(case_tbl["CreditScore"].mean()),
                    "credit_score_std": float(case_tbl["CreditScore"].std()),
                    "monthly_cost_mean": float(case_tbl["MonthlyCost"].mean()),
                    "monthly_cost_std": float(case_tbl["MonthlyCost"].std()),
                }
                use_lr = True
            except Exception as e:
                # Fallback auf verbesserte lineare Formel
                print(f"Warnung: Logistische Regression fehlgeschlagen ({e}), verwende verbesserte lineare Formel.")
                use_lr = False
        
        if not use_lr:
            # Verbesserte lineare Formel basierend auf tatsächlichen Daten
            base_rate = float(case_tbl[acc_col].mean())
            credit_score_mean = float(case_tbl["CreditScore"].mean())
            credit_score_std = float(case_tbl["CreditScore"].std())
            monthly_cost_mean = float(case_tbl["MonthlyCost"].mean())
            monthly_cost_std = float(case_tbl["MonthlyCost"].std())
            
            # Berechne Koeffizienten basierend auf Korrelationen
            # CreditScore hat moderate positive Korrelation (~0.2)
            # MonthlyCost hat sehr schwache Korrelation (~0.002)
            credit_score_coef = 0.0005  # Reduziert von 0.001
            monthly_cost_coef = -0.000005  # Reduziert von -0.00001
            
            self.model = {
                "base_rate": base_rate,
                "use_lr": False,
                "credit_score_mean": credit_score_mean,
                "credit_score_std": credit_score_std,
                "credit_score_coef": credit_score_coef,
                "monthly_cost_mean": monthly_cost_mean,
                "monthly_cost_std": monthly_cost_std,
                "monthly_cost_coef": monthly_cost_coef,
            }
        
        return self

    def predict_proba(self, monthly_cost: float, credit_score: float) -> float:
        self._require_fitted()
        m = self.model
        assert m is not None

        if m.get("use_lr", False) and "lr_model" in m:
            # Verwende logistische Regression (aus model dict)
            X = np.array([[credit_score, monthly_cost]])
            X_scaled = m["scaler"].transform(X)
            p = float(m["lr_model"].predict_proba(X_scaled)[0, 1])
        else:
            # Verbesserte lineare Formel
            base_rate = m["base_rate"]
            
            # Normalisiere Features (z-score)
            credit_score_norm = (float(credit_score) - m["credit_score_mean"]) / max(m["credit_score_std"], 1.0)
            monthly_cost_norm = (float(monthly_cost) - m["monthly_cost_mean"]) / max(m["monthly_cost_std"], 1.0)
            
            # Angepasste Formel mit normalisierten Features
            p = base_rate
            p += m["credit_score_coef"] * credit_score_norm * 100  # Skaliere zurück
            p += m["monthly_cost_coef"] * monthly_cost_norm * 100
            
            # Clip auf sinnvollen Bereich
            p = float(np.clip(p, 0.01, 0.99))
        
        return p

    def predict(self, monthly_cost: float, credit_score: float) -> bool:
        p = self.predict_proba(monthly_cost, credit_score)
        return bool(self.rng.random() < p)

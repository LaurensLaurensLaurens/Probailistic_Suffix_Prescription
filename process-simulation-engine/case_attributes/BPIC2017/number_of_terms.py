from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttributePredictorBase
from .utils import to_case_level, resolve_col


class NumberOfTermsPredictor(AttributePredictorBase):
    name = "NumberOfTerms"

    def __init__(self, seed: int = 42, sklearn_model=None):
        """
        Zwei Betriebsmodi:
          A) Ohne sklearn_model: resample aus empirischer Verteilung (init_number_of_terms_predictor)
          B) Mit sklearn_model: nutzt final_model.predict(X) wie in Ihrem Notebook
        """
        super().__init__(seed=seed)
        self.sklearn_model = sklearn_model

    def fit(
        self,
        df: pd.DataFrame,
        terms_col: str = "NumberOfTerms",
        offered_col: str = "OfferedAmount",
        credit_col: str = "CreditScore",
    ) -> "NumberOfTermsPredictor":
        self.rng = np.random.default_rng(self.seed)

        t_col = resolve_col(df, terms_col)
        o_col = resolve_col(df, offered_col)
        c_col = resolve_col(df, credit_col)
        
        # Case-level extrahieren mit allen relevanten Features
        cols = ["case:LoanGoal", "case:ApplicationType", t_col, o_col, c_col]
        case_tbl = to_case_level(df, cols).copy()

        case_tbl[t_col] = pd.to_numeric(case_tbl[t_col], errors="coerce")
        case_tbl[o_col] = pd.to_numeric(case_tbl[o_col], errors="coerce")
        case_tbl[c_col] = pd.to_numeric(case_tbl[c_col], errors="coerce")
        case_tbl = case_tbl.dropna(subset=[t_col, o_col, c_col])
        case_tbl = case_tbl[(case_tbl[o_col] > 0) & (case_tbl[c_col] >= 0)]

        # OfferedAmount in Quantil-Bins aufteilen (0-25%, 25-50%, 50-75%, 75-100%)
        if len(case_tbl) > 0:
            offered_quantiles = [0, 0.25, 0.5, 0.75, 1.0]
            offered_bins = case_tbl[o_col].quantile(offered_quantiles).tolist()
            # Sicherstellen, dass bins eindeutig sind
            offered_bins = sorted(list(set(offered_bins)))
            if len(offered_bins) < 2:
                offered_bins = [case_tbl[o_col].min(), case_tbl[o_col].max()]
            case_tbl["_offered_bin"] = pd.cut(
                case_tbl[o_col], 
                bins=offered_bins, 
                labels=False, 
                include_lowest=True,
                duplicates='drop'
            )
            case_tbl["_offered_bin"] = case_tbl["_offered_bin"].fillna(0).astype(int)

            # CreditScore in Standard-Bins aufteilen (wie in anderen Predictors)
            credit_bins = [0, 600, 650, 700, 750, 1000]
            case_tbl["_credit_bin"] = pd.cut(
                case_tbl[c_col],
                bins=credit_bins,
                labels=[0, 1, 2, 3, 4],  # <600, 600-649, 650-699, 700-749, 750-999
                include_lowest=True,
                right=False
            )
            case_tbl["_credit_bin"] = case_tbl["_credit_bin"].fillna(0).astype(int)

            # Mehrstufige Segmentierung: zuerst detailliert, dann immer allgemeiner
            # Level 1: (LoanGoal, ApplicationType, OfferedBin, CreditBin)
            dist_by_4 = {
                k: v[t_col].to_numpy()
                for k, v in case_tbl.groupby(["case:LoanGoal", "case:ApplicationType", "_offered_bin", "_credit_bin"])
            }

            # Level 2: (LoanGoal, ApplicationType, OfferedBin)
            dist_by_3 = {
                k: v[t_col].to_numpy()
                for k, v in case_tbl.groupby(["case:LoanGoal", "case:ApplicationType", "_offered_bin"])
            }

            # Level 3: (LoanGoal, ApplicationType) - Original
            dist_by_2 = {
                k: v[t_col].to_numpy()
                for k, v in case_tbl.groupby(["case:LoanGoal", "case:ApplicationType"])
            }

            # Level 4: Nur nach LoanGoal
            dist_by_1 = {
                str(k): v[t_col].to_numpy()
                for k, v in case_tbl.groupby("case:LoanGoal")
            }

            self.model = {
                "by_4": dist_by_4,  # (LoanGoal, AppType, OfferedBin, CreditBin)
                "by_3": dist_by_3,  # (LoanGoal, AppType, OfferedBin)
                "by_2": dist_by_2,  # (LoanGoal, AppType)
                "by_1": dist_by_1,  # LoanGoal
                "global": case_tbl[t_col].to_numpy(),
                "terms_col": t_col,
                "offered_bins": offered_bins,
                "credit_bins": credit_bins,
            }
        else:
            # Fallback falls keine Daten
            self.model = {
                "by_4": {},
                "by_3": {},
                "by_2": {},
                "by_1": {},
                "global": np.array([]),
                "terms_col": t_col,
                "offered_bins": [0, 1],
                "credit_bins": [0, 600, 650, 700, 750, 1000],
            }
            
        return self

    def predict(
        self,
        offered_amount: float,
        credit_score: float,
        loan_goal: str,
        application_type: str,
    ) -> int:
        """
        Wenn sklearn_model gesetzt ist: exakt Notebook-Pattern (X DataFrame -> model.predict)
        Sonst: mehrstufiges segment-basiertes Resampling aus empirischer Terms-Verteilung.
        Nutzt OfferedAmount und CreditScore für bessere Segmentierung.
        """
        if self.sklearn_model is not None:
            X = pd.DataFrame([{
                "OfferedAmount": float(offered_amount),
                "CreditScore": float(credit_score),
                "case:LoanGoal": loan_goal,
                "case:ApplicationType": application_type,
            }])
            return int(self.sklearn_model.predict(X)[0])

        self._require_fitted()
        m = self.model
        assert m is not None

        # Rückwärtskompatibilität: Prüfe ob altes oder neues Modellformat
        is_old_format = "by_pair" in m
        is_new_format = "by_4" in m

        if is_old_format:
            # Altes Format: nutze nur (LoanGoal, ApplicationType)
            arr = m["by_pair"].get((loan_goal, application_type))
            if arr is None or len(arr) == 0:
                arr = m["global"]
            val = float(self.rng.choice(arr)) if len(arr) else 0.0
            return int(round(val))

        # Neues Format: mehrstufige Segmentierung
        # Binning für OfferedAmount
        offered_bins = m.get("offered_bins", [0, float('inf')])
        offered_bin = 0
        if len(offered_bins) > 1:
            offered_val = float(offered_amount)
            # Finde das richtige Bin
            for i in range(len(offered_bins) - 1):
                if offered_val <= offered_bins[i + 1]:
                    offered_bin = i
                    break
            else:
                # Fallback: letztes Bin
                offered_bin = len(offered_bins) - 2

        # Binning für CreditScore
        credit_bins_list = m.get("credit_bins", [0, 600, 650, 700, 750, 1000])
        credit_bin = 0
        for i in range(len(credit_bins_list) - 1):
            if credit_bins_list[i] <= float(credit_score) < credit_bins_list[i + 1]:
                credit_bin = i
                break
        if float(credit_score) >= credit_bins_list[-1]:
            credit_bin = len(credit_bins_list) - 2

        # Mehrstufige Suche: versuche immer spezifischer zu werden
        # Level 1: (LoanGoal, AppType, OfferedBin, CreditBin)
        key_4 = (loan_goal, application_type, offered_bin, credit_bin)
        arr = m.get("by_4", {}).get(key_4)
        
        # Level 2: (LoanGoal, AppType, OfferedBin)
        if arr is None or len(arr) == 0:
            key_3 = (loan_goal, application_type, offered_bin)
            arr = m.get("by_3", {}).get(key_3)
        
        # Level 3: (LoanGoal, AppType) - Original
        if arr is None or len(arr) == 0:
            key_2 = (loan_goal, application_type)
            arr = m.get("by_2", {}).get(key_2)
        
        # Level 4: Nur LoanGoal
        if arr is None or len(arr) == 0:
            arr = m.get("by_1", {}).get(str(loan_goal))
        
        # Fallback: globale Verteilung
        if arr is None or len(arr) == 0:
            arr = m.get("global", np.array([]))

        val = float(self.rng.choice(arr)) if len(arr) else 0.0
        return int(round(val))

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttributePredictorBase
from .utils import to_case_level, resolve_col


class SelectedPredictor(AttributePredictorBase):
    name = "Selected"

    def fit(
        self,
        df: pd.DataFrame,
        selected_col: str = "Selected",
        tau: float = 200.0,  # Shrinkage wie im Notebook
    ) -> "SelectedPredictor":
        self.rng = np.random.default_rng(self.seed)

        sel_col = resolve_col(df, selected_col)
        cols = ["case:LoanGoal", "case:ApplicationType", "case:RequestedAmount", "CreditScore", sel_col]
        case_tbl = to_case_level(df, cols).copy()

        case_tbl[sel_col] = case_tbl[sel_col].astype(bool)

        grp = case_tbl.groupby(["case:LoanGoal", "case:ApplicationType"])
        sel_rate = grp[sel_col].mean()
        sel_n = grp[sel_col].size()
        global_rate = float(case_tbl[sel_col].mean())

        self.model = {
            "by_pair": sel_rate.to_dict(),      # (loan_goal, app_type) -> p
            "n_by_pair": sel_n.to_dict(),       # (loan_goal, app_type) -> n
            "global": global_rate,
            "tau": float(tau),
        }
        return self

    def predict_proba(self, loan_goal: str, application_type: str, credit_score: float) -> float:
        self._require_fitted()
        m = self.model
        assert m is not None

        seg = (loan_goal, application_type)

        p_global = m["global"]
        p_seg = m["by_pair"].get(seg)
        n_seg = m.get("n_by_pair", {}).get(seg, 0)
        tau = float(m.get("tau", 200.0))

        # Shrinkage wie Notebook:
        if p_seg is None:
            base_p = p_global
        else:
            base_p = (n_seg / (n_seg + tau)) * p_seg + (tau / (n_seg + tau)) * p_global

        # Logit
        logit_p = np.log(base_p / (1 - base_p))

        # CreditScore-Effekt (wie Notebook)
        beta = 0.03 / 50 if application_type == "Limit raise" else 0.015 / 50
        logit_p += beta * (float(credit_score) - 650.0)

        # Fix für Limit raise (wie Notebook)
        if application_type == "Limit raise":
            logit_p += 0.45

        p = 1.0 / (1.0 + np.exp(-logit_p))
        return float(np.clip(p, 0.0, 1.0))

    def predict(self, loan_goal: str, application_type: str, credit_score: float) -> bool:
        p = self.predict_proba(loan_goal, application_type, credit_score)
        return bool(self.rng.random() < p)

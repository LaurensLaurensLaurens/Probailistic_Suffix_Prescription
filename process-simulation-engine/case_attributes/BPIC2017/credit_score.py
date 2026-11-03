from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttributePredictorBase
from .utils import to_case_level, resolve_col


class CreditScorePredictor(AttributePredictorBase):
    name = "CreditScore"

    def fit(self, df: pd.DataFrame, credit_score_col: str = "CreditScore") -> "CreditScorePredictor":
        self.rng = np.random.default_rng(self.seed)

        cs_col = resolve_col(df, credit_score_col)
        cols = ["case:LoanGoal", "case:ApplicationType", cs_col]
        case_tbl = to_case_level(df, cols)

        case_tbl[cs_col] = pd.to_numeric(case_tbl[cs_col], errors="coerce")
        case_tbl = case_tbl.dropna(subset=[cs_col])

        cs_global = case_tbl[cs_col].to_numpy()

        cs_by_pair = {
            (lg, at): sub[cs_col].to_numpy()
            for (lg, at), sub in case_tbl.groupby(["case:LoanGoal", "case:ApplicationType"])
        }
        cs_by_lg = {lg: sub[cs_col].to_numpy() for lg, sub in case_tbl.groupby("case:LoanGoal")}
        cs_by_at = {at: sub[cs_col].to_numpy() for at, sub in case_tbl.groupby("case:ApplicationType")}

        self.model = {
            "global": cs_global,
            "by_pair": cs_by_pair,
            "by_lg": cs_by_lg,
            "by_at": cs_by_at,
            "final": self._fit_final(df, cs_col),
        }
        return self

    # Selected, CreditScore und Accepted stehen im echten Log erst später fest, als ihr Event (O_Create Offer) vermuten
    # lässt. Über die OfferID lassen sich die Angebots-Events eines Angebots verknüpfen:
    #   Selected: nur zurückgekommene Angebote (O_Returned), davon 93,4 % ("signed by the customer", BPIC2017 im ProM-Forum),
    #     je nach Ausgang des Angebots verschieden (das von der Bank angenommene ist praktisch immer Selected)
    #   CreditScore > 0: 88,6 % der von der Bank angenommenen Angebote (O_Accepted), sonst 0 (bis auf 0,02 %)
    #   Accepted: "whether the offer is still acceptable", letzter Stand der Prüfung, je Ausgang des Angebots verschieden
    # Gelernt werden diese Anteile und die echten Scores > 0. Ohne Event-Spalten (concept:name, EventID, OfferID) None.
    @staticmethod
    def _fit_final(df: pd.DataFrame, cs_col: str):
        if not {"concept:name", "EventID", "OfferID"} <= set(df.columns):
            return None
        create = df[df["concept:name"] == "O_Create Offer"][["EventID", cs_col, "Selected", "Accepted"]].copy()
        for column in ("Selected", "Accepted"):
            create[column] = create[column].astype(str).str.lower().isin(["true", "1", "1.0"])
        create[cs_col] = pd.to_numeric(create[cs_col], errors="coerce").fillna(0)
        events = df[df["concept:name"].isin(["O_Returned", "O_Accepted", "O_Refused", "O_Cancelled"]) & df["OfferID"].notna()]
        seen = events.groupby("OfferID")["concept:name"].agg(set)
        happened = create["EventID"].map(seen).apply(lambda value: value if isinstance(value, set) else set())
        returned = happened.apply(lambda value: "O_Returned" in value)
        status = happened.apply(lambda value: "accepted" if "O_Accepted" in value else "refused" if "O_Refused" in value
                                else "cancelled" if "O_Cancelled" in value else "open")
        return {
            "p_selected_returned": {name: float(create.loc[returned & (status == name), "Selected"].mean()) for name in ("accepted", "refused", "cancelled", "open")
                                    if (returned & (status == name)).any()},
            "p_score_accepted": float((create.loc[status == "accepted", cs_col] > 0).mean()),
            "positive": create.loc[create[cs_col] > 0, cs_col].to_numpy(dtype=float),
            "p_accepted": {name: float(create.loc[status == name, "Accepted"].mean()) for name in ("accepted", "refused", "cancelled", "open")
                           if (status == name).any()},
        }

    # Selected, CreditScore und Accepted eines Angebots, wenn sein Verlauf feststeht: returned (O_Returned kam) und status
    # ("accepted" / "refused" / "cancelled" / "open"). Wird aufgerufen von AttributeSimulationEngine.final_offer_attributes
    def final_attributes(self, returned: bool, status: str, seed: int) -> dict:
        self._require_fitted()
        final = self.model.get("final")
        rng = np.random.default_rng(seed)
        selected = bool(returned and rng.random() < final["p_selected_returned"].get(status, 0.934))
        score = float(rng.choice(final["positive"])) if status == "accepted" and rng.random() < final["p_score_accepted"] else 0.0
        accepted = bool(rng.random() < final["p_accepted"].get(status, final["p_accepted"].get("open", 0.5)))
        return {"Selected": selected, "CreditScore": score, "Accepted": accepted}

    def predict(self, loan_goal: str, application_type: str) -> float:
        self._require_fitted()
        m = self.model
        assert m is not None

        arr = m["by_pair"].get((loan_goal, application_type))
        if arr is None or len(arr) == 0:
            arr = m["by_lg"].get(loan_goal)
        if arr is None or len(arr) == 0:
            arr = m["by_at"].get(application_type)
        if arr is None or len(arr) == 0:
            arr = m["global"]

        return float(self.rng.choice(arr))

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttributePredictorBase
from .utils import resolve_col

OFFER_ACTIVITY = "O_Create Offer"


def _offer_group(offer_number: int) -> int:
    # 1 = erstes Angebot, 2 = jedes weitere. Im echten Log ist das erste Angebot in 74 % der Fälle genau der
    # gewünschte Betrag, das zweite in 52 %, das dritte in 38 %.
    return 1 if offer_number <= 1 else 2


class OfferedAmountPredictor(AttributePredictorBase):
    name = "OfferedAmount"

    def fit(self, df: pd.DataFrame, offered_col: str = "OfferedAmount") -> "OfferedAmountPredictor":
        """
        Verhältnis OfferedAmount / RequestedAmount je Angebot, getrennt nach (LoanGoal, ApplicationType) und
        danach, ob es das erste oder ein weiteres Angebot des Falls ist.
        """
        self.rng = np.random.default_rng(self.seed)

        off_col = resolve_col(df, offered_col)
        offers = df
        if "concept:name" in df.columns:
            offers = df[df["concept:name"] == OFFER_ACTIVITY]
        if "time:timestamp" in offers.columns:
            offers = offers.sort_values("time:timestamp", kind="stable")
        offers = offers[["case:concept:name", "case:LoanGoal", "case:ApplicationType", "case:RequestedAmount", off_col]].copy()
        offers["case:RequestedAmount"] = pd.to_numeric(offers["case:RequestedAmount"], errors="coerce")
        offers[off_col] = pd.to_numeric(offers[off_col], errors="coerce")
        offers = offers.dropna(subset=["case:LoanGoal", "case:ApplicationType", "case:RequestedAmount", off_col])
        offers["_group_"] = (offers.groupby("case:concept:name").cumcount() + 1).map(_offer_group)
        # Anträge ohne gewünschten Betrag (RequestedAmount = 0, 9,6 % der Angebote): Angebotsbetrag direkt aus diesen Angeboten
        zero = offers[offers["case:RequestedAmount"] == 0]
        zero_requested = {
            "by_pair_offer": {k: v[off_col].to_numpy() for k, v in zero.groupby(["case:LoanGoal", "case:ApplicationType", "_group_"])},
            "by_offer": {k: v[off_col].to_numpy() for k, v in zero.groupby("_group_")},
        }
        offers = offers[offers["case:RequestedAmount"] > 0]

        # Nicht bei 1 kappen: 12 % der echten Angebote liegen über dem gewünschten Betrag. Nur einzelne extreme
        # Verhältnisse (sehr kleiner gewünschter Betrag) werden auf das 99-%-Quantil begrenzt.
        ratio = (offers[off_col] / offers["case:RequestedAmount"]).clip(lower=0)
        max_ratio = float(np.quantile(ratio, 0.99))
        offers["_ratio_"] = ratio.clip(upper=max_ratio)

        # Folgeangebote hängen am vorigen Angebot: In 63 % der Fälle ist es gleich, sonst meist das 0,8- bis 1,2-fache.
        # Verhältnis zum vorigen Angebot desselben Falls, auf das 1-%- bis 99-%-Quantil begrenzt.
        previous = offers.groupby("case:concept:name")[off_col].shift(1)
        next_ratio = (offers[off_col] / previous.where(previous > 0)).dropna()
        next_ratio = next_ratio.clip(*np.quantile(next_ratio, [0.01, 0.99])).to_numpy()

        first = offers[offers["_group_"] == 1]
        self.model = {
            "next_ratio": next_ratio,
            "by_pair_offer": {k: v["_ratio_"].to_numpy() for k, v in offers.groupby(["case:LoanGoal", "case:ApplicationType", "_group_"])},
            "by_offer": {k: v["_ratio_"].to_numpy() for k, v in offers.groupby("_group_")},
            "by_pair": {k: v["_ratio_"].to_numpy() for k, v in first.groupby(["case:LoanGoal", "case:ApplicationType"])},
            "global": first["_ratio_"].to_numpy(),
            "max_ratio": max_ratio,
            "zero_requested": zero_requested,
        }
        return self

    def predict(
        self,
        loan_goal: str,
        application_type: str,
        requested_amount: float,
        mode: str = "sample",   # "sample" | "mean" | "median"
        n_draws: int = 1,
        seed: int | None = None,
        offer_number: int = 1,
        previous_offered: float | None = None,
    ) -> float:
        """
        Erstes Angebot: Offered/Requested-Verhältnis resamplen, (LoanGoal, ApplicationType) -> global.
        Weitere Angebote: das vorige Angebot mal ein echtes Verhältnis zweier aufeinanderfolgender Angebote;
        ohne vorigen Wert wie das erste, aber aus den weiteren Angeboten.
        Antrag ohne Betrag (requested_amount = 0): erstes Angebot direkt aus den echten Angeboten ohne gewünschten
        Betrag, (LoanGoal, ApplicationType) -> alle; weitere wie oben am vorigen Angebot.
        """
        self._require_fitted()
        m = self.model
        assert m is not None

        req = float(requested_amount)
        if not np.isfinite(req) or req < 0:
            return float("nan")

        group = _offer_group(offer_number)
        rng = self.rng if seed is None else np.random.default_rng(int(seed))
        next_ratio = m.get("next_ratio")
        if req == 0 and not (offer_number > 1 and next_ratio is not None and previous_offered is not None and np.isfinite(previous_offered) and previous_offered > 0):
            # Antrag ohne Betrag: Angebotsbetrag aus den echten Angeboten ohne gewünschten Betrag, (LoanGoal, ApplicationType) -> alle
            zero = m.get("zero_requested") or {}
            amounts = zero.get("by_pair_offer", {}).get((loan_goal, application_type, group))
            if amounts is None or len(amounts) < 5:
                amounts = zero.get("by_offer", {}).get(group)
            if amounts is None or len(amounts) == 0:
                return float("nan")
            if mode == "mean":
                return float(np.mean(amounts))
            draws = rng.choice(amounts, size=max(1, int(n_draws)), replace=True).astype(float)
            return float(np.median(draws)) if mode == "median" else float(draws[0])
        arr = m.get("by_pair_offer", {}).get((loan_goal, application_type, group))
        if arr is None or len(arr) < 5:
            arr = m.get("by_offer", {}).get(group) if group > 1 else m["by_pair"].get((loan_goal, application_type))
        if arr is None or len(arr) == 0:
            arr = m["global"]
        # Ältere gespeicherte Modelle (ohne max_ratio) waren auf [0, 1] gekappt
        max_ratio = m.get("max_ratio", 1.0)

        if mode == "mean":
            r = float(np.mean(arr)) if len(arr) else 0.0
            return float(np.clip(r, 0.0, max_ratio) * req)

        n = max(1, int(n_draws))

        if offer_number > 1 and next_ratio is not None and previous_offered is not None and np.isfinite(previous_offered) and previous_offered > 0:
            if mode == "mean":
                return float(previous_offered * np.mean(next_ratio))
            offered = previous_offered * rng.choice(next_ratio, size=n, replace=True)
            return float(np.median(offered)) if mode == "median" else float(offered[0])

        draws = rng.choice(arr, size=n, replace=True) if len(arr) else np.zeros(n, dtype=float)
        offered = np.clip(draws.astype(float), 0.0, max_ratio) * req

        if mode == "median":
            return float(np.median(offered))
        return float(offered[0])

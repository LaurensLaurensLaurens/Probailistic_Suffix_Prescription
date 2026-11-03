from __future__ import annotations

import numpy as np
import pandas as pd

from .base import AttributePredictorBase


class MonthlyCostPredictor(AttributePredictorBase):
    """
    Ratio-Resampling Predictor für MonthlyCost.

    Kernidee: MonthlyCost ≈ OfferedAmount × ratio(NumberOfTerms, ApplicationType)
    Das Ratio MonthlyCost/OfferedAmount ist fast vollständig durch NumberOfTerms
    bestimmt (CV ≈ 1–3% innerhalb jeder NumberOfTerms-Gruppe). Statt einer linearen
    Regression (die die Verteilung verzerrt) samplen wir direkt aus der empirischen
    Ratio-Verteilung, segmentiert nach (NumberOfTerms, ApplicationType).

    Fallback-Hierarchie:
        Level 1: (ApplicationType, NumberOfTerms) – feingranular
        Level 2: NumberOfTerms allein
        Level 3: (ApplicationType, NumberOfTerms_bin)  – 12-Monats-Buckets
        Level 4: GlobalRatio
    """

    name = "MonthlyCost"

    def __init__(self, seed: int = 42, artifact: dict | None = None):
        super().__init__(seed=seed)
        self.model = artifact

    # ──────────────────────────────────────────────────────────────────────────
    # fit
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "MonthlyCostPredictor":
        """
        Trainiert das Ratio-Modell aus dem Event-Log df.
        Benötigt: case:concept:name, OfferedAmount, NumberOfTerms,
                  MonthlyCost, case:ApplicationType
        """
        if self.model is not None:
            return self

        required = [
            "case:concept:name",
            "OfferedAmount",
            "NumberOfTerms",
            "MonthlyCost",
            "case:ApplicationType",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Fehlende Spalten für MonthlyCostPredictor.fit(): {missing}")

        # Case-level Tabelle
        d = (
            df.groupby("case:concept:name")[required[1:]]
              .first()
              .reset_index(drop=True)
        )

        for c in ["OfferedAmount", "NumberOfTerms", "MonthlyCost"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")

        d = d.dropna(subset=["OfferedAmount", "NumberOfTerms", "MonthlyCost"]).copy()
        d = d[(d["OfferedAmount"] > 0) & (d["NumberOfTerms"] > 0)]
        d["NumberOfTerms"] = d["NumberOfTerms"].round().astype(int)

        # Ratio berechnen
        d["_ratio_"] = d["MonthlyCost"] / d["OfferedAmount"]

        # Outlier entfernen: Ratio muss physikalisch sinnvoll sein
        # (Monatszahlung zwischen 0,1% und 50% des Kreditbetrags)
        d = d[(d["_ratio_"] > 0.001) & (d["_ratio_"] < 0.5)]

        # ── Level 1: (ApplicationType, NumberOfTerms) ──────────────────────────
        by_at_n = {
            k: v["_ratio_"].to_numpy()
            for k, v in d.groupby(["case:ApplicationType", "NumberOfTerms"])
            if len(v) >= 1
        }

        # ── Level 2: NumberOfTerms allein ─────────────────────────────────────
        by_n = {
            int(k): v["_ratio_"].to_numpy()
            for k, v in d.groupby("NumberOfTerms")
            if len(v) >= 1
        }

        # ── Level 3: (ApplicationType, NumberOfTerms-Bucket à 12 Monate) ──────
        d["_n_bucket_"] = (d["NumberOfTerms"] // 12) * 12   # z.B. 60→60, 58→48
        by_at_bucket = {
            k: v["_ratio_"].to_numpy()
            for k, v in d.groupby(["case:ApplicationType", "_n_bucket_"])
            if len(v) >= 3
        }

        # ── Level 4: Global ────────────────────────────────────────────────────
        global_ratios = d["_ratio_"].to_numpy()

        self.model = {
            "by_at_n":     by_at_n,       # (app_type, n_terms) → ratios
            "by_n":        by_n,           # n_terms → ratios
            "by_at_bucket": by_at_bucket,  # (app_type, n_bucket) → ratios
            "global":      global_ratios,
        }
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # set_artifact (Kompatibilität mit altem Interface)
    # ──────────────────────────────────────────────────────────────────────────

    def set_artifact(self, artifact: dict) -> "MonthlyCostPredictor":
        self.model = artifact
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # predict
    # ──────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        offered_amount: float,
        number_of_terms: int,
        credit_score: float,           # wird für Kompatibilität akzeptiert
        application_type: str,
    ) -> float:
        self._require_fitted()
        m = self.model
        assert m is not None

        oa = float(offered_amount)
        n  = int(round(number_of_terms))
        at = str(application_type)

        if not np.isfinite(oa) or oa <= 0 or n <= 0:
            return float("nan")

        # Ratio-Array in Fallback-Hierarchie suchen
        arr = m["by_at_n"].get((at, n))

        if arr is None or len(arr) == 0:
            arr = m["by_n"].get(n)

        if arr is None or len(arr) == 0:
            n_bucket = (n // 12) * 12
            arr = m["by_at_bucket"].get((at, n_bucket))

        if arr is None or len(arr) == 0:
            arr = m["global"]

        ratio = float(self.rng.choice(arr))
        return float(max(oa * ratio, 0.0))

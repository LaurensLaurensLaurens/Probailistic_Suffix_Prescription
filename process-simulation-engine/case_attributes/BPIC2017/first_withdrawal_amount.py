from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit, expit

from .base import AttributePredictorBase
from .utils import to_case_level, resolve_col, detect_rounding_step


class FirstWithdrawalAmountPredictor(AttributePredictorBase):
    """
    Mixture-Modell für FirstWithdrawalAmount:

      FWA_pct = (FirstWithdrawalAmount / OfferedAmount) * 100

      Verteilung = P(zero)  * δ(0)          ← kein Auszahlungsauftakt  (≤ ZERO_THRESHOLD)
                 + P(full)  * δ(100)         ← volle Auszahlung         (≥ FULL_THRESHOLD)
                 + P(part.) * KDE_logit(x)   ← Teilauszahlung (1–99 %)

    Segmentierung nach (LoanGoal, ApplicationType), Fallback auf LoanGoal → global.

    Kernverbesserungen gegenüber dem alten Ansatz:
      1. Explizite Massenatome bei 0 % und 100 % statt breiter 0–5 %/95–100 %-Bins.
      2. Logit-KDE für den kontinuierlichen Teilbereich (bounded interpolation).
      3. Gruppenspezifische Wahrscheinlichkeiten (z. B. „Limit raise" → p_full ≈ 0).
    """

    name = "FirstWithdrawalAmount"

    # Klassenkonstanten – werden auch im Modell-Dict gespeichert
    ZERO_THRESHOLD: float = 1.0   # FWA_pct ≤ 1 %  → „zero"
    FULL_THRESHOLD: float = 99.0  # FWA_pct ≥ 99 % → „full"
    KDE_MIN_SAMPLES: int = 8      # Mindest-Stichprobe für Logit-KDE

    def __init__(self, seed: int = 42, apply_rounding: bool = False):
        super().__init__(seed=seed)
        self.apply_rounding = bool(apply_rounding)

    # ─── interne Hilfsmethoden ────────────────────────────────────────────────

    def _build_segment(
        self,
        arr_pct: np.ndarray,
        zero_thr: float,
        full_thr: float,
    ) -> dict | None:
        """
        Baut ein Mixture-Segment aus einem Array von FWA-Prozentwerten (0–100).

        Rückgabe-Dict:
            p_zero       – Anteil ≤ zero_thr
            p_full       – Anteil ≥ full_thr
            p_partial    – Anteil (zero_thr, full_thr)
            partial_vals – Rohe Prozentwerte im partiellen Bereich
            kde_logit    – scipy gaussian_kde auf logit-transformierten Werten (oder None)
            logit_vals   – gespeicherte logit-Werte für Bandbreiten-Reproduzierbarkeit
            bw_factor    – KDE-Bandbreitenfaktor (Scott)
        """
        arr_pct = np.asarray(arr_pct, dtype=float)
        n = len(arr_pct)
        if n == 0:
            return None

        mask_zero = arr_pct <= zero_thr
        mask_full = arr_pct >= full_thr
        mask_part = ~mask_zero & ~mask_full

        p_zero = float(mask_zero.sum() / n)
        p_full = float(mask_full.sum() / n)
        p_part = float(mask_part.sum() / n)

        partial_vals = arr_pct[mask_part].copy()

        # Logit-KDE für den partiellen Bereich (bounded distribution → logit-transform)
        kde_logit = None
        logit_vals = np.array([], dtype=float)
        bw_factor = np.nan

        if len(partial_vals) >= self.KDE_MIN_SAMPLES:
            try:
                range_w = full_thr - zero_thr
                eps = 1e-6
                normalized = (partial_vals - zero_thr) / range_w
                normalized = np.clip(normalized, eps, 1.0 - eps)
                logit_vals = logit(normalized)
                kde_logit = stats.gaussian_kde(logit_vals)  # Scott's Regel
                bw_factor = float(kde_logit.factor)
            except Exception:
                kde_logit = None
                logit_vals = np.array([], dtype=float)
                bw_factor = np.nan

        return {
            "p_zero": p_zero,
            "p_full": p_full,
            "p_partial": p_part,
            "partial_vals": partial_vals,
            "kde_logit": kde_logit,
            "logit_vals": logit_vals,
            "bw_factor": bw_factor,
        }

    def _sample_from_segment(
        self,
        seg: dict,
        rng: np.random.Generator,
        zero_thr: float,
        full_thr: float,
        mode: str = "sample",
    ) -> float:
        """
        Zieht einen FWA-Prozentwert (0–100) aus dem gegebenen Segment.

        mode="sample" → stochastische Ziehung
        mode="mean"   → Erwartungswert
        """
        p_zero = seg["p_zero"]
        p_full = seg["p_full"]
        p_part = seg["p_partial"]
        partial_vals = seg["partial_vals"]
        kde_logit = seg.get("kde_logit")

        if mode == "mean":
            part_mean = float(np.mean(partial_vals)) if len(partial_vals) > 0 else (zero_thr + full_thr) / 2
            return float(p_zero * 0.0 + p_full * 100.0 + p_part * part_mean)

        # ── Kategorie ziehen ─────────────────────────────────────────────────
        cats = ["zero", "full", "partial"]
        probs = np.array([p_zero, p_full, p_part], dtype=float)

        # Sonderfall: p_partial > 0 aber keine Werte → partial auf full schieben
        if p_part > 0 and len(partial_vals) == 0:
            probs[1] += probs[2]
            probs[2] = 0.0

        total = probs.sum()
        if total <= 0:
            return 0.0
        probs /= total

        cat = str(rng.choice(cats, p=probs))

        if cat == "zero":
            return 0.0
        if cat == "full":
            return 100.0

        # ── Partial: Logit-KDE oder Bootstrap ────────────────────────────────
        if kde_logit is not None:
            range_w = full_thr - zero_thr
            MAX_ATTEMPTS = 30
            for _ in range(MAX_ATTEMPTS):
                seed_i = int(rng.integers(0, 2 ** 31))
                logit_s = float(kde_logit.resample(1, seed=seed_i)[0, 0])
                pct = float(expit(logit_s)) * range_w + zero_thr
                if zero_thr < pct < full_thr:
                    return float(pct)
            # Fallback bei gescheitertem Rejection Sampling

        if len(partial_vals) > 0:
            return float(rng.choice(partial_vals))
        return float((zero_thr + full_thr) / 2)

    # ─── öffentliche API ──────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        fwa_col: str = "FirstWithdrawalAmount",
        offered_col: str = "OfferedAmount",
        group_cols=("case:LoanGoal", "case:ApplicationType"),
    ) -> "FirstWithdrawalAmountPredictor":
        """
        Fittet das Mixture-KDE-Modell.

        Ablauf:
          1. FWA_pct = (FWA / OfferedAmount) × 100 auf Case-Ebene berechnen.
          2. Für jedes Segment (by_2, by_1, global) ein Mixture-Segment bauen.
          3. Modell als Dict persistieren (Format-Tag: "mixture_kde_v2").
        """
        self.rng = np.random.default_rng(self.seed)

        fwa_c = resolve_col(df, fwa_col)
        off_c = resolve_col(df, offered_col)

        cols = list(group_cols) + [off_c, fwa_c]
        case_tbl = to_case_level(df, cols).copy()

        case_tbl[off_c] = pd.to_numeric(case_tbl[off_c], errors="coerce")
        case_tbl[fwa_c] = pd.to_numeric(case_tbl[fwa_c], errors="coerce")

        mask = (
            case_tbl[off_c].notna() & (case_tbl[off_c] > 0) &
            case_tbl[fwa_c].notna() & (case_tbl[fwa_c] >= 0)
        )
        d = case_tbl.loc[mask].copy()

        d["_fwa_pct"] = (d[fwa_c] / d[off_c]) * 100.0
        d = d[(d["_fwa_pct"] >= 0.0) & (d["_fwa_pct"] <= 100.0)].copy()

        zt = self.ZERO_THRESHOLD
        ft = self.FULL_THRESHOLD

        # ── Level-2-Segmente: (LoanGoal, ApplicationType) ────────────────────
        segs_by_2: dict = {}
        for key, grp in d.groupby([group_cols[0], group_cols[1]]):
            seg = self._build_segment(grp["_fwa_pct"].to_numpy(), zt, ft)
            if seg is not None:
                segs_by_2[key] = seg

        # ── Level-1-Segmente: LoanGoal ────────────────────────────────────────
        segs_by_1: dict = {}
        for key, grp in d.groupby(group_cols[0]):
            seg = self._build_segment(grp["_fwa_pct"].to_numpy(), zt, ft)
            if seg is not None:
                segs_by_1[str(key)] = seg

        # ── Globales Segment ──────────────────────────────────────────────────
        global_seg = self._build_segment(d["_fwa_pct"].to_numpy(), zt, ft)

        rounding_step = detect_rounding_step(d[fwa_c].to_numpy()) if self.apply_rounding else None

        self.model = {
            "format": "mixture_kde_v2",
            "ZERO_THRESHOLD": zt,
            "FULL_THRESHOLD": ft,
            "segs_by_2": segs_by_2,
            "segs_by_1": segs_by_1,
            "global_seg": global_seg,
            "rounding_step": rounding_step,
        }
        return self

    def predict(
        self,
        loan_goal: str,
        application_type: str,
        credit_score: float,          # ← für Rückwärtskompatibilität, wird nicht mehr genutzt
        requested_amount: float = None,  # ← für Rückwärtskompatibilität, wird nicht mehr genutzt
        offered_amount: float | None = None,
        mode: str = "sample",
        seed: int | None = None,
    ) -> float:
        """
        Sagt FirstWithdrawalAmount vorher.

        Unterstützt drei Modell-Formate (Rückwärtskompatibilität):
          - "mixture_kde_v2"  → neues Mixture-KDE-Modell (dieser Code)
          - altes Prozent-Bootstrap-Format (no "format" key)
          - ältestes Log-Ratio-Format ("mu_by_pair" key)
        """
        self._require_fitted()
        m = self.model
        assert m is not None

        # ── ÄLTESTES FORMAT: Log-Ratio ────────────────────────────────────────
        if "mu_by_pair" in m:
            req = float(requested_amount) if requested_amount is not None else 0.0
            if not np.isfinite(req) or req <= 0:
                return 0.0

            seg = (loan_goal, application_type)
            mu_seg = m["mu_by_pair"].get(seg)
            sig_seg = m["sigma_by_pair"].get(seg)
            n_seg = float(m["n_by_pair"].get(seg, 0))
            tau = float(m.get("tau", 200.0))

            global_mu = float(m["global_mu"])
            global_sigma = float(m["global_sigma"])

            if mu_seg is None or sig_seg is None:
                mu, sigma = global_mu, global_sigma
            else:
                w = n_seg / (n_seg + tau)
                mu = float(w * mu_seg + (1.0 - w) * global_mu)
                sigma = float(w * sig_seg + (1.0 - w) * global_sigma)

            mu_adj = mu + float(m["beta_cs"]) * ((float(credit_score) - 650.0) / 50.0)

            if mode == "mean":
                ratio = float(np.clip(np.exp(mu_adj + 0.5 * sigma ** 2), float(m["eps"]), 1.0))
            else:
                rng = self.rng if seed is None else np.random.default_rng(int(seed))
                ratio = float(np.clip(rng.lognormal(mean=mu_adj, sigma=max(1e-9, sigma)), float(m["eps"]), 1.0))

            fwa = ratio * req
            step = m.get("rounding_step")
            if step is not None and np.isfinite(step) and step > 0:
                fwa = round(fwa / step) * step
            fwa = float(np.clip(fwa, 0.0, req))
            if offered_amount is not None and np.isfinite(offered_amount):
                fwa = float(min(fwa, float(offered_amount)))
            return fwa

        # ── NEUES MIXTURE-KDE-FORMAT ──────────────────────────────────────────
        if m.get("format") == "mixture_kde_v2":
            if offered_amount is None or not np.isfinite(float(offered_amount)) or float(offered_amount) <= 0:
                return 0.0

            offered = float(offered_amount)
            rng = self.rng if seed is None else np.random.default_rng(int(seed))

            zt = float(m.get("ZERO_THRESHOLD", self.ZERO_THRESHOLD))
            ft = float(m.get("FULL_THRESHOLD", self.FULL_THRESHOLD))

            # Segment-Hierarchie: by_2 → by_1 → global
            key_2 = (str(loan_goal), str(application_type))
            seg = m["segs_by_2"].get(key_2)
            if seg is None:
                seg = m["segs_by_1"].get(str(loan_goal))
            if seg is None:
                seg = m.get("global_seg")
            if seg is None:
                return 0.0

            fwa_pct = self._sample_from_segment(seg, rng, zt, ft, mode=mode)

            # Prozentwert → Absolute Betrag
            if fwa_pct >= 99.9:
                fwa = offered          # exakt OfferedAmount
            elif fwa_pct <= 0.1:
                fwa = 0.0
            else:
                fwa = (fwa_pct / 100.0) * offered

            # Optionales Rounding
            step = m.get("rounding_step")
            if step is not None and np.isfinite(step) and step > 0:
                fwa = round(fwa / step) * step

            return float(np.clip(fwa, 0.0, offered))

        # ── ZWISCHENFORMAT: Prozent-Bootstrap (altes "by_2/by_1/global"-Format) ─
        if offered_amount is None or not np.isfinite(float(offered_amount)) or float(offered_amount) <= 0:
            return 0.0

        offered = float(offered_amount)
        rng = self.rng if seed is None else np.random.default_rng(int(seed))

        key_2 = (loan_goal, application_type)
        cat_probs = m.get("cat_probs_by_2", {}).get(key_2)
        if cat_probs is None:
            cat_probs = m.get("cat_probs_by_1", {}).get(str(loan_goal))
        if cat_probs is None:
            cat_probs = m.get("global_cat_probs", {"zero_low": 0.33, "medium": 0.34, "high_full": 0.33})

        categories = list(cat_probs.keys())
        probs = [cat_probs[cat] for cat in categories]
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        else:
            probs = [1.0 / len(categories)] * len(categories)

        selected_category = str(rng.choice(categories, p=probs))

        arr = None
        key_2_cat = (loan_goal, application_type, selected_category)
        arr = m.get("dist_by_2_cat", {}).get(key_2_cat)

        if arr is None or len(arr) == 0:
            for key in m.get("dist_by_1_cat", {}):
                if len(key) == 2 and key[0] == loan_goal and key[1] == selected_category:
                    arr = m["dist_by_1_cat"][key]
                    break

        if arr is None or len(arr) == 0:
            arr = m.get("global_dist_by_cat", {}).get(selected_category)

        if arr is None or len(arr) == 0:
            if selected_category == "zero_low":
                arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            elif selected_category == "high_full":
                arr = np.array([95.0, 96.0, 97.0, 98.0, 99.0, 100.0])
            else:
                arr = m.get("global", np.array([50.0]))

        fwa_pct = float(np.mean(arr)) if mode == "mean" else float(rng.choice(arr))
        fwa = (fwa_pct / 100.0) * offered

        step = m.get("rounding_step")
        if step is not None and np.isfinite(step) and step > 0:
            fwa = round(fwa / step) * step

        return float(np.clip(fwa, 0.0, offered))

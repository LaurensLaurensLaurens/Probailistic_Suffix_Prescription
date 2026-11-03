from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import zlib
from typing import Callable, Optional, Any, Dict
import numpy as np
import pandas as pd

from .registry import (
    build_default_registry,
    PredictorRegistry,
    registry_models_to_dict,
    load_models_into_registry,
)
from .utils import resolve_col

# Fallattribute kommen aus den echten Fällen höchstens so viele Wochen um dasselbe Kalenderdatum (wie die Ankünfte)
WINDOW_WEEKS = 3
# Weniger passende Beträge im Fenster: RequestedAmount aus dem ganzen Jahr
MIN_WINDOW_AMOUNTS = 5


def _compute_case_sampler_artifact(
    df: pd.DataFrame,
    case_col: str = "case:concept:name",
    loan_goal_col: str = "case:LoanGoal",
    app_type_col: str = "case:ApplicationType",
    requested_col: str = "case:RequestedAmount",
    timestamp_col: str = "time:timestamp",
) -> dict:
    """
    Berechnet alle benötigten Verteilungen für die Case-Sampler und gibt sie
    als serialisierbares Artefakt zurück.
    """
    for c in (loan_goal_col, app_type_col, requested_col):
        if c not in df.columns:
            raise KeyError(f"Fehlende Spalte im Input df: '{c}'")

    case_tbl_raw = df.groupby(case_col)[[loan_goal_col, app_type_col, requested_col]].first().copy()
    case_tbl_raw[requested_col] = pd.to_numeric(case_tbl_raw[requested_col], errors="coerce")

    # ── Vollständige Tabelle für Kategorial-Verteilungen ──────────────────────
    # Nur NaN-Zeilen in LoanGoal/AppType ausschließen – RequestedAmount=0 ist
    # kein Ausschlussgrund für die Kategorial-Verteilungen (z.B. "Unknown"-Fälle
    # haben häufig RequestedAmount=0, sollen aber korrekt repräsentiert werden).
    case_tbl_all = case_tbl_raw.dropna(subset=[loan_goal_col, app_type_col])

    # ── Gefilterte Tabelle für RequestedAmount-Bootstrap ──────────────────────
    # Nur Fälle mit RequestedAmount > 0 verwenden, damit der Bootstrap-Sampler
    # stets positive Beträge liefert (verhindert NaN-Weitergabe an Offer-Prädiktoren).
    # Für Gruppen ohne positive Werte (z.B. "Unknown") greift der globale Fallback.
    case_tbl_req = case_tbl_raw.dropna(subset=[requested_col])
    case_tbl_req = case_tbl_req[case_tbl_req[requested_col] > 0]

    # P(LoanGoal) – aus vollständiger Tabelle ohne RequestedAmount-Filter
    lg_p = case_tbl_all[loan_goal_col].value_counts(normalize=True)
    lg_vals = lg_p.index.to_numpy()
    lg_probs = lg_p.to_numpy()

    # P(AppType | LoanGoal)
    at_marg = case_tbl_all[app_type_col].value_counts(normalize=True)
    at_vals = at_marg.index.to_numpy()
    at_probs = at_marg.to_numpy()

    # Bedingte Verteilung: P(AppType | LoanGoal)
    # Beispiel-Struktur von at_cond:
    # {
    #   "Home": (array(['New', 'Renewal']), array([0.6, 0.4])),
    #   "Car": (array(['New', 'Renewal']), array([0.3, 0.7])),
    #   "Education": (array(['New']), array([1.0]))
    # }
    # Für jedes LoanGoal: (mögliche AppType-Werte, zugehörige Wahrscheinlichkeiten)
    at_cond: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for lg, sub in case_tbl_all.groupby(loan_goal_col):
        p = sub[app_type_col].value_counts(normalize=True)
        at_cond[str(lg)] = (p.index.to_numpy(), p.to_numpy())

    # RequestedAmount | (lg, at) + Fallbacks
    # Erstellt ein Dictionary mit gruppierten RequestedAmount-Werten
    # Gruppiert nach Kombinationen aus LoanGoal und ApplicationType
    # Schlüssel: Tupel (LoanGoal, ApplicationType) (z. B. ("Home", "New"))
    req_global = case_tbl_req[requested_col].to_numpy(dtype=float)
    req_by_pair = {
        (str(lg), str(at)): sub[requested_col].to_numpy(dtype=float)
        for (lg, at), sub in case_tbl_req.groupby([loan_goal_col, app_type_col])
    }
    req_by_lg = {
        str(lg): sub[requested_col].to_numpy(dtype=float)
        for lg, sub in case_tbl_req.groupby(loan_goal_col)
    }

    # Für das Ziehen nach Jahreszeit: je Fall Starttag, LoanGoal, ApplicationType und RequestedAmount, nach Starttag sortiert
    window = None
    if timestamp_col in df.columns:
        starts = pd.to_datetime(df.groupby(case_col)[timestamp_col].min())
        if starts.dt.tz is not None:
            starts = starts.dt.tz_convert("UTC").dt.tz_localize(None)
        window_tbl = case_tbl_all.join(starts.rename("_start_")).sort_values("_start_")
        log_start = window_tbl["_start_"].min().floor("D")
        window = {
            "log_start": log_start,
            "n_days": int((window_tbl["_start_"].max().floor("D") - log_start).days) + 1,
            "day": (window_tbl["_start_"].dt.floor("D") - log_start).dt.days.to_numpy(),
            "lg": window_tbl[loan_goal_col].astype(str).to_numpy(),
            "at": window_tbl[app_type_col].astype(str).to_numpy(),
            "req": window_tbl[requested_col].to_numpy(dtype=float),
        }

    return {
        "window": window,
        "lg_vals": lg_vals,
        "lg_probs": lg_probs,
        "at_vals": at_vals,
        "at_probs": at_probs,
        "at_cond": at_cond,
        "req_global": req_global,
        "req_by_pair": req_by_pair,
        "req_by_lg": req_by_lg,
    }


def _build_case_samplers_from_artifact(
    artifact: dict,
    seed: int = 42,
    weeks: int = WINDOW_WEEKS,
) -> tuple[
    Callable[[], str],
    Callable[[str], str],
    Callable[[str, str], float],
    Callable[[Any], Optional[tuple]],
]:
    """
    Baut die Case-Sampler-Funktionen aus einem zuvor berechneten Artefakt.
    """
    rng = np.random.default_rng(seed)

    lg_vals = artifact["lg_vals"]
    lg_probs = artifact["lg_probs"]
    at_vals = artifact["at_vals"]
    at_probs = artifact["at_probs"]
    at_cond = artifact["at_cond"]
    req_global = artifact["req_global"]
    req_by_pair = artifact["req_by_pair"]
    req_by_lg = artifact["req_by_lg"]

    def draw_lg() -> str:
        return str(rng.choice(lg_vals, p=lg_probs))

    def draw_at(loan_goal: str) -> str:
        vals_probs = at_cond.get(str(loan_goal))
        if vals_probs is None:
            return str(rng.choice(at_vals, p=at_probs))
        vals, probs = vals_probs
        return str(rng.choice(vals, p=probs))

    def draw_requested_amount(loan_goal: str, application_type: str) -> float:
        arr = req_by_pair.get((str(loan_goal), str(application_type)))
        if arr is None or len(arr) == 0:
            arr = req_by_lg.get(str(loan_goal))
        if arr is None or len(arr) == 0:
            arr = req_global
        return float(rng.choice(arr))

    window = artifact.get("window")

    def draw_in_window(time) -> Optional[tuple]:
        """
        Wie draw_lg -> draw_at -> draw_requested_amount, aber nur aus den echten Fällen, die höchstens
        `weeks` Wochen um dasselbe Kalenderdatum begonnen haben (wie die Ankünfte in case_arrivals.py).
        So bleibt die Veränderung über das Jahr erhalten. None ohne Zeit oder ohne Fenster im Artefakt.
        """
        if window is None or time is None:
            return None
        day = (pd.Timestamp(time).floor("D") - window["log_start"]).days
        # Tage außerhalb des Logs in Schritten von 52 Wochen hineinlegen (Wochentag und Jahreszeit bleiben)
        while day >= window["n_days"]:
            day -= 364
        while day < 0:
            day += 364
        low = np.searchsorted(window["day"], day - 7 * weeks, side="left")
        high = np.searchsorted(window["day"], day + 7 * weeks, side="right")
        if high <= low:
            return None
        lg_w, at_w, req_w = window["lg"][low:high], window["at"][low:high], window["req"][low:high]
        lg = str(rng.choice(lg_w))
        at = str(rng.choice(at_w[lg_w == lg]))
        # Auch Beträge von 0: Im echten Log haben 9,6 % der Angebote keinen gewünschten Betrag (vor allem LoanGoal Unknown
        # und Limit raise); das Angebot wird dann ohne Bezug zum Antrag gezogen (OfferedAmountPredictor).
        # Zu wenige Fälle im Fenster -> Beträge > 0 des ganzen Jahres
        pool = req_w[(lg_w == lg) & (at_w == at)]
        req = float(rng.choice(pool)) if len(pool) >= MIN_WINDOW_AMOUNTS else draw_requested_amount(lg, at)
        return lg, at, req

    return draw_lg, draw_at, draw_requested_amount, draw_in_window


def _build_case_samplers(
    df: pd.DataFrame,
    seed: int = 42,
    case_col: str = "case:concept:name",
    loan_goal_col: str = "case:LoanGoal",
    app_type_col: str = "case:ApplicationType",
    requested_col: str = "case:RequestedAmount",
) -> tuple[
    Callable[[], str],
    Callable[[str], str],
    Callable[[str, str], float],
    Callable[[Any], Optional[tuple]],
]:
    """
    Convenience-Wrapper: berechnet das Artefakt aus dem df und baut daraus
    die Sampler-Funktionen.
    """
    artifact = _compute_case_sampler_artifact(
        df,
        case_col=case_col,
        loan_goal_col=loan_goal_col,
        app_type_col=app_type_col,
        requested_col=requested_col,
    )
    return _build_case_samplers_from_artifact(artifact, seed=seed)


def _compute_monthly_rate_artifact(
    df: pd.DataFrame,
    case_col: str = "case:concept:name",
    monthly_col: str = "MonthlyCost",
    offered_col: str = "OfferedAmount",
) -> dict:
    """
    Berechnet die Verteilung der Rate = MonthlyCost/OfferedAmount auf Case-Level.
    """
    mc = resolve_col(df, monthly_col)
    oa = resolve_col(df, offered_col)

    case_tbl = df.groupby(case_col)[[mc, oa]].first().copy()
    case_tbl[mc] = pd.to_numeric(case_tbl[mc], errors="coerce")
    case_tbl[oa] = pd.to_numeric(case_tbl[oa], errors="coerce")
    case_tbl = case_tbl.dropna(subset=[mc, oa])
    case_tbl = case_tbl[(case_tbl[oa] > 0) & (case_tbl[mc] >= 0)]

    rate = (case_tbl[mc] / case_tbl[oa]).to_numpy(dtype=float)
    rate = rate[np.isfinite(rate)]
    if rate.size == 0:
        rate = np.array([0.01], dtype=float)

    return {"rate": rate}


def _build_monthly_rate_sampler_from_artifact(
    artifact: dict,
    seed: int = 42,
) -> Callable[[float], float]:
    """
    Baut den Monthly-Rate-Sampler aus einem zuvor berechneten Artefakt.
    """
    rng = np.random.default_rng(seed)
    rate = np.asarray(artifact["rate"], dtype=float)

    def draw_monthly(offered_amount: float) -> float:
        r = float(rng.choice(rate))
        return float(max(r * float(offered_amount), 0.0))

    return draw_monthly


def _build_monthly_rate_sampler(
    df: pd.DataFrame,
    seed: int = 42,
    case_col: str = "case:concept:name",
    monthly_col: str = "MonthlyCost",
    offered_col: str = "OfferedAmount",
) -> Callable[[float], float]:
    """
    Convenience-Wrapper: berechnet das Artefakt aus dem df und baut daraus
    den Monthly-Rate-Sampler.
    """
    artifact = _compute_monthly_rate_artifact(
        df,
        case_col=case_col,
        monthly_col=monthly_col,
        offered_col=offered_col,
    )
    return _build_monthly_rate_sampler_from_artifact(artifact, seed=seed)


@dataclass
class CaseState:
    case_id: str
    loan_goal: str
    application_type: str
    requested_amount: float

    # Offer-abhängige Attribute (werden einmalig gezogen)
    credit_score: float = np.nan
    offered_amount: float = np.nan
    first_withdrawal_amount: float = np.nan
    number_of_terms: float = np.nan
    monthly_cost: float = np.nan
    selected: Optional[bool] = None
    accepted: Optional[bool] = None
    n_offers: int = 0  # bisher gezogene Angebote


class AttributeSimulationEngine:
    """
    Einmal initialisieren (Modelle und Verteilungen laden oder aus dem Log fitten), danach:
      - start_new_case(case_id, time): bei der Ankunft eines Falls die Fallattribute
        (LoanGoal, ApplicationType, RequestedAmount), aus den echten Fällen derselben Jahreszeit
      - populate_offer_attributes(state): bei jedem O_Create Offer des Falls ein neues Angebot
        (Accepted bleibt ab dem ersten Angebot gleich)
    """

    def __init__(
        self,
        df: pd.DataFrame | None,
        seed: int = 42,
        monthly_artifact: dict | None = None,
        model_store_path: str | Path | None = None,
        retrain_models: bool = False,
    ):
        """
        df kann None sein, wenn ausschließlich aus gespeicherten Artefakten
        (Modelle + Verteilungen) simuliert werden soll.
        """
        self.seed = int(seed)

        # Pfad für Modellartefakte (Pickle); default neben dieser Datei
        self.model_store_path = Path(model_store_path) if model_store_path is not None else Path(__file__).with_name("attribute_models.pkl")
        # Pfad für Verteilungs-Artefakte (Case-Sampler + Monthly-Rate)
        self.dist_store_path = Path(__file__).with_name("attribute_distributions.pkl")
        self.retrain_models = bool(retrain_models)

        # Registry + Fit (einmal)
        self.registry: PredictorRegistry = build_default_registry(seed=self.seed)
        self._init_models(df)

        # MonthlyCost: Artefakt oder fallback rate sampler
        # Wenn kein explizites Artefakt übergeben, aber das Modell im Modell-Store
        # geladen wurde, bevorzuge das Registry-Modell gegenüber dem Rate-Sampler.
        if monthly_artifact is None and self.registry.monthly_cost.model is not None:
            monthly_artifact = self.registry.monthly_cost.model

        self.monthly_artifact = monthly_artifact
        self.monthly_rate_sampler = None

        if monthly_artifact is not None:
            self.registry.monthly_cost.set_artifact(monthly_artifact)
        # Fallback-Rate-Sampler oder Laden aus Artefakt
        # sowie Case-Sampler:
        if df is not None:
            # Sampler direkt aus df bauen und Artefakt persistieren
            self.monthly_rate_sampler = _build_monthly_rate_sampler(df, seed=self.seed)
            self.draw_lg, self.draw_at, self.draw_requested_amount, self.draw_in_window = _build_case_samplers(df, seed=self.seed)

            dist_artifact = {
                "case": _compute_case_sampler_artifact(df),
                "monthly_rate": _compute_monthly_rate_artifact(df),
            }
            try:
                with self.dist_store_path.open("wb") as f:
                    pickle.dump(dist_artifact, f)
            except Exception:
                # Persistence-Fehler sollen die Simulation nicht verhindern
                pass
        else:
            # df ist None: wir erwarten vortrainierte Verteilungs-Artefakte
            if not self.dist_store_path.exists():
                raise ValueError(
                    f"Kein df übergeben und Verteilungs-Artefakt '{self.dist_store_path}' nicht gefunden. "
                    f"Bitte zuerst einmal mit retrain_models=True und df trainieren."
                )
            with self.dist_store_path.open("rb") as f:
                dist_artifact = pickle.load(f)

            case_art = dist_artifact.get("case")
            monthly_art = dist_artifact.get("monthly_rate")
            if case_art is None or monthly_art is None:
                raise ValueError(
                    f"Verteilungs-Artefakt '{self.dist_store_path}' ist unvollständig oder korrupt."
                )

            self.draw_lg, self.draw_at, self.draw_requested_amount, self.draw_in_window = _build_case_samplers_from_artifact(
                case_art,
                seed=self.seed,
            )

            if monthly_artifact is None:
                self.monthly_rate_sampler = _build_monthly_rate_sampler_from_artifact(
                    monthly_art,
                    seed=self.seed,
                )

        # Zähler für Fall-IDs, wenn der Aufrufer keine übergibt
        self._case_counter = 0

    def _init_models(self, df: pd.DataFrame | None):
        """
        Modelle entweder laden oder neu fitten + persistieren.
        """
        models_loaded = False
        if not self.retrain_models and self.model_store_path.exists():
            try:
                with self.model_store_path.open("rb") as f:
                    models = pickle.load(f)
                load_models_into_registry(self.registry, models)
                models_loaded = True
            except Exception:
                # Fallback auf Neu-Training, wenn Laden fehlschlägt
                models_loaded = False

        # Ältere gespeicherte OfferedAmount-Modelle kennen die Angebotsnummer oder Anträge ohne Betrag noch nicht: nur dieses Modell neu fitten
        offered_model = self.registry.offered_amount.model or {}
        refit = False
        if models_loaded and df is not None and ("next_ratio" not in offered_model or "zero_requested" not in offered_model):
            self.registry.offered_amount.fit(df)
            refit = True
        # Ebenso CreditScore-Modelle ohne die Regeln für Selected, CreditScore und Accepted je Angebot (final)
        if models_loaded and df is not None and not isinstance(((self.registry.credit_score.model or {}).get("final") or {}).get("p_selected_returned"), dict):
            self.registry.credit_score.fit(df)
            refit = True
        if refit:
            try:
                with self.model_store_path.open("wb") as f:
                    pickle.dump(registry_models_to_dict(self.registry), f)
            except Exception:
                pass

        if not models_loaded:
            # Fit einmalig – benötigt ein df
            if df is None:
                raise ValueError(
                    "Kein df übergeben, aber Modelle müssen neu trainiert werden. "
                    "Bitte ein Trainings-DataFrame übergeben oder retrain_models=False setzen."
                )
            self.registry.credit_score.fit(df)
            self.registry.offered_amount.fit(df)
            self.registry.first_withdrawal_amount.fit(df)
            self.registry.number_of_terms.fit(df)
            self.registry.selected.fit(df)
            self.registry.accepted.fit(df)
            self.registry.monthly_cost.fit(df)
            # persistieren
            try:
                with self.model_store_path.open("wb") as f:
                    pickle.dump(registry_models_to_dict(self.registry), f)
            except Exception:
                # Persistence failures sollen Simulation nicht stoppen
                pass

    # Wird aufgerufen von CaseEngine._arrive -> start_new_case (mit Fall-ID und Ankunftszeit)
    def start_new_case(self, case_id: str | None = None, time=None) -> CaseState:
        """
        Zieht Basis-Case-Attribute (LoanGoal, ApplicationType, RequestedAmount) genau einmal.
        Mit time aus den echten Fällen derselben Jahreszeit, sonst aus dem ganzen Jahr.
        """
        self._case_counter += 1
        if case_id is None:
            case_id = f"SIM_{self._case_counter:07d}"

        drawn = self.draw_in_window(time)
        if drawn is not None:
            lg, at, req = drawn
        else:
            lg = self.draw_lg()
            at = self.draw_at(lg)
            req = self.draw_requested_amount(lg, at)

        return CaseState(
            case_id=str(case_id),
            loan_goal=str(lg),
            application_type=str(at),
            requested_amount=float(req),
        )

    def final_offer_attributes(self, events: list, exclude=()) -> list:
        """
        Setzt Selected, CreditScore und Accepted der Angebote (O_Create Offer) beendeter Fälle, wie im echten Log erst,
        wenn der Verlauf des Angebots feststeht (CreditScorePredictor.final_attributes). Die Angebots-Events eines Falls
        gehören zum zuletzt erstellten Angebot (O_Returned, O_Accepted, O_Refused); O_Cancelled zum ältesten noch offenen.
        events: alle Events der Fälle (Liste von dicts); Fälle in exclude (nicht beendet) bleiben unverändert. Liefert eine
        neue Liste; geänderte Events werden kopiert, damit Events, die sich mehrere Fortsetzungen teilen, unverändert bleiben.
        """
        offers = {}  # Fall -> Liste [Index des O_Create Offer, zurückgekommen, Ausgang]
        for index, event in enumerate(events):
            case_id = event["case:concept:name"]
            if case_id in exclude or event.get("lifecycle:transition", "complete") != "complete":
                continue
            activity, case_offers = event["concept:name"], offers.setdefault(case_id, [])
            if activity == "O_Create Offer":
                case_offers.append([index, False, "open"])
            elif case_offers and activity == "O_Returned":
                case_offers[-1][1] = True
            elif case_offers and activity in ("O_Accepted", "O_Refused") and case_offers[-1][2] == "open":
                case_offers[-1][2] = "accepted" if activity == "O_Accepted" else "refused"
            elif activity == "O_Cancelled":
                still_open = [offer for offer in case_offers if offer[2] == "open"]
                if still_open:
                    still_open[0][2] = "cancelled"
        result = list(events)
        for case_id, case_offers in offers.items():
            for number, (index, returned, status) in enumerate(case_offers):
                seed = int(np.random.SeedSequence([self.seed, zlib.crc32(str(case_id).encode()), 1000 + number]).generate_state(1)[0])
                result[index] = {**result[index], **self.registry.credit_score.final_attributes(returned, status, seed)}
        return result

    def populate_offer_attributes(self, case_state: CaseState) -> None:
        """
        Populiert Offer-abhängige Attribute direkt auf dem übergebenen CaseState.
        
        Für die Engines gedacht, die viele Cases parallel verarbeiten: Sie arbeitet
        direkt auf dem übergebenen CaseState-Objekt. Aufruf bei jedem O_Create Offer des Falls.
        
        Args:
            case_state: Das CaseState-Objekt, auf dem die Attribute gesetzt werden.
        """
        cs = case_state

        # Jedes O_Create Offer zieht ein neues Angebot: Im echten Log unterscheiden sich die Angebote eines Falls
        # meistens (z.B. OfferedAmount in 42 %, MonthlyCost in 75 % der Fälle mit mehreren Angeboten).
        cs.n_offers += 1

        # Seed aus Seed, Fall-ID und Angebotsnummer. crc32 statt hash(): hash() ändert sich bei jedem Python-Start.
        case_seed = int(np.random.SeedSequence([self.seed, zlib.crc32(cs.case_id.encode()), cs.n_offers]).generate_state(1)[0])

        cs.credit_score = float(self.registry.credit_score.predict(cs.loan_goal, cs.application_type))

        cs.offered_amount = float(
            self.registry.offered_amount.predict(
                cs.loan_goal, cs.application_type, cs.requested_amount,
                mode="sample", n_draws=1, seed=case_seed, offer_number=cs.n_offers,
                # Folgeangebote hängen am vorigen Angebot (steht noch auf cs)
                previous_offered=cs.offered_amount if cs.n_offers > 1 else None,
            )
        )

        cs.first_withdrawal_amount = float(
            self.registry.first_withdrawal_amount.predict(
                cs.loan_goal, cs.application_type, cs.credit_score,
                requested_amount=cs.requested_amount,
                offered_amount=cs.offered_amount,
                mode="sample",
                seed=case_seed,
            )
        )

        cs.number_of_terms = int(
            self.registry.number_of_terms.predict(
                offered_amount=cs.offered_amount,
                credit_score=cs.credit_score,
                loan_goal=cs.loan_goal,
                application_type=cs.application_type,
            )
        )

        if self.monthly_artifact is not None:
            cs.monthly_cost = float(
                self.registry.monthly_cost.predict(
                    offered_amount=cs.offered_amount,
                    number_of_terms=int(cs.number_of_terms),
                    credit_score=cs.credit_score,
                    application_type=cs.application_type,
                )
            )
        else:
            assert self.monthly_rate_sampler is not None
            cs.monthly_cost = float(self.monthly_rate_sampler(cs.offered_amount))

        cs.selected = bool(self.registry.selected.predict(cs.loan_goal, cs.application_type, cs.credit_score))
        # Accepted gehört eher zum Fall als zum Angebot: Im echten Log unterscheidet es sich nur in 14 % der
        # Fälle mit mehreren Angeboten. Es wird deshalb nur beim ersten Angebot gezogen.
        if cs.accepted is None:
            cs.accepted = bool(self.registry.accepted.predict(cs.monthly_cost, cs.credit_score))

from __future__ import annotations

from dataclasses import dataclass

from .credit_score import CreditScorePredictor
from .selected import SelectedPredictor
from .accepted import AcceptedPredictor
from .offered_amount import OfferedAmountPredictor
from .number_of_terms import NumberOfTermsPredictor
from .monthly_cost import MonthlyCostPredictor
from .first_withdrawal_amount import FirstWithdrawalAmountPredictor


@dataclass
class PredictorRegistry:
    credit_score: CreditScorePredictor
    selected: SelectedPredictor
    accepted: AcceptedPredictor
    offered_amount: OfferedAmountPredictor
    number_of_terms: NumberOfTermsPredictor
    monthly_cost: MonthlyCostPredictor
    first_withdrawal_amount: FirstWithdrawalAmountPredictor


def build_default_registry(seed: int = 42) -> PredictorRegistry:
    return PredictorRegistry(
        credit_score=CreditScorePredictor(seed=seed),
        selected=SelectedPredictor(seed=seed),
        accepted=AcceptedPredictor(seed=seed),
        offered_amount=OfferedAmountPredictor(seed=seed),
        number_of_terms=NumberOfTermsPredictor(seed=seed),
        monthly_cost=MonthlyCostPredictor(seed=seed),
        first_withdrawal_amount=FirstWithdrawalAmountPredictor(seed=seed),
    )


def registry_models_to_dict(registry: PredictorRegistry) -> dict:
    """
    Serialisiert lediglich die internen Artefakte (model-Attribute) aller Predictor.
    """
    return {
        "credit_score": registry.credit_score.model,
        "selected": registry.selected.model,
        "accepted": registry.accepted.model,
        "offered_amount": registry.offered_amount.model,
        "number_of_terms": registry.number_of_terms.model,
        "monthly_cost": registry.monthly_cost.model,
        "first_withdrawal_amount": registry.first_withdrawal_amount.model,
    }


def load_models_into_registry(registry: PredictorRegistry, models: dict | None) -> PredictorRegistry:
    """
    Befüllt die Predictor-Instanzen mit bereits trainierten Artefakten.
    """
    if not models:
        return registry

    registry.credit_score.model = models.get("credit_score")
    registry.selected.model = models.get("selected")
    registry.accepted.model = models.get("accepted")
    registry.offered_amount.model = models.get("offered_amount")
    registry.number_of_terms.model = models.get("number_of_terms")
    registry.monthly_cost.model = models.get("monthly_cost")
    registry.first_withdrawal_amount.model = models.get("first_withdrawal_amount")
    return registry

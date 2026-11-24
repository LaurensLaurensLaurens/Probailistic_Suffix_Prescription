"""EV choice in EUR, accepting only models fitted to the matching event tariff."""
import numpy as np

from prescriptive.policies.ev_policy.cate_policy import CATEPolicy
from .costs import EventCosts


class CATEEventCostPolicy(CATEPolicy):
    def __init__(self, model_path, seed=42, threshold=0.0, cost_config=None, first_only=()):
        super().__init__(model_path, seed=seed, threshold=threshold, first_only=first_only)
        if getattr(self.model, "objective", None) != "negative_event_costs":
            raise ValueError("An EV model trained on negative event costs is required")
        self.costs = EventCosts(self.model.costs if cost_config is None else cost_config)
        if self.costs.fingerprint != self.model.cost_fingerprint:
            raise ValueError("Event tariffs differ from training; retrain the CATE model for this configuration")
        if not np.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("threshold must be a finite, nonnegative amount in EUR")

    def choose(self, point, case, options, time, engine):
        if not options or not self.model.knows(point) or self._repeated(point, case.prefix):
            return None
        values = self.model.values(point, case.prefix, options, time)
        reference = self.model.options[point.name][0]
        values = np.where([option == reference for option in options], values, values - self.threshold)
        best = np.flatnonzero(np.isfinite(values) & (values == values.max()))
        if not len(best):
            return None
        return options[int(best[0] if len(best) == 1 else self.rng.choice(best))]

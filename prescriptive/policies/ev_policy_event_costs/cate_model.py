"""The existing S/X learners and causal forest, trained on negative event costs."""
import numpy as np

from prescriptive.estimators.cate_model import CATEModel
from .costs import EventCosts


class EventCostCATEModel(CATEModel):
    objective = "negative_event_costs"

    def __init__(self, dataset, points=None, seed=42, learner="x", cost_config=None):
        super().__init__(dataset, points, seed, learner)
        self.costs = EventCosts(cost_config)
        self.cost_fingerprint = self.costs.fingerprint

    def fit(self, log, samples=None):
        return super().fit(self.costs.label_log(log), samples=samples)

    def values(self, point, events, options, now):
        values = super().values(point, events, options, now)
        # An untrained action must not receive an apparently favorable zero effect.
        trained = {self.options[point.name][0]} | {
            self.options[point.name][index] for index in self.models[point.name]}
        return np.where([option in trained for option in options], values, -np.inf)

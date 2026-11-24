"""Train-only empirical gap calibration conditional on event and work state.

This is an explicit additional time model, not a change to frozen PSP weights.
Backoff retains event and work state; it never pools active work with waiting.
"""
from collections import defaultdict

import numpy as np
import torch


class SuffixSamplingError(ValueError):
    """Constrained sampling could not produce complete supported paths."""


class ConditionalGaps:
    def __init__(self, min_observations=30):
        self.min_observations = min_observations

    def fit(self, log):
        from interventions.work_state import WorkState
        pools = defaultdict(list)
        self.training_cases = sorted(map(str, log["case:concept:name"].unique()))
        for _, part in log.sort_values(["case:concept:name", "time:timestamp"], kind="stable").groupby("case:concept:name", sort=False):
            state, previous_time = WorkState(), None
            for event in part.to_dict("records"):
                token = f"{event['concept:name']}|{event.get('lifecycle:transition', 'complete')}"
                at = event["time:timestamp"]
                if previous_time is not None and state.permits(token):
                    gap = max((at-previous_time).total_seconds(), 0.)
                    own, active = state.context(token)
                    pools[(state.previous, token, own, active)].append(gap)
                    pools[(None, token, own, active)].append(gap)
                state.push(token)
                previous_time = at
        self.pools = {key: np.asarray(values, dtype=np.float64) for key, values in pools.items()}
        return self

    def distribution(self, state, token):
        own, active = state.context(token)
        specific = self.pools.get((state.previous, token, own, active))
        general = self.pools.get((None, token, own, active))
        if specific is not None and len(specific) >= self.min_observations:
            return specific
        return general

    def draw(self, states, labels):
        uniforms = torch.rand(len(labels)).numpy()
        gaps = np.zeros(len(labels), dtype=float)
        for i, (state, token) in enumerate(zip(states, labels)):
            values = self.distribution(state, token)
            if values is None or not len(values):
                raise ValueError(f"No training gap support for {token} in work state {state.context(token)}")
            gaps[i] = values[min(int(uniforms[i]*len(values)), len(values)-1)]
        return gaps

"""Standard PSP EV choice, with sampled suffixes valued by remaining event costs."""
import numpy as np

from prescriptive.policies.ev_policy.psp_policy import PSPPolicy as BasePSPPolicy
from .costs import ACTIVITY, LIFECYCLE, TIMESTAMP, EventCosts, seconds


class EventCostPSPModel:
    """Adapter: keep the fitted suffix model and its sampling unchanged."""
    objective = "negative_event_costs"

    def __init__(self, model, cost_config=None):
        if not model.lifecycle_tokens:
            raise ValueError("Event costs require a PSP trained with activity|lifecycle tokens")
        self.base = model
        self.costs = EventCosts(cost_config)

    def knows(self, point):
        return self.base.knows(point)

    def outcome_samples(self, point, events, options, now, n=None):
        if not events:
            raise ValueError("A nonempty observed prefix is required")
        n = self.base.samples_per_option if n is None else n
        if type(n) is not int or n < 1:
            raise ValueError("samples must be a positive integer")
        drawn, elapsed = self.base.option_suffixes(point, events, options, now, n)
        # Some PSP variants return the option as the first suffix event. The
        # original encoder variant instead includes it in the encoded prefix.
        included = point.kind == "control_flow" and (
            self.base.option_in_encoder() or self.base.option_in_decoder()
            or getattr(self.base, "option_gap_distributions", None) is not None)
        origin = seconds(events[0][TIMESTAMP])
        observed = self.costs.state(events)
        horizon = max(float(now), observed.last)
        sunk = observed.total(horizon)
        values = []
        for option, suffixes in zip(options, drawn):
            prefix = observed.copy()
            if not included:
                event = self.base._option_event(point, events, option, now)
                # The event is an observed-format dictionary, not a sampled token.
                prefix.push(str(event[ACTIVITY]).strip(), str(event.get(LIFECYCLE, "complete")).lower(),
                            seconds(event[TIMESTAMP]))
            scores = []
            if len(suffixes) != n:
                raise ValueError("Unexpected number of PSP suffixes")
            for suffix in suffixes:
                state = prefix.copy()
                at = max(origin + float(elapsed), state.last, horizon)
                for token, _, gap in suffix:
                    if not np.isfinite(gap) or gap < 0:
                        raise ValueError("Sampled event gaps must be finite and nonnegative")
                    activity, separator, lifecycle = str(token).rpartition("|")
                    if not separator:
                        raise ValueError(f"Missing lifecycle in sampled token {token!r}")
                    at += float(gap)
                    state.push(activity, lifecycle, at)
                scores.append(-max(state.total(at) - sunk, 0.))
            values.append(scores)
        return np.asarray(values, dtype=float)

    def values(self, point, events, options, now):
        return self.outcome_samples(point, events, options, now).mean(axis=1)


class PSPEventCostPolicy(BasePSPPolicy):
    def __init__(self, model_path, seed=42, samples=30, cost_config=None, first_only=()):
        if type(samples) is not int or samples < 1:
            raise ValueError("samples must be a positive integer")
        super().__init__(model_path=model_path, seed=seed, samples=samples, first_only=first_only)
        self.model = EventCostPSPModel(self.model, cost_config)

    def choose(self, point, case, options, time, engine):
        if not options:
            return None
        return super().choose(point, case, options, time, engine)

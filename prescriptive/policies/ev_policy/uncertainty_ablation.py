"""Isolate the final uncertainty fallback without changing adaptive sampling.

Both arms execute the existing PSPPolicy, including its 30→100 escalation,
feasibility checks and sampling-error fallback. Only its final return on an
uncertain comparison is optionally replaced by the best sampled mean.
"""
import numpy as np

from .psp_policy import PSPPolicy


class UncertaintyAblationPolicy(PSPPolicy):
    def __init__(self, *args, bank_on_uncertain=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.bank_on_uncertain = bank_on_uncertain
        self.uncertainty_overrides = 0
        self.decision_rows = []

    def choose(self, point, case, options, time, engine):
        before = getattr(self, 'decision_audit', {}).get('uncertain_fallbacks', 0)
        failures_before = getattr(self, 'sampling_fallbacks', 0)
        estimates = []
        original = self.model.outcome_samples
        had_override = 'outcome_samples' in self.model.__dict__

        def capture(*args, **kwargs):
            scores = original(*args, **kwargs)
            delta = scores[0] - scores[1] if len(scores) == 2 else None
            estimates.append({'samples_per_option': int(scores.shape[1]),
                              'mean_values': scores.mean(axis=1).tolist(),
                              'paired_mcse': float(delta.std(ddof=1) / np.sqrt(len(delta)))
                              if delta is not None and len(delta) > 1 else None})
            return scores

        self.model.outcome_samples = capture
        try:
            selected = super().choose(point, case, options, time, engine)
        finally:
            if had_override:
                self.model.outcome_samples = original
            else:
                del self.model.outcome_samples
        uncertain = getattr(self, 'decision_audit', {}).get('uncertain_fallbacks', 0) > before
        sampling_failure = getattr(self, 'sampling_fallbacks', 0) > failures_before
        if uncertain and not self.bank_on_uncertain:
            assert selected is None and estimates and not sampling_failure
            values = np.asarray(estimates[-1]['mean_values'])
            best = np.flatnonzero(values >= values.max())
            selected = options[int(best[0] if len(best) == 1 else self.rng.choice(best))]
            self.uncertainty_overrides += 1
        self.decision_rows.append({'case': case.case_id, 'point': point.name, 'time': float(time),
                                   'options': list(options), 'selected': selected,
                                   'uncertain_estimate': uncertain, 'sampling_failure': sampling_failure,
                                   'bank_on_uncertain': self.bank_on_uncertain, 'estimates': estimates})
        return selected

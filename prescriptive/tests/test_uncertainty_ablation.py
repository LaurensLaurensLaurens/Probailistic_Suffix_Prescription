"""The treatment changes only the final uncertain choice, retaining its draws."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()
from prescriptive.estimators.conditional_gaps import SuffixSamplingError
from prescriptive.policies.ev_policy.uncertainty_ablation import UncertaintyAblationPolicy


class Model:
    def __init__(self, confident=False, fail=False, tied=False):
        self.params = {'sampling_control': {'max_samples': 100, 'z': 2.}}
        self.samples_per_option = 30
        self.calls = []
        self.confident, self.fail, self.tied = confident, fail, tied

    def knows(self, point):
        return True

    def outcome_samples(self, point, prefix, options, time, n):
        self.calls.append(n)
        if self.fail and n == 100:
            raise SuffixSamplingError('unsupported suffix')
        delta = np.full(n, 5.) if self.confident else np.tile([-10., 10.], n // 2) + .1
        if self.tied:
            delta = np.zeros(n)
        return np.stack([np.zeros(n), delta])


def policy(bank, **kwargs):
    result = UncertaintyAblationPolicy.__new__(UncertaintyAblationPolicy)
    result.model = Model(**kwargs)
    result.first_only = set()
    result.rng = np.random.default_rng(0)
    result.bank_on_uncertain = bank
    result.uncertainty_overrides = 0
    result.decision_rows = []
    return result


class UncertaintyAblationTests(unittest.TestCase):
    point = SimpleNamespace(name='point', activity='Choice', lifecycle='complete')
    case = SimpleNamespace(case_id='Application_700', prefix=[{}])

    def choose(self, candidate, options=('A', 'B')):
        return candidate.choose(self.point, self.case, list(options), 0., None)

    def test_uncertain_arms_use_same_30_and_100_draws_and_only_then_diverge(self):
        guarded, unguarded = policy(True), policy(False)
        self.assertIsNone(self.choose(guarded))
        self.assertEqual(self.choose(unguarded), 'B')
        self.assertEqual(guarded.model.calls, [30, 100])
        self.assertEqual(unguarded.model.calls, [30, 100])
        self.assertEqual(guarded.decision_rows[0]['estimates'], unguarded.decision_rows[0]['estimates'])
        self.assertEqual(guarded.uncertainty_overrides, 0)
        self.assertEqual(unguarded.uncertainty_overrides, 1)
        self.assertNotIn('outcome_samples', unguarded.model.__dict__)

    def test_confident_first_stage_is_unchanged_and_does_not_draw_100(self):
        for bank in (True, False):
            candidate = policy(bank, confident=True)
            self.assertEqual(self.choose(candidate), 'B')
            self.assertEqual(candidate.model.calls, [30])
            self.assertEqual(candidate.uncertainty_overrides, 0)

    def test_sampling_failure_keeps_bank_fallback_in_both_arms(self):
        for bank in (True, False):
            candidate = policy(bank, fail=True)
            self.assertIsNone(self.choose(candidate))
            self.assertEqual(candidate.sampling_fallbacks, 1)
            self.assertEqual(candidate.uncertainty_overrides, 0)
            self.assertTrue(candidate.decision_rows[0]['sampling_failure'])

    def test_single_feasible_option_never_invokes_sampling(self):
        for bank in (True, False):
            candidate = policy(bank)
            self.assertEqual(self.choose(candidate, ['only']), 'only')
            self.assertEqual(candidate.model.calls, [])

    def test_unguarded_tie_chooses_a_feasible_option_reproducibly(self):
        first, second = policy(False, tied=True), policy(False, tied=True)
        self.assertEqual(self.choose(first), self.choose(second))
        self.assertIn(first.decision_rows[0]['selected'], ['A', 'B'])


if __name__ == '__main__':
    unittest.main()

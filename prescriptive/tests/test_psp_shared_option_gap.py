"""One timing draw per option is shared by all suffixes and counted exactly once."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from prescriptive.helpers.datasets import add_paths
add_paths()
from prescriptive.estimators.psp_model import ACTIVITY, TIME_COLUMNS, PSPModel


class RecordingModel(PSPModel):
    def __init__(self):
        self.params = {}
        self.lifecycle_tokens = True
        self.recorded = []
        scaler = SimpleNamespace(mean_=np.array([0.]), scale_=np.array([1.]))
        self.encoder_decoder = SimpleNamespace(continuous_encoders={key: scaler for key in TIME_COLUMNS})
        self.option_gap_distributions = {"p": {"A": {"mean_seconds": 10., "std_seconds": 0.},
                                                "B": {"mean_seconds": 30., "std_seconds": 0.}}}

    def _evaluator(self):
        return SimpleNamespace(prefix_num_attributes=TIME_COLUMNS)

    def _encode(self, events, option_event=None):
        length = len(events) + int(option_event is not None)
        prefix = ([torch.ones(1, length, dtype=torch.long), torch.ones(1, length, dtype=torch.long)],
                  [torch.zeros(1, length) for _ in TIME_COLUMNS])
        return prefix, (None, None), torch.zeros(1, length, dtype=torch.bool), length, 100. + (0.001 if option_event else 0.)

    def _option_event(self, point, events, option, now):
        return {"option": option}

    def _option_ids(self, point, options):
        return [(0, np.array([1, 2]))]

    def draw_cases(self, encoded, n, **kwargs):
        cats, nums, _, _ = self._batch(encoded, n)
        self.recorded.append((cats, nums))
        return [[[('R|complete', float(entry[0][1][0][0, -1]) + 3., 3.)] for _ in range(n)]
                for entry in encoded]


class SharedOptionGapTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.point = SimpleNamespace(name="p", kind="control_flow")
        self.cases = [[{ACTIVITY: "Decision"}]]

    def test_one_prefix_and_gap_per_option_for_all_thirty_suffixes(self):
        model = RecordingModel()
        drawn, elapsed = model.option_suffixes_many(self.point, self.cases, ['A', 'B'], 1000., 30)
        self.assertEqual(elapsed, [100.])
        for k, gap in enumerate((10., 30.)):
            self.assertEqual(len(drawn[k][0]), 30)
            self.assertTrue(all(s[0][2] == gap for s in drawn[k][0]))
            cats, nums = model.recorded[k]
            self.assertEqual(len(cats[0]), 30)
            self.assertTrue(torch.all(cats[0][:, -1] == k + 1))
            for values in nums:
                self.assertTrue(torch.all(values[:, -1] == values[0, -1]))
            self.assertTrue(torch.all(nums[0][:, -1] == 100. + gap))
            self.assertTrue(torch.all(nums[1][:, -1] == gap))

    def test_first_gap_is_included_exactly_once_in_total_duration(self):
        model = RecordingModel()
        drawn, elapsed = model.option_suffixes_many(self.point, self.cases, ['A', 'B'], 1000., 30)
        for k, gap in enumerate((10., 30.)):
            for suffix in drawn[k][0]:
                days = -model.outcome(elapsed[0], suffix, 'negative_cycle_days', gaps=True)
                self.assertAlmostEqual(days * 86400., 100. + gap + 3.)

    def test_number_of_suffixes_does_not_change_the_single_timing_draw(self):
        model = RecordingModel()
        for option in ('A', 'B'):
            model.option_gap_distributions['p'][option] = {"mean_seconds": 1., "std_seconds": 100.}
        torch.manual_seed(31)
        one, _ = model.option_suffixes_many(self.point, self.cases, ['A', 'B'], 1000., 1)
        torch.manual_seed(31)
        thirty, _ = model.option_suffixes_many(self.point, self.cases, ['A', 'B'], 1000., 30)
        for k in range(2):
            gap = one[k][0][0][0][2]
            self.assertGreaterEqual(gap, 0.)
            self.assertTrue(all(s[0][2] == gap for s in thirty[k][0]))
        self.assertNotEqual(one[0][0][0][0][2], one[1][0][0][0][2])

    def test_default_model_keeps_original_option_path(self):
        model = RecordingModel()
        del model.option_gap_distributions
        model._per_option = Mock(return_value='original')
        self.assertEqual(model.option_suffixes_many(self.point, self.cases, ['A', 'B'], 1000., 30), 'original')


if __name__ == '__main__':
    unittest.main()

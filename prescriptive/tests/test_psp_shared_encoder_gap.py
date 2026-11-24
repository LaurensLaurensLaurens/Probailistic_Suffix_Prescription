"""A model-drawn first gap may be shared without changing future suffix sampling."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from prescriptive.helpers.datasets import add_paths
add_paths()
from prescriptive.estimators.psp_model import ACTIVITY, GAP_CLASS, GAP_NAMES, TIME_COLUMNS, PSPModel


class Categories:
    def __init__(self, names):
        self.ids = {name: i for i, name in enumerate(names)}

    def transform(self, values):
        return np.array([[self.ids[str(row[0])]] for row in values])


class RecordingModel(PSPModel):
    def __init__(self, shared):
        self.params = {"option_as": "encoder", "gap_tokens": True, "shared_option_gap": shared}
        self.lifecycle_tokens = True
        scaler = SimpleNamespace(mean_=np.array([0.]), scale_=np.array([1.]))
        self.encoder_decoder = SimpleNamespace(
            continuous_encoders={name: scaler for name in TIME_COLUMNS},
            categorical_encoders={ACTIVITY: Categories(['A|complete|<10min', 'B|complete|<10min']),
                                  GAP_CLASS: Categories(GAP_NAMES)})
        self.calls = []

    def _evaluator(self):
        return SimpleNamespace(prefix_cat_attributes=[ACTIVITY, GAP_CLASS], prefix_num_attributes=TIME_COLUMNS)

    def _encode(self, events, option_event=None):
        length = len(events) + int(option_event is not None)
        elapsed = float(events[-1].get('elapsed', 100.))
        nums = [torch.zeros(1, length) for _ in TIME_COLUMNS]
        nums[0][:, -1] = elapsed + (0.001 if option_event else 0.)
        return ([torch.ones(1, length, dtype=torch.long) for _ in range(2)], nums), (None, None), torch.zeros(1, length), length, elapsed

    def _option_event(self, point, events, option, now):
        return {'option': option}

    def draw_cases(self, encoded, n, first=None, last=None, **kwargs):
        cats, nums, _, _ = self._batch(encoded, n)
        if last:
            for at, values in last['cats'].items():
                cats[at][:, -1] = torch.as_tensor(values, dtype=cats[at].dtype)
            for at, values in last['nums'].items():
                nums[at][:, -1] = values.reshape(-1)
        self.calls.append({'n': n, 'first': first, 'cats': cats, 'nums': nums})
        if first is not None:
            gaps = (torch.rand(len(encoded) * n) + (10. if first[0].startswith('A') else 50.)).numpy()
            return [[[(first[c], float(nums[0][c*n+k, -1]) + float(gaps[c*n+k]), float(gaps[c*n+k]))]
                     for k in range(n)] for c in range(len(encoded))]
        return [[[('R|complete', float(nums[0][c*n+k, -1]) + 3., 3.)] for k in range(n)]
                for c in range(len(encoded))]


class SharedEncoderGapTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.point = SimpleNamespace(name='p', kind='control_flow')
        self.cases = [[{ACTIVITY: 'Decision', 'elapsed': 100.}]]

    def test_one_first_draw_and_identical_extended_prefix_for_thirty_suffixes(self):
        model = RecordingModel(shared=True)
        drawn, elapsed = model.option_suffixes_many(self.point, self.cases, ['A', 'B'], 1000., 30)
        self.assertEqual([call['n'] for call in model.calls], [1, 30, 1, 30])
        self.assertEqual(elapsed, [100.])
        for at, option in enumerate(('A', 'B')):
            suffixes = drawn[at][0]
            gap = suffixes[0][0][2]
            self.assertTrue(all(s[0] == suffixes[0][0] for s in suffixes))
            self.assertEqual(suffixes[0][0][0], option + '|complete')
            self.assertAlmostEqual(suffixes[0][0][1], 100. + gap)
            call = model.calls[2*at+1]
            for tensor in call['cats'] + call['nums']:
                self.assertTrue(torch.all(tensor == tensor[0:1]))
            self.assertAlmostEqual(float(call['nums'][1][0, -1]), gap, places=5)
            self.assertAlmostEqual(float(call['nums'][0][0, -1]), 100. + gap, places=4)
            self.assertTrue(all(s[1][2] == 3. and len(s) == 2 for s in suffixes))
        self.assertNotEqual(drawn[0][0][0][0][2], drawn[1][0][0][0][2])
        self.assertEqual(model.last_option_gap_draws['draws_per_case_and_option'], 1)

    def test_default_still_draws_one_gap_for_each_suffix(self):
        model = RecordingModel(shared=False)
        del model.params['shared_option_gap']
        drawn, _ = model.option_suffixes_many(self.point, self.cases, ['A', 'B'], 1000., 30)
        self.assertEqual([call['n'] for call in model.calls], [30, 30, 30, 30])
        self.assertGreater(len(set(s[0][2] for s in drawn[0][0])), 1)
        self.assertFalse(hasattr(model, 'last_option_gap_draws'))

    def test_first_gap_does_not_depend_on_number_of_future_suffixes(self):
        model = RecordingModel(shared=True)
        torch.manual_seed(31)
        one, _ = model.option_suffixes_many(self.point, self.cases, ['A', 'B'], 1000., 1)
        torch.manual_seed(31)
        thirty, _ = model.option_suffixes_many(self.point, self.cases, ['A', 'B'], 1000., 30)
        for at in range(2):
            self.assertTrue(all(s[0][2] == one[at][0][0][0][2] for s in thirty[at][0]))

    def test_one_suffix_is_identical_with_or_without_sharing(self):
        model = RecordingModel(shared=False)
        torch.manual_seed(42)
        before = model.option_suffixes_many(self.point, self.cases, ['A', 'B'], 1000., 1)
        model.params['shared_option_gap'] = True
        torch.manual_seed(42)
        after = model.option_suffixes_many(self.point, self.cases, ['A', 'B'], 1000., 1)
        self.assertEqual(before, after)

    def test_each_case_has_its_own_single_gap(self):
        model = RecordingModel(shared=True)
        cases = self.cases + [[{ACTIVITY: 'Decision', 'elapsed': 200.}]]
        drawn, elapsed = model.option_suffixes_many(self.point, cases, ['A', 'B'], 1000., 30)
        self.assertEqual(elapsed, [100., 200.])
        for at in range(2):
            gaps = []
            for case, suffixes in enumerate(drawn[at]):
                gap = suffixes[0][0][2]
                gaps.append(gap)
                self.assertTrue(all(s[0][2] == gap for s in suffixes))
                self.assertAlmostEqual(suffixes[0][0][1], elapsed[case] + gap)
            self.assertNotEqual(gaps[0], gaps[1])
        self.assertEqual(len(model.last_option_gap_draws['gaps_seconds']['A']), 2)


if __name__ == '__main__':
    unittest.main()

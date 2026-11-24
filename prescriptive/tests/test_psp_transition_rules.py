"""Observed-transition invariants of suffix sampling, including forced option events."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from prescriptive.helpers.datasets import add_paths
add_paths()
from prescriptive.estimators.psp_model import ACTIVITY, TIME_COLUMNS, PSPModel
from evaluation_v2.probabilistic_evaluation import ProbabilisticEvaluation


class Evaluator:
    inverted_suffix_categories = [{1: "A|complete|0s", 2: "B|complete|<10min",
                                   3: "B|complete|<1d", 4: "EOS"}]
    prefix_cat_attributes = [ACTIVITY]
    prefix_num_attributes = TIME_COLUMNS[:2]
    all_cat_attributes = [ACTIVITY]
    all_num_attributes = TIME_COLUMNS[:2]
    eos_id = 4
    use_variance_cat = True
    sample_argmax = False
    sample_cat_predictions_optim = staticmethod(ProbabilisticEvaluation.sample_cat_predictions_optim)


class Network:
    def inference(self, prefix=None, **kwargs):
        if prefix is not None:
            self.rows = len(prefix[0][0])
        # Invalid A/padding and premature EOS dominate unless probabilities are masked.
        logits = torch.tensor([1000., 900., -100., -100., 800.]).repeat(self.rows, 1)
        categorical = {ACTIVITY + "_mean": logits}
        cat_variances = {ACTIVITY + "_var": torch.zeros_like(logits)}
        numbers = {name + "_mean": torch.zeros(self.rows, 1) for name in TIME_COLUMNS[:2]}
        num_variances = {name + "_var": torch.full((self.rows, 1), -6.) for name in TIME_COLUMNS[:2]}
        prediction = ((categorical, numbers), (cat_variances, num_variances))
        return (prediction, None, None) if prefix is not None else (prediction, None)


def model():
    instance = PSPModel.__new__(PSPModel)
    instance.params = {"gap_tokens": True}
    # Deliberately different input/decoder IDs: A has prefix ID 3 but decoder ID 1.
    categories = np.array(["EOS", "B|complete|<1d", "A|complete|0s", "B|complete|<10min"], dtype=object)
    scaler = StandardScaler().fit(np.array([[0.], [1.]]))
    instance.encoder_decoder = SimpleNamespace(
        window_size=20, min_suffix_size=5,
        categorical_encoders={ACTIVITY: SimpleNamespace(categories_=[categories])},
        continuous_encoders={name: scaler for name in TIME_COLUMNS[:2]})
    instance.net = Network()
    instance._evaluator = lambda: Evaluator()
    instance.transition_rules = {"pairs": [("A|complete", "B|complete")], "ends": ["B|complete"]}
    return instance


def encoded(prefix_id):
    return (([torch.tensor([[prefix_id]])], [torch.zeros(1, 1), torch.zeros(1, 1)]),
            (None, None), torch.zeros(1, 1, dtype=torch.bool), 1)


class TransitionRulesTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_forbidden_steps_and_premature_eos_have_zero_probability(self):
        m = model()
        suffixes = m.draw_cases([encoded(3)], 300)[0]
        self.assertTrue(all(len(s) == 1 and s[0][0] == "B|complete" for s in suffixes))

    def test_all_time_classes_of_allowed_successor_remain_available(self):
        m = model()
        masks = m._transition_masks(Evaluator())
        self.assertEqual(masks["prefix"][3].tolist(), [False, False, True, True, False])
        suffixes = m.draw_cases([encoded(3)], 300, first=["B|complete"], max_steps=1)[0]
        gaps = np.array([s[0][2] for s in suffixes])
        self.assertTrue((gaps < 600).any())
        self.assertTrue((gaps >= 14400).any())

    def test_only_observed_case_ends_allow_eos(self):
        m = model()
        self.assertEqual(m.draw_cases([encoded(2)], 30)[0], [[] for _ in range(30)])

    def test_forced_option_and_transition_rules_are_intersected(self):
        m = model()
        with self.assertRaisesRegex(ValueError, "kein erlaubter Nachfolger"):
            m.draw_cases([encoded(3)], 3, first=["A|complete"], max_steps=1)

    def test_no_successor_fails_instead_of_relaxing_rule(self):
        m = model()
        m.transition_rules = {"pairs": [], "ends": ["B|complete"]}
        with self.assertRaisesRegex(ValueError, "kein erlaubter Nachfolger"):
            m.draw_cases([encoded(3)], 3)

    def test_last_event_override_selects_actual_option_successors(self):
        m = model()
        result = m.draw_cases([encoded(3)], 3, last={"cats": {0: np.full(3, 2)}})[0]
        self.assertEqual(result, [[], [], []])

    def test_unknown_time_class_uses_raw_activity_for_transition_rules(self):
        m = model()
        # A known activity in an unseen time class has encoded input ID 0.
        suffixes = m.draw_cases([encoded(0)], 30, previous_tokens=["A|complete"])[0]
        self.assertTrue(all(len(s) == 1 and s[0][0] == "B|complete" for s in suffixes))

    def test_unknown_raw_activity_does_not_relax_transition_rules(self):
        m = model()
        with self.assertRaisesRegex(ValueError, "unbekanntes letztes Token"):
            m.draw_cases([encoded(0)], 3, previous_tokens=["Unseen|complete"])

    def test_absent_rules_preserve_original_random_draws(self):
        m = model()
        del m.transition_rules
        torch.manual_seed(99)
        original = m.draw_cases([encoded(3)], 4, max_steps=2)
        state = torch.random.get_rng_state().clone()
        m.transition_rules = None
        torch.manual_seed(99)
        self.assertEqual(m.draw_cases([encoded(3)], 4, max_steps=2), original)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))


if __name__ == "__main__":
    unittest.main()

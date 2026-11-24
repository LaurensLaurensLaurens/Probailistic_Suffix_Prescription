from datetime import datetime, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from prescriptive.helpers.datasets import add_paths
add_paths()
from interventions import ChoicePoint
from interventions.work_state import WorkState
from simulation.case_engine_lifecycle_aware import Case, CaseEngine
from prescriptive.estimators.conditional_gaps import ConditionalGaps
from prescriptive.estimators.psp_model import PSPModel, ACTIVITY, TIME_COLUMNS
from prescriptive.policies.ev_policy.psp_policy import PSPPolicy
from evaluation_v2.probabilistic_evaluation import ProbabilisticEvaluation


def event(activity, lifecycle, seconds=0):
    return {"case:concept:name": "Application_750", "concept:name": activity,
        "lifecycle:transition": lifecycle, "time:timestamp": datetime(2020, 1, 1)+timedelta(seconds=seconds)}


class WorkStateTests(unittest.TestCase):
    def test_duplicate_opening_and_resume_after_completion_are_infeasible(self):
        state = WorkState.from_events([event("W_task", "start")])
        self.assertFalse(state.permits("W_task|start"))
        self.assertFalse(state.permits("W_task|resume"))
        state.push("W_task|suspend")
        self.assertTrue(state.permits("W_task|resume"))
        self.assertFalse(state.permits("W_task|start"))
        state.push("W_task|complete")
        self.assertFalse(state.permits("W_task|resume"))
        self.assertTrue(state.permits("W_task|start"))

    def test_planned_task_abort_is_allowed_and_eos_needs_closed_work(self):
        state = WorkState()
        self.assertTrue(state.permits("W_task|withdraw"))
        state.push("W_task|start")
        state.push("A_Pending|complete")
        self.assertFalse(state.complete)
        state.push("W_task|complete")
        self.assertTrue(state.complete)

    def test_case750_option_is_filtered_before_its_time_is_drawn(self):
        engine = CaseEngine.__new__(CaseEngine)
        case = Case("Application_750", {}, None, {}, last_token=0,
            sessions={"W_Complete application": "User_25"}, rng=np.random.default_rng(0),
            prefix=[event("W_Complete application", "start"), event("A_Concept", "complete")])
        point = ChoicePoint("annehmen_bei_konzept", "control_flow", "A_Concept", "complete",
            ("W_Complete application|start", "A_Accepted"))
        engine._control_flow = {0: (point, [3, 4])}
        engine._tokens = {3: ("W_Complete application", "start"), 4: ("A_Accepted", "complete")}
        engine.work_item_prefix = "W_"
        engine.min_probability, engine.choices, engine.window = 0., [], (700, 900)
        engine.policy = SimpleNamespace(choose=Mock(return_value="A_Accepted"))
        engine._schedule = Mock()
        engine.model = SimpleNamespace(use_offers=False, max_length=100,
            step=lambda *a, **k: np.array([[-np.inf, -np.inf, -np.inf, 10., 1.]]),
            draw=lambda logits, *a: np.argmax(logits, axis=-1),
            time_logits=Mock(return_value=np.array([[0.]])),
            encoder=SimpleNamespace(resource_id=lambda r: 0, time_edges=np.array([0., 1.])))
        with patch("simulation.case_engine_lifecycle_aware.sample_seconds", return_value=np.array([10.])):
            engine._advance(case, 0.)
        self.assertEqual(engine.policy.choose.call_args.args[2], ["A_Accepted"])
        self.assertEqual(case.pending[0], 4)
        np.testing.assert_array_equal(engine.model.time_logits.call_args.args[1], [4])
        engine.policy = None
        with patch("simulation.case_engine_lifecycle_aware.sample_seconds", return_value=np.array([10.])):
            engine._advance(case, 10.)
        self.assertEqual(case.pending[0], 4)

    def test_uncertain_difference_expands_sampling_then_preserves_bank(self):
        policy = PSPPolicy.__new__(PSPPolicy)
        policy.first_only = set()
        calls = []
        def samples(point, events, options, now, n):
            calls.append(n)
            return np.array([np.resize([-1., 1.], n), np.zeros(n)])
        policy.model = SimpleNamespace(knows=lambda point: True, samples_per_option=4,
            params={"sampling_control": {"max_samples": 8, "z": 2.}}, outcome_samples=samples)
        result = policy.choose(SimpleNamespace(), SimpleNamespace(prefix=[]), ["A", "B"], 0., None)
        self.assertIsNone(result)
        self.assertEqual(calls, [4, 8])
        self.assertEqual(policy.decision_audit["uncertain_fallbacks"], 1)

    def test_clear_difference_does_not_spend_maximum_sampling_budget(self):
        policy = PSPPolicy.__new__(PSPPolicy)
        policy.first_only = set()
        policy.model = SimpleNamespace(knows=lambda point: True, samples_per_option=4,
            params={"sampling_control": {"max_samples": 8, "z": 2.}},
            outcome_samples=Mock(return_value=np.array([[2.]*4, [0.]*4])))
        self.assertEqual(policy.choose(SimpleNamespace(), SimpleNamespace(prefix=[]), ["A", "B"], 0., None), "A")
        self.assertEqual(policy.model.outcome_samples.call_count, 1)

    def test_gap_calibration_keeps_active_and_waiting_contexts_separate(self):
        records = [event("W_task", "start"), event("A_Complete", "complete", 2),
            event("W_task", "complete", 3), event("A_Complete", "complete", 1003)]
        calibration = ConditionalGaps(min_observations=1).fit(pd.DataFrame(records))
        active = WorkState.from_events(records[:1])
        closed = WorkState.from_events(records[:3])
        self.assertEqual(calibration.distribution(active, "A_Complete|complete").tolist(), [2.])
        self.assertEqual(calibration.distribution(closed, "A_Complete|complete").tolist(), [1000.])
        self.assertEqual(calibration.training_cases, ["Application_750"])

    def test_draws_block_duplicate_start_and_premature_eos(self):
        names = {1: "W_task|start|<10min", 2: "W_task|suspend|<10min", 3: "A_Pending|complete|<10min", 4: "EOS"}
        class Evaluator:
            inverted_suffix_categories = [names]
            prefix_cat_attributes, prefix_num_attributes = [ACTIVITY], TIME_COLUMNS[:2]
            all_cat_attributes, all_num_attributes = [ACTIVITY], TIME_COLUMNS[:2]
            eos_id, use_variance_cat, sample_argmax = 4, True, False
            sample_cat_predictions_optim = staticmethod(ProbabilisticEvaluation.sample_cat_predictions_optim)
        class Network:
            def inference(self, prefix=None, **kwargs):
                if prefix is not None:
                    self.rows, self.step = len(prefix[0][0]), 0
                else:
                    self.step += 1
                scores = [[-1000., 900., 800., -100., 1000.],
                    [-1000., 900., -100., 800., 1000.], [-1000., -100., -100., -100., 1000.]][min(self.step, 2)]
                cat = {ACTIVITY+"_mean": torch.tensor(scores).repeat(self.rows, 1)}
                var = {ACTIVITY+"_var": torch.full_like(cat[ACTIVITY+"_mean"], -6.)}
                nums = {name+"_mean": torch.zeros(self.rows, 1) for name in TIME_COLUMNS[:2]}
                nvar = {name+"_var": torch.zeros(self.rows, 1) for name in TIME_COLUMNS[:2]}
                prediction = ((cat, nums), (var, nvar))
                return (prediction, None, None) if prefix is not None else (prediction, None)
        model = PSPModel.__new__(PSPModel)
        model.params, model.work_state_constraints = {"gap_tokens": True}, True
        model.net, model._evaluator = Network(), lambda: Evaluator()
        scaler = StandardScaler().fit(np.array([[0.], [1.]]))
        model.encoder_decoder = SimpleNamespace(window_size=20, min_suffix_size=5,
            categorical_encoders={ACTIVITY: SimpleNamespace(categories_= [np.array(list(names.values()))])},
            continuous_encoders={name: scaler for name in TIME_COLUMNS[:2]})
        encoded = (([torch.tensor([[1]])], [torch.zeros(1, 1), torch.zeros(1, 1)]),
            (None, None), torch.zeros(1, 1, dtype=torch.bool), 1)
        state = WorkState.from_events([event("W_task", "start")])
        torch.manual_seed(12)
        result = model.draw_cases([encoded], 20, work_states=[state])[0]
        self.assertTrue(all([e[0] for e in suffix] == ["W_task|suspend", "A_Pending|complete"] for suffix in result))
        self.assertEqual(state.states["W_task"], "active")


if __name__ == "__main__":
    unittest.main()

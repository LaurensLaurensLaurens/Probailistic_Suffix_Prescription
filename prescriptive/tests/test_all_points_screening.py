"""Randomization, exact AIPW weights, grouped inference and deadline semantics."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from prescriptive.helpers.datasets import add_paths
add_paths()
from screening.all_points import aipw_scores, crossfit_folds, estimate_interval, fit_nuisances
from screening.rct_all_points import AllPointsRCTEngine, RecordingRCT, RandomizedOffers
from simulation.case_engine_lifecycle_aware import CaseEngine, EVENT


class AllPointsScreeningTests(unittest.TestCase):
    def test_known_randomization_probabilities_are_not_clipped(self):
        e = np.array([[.005, .995], [.005, .995]])
        values = aipw_scores(np.array([0, 1]), np.array([1., 2.]), e, np.zeros((2, 2)), randomized=True)
        np.testing.assert_allclose(values, [[200., 0.], [0., 2 / .995]])
        self.assertAlmostEqual(aipw_scores(np.array([0, 1]), np.array([1., 2.]), e,
                               np.zeros((2, 2)), randomized=False)[0, 0], 100.)

    def test_zero_probability_for_unavailable_arm_is_safe_but_observed_zero_is_error(self):
        values = aipw_scores(np.array([0, 1]), np.array([3., 4.]), np.eye(2), np.zeros((2, 2)), randomized=True)
        np.testing.assert_allclose(values, [[3., 0.], [0., 4.]])
        with self.assertRaises(ValueError):
            aipw_scores(np.array([1, 1]), np.ones(2), np.eye(2), np.zeros((2, 2)), randomized=True)

    def test_whole_simulation_runs_stay_in_one_crossfit_fold(self):
        groups = np.repeat(np.arange(10), 17)
        folds = crossfit_folds(len(groups), 5, 42, groups)
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(folds[groups == group])), 1)
        self.assertEqual(len(np.unique(folds)), 5)

    def test_clustered_interval_does_not_treat_shared_cases_as_independent(self):
        groups = np.repeat(np.arange(10), 100)
        values = np.repeat(np.arange(10, dtype=float), 100)
        iid = estimate_interval(values)
        clustered = estimate_interval(values, groups)
        self.assertEqual(clustered[0], 4.5)
        self.assertGreater(clustered[3], 9 * iid[3])

    def test_known_propensity_does_not_fit_classifier_and_recovers_synthetic_effect(self):
        rng = np.random.default_rng(7)
        n = 1000
        X = pd.DataFrame({"x": rng.normal(size=n)})
        codes = rng.integers(2, size=n)
        Y = 3 * codes + 2 * X.x.to_numpy()
        e = np.full((n, 2), .5)
        folds = crossfit_folds(n, 5, 4)
        with patch("xgboost.XGBClassifier", side_effect=AssertionError("RCT propensity must not be fitted")):
            fitted_e, mu = fit_nuisances(X, codes, Y, 2, folds, e, threads=1)
        psi = aipw_scores(codes, Y, fitted_e, mu, randomized=True)
        self.assertAlmostEqual(float((psi[:, 1] - psi[:, 0]).mean()), 3., delta=.2)

    def test_recorded_probability_and_option_match_assignment(self):
        policy = RecordingRCT(7)
        case = SimpleNamespace(case_id="Application_0", n_events=4)
        point = SimpleNamespace(name="staff", kind="employee")
        chosen = policy.choose(point, case, ["A", "B", "C"], 20, None)
        self.assertEqual(policy.records[0]["probability"], 1 / 3)
        self.assertEqual(policy.records[0]["choice"], chosen)
        self.assertEqual(policy.records[0]["event_pos"], 4)

    def test_randomized_offer_updates_state_and_dependent_monthly_payment(self):
        policy = RecordingRCT(7)
        state = SimpleNamespace(case_id="Application_0", credit_score=500, application_type="New")
        case = SimpleNamespace(case_id=state.case_id, n_events=4, pending=(1, 20), attributes={"case:RequestedAmount": 10000})
        monthly = SimpleNamespace(predict=lambda **kw: kw["offered_amount"] / kw["number_of_terms"])
        original = SimpleNamespace(
            dynamic_attributes=lambda state: ({"OfferedAmount": 1, "NumberOfTerms": 60, "FirstWithdrawalAmount": 0, "MonthlyCost": 0}, {}),
            generator=SimpleNamespace(registry=SimpleNamespace(monthly_cost=monthly)),
            spec={"dynamic": {"OfferedAmount": "offered_amount", "NumberOfTerms": "number_of_terms", "MonthlyCost": "monthly_cost"}, "open": {}})
        design = {"attributes": {"angebotshoehe": {"attribute": "OfferedAmount", "ratio_to": "case:RequestedAmount",
                         "labels": ["low", "same", "high"], "samples": {"low": [.9], "same": [1.], "high": [1.1]}}}}
        engine = SimpleNamespace(cases={state.case_id: case}, policy=policy, study_design=design)
        values, event = RandomizedOffers(original, engine).dynamic_attributes(state)
        expected = {"low": 9000, "same": 10000, "high": 11000}[policy.records[0]["choice"]]
        self.assertAlmostEqual(values["OfferedAmount"], expected)
        self.assertAlmostEqual(state.offered_amount, expected)
        self.assertAlmostEqual(event["MonthlyCost"], expected / 60)

    def test_deadline_changes_first_target_start_and_never_moves_before_previous_event(self):
        engine = AllPointsRCTEngine.__new__(AllPointsRCTEngine)
        case = SimpleNamespace(pending=(0, 300), last_time=100)
        engine.cases = {"c": case}
        engine.deadlines, engine.deadline_applied = {"c": 50}, set()
        engine._tokens = [("W_Complete application", "start")]
        with patch.object(CaseEngine, "_schedule") as scheduled:
            engine._schedule(300, EVENT, "c")
            scheduled.assert_called_once_with(100, EVENT, "c")
        self.assertEqual(case.pending, (0, 100))
        self.assertIn("c", engine.deadline_applied)


if __name__ == "__main__":
    unittest.main()

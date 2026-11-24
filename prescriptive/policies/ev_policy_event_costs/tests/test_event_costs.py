from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from prescriptive.helpers.datasets import add_paths
add_paths()
import interventions
from prescriptive.estimators import cate_model
from prescriptive.estimators.cate_model import CATEModel
from prescriptive.policies.ev_policy_event_costs.costs import ACTIVITY, CASE, LIFECYCLE, TIMESTAMP, EventCosts
from prescriptive.policies.ev_policy_event_costs.psp_policy import EventCostPSPModel, PSPEventCostPolicy
from prescriptive.policies.ev_policy_event_costs.cate_model import EventCostCATEModel
from prescriptive.policies.ev_policy_event_costs.cate_policy import CATEEventCostPolicy

START = pd.Timestamp("2025-01-01")
POINT = interventions.ChoicePoint("cost", "control_flow", "Decide", "complete", ("A", "B", "unseen"))


def config():
    return {"defaults": {"base_eur": 1, "eur_per_work_hour": 0,
                         "lifecycle_eur": {"default": .5, "start": 1, "resume": 2, "suspend": 3,
                                           "complete": 4, "ate_abort": 5, "withdraw": 6}},
            "events": {"Work": {"base_eur": 2, "eur_per_work_hour": 60},
                       "Other": {"base_eur": 3, "eur_per_work_hour": 30}, "A_Accepted": {"base_eur": 5}}}


def event(activity, lifecycle, hour, case="1"):
    return {CASE: case, ACTIVITY: activity, LIFECYCLE: lifecycle, TIMESTAMP: START + pd.Timedelta(hours=hour)}


class CostTests(unittest.TestCase):
    def setUp(self):
        self.costs = EventCosts(config())

    def test_start_suspend_resume_complete_excludes_the_pause(self):
        log = pd.DataFrame([event("Work", "start", 0), event("Work", "suspend", 1),
                            event("Work", "resume", 5), event("Work", "complete", 7)])
        ledger = self.costs.event_costs(log)
        self.assertEqual(ledger["total_eur"].tolist(), [3, 65, 4, 126])
        self.assertEqual(ledger["active_hours"].tolist(), [0, 1, 0, 2])
        bill = self.costs.case_costs(log).loc["1"]
        self.assertEqual(bill["active_hours"], 3)
        self.assertEqual(bill["total_eur"], 198)

    def test_complete_only_events_have_base_and_lifecycle_costs(self):
        log = pd.DataFrame([event("A_Accepted", "complete", 100)])
        bill = self.costs.case_costs(log).loc["1"]
        self.assertEqual(bill["total_eur"], 9)
        self.assertEqual(bill["time_eur"], 0)

    def test_intervening_events_and_parallel_activities_do_not_close_work(self):
        log = pd.DataFrame([event("Work", "start", 0), event("Other", "start", .5),
                            event("A_Accepted", "complete", 1), event("Work", "complete", 2),
                            event("Other", "complete", 3)])
        bill = self.costs.case_costs(log).loc["1"]
        self.assertEqual(bill["active_hours"], 4.5)
        self.assertEqual(bill["time_eur"], 195)

    def test_abort_and_withdraw_close_the_active_segment(self):
        for lifecycle in ("ate_abort", "withdraw"):
            with self.subTest(lifecycle=lifecycle):
                log = pd.DataFrame([event("Work", "start", 0), event("Work", lifecycle, 1),
                                    event("A_Accepted", "complete", 50)])
                bill = self.costs.case_costs(log).loc["1"]
                self.assertEqual(bill["time_eur"], 60)
                self.assertEqual(bill["unclosed_segments"], 0)

    def test_missing_begin_has_fixed_fees_and_no_invented_work_time(self):
        bill = self.costs.case_costs(pd.DataFrame([event("Work", "complete", 10)])).loc["1"]
        self.assertEqual(bill["total_eur"], 6)
        self.assertEqual(bill["orphan_closures"], 1)

    def test_open_work_accrues_to_horizon_without_a_fake_completion_fee(self):
        log = pd.DataFrame([event("Work", "start", 0), event("A_Accepted", "complete", 2)])
        ledger = self.costs.event_costs(log)
        bill = self.costs.case_costs(log).loc["1"]
        self.assertEqual(bill["total_eur"], 3 + 9 + 120)
        self.assertEqual(bill["unclosed_segments"], 1)
        self.assertEqual(ledger.iloc[0]["time_eur"], 120)
        self.assertEqual(ledger.iloc[0]["unclosed_segments"], 1)
        self.assertEqual(ledger.iloc[-1]["time_eur"], 0)
        self.assertEqual(ledger.iloc[-1]["total_eur"], 9)

    def test_repeated_opening_keeps_elapsed_work_and_records_the_anomaly(self):
        log = pd.DataFrame([event("Work", "start", 0), event("Work", "resume", 1), event("Work", "complete", 3)])
        bill = self.costs.case_costs(log).loc["1"]
        self.assertEqual(bill["time_eur"], 180)
        self.assertEqual(bill["reopened_segments"], 1)

    def test_stable_order_duplicate_indices_and_microsecond_timestamps(self):
        log = pd.DataFrame([event("Work", "complete", 1), event("Work", "start", 0),
                            event("A_Accepted", "complete", 2, "2")], index=[7, 7, 2])
        log[TIMESTAMP] = log[TIMESTAMP].astype("datetime64[us]")
        ledger = self.costs.event_costs(log)
        self.assertEqual(ledger.index.tolist(), [7, 7, 2])
        self.assertEqual(ledger["time_eur"].tolist(), [60, 0, 0])
        self.assertEqual(self.costs.case_costs(log).loc["2", "total_eur"], 9)

    def test_case_boundaries_prevent_cross_case_matching(self):
        log = pd.DataFrame([event("Work", "start", 0), event("Work", "complete", 1, "2")])
        bill = self.costs.case_costs(log)
        self.assertTrue((bill["time_eur"] == 0).all())
        self.assertEqual(bill.loc["2", "orphan_closures"], 1)

    def test_empty_logs_and_unused_case_categories_do_not_add_cases(self):
        empty = pd.DataFrame(columns=[CASE, ACTIVITY, TIMESTAMP])
        self.assertTrue(self.costs.event_costs(empty).empty)
        self.assertTrue(self.costs.case_costs(empty).empty)
        log = pd.DataFrame([event("Work", "start", 0), event("Work", "complete", 1)])
        log[CASE] = pd.Categorical(log[CASE], categories=["unused", "1"])
        bill = self.costs.case_costs(log)
        self.assertEqual(bill.index.tolist(), ["1"])
        self.assertEqual(bill.loc["1", "time_eur"], 60)

    def test_timezone_and_naive_utc_have_the_same_costs(self):
        log = pd.DataFrame([event("Work", "start", 0), event("Work", "complete", 1)])
        aware = log.copy()
        aware[TIMESTAMP] = aware[TIMESTAMP].dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
        pd.testing.assert_frame_equal(self.costs.case_costs(log), self.costs.case_costs(aware))
        self.assertEqual(self.costs.state(log.to_dict("records")).total(),
                         self.costs.state(aware.to_dict("records")).total())

    def test_labelled_log_has_per_event_and_per_case_costs_without_mutating_input(self):
        log = pd.DataFrame([event("Work", "start", 0), event("Work", "complete", 1)])
        log["outcome"] = 99.
        before = log.copy(deep=True)
        labelled = self.costs.label_log(log)
        self.assertEqual(labelled["event_cost_eur"].tolist(), [3, 66])
        self.assertEqual(labelled["case_event_cost_eur"].tolist(), [69, 69])
        self.assertEqual(labelled["outcome"].tolist(), [-69, -69])
        pd.testing.assert_frame_equal(log, before)

    def test_resource_identity_does_not_change_event_costs(self):
        log = pd.DataFrame([event("Work", "start", 0), event("Work", "complete", 1)])
        log["org:resource"] = ["expensive_employee", "another_employee"]
        before = self.costs.case_costs(log)
        log["org:resource"] = "anybody"
        pd.testing.assert_frame_equal(before, self.costs.case_costs(log))

    def test_config_validation_fingerprint_and_unknown_activities(self):
        self.assertEqual(self.costs.fingerprint, EventCosts(self.costs.to_dict()).fingerprint)
        data = self.costs.to_dict()
        data["events"]["Work"]["base_eur"] = 999
        self.assertNotEqual(self.costs.fingerprint, EventCosts(data).fingerprint)
        self.assertEqual(self.costs.tariff("Work")["base_eur"], 2)
        for bad in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                EventCosts({"defaults": {"base_eur": bad}})
        with self.assertRaises(ValueError):
            EventCosts({"defaults": {"bas_eur": 1}})
        strict = EventCosts({**config(), "unknown_activity": "error"})
        with self.assertRaisesRegex(ValueError, "No event tariff"):
            strict.case_costs(pd.DataFrame([event("missing", "complete", 0)]))
        default = self.costs.case_costs(pd.DataFrame([event("missing", "complete", 0)])).loc["1"]
        self.assertEqual(default["total_eur"], 5)
        self.assertEqual(default["unconfigured_events"], 1)

    def test_default_configuration_matches_the_agreed_example(self):
        costs = EventCosts()
        log = pd.DataFrame([event("W_Validate application", "start", 0),
                            event("W_Validate application", "complete", .5), event("A_Accepted", "complete", 10)])
        self.assertEqual(costs.event_costs(log)["total_eur"].tolist(), [3, 35, 7])


class FakePSP:
    lifecycle_tokens = True
    samples_per_option = 2

    def __init__(self, mode="encoder"):
        self.mode = mode
        self.option_gap_distributions = {"test": True} if mode == "shared" else None

    def knows(self, point):
        return point.name == POINT.name

    def set_threads(self, n):
        pass

    def option_in_encoder(self):
        return self.mode == "encoder"

    def option_in_decoder(self):
        return self.mode == "decoder"

    def _option_event(self, point, events, option, now):
        return {CASE: events[0][CASE], ACTIVITY: option, LIFECYCLE: "complete",
                TIMESTAMP: pd.Timestamp(now + .001, unit="s")}

    def option_suffixes(self, point, events, options, now, n):
        included = self.mode != "legacy"
        elapsed = (pd.Timestamp(now, unit="s") - events[0][TIMESTAMP]).total_seconds() + (0 if included else .001)
        result = []
        for option in options:
            suffixes = []
            for i in range(n):
                # A has lower minimum but higher mean cost: EV must choose B.
                hours = (0 if i % 2 == 0 else 4) if option == "A" else 1
                prefix = [(option + "|complete", elapsed, 0)] if included else []
                suffixes.append(prefix + [("Work|start", elapsed, 0), ("Work|complete", elapsed + hours*3600, hours*3600)])
            result.append(suffixes)
        return result, elapsed


class PSPTests(unittest.TestCase):
    def setUp(self):
        self.events = [event("Decide", "complete", 0)]
        self.now = START.timestamp()

    def test_costs_match_a_complete_log_for_every_existing_option_mode(self):
        costs = EventCosts(config())
        for mode in ("encoder", "decoder", "shared", "legacy"):
            with self.subTest(mode=mode):
                wrapped = EventCostPSPModel(FakePSP(mode), costs)
                values = wrapped.outcome_samples(POINT, self.events, ["A", "B"], self.now)
                # Future fees: option 5, Work start 3, Work complete 6 = 14.
                np.testing.assert_allclose(values, [[-14, -254], [-74, -74]])
                self.assertGreater(values[1].mean(), values[0].mean())

    def test_same_ev_choose_rule_selects_the_lower_mean_cost(self):
        with patch("prescriptive.estimators.psp_model.PSPModel.load", return_value=FakePSP()):
            policy = PSPEventCostPolicy("unused", cost_config=config(), samples=2)
        before = deepcopy(self.events)
        chosen = policy.choose(POINT, SimpleNamespace(prefix=self.events), ["A", "B"], self.now, None)
        self.assertEqual(chosen, "B")
        self.assertEqual(self.events, before)
        self.assertIsNone(policy.choose(POINT, SimpleNamespace(prefix=self.events), [], self.now, None))

    def test_prefix_open_segment_continues_and_past_work_is_subtracted_once(self):
        fake = FakePSP()
        fake.option_suffixes = lambda *args: ([[[("A|complete", 1800, 0), ("Work|complete", 3600, 1800)]]], 1800.)
        events = [event("Work", "start", 0), event("Decide", "complete", .5)]
        result = EventCostPSPModel(fake, config()).outcome_samples(POINT, events, ["A"], START.timestamp() + 1800, n=1)
        # Only future half-hour (30 EUR), option fee (5), close fee (6).
        self.assertAlmostEqual(result[0, 0], -41)

    def test_non_lifecycle_models_are_rejected(self):
        fake = FakePSP()
        fake.lifecycle_tokens = False
        with self.assertRaises(ValueError):
            EventCostPSPModel(fake, config())


def cost_log():
    rows = []
    for arm, option in enumerate(("A", "B")):
        for i in range(240):
            attributes = {CASE: f"case_{arm}_{i:04}", "case:LoanGoal": "test", "case:ApplicationType": "test",
                          "case:RequestedAmount": float(i % 4), "OfferedAmount": 0., "NumberOfTerms": 0.,
                          "MonthlyCost": 0., "FirstWithdrawalAmount": 0., "outcome": -float(1 + arm)}
            path = [("Decide", "complete", 0), (option, "complete", .01), ("Work", "start", .02),
                    ("Work", "complete", 1.02 if not arm else .12), ("End", "complete", 24 if not arm else 48)]
            for activity, lifecycle, hour in path:
                rows.append({**event(activity, lifecycle, hour, attributes[CASE]), **attributes})
    return pd.DataFrame(rows)


class LearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.log = cost_log()
        cls.models, cls.paths = {}, {}
        fast = {"n_estimators": 80, "max_depth": 2, "learning_rate": .1, "n_jobs": 1,
                "early_stopping_rounds": 10, "subsample": 1., "colsample_bytree": 1.}
        with warnings.catch_warnings(), patch.dict(cate_model.PARAMS, fast), \
                patch.dict(cate_model.FOREST, {"n_estimators": 32, "cv": 2, "n_jobs": 1}):
            warnings.simplefilter("ignore", RuntimeWarning)
            for learner in ("s", "x", "cf"):
                model = EventCostCATEModel("bpic17", [POINT], learner=learner, cost_config=config()).fit(cls.log)
                path = str(Path(cls.folder.name) / f"{learner}.pkl")
                model.save(path)
                cls.models[learner], cls.paths[learner] = model, path
            ordinary = CATEModel("bpic17", [POINT], learner="s").fit(cls.log)
            cls.ordinary_path = str(Path(cls.folder.name) / "ordinary.pkl")
            ordinary.save(cls.ordinary_path)
        cls.case = SimpleNamespace(prefix=cls.log[cls.log[CASE] == "case_0_0000"].iloc[:1].to_dict("records"))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_all_three_real_learners_choose_lower_cost_even_with_longer_cycle_time(self):
        for learner, model in self.models.items():
            with self.subTest(learner=learner):
                values = model.values(POINT, self.case.prefix, list(POINT.options), START.timestamp())
                self.assertGreater(values[1], values[0])
                self.assertEqual(values[2], -np.inf)
                policy = CATEEventCostPolicy(self.paths[learner], cost_config=config())
                self.assertEqual(policy.choose(POINT, self.case, list(POINT.options), START.timestamp(), None), "B")
                self.assertIsNone(policy.choose(POINT, self.case, ["unseen"], START.timestamp(), None))

    def test_training_relabels_a_copy_and_preserves_the_choice_points(self):
        self.assertTrue((self.log.groupby(CASE)["outcome"].first().isin([-1, -2])).all())
        for model in self.models.values():
            self.assertEqual(model.points[POINT.name], POINT)
            self.assertEqual(model.objective, "negative_event_costs")
            self.assertEqual(model.cost_fingerprint, EventCosts(config()).fingerprint)

    def test_policy_rejects_cycle_models_and_changed_tariffs(self):
        with self.assertRaisesRegex(ValueError, "negative event costs"):
            CATEEventCostPolicy(self.ordinary_path)
        changed = config()
        changed["events"]["Work"]["eur_per_work_hour"] = 99
        with self.assertRaisesRegex(ValueError, "differ from training"):
            CATEEventCostPolicy(self.paths["s"], cost_config=changed)
        with self.assertRaises(ValueError):
            CATEEventCostPolicy(self.paths["s"], threshold=-1)


if __name__ == "__main__":
    unittest.main()

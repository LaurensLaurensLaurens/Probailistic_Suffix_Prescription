"""ERM-Ziel, alle drei echten Schätzer, gespeicherte Policies und CLI-Einbindung.

Aufruf im Projektordner:
    process-simulation-engine/.venv/bin/python -m unittest discover -s prescriptive/tests -v
"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from prescriptive.helpers.datasets import add_paths

add_paths()

import interventions
from prescriptive.estimators import cate_model
from prescriptive.estimators.cate_erm_model import CATEERMModel
from prescriptive.estimators.cate_model import CATEModel
from prescriptive.estimators.xgb_model import ACTIVITY, CASE, LIFECYCLE, TIMESTAMP
from prescriptive.evaluation import online_evaluation as online
from prescriptive.policies.erm_policy.cate_policy import CATEERMPolicy
from prescriptive.policies.erm_policy.psp_erm_policy import entropic_risk
from prescriptive.policies.ev_policy.cate_policy import CATEPolicy
from prescriptive.scripts import main as command
from prescriptive.scripts import train_cate_real


POINT = interventions.ChoicePoint("risk", "control_flow", "Decide", "complete", ("A", "B", "unseen"))
FAST_XGB = {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.1, "subsample": 1.0,
            "colsample_bytree": 1.0, "n_jobs": 1, "early_stopping_rounds": 10}


def risk_log():
    """A: 95 % 5 Tage, 5 % 70 Tage (Mittel 8,25); B: immer 10 Tage. Bei γ=0,1 ist B risikoärmer.

    Jede der fünf Prefix-Ausprägungen hat dieselben Anteile und beide Aktionen; keine Confounder.
    Die dritte Option hat keine Überdeckung und darf nicht als scheinbar risikofrei gelten.
    """
    rows = []
    start = pd.Timestamp("2025-01-01")
    for arm, option in enumerate(POINT.options[:2]):
        for i in range(1000):
            days = 10 if arm else (70 if i % 20 == 0 else 5)
            attributes = {CASE: f"case_{arm}_{i:04}", "case:LoanGoal": "test", "case:ApplicationType": "test",
                          "case:RequestedAmount": float((i // 20) % 5), "outcome": -float(days), LIFECYCLE: "complete",
                          "OfferedAmount": 0.0, "NumberOfTerms": 0.0, "MonthlyCost": 0.0, "FirstWithdrawalAmount": 0.0}
            for activity, at in (("Decide", start), (option, start + pd.Timedelta(hours=1)), ("End", start + pd.Timedelta(days=days))):
                rows.append({**attributes, ACTIVITY: activity, TIMESTAMP: at})
    return pd.DataFrame(rows)


class CATEERMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.log = risk_log()
        first = CATEERMModel("bpic17", [POINT], gamma=0.1)
        cls.samples = first.samples(cls.log)
        cls.features = [row[1] for row in cls.samples[POINT.name]][::20]
        cls.models, cls.paths = {}, {}
        with patch.dict(cate_model.PARAMS, FAST_XGB), patch.dict(cate_model.FOREST, {"n_estimators": 32, "cv": 2, "n_jobs": 1}):
            for learner in ("s", "x", "cf"):
                model = CATEERMModel("bpic17", [POINT], learner=learner, gamma=0.1).fit(cls.log, cls.samples)
                path = str(Path(cls.folder.name) / f"{learner}.pkl")
                model.save(path)
                cls.models[learner], cls.paths[learner] = model, path
            cls.ev = CATEModel("bpic17", [POINT], learner="s").fit(cls.log, cls.samples)
            cls.ev_path = str(Path(cls.folder.name) / "ev.pkl")
            cls.ev.save(cls.ev_path)
        case = cls.log[cls.log[CASE] == "case_0_0000"].iloc[:1]
        cls.case = SimpleNamespace(prefix=case.to_dict("records"))
        cls.now = pd.Timestamp("2025-01-01").timestamp()

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_all_three_learners_prefer_lower_erm_despite_worse_mean(self):
        a = np.array([70.0] + [5.0] * 19)
        b = np.full(20, 10.0)
        self.assertLess(a.mean(), b.mean())
        self.assertGreater(entropic_risk([a, b], 0.1)[0], entropic_risk([a, b], 0.1)[1])
        # EV und ERM müssen bei diesen Daten entgegengesetzte Richtungen lernen.
        self.assertGreater(np.mean(self.ev.effects(POINT, self.features)[:, 0] < 0), 0.9)
        for learner, model in self.models.items():
            with self.subTest(learner=learner):
                effects = model.effects(POINT, self.features)
                self.assertGreater(np.mean(effects[:, 0] < 0), 0.9)
                policy = CATEERMPolicy(self.paths[learner], gamma=0.1)
                self.assertEqual(policy.choose(POINT, self.case, ["A", "B"], self.now, None), "B")

    def test_translation_invariance_without_exponential_overflow(self):
        model = CATEERMModel("bpic17", [POINT], gamma=0.1)
        held = np.array([False, False, True])
        losses = np.array([5.0, 70.0, 10.0])
        targets = model._targets("original", -losses, held)
        shifted = model._targets("shifted", -(losses + 10000), held)
        np.testing.assert_allclose(targets, shifted, rtol=1e-10)
        self.assertAlmostEqual(targets[~held].mean(), 1.0)
        self.assertAlmostEqual(model.log_scales["shifted"] - model.log_scales["original"], 1000.0)

    def test_inner_validation_does_not_set_scaling(self):
        model = CATEERMModel("bpic17", [POINT], gamma=0.1)
        held = np.array([False, False, True])
        model._targets("point", -np.array([5.0, 10.0, 20.0]), held)
        scale = model.log_scales["point"]
        model._targets("point", -np.array([5.0, 10.0, 70.0]), held)
        self.assertEqual(scale, model.log_scales["point"])

    def test_three_supported_actions_share_scale_despite_different_sample_sizes(self):
        point = interventions.ChoicePoint("three", "control_flow", "Decide", "complete", ("A", "B", "C"))
        a = self.log[self.log[CASE].str.startswith("case_0_")].iloc[:300].copy()  # 100 Referenzfälle
        b = self.log[self.log[CASE].str.startswith("case_1_")].copy()  # 1000 Fälle der zweiten Option
        c = a.copy()  # 100 Fälle der dritten Option
        c[CASE] = "third_" + c[CASE]
        c.loc[c[ACTIVITY] == "A", ACTIVITY] = "C"
        for part, days in ((a, 20), (b, 10), (c, 5)):
            part["outcome"] = -float(days)
            part.loc[part[ACTIVITY] == "End", TIMESTAMP] = pd.Timestamp("2025-01-01") + pd.Timedelta(days=days)
        with patch.dict(cate_model.PARAMS, FAST_XGB):
            model = CATEERMModel("bpic17", [point], learner="s", gamma=0.1).fit(pd.concat([a, b, c]))
        effects = model.effects(point, self.features)
        self.assertTrue((effects[:, 1] < effects[:, 0]).all())
        self.assertTrue((effects[:, 0] < 0).all())
        self.assertEqual(set(model.log_scales), {"three"})
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "three.pkl")
            model.save(path)
            policy = CATEERMPolicy(path)
            self.assertEqual(policy.choose(point, self.case, ["B", "A", "C"], self.now, None), "C")

    def test_invalid_gamma_or_unrepresentable_targets_fail_explicitly(self):
        for gamma in (0, -0.1, np.nan, np.inf):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                CATEERMModel("bpic17", [POINT], gamma=gamma)
        model = CATEERMModel("bpic17", [POINT], gamma=0.1)
        for outcome in ([1.0, -5.0], [-np.inf, -5.0], [np.nan, -5.0], [-10000.0, -5.0]):
            with self.subTest(outcome=outcome), self.assertRaises(ValueError):
                model._targets("point", np.array(outcome), np.array([False, False]))

    def test_untrained_and_unknown_actions_are_excluded(self):
        for learner, path in self.paths.items():
            with self.subTest(learner=learner):
                policy = CATEERMPolicy(path)
                values = policy.model.values(POINT, self.case.prefix, ["unseen", "other", "B", "A"], self.now)
                self.assertTrue(np.isposinf(values[:2]).all())
                self.assertEqual(values[3], 0.0)
                self.assertIsNone(policy.choose(POINT, self.case, ["unseen", "other"], self.now, None))
                self.assertEqual(policy.choose(POINT, self.case, ["unseen", "A"], self.now, None), "A")
                self.assertEqual(policy.choose(POINT, self.case, ["other", "B"], self.now, None), "B")

    def test_gamma_metadata_and_ev_erm_model_guards(self):
        for learner, path in self.paths.items():
            with self.subTest(learner=learner):
                policy = CATEERMPolicy(path)
                self.assertEqual(policy.gamma, 0.1)
                self.assertEqual(policy.model.log_scales, self.models[learner].log_scales)
                for gamma in (0, 0.2, np.nan, np.inf):
                    with self.assertRaises(ValueError):
                        CATEERMPolicy(path, gamma=gamma)
                with self.assertRaises(ValueError):
                    CATEPolicy(path)
        with self.assertRaises(ValueError):
            CATEERMPolicy(self.ev_path)

    def test_first_only_including_merged_points_and_unknown_points(self):
        policy = CATEERMPolicy(self.paths["s"], first_only=["risk"])
        repeated = SimpleNamespace(prefix=self.case.prefix * 2)
        self.assertIsNone(policy.choose(POINT, repeated, ["A", "B"], self.now, None))
        merged = interventions.ChoicePoint("risk+retry", "control_flow", "Decide", "complete", ("A", "B"))
        self.assertTrue(policy._repeated(merged, repeated.prefix))
        unknown = interventions.ChoicePoint("unknown", "control_flow", "Elsewhere", "complete", ("A", "B"))
        self.assertIsNone(policy.choose(unknown, self.case, ["A", "B"], self.now, None))

    def test_online_factory_selects_each_learner_and_checks_model(self):
        params = {name: {"model": self.paths[learner], "gamma": 0.1} for name, learner in
                  zip(online.CATE_ERM_POLICIES, ("s", "x", "cf"))}
        with patch.dict(online._SHARED, {"policy_params": params}, clear=True):
            for name, learner in zip(online.CATE_ERM_POLICIES, ("s", "x", "cf")):
                policy = online._policy(name, 42)
                self.assertEqual(policy.model.learner, learner)
                self.assertEqual(policy.choose(POINT, self.case, ["A", "B"], self.now, None), "B")
            params["x_learner_erm"]["model"] = self.paths["s"]
            with self.assertRaises(ValueError):
                online._policy("x_learner_erm", 42)

    def test_cli_merges_gamma_models_and_psp_settings(self):
        argv = ["main.py", "--steps", "online", "--online-policies", *online.CATE_ERM_POLICIES, "psp_erm",
                "--cate-model", f"s_learner_erm={self.paths['s']}", f"s_learner={self.ev_path}",
                "--erm-gamma", "0.1", "--cate-threshold", "2", "--cate-first-only", "risk",
                "--psp-model", "custom_psp.pkl", "--psp-samples", "50"]
        with patch.object(sys, "argv", argv):
            args = command.parse()
        params = command.online_policy_params(args)
        self.assertEqual(params["s_learner_erm"], {"model": self.paths["s"], "gamma": 0.1, "first_only": ["risk"]})
        self.assertEqual(params["s_learner"]["threshold"], 2)
        self.assertEqual(params["psp_erm"], {"model": "custom_psp.pkl", "gamma": 0.1, "samples": 50})

    def test_training_cli_writes_erm_model_and_separate_test_report(self):
        from helpers import datasets, training
        import preprocessing
        with tempfile.TemporaryDirectory() as directory:
            argv = ["train_cate_real.py", "--tag", "smoke", "--learner", "s", "--erm-gamma", "0.1"]
            with patch.object(sys, "argv", argv), patch.object(train_cate_real, "PRESCRIPTIVE", directory), \
                    patch.object(training, "MODELS", directory), patch.object(interventions, "choice_points", return_value=[POINT]), \
                    patch.object(datasets, "load_dataset", return_value=(self.log, None, None)), \
                    patch.object(preprocessing, "preprocess", return_value=self.log), \
                    patch.dict(cate_model.PARAMS, FAST_XGB), contextlib.redirect_stdout(io.StringIO()):
                train_cate_real.main()
            path = Path(directory) / "cate_erm_model_s_gamma0.1_durchlaufzeit_smoke.pkl"
            model = CATEERMModel.load(path)
            self.assertIsInstance(model, CATEERMModel)
            self.assertEqual(model.test_cases, training.test_cases(self.log))
            self.assertEqual(model.trained_on["gamma"], 0.1)
            self.assertEqual(model.trained_on["cases"], 1600)
            report = pd.read_csv(Path(directory) / "scripts/results_100k/cate_erm_gamma0.1_smoke/effekte_testfaelle.csv")
            self.assertIn("Effekt Mittel (skalierter exp-Verlust)", report)
            self.assertNotIn("Effekt Mittel (Tage)", report)
            self.assertNotIn("Effekt laut AIPW (Tage)", report)


class ERMComparisonTests(unittest.TestCase):
    def test_empirical_erm_includes_paired_bank_comparison_and_confidence_interval(self):
        rows = []
        for seed in range(3):
            for name, losses in (("bank", np.array([5., 5., 5., 70.])), ("s_learner_erm", np.full(4, 10.))):
                days = losses + seed
                p90, p95 = np.quantile(days, [0.9, 0.95])
                row = {"Policy": name, "seed": seed, "Outcome je Fall": -days.mean(), "_days": days,
                       online.TAIL[0]: p90, online.TAIL[1]: p95, online.TAIL[2]: days[days >= p90].mean()}
                if name.endswith("_erm"):
                    row["ERM γ (1/Tag)"] = 0.1
                rows.append(row)
        table, runs = online.compare(rows)
        metric = "ERM γ=0.1 (Tage)"
        self.assertAlmostEqual(table.loc["s_learner_erm", metric], 11.0)
        expected = 10 - entropic_risk([[5, 5, 5, 70]], 0.1)[0]
        self.assertAlmostEqual(table.loc["s_learner_erm", f"{metric} ggü. Bank"], expected)
        self.assertAlmostEqual(table.loc["s_learner_erm", f"{metric} ggü. Bank 95-%-KI unten"], expected)
        self.assertAlmostEqual(table.loc["s_learner_erm", f"{metric} ggü. Bank 95-%-KI oben"], expected)
        self.assertEqual(table.loc["bank", f"{metric} ggü. Bank"], 0)
        self.assertNotIn("_days", runs)


if __name__ == "__main__":
    unittest.main()

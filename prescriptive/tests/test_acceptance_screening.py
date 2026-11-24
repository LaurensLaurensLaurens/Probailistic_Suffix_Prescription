"""Prospective acceptance targets, grouped risk sets, binary nuisances and Holm."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from prescriptive.helpers.datasets import add_paths
add_paths()
from helpers.datasets import ACTIVITY, CASE, LIFECYCLE
from screening.acceptance import prospective_outcome, holm, contrast_pvalues, annotate_results
from screening.all_points import crossfit_folds, fit_nuisances, aipw_scores, evaluate_point
from screening.candidates import CandidateData


def candidate_data(index, positions, Y=None):
    index = pd.Index(index, name=CASE)
    return CandidateData("point", pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index),
                         pd.Series(np.arange(len(index)) % 2, index=index),
                         pd.Series(np.full(len(index), 123.) if Y is None else Y, index=index),
                         pd.Series(dtype=float), pd.Series(dtype=bool),
                         pool=pd.Series([frozenset([0, 1])] * len(index), index=index),
                         decision_pos=pd.Series(positions, index=index), kind="resource")


class AcceptanceScreeningTests(unittest.TestCase):
    def test_future_event_target_ignores_retrospective_accepted_and_application_status(self):
        data = candidate_data(["past", "future", "never", "same"], [2, 2, 2, 2])
        log = pd.DataFrame({CASE: ["past", "future", "never", "same"],
                            ACTIVITY: ["O_Accepted", "O_Accepted", "A_Accepted", "O_Accepted"],
                            LIFECYCLE: ["complete"] * 4, "_pos": [1, 3, 1, 2], "Accepted": [True] * 4})
        result, groups = prospective_outcome(data, log, "O_Accepted", np.array([0, 1, 2, 3]))
        self.assertEqual(result.X.index.tolist(), ["future", "never"])
        self.assertEqual(result.Y.tolist(), [1., 0.])
        np.testing.assert_array_equal(groups, [1, 2])
        self.assertEqual(result.pool.index.tolist(), result.X.index.tolist())
        self.assertEqual(result.T.index.tolist(), result.X.index.tolist())
        self.assertEqual(data.Y.tolist(), [123.] * 4)  # original cycle-time input is untouched

    def test_only_completed_targets_count(self):
        data = candidate_data(["a", "b"], [0, 0])
        log = pd.DataFrame({CASE: ["a", "b"], ACTIVITY: ["O_Accepted"] * 2,
                            LIFECYCLE: ["start", "complete"], "_pos": [1, 1]})
        result, _ = prospective_outcome(data, log, "O_Accepted")
        self.assertEqual(result.Y.tolist(), [0., 1.])

    def test_previously_returned_offer_is_not_an_at_risk_return_outcome(self):
        data = candidate_data(["a", "b"], [3, 3])
        log = pd.DataFrame({CASE: ["a", "a", "b"], ACTIVITY: ["O_Returned", "O_Accepted", "O_Returned"],
                            LIFECYCLE: ["complete"] * 3, "_pos": [2, 5, 4]})
        returned, _ = prospective_outcome(data, log, "O_Returned")
        accepted, _ = prospective_outcome(data, log, "O_Accepted")
        self.assertEqual(returned.X.index.tolist(), ["b"])
        self.assertEqual(accepted.X.index.tolist(), ["a", "b"])

    def test_binary_nuisance_single_class_is_finite_and_known_propensity_is_preserved(self):
        X = pd.DataFrame({"x": np.arange(30, dtype=float)})
        T = np.arange(30) % 2
        e = np.full((30, 2), .5)
        fold = crossfit_folds(30, 5, 42)
        for target in (0., 1.):
            with self.subTest(target=target):
                fitted_e, mu = fit_nuisances(X, T, np.full(30, target), 2, fold, e,
                                             threads=1, binary_outcome=True)
                np.testing.assert_array_equal(mu, np.full((30, 2), target))
                np.testing.assert_array_equal(fitted_e, e)
                psi = aipw_scores(T, np.full(30, target), fitted_e, mu, randomized=True)
                np.testing.assert_array_equal(psi[:, 1] - psi[:, 0], np.zeros(30))

    def test_binary_logistic_models_recover_a_risk_difference(self):
        rng = np.random.default_rng(27)
        n = 5000
        X = pd.DataFrame({"x": rng.normal(size=n)})
        T = rng.integers(2, size=n)
        Y = (rng.random(n) < .2 + .35 * T).astype(float)
        e, mu = fit_nuisances(X, T, Y, 2, crossfit_folds(n, 5, 42), np.full((n, 2), .5),
                             threads=1, binary_outcome=True)
        self.assertTrue(((0 <= mu) & (mu <= 1)).all())
        psi = aipw_scores(T, Y, e, mu, randomized=True)
        self.assertAlmostEqual(float((psi[:, 1] - psi[:, 0]).mean()), .35, delta=.04)

    def test_holm_handles_missing_values_and_known_example(self):
        adjusted = holm([.01, .04, np.nan, .03, .001])
        np.testing.assert_allclose(adjusted, [.03, .06, np.nan, .06, .004], equal_nan=True)
        self.assertTrue(np.isnan(holm([np.nan]))[0])

    def test_zero_effect_has_p_one_and_rct_uses_number_of_runs(self):
        details = pd.DataFrame({"Effekt (AIPW)": [0., .1], "Std.-Fehler": [0., .05],
                                "Vergleichsfälle": [10000, 10000], "RCT-Läufe im Vergleich": [10, 10]})
        rct = contrast_pvalues(details, True)
        iid = contrast_pvalues(details, False)
        self.assertEqual(rct.iloc[0]["p-Wert"], 1.)
        self.assertGreater(rct.iloc[1]["p-Wert"], iid.iloc[1]["p-Wert"])

    def test_all_options_can_indicate_influence_even_when_shown_option_does_not(self):
        summary = pd.DataFrame([{"Interventionspunkt": "p", "Vergleich": "last vs. first", "Effekt (AIPW)": .01,
                                 "KI unten": -.01, "KI oben": .03}])
        details = pd.DataFrame({"Interventionspunkt": ["p", "p"], "Option": ["middle", "last"],
                                "Referenz": ["first", "first"], "Effekt (AIPW)": [.2, .01],
                                "KI unten": [.1, -.01], "KI oben": [.3, .03], "Std.-Fehler": [.05, .01],
                                "p-Wert": [.0001, .3]})
        overview, full = annotate_results(summary, details)
        self.assertEqual(overview.iloc[0]["Kontraste mit Hinweis nach Holm"], 1)
        self.assertEqual(overview.iloc[0]["p Holm des gezeigten Vergleichs"], .3)
        self.assertEqual(overview.iloc[0]["Effekt (AIPW) (Prozentpunkte)"], 1.)


if __name__ == "__main__":
    unittest.main()

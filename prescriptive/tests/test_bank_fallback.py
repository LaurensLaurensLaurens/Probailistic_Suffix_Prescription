"""Die gelernte Bankwahl ist die Basis; eine Policy überschreibt nur mit konkreter Auswahl."""
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from prescriptive.helpers.datasets import add_paths
add_paths()

import interventions
from resources.case_allocation import Allocation, Task, create_allocation, default_allocation
from resources.BPIC2017.allocation.learned_allocation import LearnedAllocation
from simulation.case_engine_lifecycle_aware import CaseEngine, MAX_WAIT


def bank_model():
    # Der letzte Mitarbeiter A wird hier bewusst nicht gewählt. Eine harte Kontinuitätsregel wäre falsch.
    return {"weights": {("*", "start"): np.array([0., 0., -1000., 0., 0.])},
            "counts": {("A", "W_task"): 10, ("B", "W_task"): 10}}


class BankFallbackTests(unittest.TestCase):
    def engine(self, answer=None, employee_point=True, window=(0, 1)):
        engine = CaseEngine.__new__(CaseEngine)
        case = SimpleNamespace(case_id="Application_0", handlers={"A"}, item_holders={"W_task": "A"}, last_opener="A")
        task = Task(case.case_id, "W_task", "start", 0., "A")
        point = interventions.ChoicePoint("employee", "employee", "W_task", "start")
        engine.cases = {case.case_id: case}
        engine.waiting, engine.busy, engine.held = [task], {}, {}
        engine.choices, engine.forced_assignments = [], 0
        engine._employee = {("W_task", "start"): point} if employee_point else {}
        engine.policy = None if answer is None else SimpleNamespace(choose=Mock(return_value=answer))
        engine.window = window
        engine.allocation = LearnedAllocation(bank_model())
        engine.allocation.bind(engine)
        engine.free_candidates = lambda activity, at: [name for name in ("A", "B") if name not in engine.busy]
        engine.eligible_for = lambda activity: ["A", "B"]
        engine.committed = []
        engine.retries = []
        engine._schedule_retry = engine.retries.append
        def commit(chosen_task, resource, at, new_task):
            engine.committed.append((chosen_task.case_id, resource))
            engine.waiting.remove(chosen_task)
            engine.busy[resource] = (chosen_task.case_id, chosen_task.activity, at)
        engine._commit = commit
        return engine, task

    def test_no_policy_uses_bank_even_when_last_employee_is_free(self):
        engine, task = self.engine()
        self.assertFalse(engine.allocation.keep_last_employee)
        engine._assign(0., new_task=task)
        self.assertEqual(engine.committed, [("Application_0", "B")])

    def test_policy_returning_none_preserves_bank_draw(self):
        engine, task = self.engine()
        policy = SimpleNamespace(choose=Mock(return_value=None))
        engine.policy = policy
        engine._assign(0., new_task=task)
        self.assertEqual(engine.committed, [("Application_0", "B")])
        policy.choose.assert_called_once()

    def test_explicit_policy_choice_overrides_bank(self):
        engine, task = self.engine(answer="A")
        engine._assign(0., new_task=task)
        self.assertEqual(engine.committed, [("Application_0", "A")])

    def test_bank_is_used_outside_policy_points_and_evaluation_window(self):
        for employee_point, window in ((False, (0, 1)), (True, (1, 2))):
            with self.subTest(point=employee_point, window=window):
                engine, task = self.engine(answer="A", employee_point=employee_point, window=window)
                engine._assign(0., new_task=task)
                self.assertEqual(engine.committed, [("Application_0", "B")])
                engine.policy.choose.assert_not_called()

    def test_busy_workers_are_excluded_and_no_candidate_means_waiting(self):
        engine, task = self.engine()
        engine.busy["B"] = ("other", "W_task", 0.)
        engine._assign(0., new_task=task)
        self.assertEqual(engine.committed, [("Application_0", "A")])
        engine, task = self.engine()
        engine.free_candidates = lambda activity, at: []
        engine._assign(0., new_task=task)
        self.assertFalse(engine.committed)
        self.assertEqual(engine.waiting, [task])
        self.assertEqual(engine.retries, [0.])

    def test_policy_wait_still_overrides_bank(self):
        engine, task = self.engine(answer=interventions.WAIT)
        engine._assign(0., new_task=task)
        self.assertEqual(engine.waiting, [task])
        self.assertFalse(engine.committed)
        self.assertEqual(engine.held[task], {"A", "B"})

    def test_maximum_wait_fallback_uses_bank_instead_of_uniform_random(self):
        engine, task = self.engine()
        engine.free_candidates = lambda activity, at: []  # niemand im Dienst
        engine._assign(MAX_WAIT + 1.)
        self.assertEqual(engine.committed, [("Application_0", "B")])
        self.assertEqual(engine.forced_assignments, 1)

    def test_catalog_default_and_explicit_experiment(self):
        self.assertEqual(default_allocation("bpic17"), "learned")
        self.assertEqual(default_allocation("bpic19"), "random")
        with patch.object(interventions, "catalog", return_value={"bpic17": {}, "bpic19": {}}):
            self.assertEqual(interventions.run_allocation("bpic17"), "learned")
            self.assertEqual(interventions.run_allocation("bpic19"), "random")
        with patch.object(interventions, "catalog", return_value={"bpic17": {"lauf": {"zuteilung": "random"}}}):
            self.assertEqual(interventions.run_allocation("bpic17"), "random")

    def test_factory_loads_saved_bank_model_without_a_new_policy_or_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.pkl"
            path.write_bytes(pickle.dumps(bank_model()))
            with patch("resources.datasets.resource_spec", return_value={"learned_allocation": str(path)}):
                allocation = create_allocation(None, None, "bpic17")
                self.assertIsInstance(allocation, LearnedAllocation)
                self.assertFalse(allocation.keep_last_employee)
            with patch("resources.datasets.resource_spec", return_value={"learned_allocation": str(path.with_name('missing.pkl'))}):
                with self.assertRaisesRegex(ValueError, "Bank-Zuteilung fehlt"):
                    create_allocation(None, None, "bpic17")

    def test_direct_engine_constructor_uses_bank_default(self):
        allocation = Mock()
        model = SimpleNamespace(encoder=SimpleNamespace(tokens=["<BOS>"], token_to_id={"<BOS>": 0}, work_item_prefix="W_"))
        resources = SimpleNamespace(system_users=set(), system_user=None)
        with patch("simulation.case_engine_lifecycle_aware.create_allocation", return_value=allocation) as create:
            engine = CaseEngine(model, resources, None, [])
        create.assert_called_once_with(None, None, "bpic17")
        self.assertIs(engine.allocation, allocation)
        self.assertIsNone(engine.policy)


if __name__ == "__main__":
    unittest.main()

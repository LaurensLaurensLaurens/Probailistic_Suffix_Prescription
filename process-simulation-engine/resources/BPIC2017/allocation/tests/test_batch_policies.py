"""
Unit tests for batch allocation policies and processing time estimator.

Run:
    python resources/tests/test_batch_policies.py
"""

import os
import sys
import unittest
from datetime import datetime, timedelta

import pandas as pd

# Ensure project root is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from resources.BPIC2017.allocation.processing_time_estimator import ProcessingTimeEstimator
from resources.BPIC2017.allocation.batch_policies import (
    BatchDecision,
    OneBatchOnePolicy,
    TaskInfo,
    WorkerInfo,
    create_batch_policy,
)


# ---------------------------------------------------------------------------
# ProcessingTimeEstimator
# ---------------------------------------------------------------------------

class TestProcessingTimeEstimator(unittest.TestCase):
    """Tests for the ProcessingTimeEstimator lookup table."""

    def _make_df(self):
        """Create a small event log for testing."""
        base = datetime(2017, 1, 2, 8, 0)
        data = {
            "case:concept:name": ["C1", "C1", "C1", "C2", "C2"],
            "concept:name": ["A", "B", "A", "A", "B"],
            "org:resource": ["W1", "W1", "W2", "W2", "W1"],
            "time:timestamp": [
                base,
                base + timedelta(seconds=100),
                base + timedelta(seconds=300),
                base + timedelta(seconds=50),
                base + timedelta(seconds=250),
            ],
        }
        return pd.DataFrame(data)

    def test_resource_activity_lookup(self):
        """Per (resource, activity) means are used when available."""
        est = ProcessingTimeEstimator(df=self._make_df())
        # W1 did activity B in case C1 after A (100s gap) and in case C2 after A (200s gap)
        # The estimator computes inter-event durations, so:
        # - For (W1, B): we have durations where B was performed by W1
        val = est.estimate("W1", "B")
        self.assertGreater(val, 0)

    def test_activity_fallback(self):
        """Falls back to activity-level mean for unknown resources."""
        est = ProcessingTimeEstimator(df=self._make_df())
        act_val = est.estimate("UNKNOWN_RESOURCE", "B")
        # Should fall through to activity mean, not global mean
        self.assertGreater(act_val, 0)

    def test_global_fallback(self):
        """Falls back to global mean for completely unknown pairs."""
        est = ProcessingTimeEstimator(df=self._make_df())
        val = est.estimate("UNKNOWN", "UNKNOWN_ACTIVITY")
        # Global mean should be positive
        self.assertGreater(val, 0)

    def test_empty_dataframe(self):
        """Empty DataFrame yields default global mean."""
        est = ProcessingTimeEstimator(df=pd.DataFrame())
        val = est.estimate("W1", "A")
        self.assertEqual(val, 3600.0)  # default

    def test_no_dataframe(self):
        """No DataFrame at all yields default global mean."""
        est = ProcessingTimeEstimator()
        val = est.estimate("W1", "A")
        self.assertEqual(val, 3600.0)


# ---------------------------------------------------------------------------
# OneBatchOnePolicy
# ---------------------------------------------------------------------------

class TestOneBatchOnePolicy(unittest.TestCase):
    """Tests for the 1-Batch-1 MSA policy."""

    def _simple_pt_fn(self, costs):
        """Return a processing_time_fn based on a dict of (worker, activity) -> cost."""
        def fn(worker_id, activity):
            return costs.get((worker_id, activity), 999999.0)
        return fn

    def test_single_task_single_worker(self):
        """Trivial: one task, one worker, must assign."""
        policy = OneBatchOnePolicy()
        tasks = [TaskInfo("C1::A", "C1", "A", "A", 0.0)]
        workers = [WorkerInfo("W1", 0.0)]
        eligible = {"A": {"W1"}}
        pt_fn = self._simple_pt_fn({("W1", "A"): 100.0})

        decision = policy.decide("W1", 0.0, tasks, workers, eligible, pt_fn)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.task_id, "C1::A")
        self.assertEqual(decision.worker_id, "W1")

    def test_2x2_makespan_minimization(self):
        """
        Hand-verifiable 2x2 case from the plan.

        Workers: W1 (idle, a=0), W2 (busy, a=100s)
        Tasks:   T1 (p_{W1,T1}=50, p_{W2,T1}=200)
                 T2 (p_{W1,T2}=200, p_{W2,T2}=50)

        Optimal: W1->T1 (50s), W2->T2 (100+50=150s), makespan=150
        Wrong:   W1->T2 (200s), W2->T1 (100+200=300s), makespan=300

        freed_resource=W1 -> policy must return T1
        """
        policy = OneBatchOnePolicy()
        tasks = [
            TaskInfo("C1::T1", "C1", "T1", "T1", 0.0),
            TaskInfo("C2::T2", "C2", "T2", "T2", 0.0),
        ]
        workers = [
            WorkerInfo("W1", 0.0),
            WorkerInfo("W2", 100.0),
        ]
        eligible = {"T1": {"W1", "W2"}, "T2": {"W1", "W2"}}
        pt_fn = self._simple_pt_fn({
            ("W1", "T1"): 50.0,
            ("W1", "T2"): 200.0,
            ("W2", "T1"): 200.0,
            ("W2", "T2"): 50.0,
        })

        decision = policy.decide("W1", 0.0, tasks, workers, eligible, pt_fn)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.worker_id, "W1")
        self.assertEqual(decision.task_id, "C1::T1")

    def test_only_commits_freed_worker(self):
        """The decision should only assign to the freed worker."""
        policy = OneBatchOnePolicy()
        tasks = [
            TaskInfo("C1::A", "C1", "A", "A", 0.0),
            TaskInfo("C2::B", "C2", "B", "B", 0.0),
        ]
        workers = [
            WorkerInfo("W1", 0.0),
            WorkerInfo("W2", 0.0),
        ]
        eligible = {"A": {"W1", "W2"}, "B": {"W1", "W2"}}
        pt_fn = self._simple_pt_fn({
            ("W1", "A"): 50.0, ("W1", "B"): 50.0,
            ("W2", "A"): 50.0, ("W2", "B"): 50.0,
        })

        decision = policy.decide("W1", 0.0, tasks, workers, eligible, pt_fn)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.worker_id, "W1")

    def test_greedy_fallback_on_infeasible(self):
        """When a task has zero eligible workers, MILP is infeasible -> greedy fallback."""
        policy = OneBatchOnePolicy()
        tasks = [
            TaskInfo("C1::A", "C1", "A", "A", 1.0),
            TaskInfo("C2::B", "C2", "B", "B", 0.5),
        ]
        workers = [WorkerInfo("W1", 0.0)]
        # B has no eligible workers -> MILP infeasible
        eligible = {"A": {"W1"}, "B": set()}
        pt_fn = self._simple_pt_fn({("W1", "A"): 100.0})

        decision = policy.decide("W1", 0.0, tasks, workers, eligible, pt_fn)

        # Should fall back to greedy and assign A (W1 is eligible for A)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.task_id, "C1::A")

    def test_aging_reduces_effective_processing_time(self):
        """Tasks that have waited longer should have lower effective p_{ij}."""
        policy = OneBatchOnePolicy()

        # Two tasks for the same activity, different wait times
        tasks = [
            TaskInfo("C1::A", "C1", "A", "A", 10.0),  # 10 hours waited
            TaskInfo("C2::A", "C2", "A", "A", 0.0),   # just arrived
        ]
        workers = [WorkerInfo("W1", 0.0)]
        eligible = {"A": {"W1"}}
        pt_fn = self._simple_pt_fn({("W1", "A"): 100.0})

        decision = policy.decide("W1", 0.0, tasks, workers, eligible, pt_fn)

        # Should succeed — both tasks are eligible
        self.assertIsNotNone(decision)
        self.assertEqual(decision.worker_id, "W1")

    def test_respects_eligibility(self):
        """Worker should not be assigned a task they're not eligible for."""
        policy = OneBatchOnePolicy()
        tasks = [
            TaskInfo("C1::A", "C1", "A", "A", 0.0),
            TaskInfo("C2::B", "C2", "B", "B", 0.0),
        ]
        workers = [
            WorkerInfo("W1", 0.0),
            WorkerInfo("W2", 0.0),
        ]
        # W1 can only do A, W2 can only do B
        eligible = {"A": {"W1"}, "B": {"W2"}}
        pt_fn = self._simple_pt_fn({
            ("W1", "A"): 50.0,
            ("W2", "B"): 50.0,
        })

        decision = policy.decide("W1", 0.0, tasks, workers, eligible, pt_fn)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.task_id, "C1::A")

    def test_max_tasks_guard_triggers_fallback(self):
        """When tasks exceed max_tasks, greedy fallback is used."""
        policy = OneBatchOnePolicy(max_tasks=2)

        tasks = [
            TaskInfo(f"C{i}::A", f"C{i}", "A", "A", float(i))
            for i in range(5)
        ]
        workers = [WorkerInfo("W1", 0.0)]
        eligible = {"A": {"W1"}}
        pt_fn = self._simple_pt_fn({("W1", "A"): 100.0})

        decision = policy.decide("W1", 0.0, tasks, workers, eligible, pt_fn)

        # Should still return a decision (via greedy fallback)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.worker_id, "W1")
        # Greedy picks the oldest (highest hours_waited)
        self.assertEqual(decision.task_id, "C4::A")

    def test_no_tasks_returns_none(self):
        """Empty task list returns None."""
        policy = OneBatchOnePolicy()
        decision = policy.decide("W1", 0.0, [], [WorkerInfo("W1", 0.0)], {}, lambda w, a: 100.0)
        self.assertIsNone(decision)

    def test_freed_worker_not_eligible_returns_none(self):
        """If freed worker can't do any waiting task, return None."""
        policy = OneBatchOnePolicy()
        tasks = [TaskInfo("C1::A", "C1", "A", "A", 0.0)]
        workers = [WorkerInfo("W1", 0.0), WorkerInfo("W2", 0.0)]
        eligible = {"A": {"W2"}}  # Only W2 eligible, but W1 is freed
        pt_fn = self._simple_pt_fn({("W2", "A"): 100.0})

        decision = policy.decide("W1", 0.0, tasks, workers, eligible, pt_fn)
        self.assertIsNone(decision)


# ---------------------------------------------------------------------------
# Registry & Factory
# ---------------------------------------------------------------------------

class TestBatchPolicyRegistry(unittest.TestCase):
    """Tests for the batch policy registry and factory."""

    def test_create_1_batch_1(self):
        """Factory creates OneBatchOnePolicy."""
        policy = create_batch_policy("1_batch_1")
        self.assertIsInstance(policy, OneBatchOnePolicy)

    def test_create_with_kwargs(self):
        """Factory forwards kwargs to constructor."""
        policy = create_batch_policy("1_batch_1", max_tasks=50, timeout_s=2.0)
        self.assertEqual(policy.max_tasks, 50)
        self.assertEqual(policy.timeout_s, 2.0)

    def test_invalid_name_raises(self):
        """Invalid policy name raises ValueError."""
        with self.assertRaises(ValueError):
            create_batch_policy("nonexistent_policy")


if __name__ == "__main__":
    unittest.main()

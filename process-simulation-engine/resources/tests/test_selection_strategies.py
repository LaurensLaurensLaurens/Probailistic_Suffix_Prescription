"""
Tests for resource selection strategies (R-RMA, R-RRA, R-SHQ).

Run: python resources/tests/test_selection_strategies.py
"""
import unittest
import sys
import os
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from resources.selection_strategies import (
    RandomStrategy,
    RoundRobinStrategy,
    ShortestQueueStrategy,
    create_strategy,
    STRATEGY_REGISTRY,
)


class TestRandomStrategy(unittest.TestCase):
    """Tests for R-RMA: Random Allocation."""

    def test_selects_from_available(self):
        strategy = RandomStrategy()
        available = ["R1", "R2", "R3"]
        for _ in range(50):
            result = strategy.select(available, "A_Create")
            self.assertIn(result, available)

    def test_single_resource(self):
        strategy = RandomStrategy()
        result = strategy.select(["R1"], "A_Create")
        self.assertEqual(result, "R1")

    def test_distribution_is_roughly_uniform(self):
        random.seed(42)
        strategy = RandomStrategy()
        available = ["R1", "R2", "R3"]
        counts = {r: 0 for r in available}
        n = 3000
        for _ in range(n):
            selected = strategy.select(available, "A")
            counts[selected] += 1
        # Each should get roughly 1000 +/- 15%
        for r in available:
            self.assertGreater(counts[r], n / 3 * 0.8)
            self.assertLess(counts[r], n / 3 * 1.2)

    def test_notify_assignment_is_noop(self):
        strategy = RandomStrategy()
        # Should not raise
        strategy.notify_assignment("R1", "A")

    def test_reset_is_noop(self):
        strategy = RandomStrategy()
        strategy.reset()


class TestRoundRobinStrategy(unittest.TestCase):
    """Tests for R-RRA: Round Robin Allocation."""

    def test_cycles_through_sorted_resources(self):
        strategy = RoundRobinStrategy()
        available = ["R3", "R1", "R2"]  # unsorted input
        results = [strategy.select(available, "A") for _ in range(6)]
        # Sorted order: R1, R2, R3 -> cycles twice
        self.assertEqual(results, ["R1", "R2", "R3", "R1", "R2", "R3"])

    def test_per_activity_independent_state(self):
        strategy = RoundRobinStrategy()
        # Activity A and B should each start from index 0
        res_a1 = strategy.select(["R1", "R2"], "Activity_A")
        res_b1 = strategy.select(["R1", "R2"], "Activity_B")
        self.assertEqual(res_a1, "R1")
        self.assertEqual(res_b1, "R1")
        # Advance A, B stays independent
        res_a2 = strategy.select(["R1", "R2"], "Activity_A")
        res_b2 = strategy.select(["R1", "R2"], "Activity_B")
        self.assertEqual(res_a2, "R2")
        self.assertEqual(res_b2, "R2")

    def test_wraps_around(self):
        strategy = RoundRobinStrategy()
        available = ["R1", "R2"]
        results = [strategy.select(available, "A") for _ in range(5)]
        self.assertEqual(results, ["R1", "R2", "R1", "R2", "R1"])

    def test_single_resource_always_selected(self):
        strategy = RoundRobinStrategy()
        for _ in range(5):
            self.assertEqual(strategy.select(["R1"], "A"), "R1")

    def test_reset_clears_state(self):
        strategy = RoundRobinStrategy()
        strategy.select(["R1", "R2"], "A")  # R1 (index 0)
        strategy.reset()
        result = strategy.select(["R1", "R2"], "A")
        self.assertEqual(result, "R1")  # Starts over from index 0

    def test_varying_available_set(self):
        """When available set changes, cycling adapts via modular arithmetic."""
        strategy = RoundRobinStrategy()
        # Full set: R1, R2, R3
        r1 = strategy.select(["R1", "R2", "R3"], "A")  # index 0 -> R1
        self.assertEqual(r1, "R1")
        # R2 unavailable: sorted = [R1, R3]
        r2 = strategy.select(["R1", "R3"], "A")  # (0+1)%2=1 -> R3
        self.assertEqual(r2, "R3")
        # Full set back: sorted = [R1, R2, R3]
        r3 = strategy.select(["R1", "R2", "R3"], "A")  # (1+1)%3=2 -> R3
        self.assertEqual(r3, "R3")


class TestShortestQueueStrategy(unittest.TestCase):
    """Tests for R-SHQ: Shortest Queue / Least Loaded Allocation.

    R-SHQ tracks cumulative assignments per resource and selects the
    least-loaded resource, with R-RMA tiebreaker.  notify_release is
    a no-op (cumulative, not instantaneous).
    """

    def test_prefers_least_loaded(self):
        """Resource with fewer cumulative assignments is preferred."""
        strategy = ShortestQueueStrategy()
        strategy.notify_assignment("R1", "A")
        strategy.notify_assignment("R1", "A")
        strategy.notify_assignment("R2", "A")
        # R3 has 0 cumulative assignments -> should be selected
        result = strategy.select(["R1", "R2", "R3"], "A")
        self.assertEqual(result, "R3")

    def test_release_is_noop(self):
        """notify_release does not decrement cumulative assignments."""
        strategy = ShortestQueueStrategy()
        strategy.notify_assignment("R1", "A")
        strategy.notify_assignment("R1", "B")
        self.assertEqual(strategy._cumulative_assignments["R1"], 2)
        strategy.notify_release("R1")
        self.assertEqual(strategy._cumulative_assignments["R1"], 2)

    def test_cumulative_tracks_total_load(self):
        """After assignment + release, cumulative count persists."""
        strategy = ShortestQueueStrategy()
        strategy.notify_assignment("R1", "A")
        strategy.notify_release("R1")
        # R1 has 1 cumulative, R2 has 0 -> R2 preferred
        result = strategy.select(["R1", "R2"], "A")
        self.assertEqual(result, "R2")

    def test_tied_candidates_random_tiebreak(self):
        """When all candidates have equal cumulative counts, select is random."""
        random.seed(42)
        strategy = ShortestQueueStrategy()
        available = ["R1", "R2", "R3"]
        # All at 0 cumulative assignments — tied, should see randomness
        results = set()
        for _ in range(100):
            results.add(strategy.select(available, "A"))
        self.assertGreater(len(results), 1)

    def test_single_resource(self):
        strategy = ShortestQueueStrategy()
        result = strategy.select(["R1"], "A")
        self.assertEqual(result, "R1")

    def test_reset_clears_state(self):
        strategy = ShortestQueueStrategy()
        strategy.notify_assignment("R1", "A")
        strategy.notify_assignment("R1", "A")
        strategy.reset()
        self.assertEqual(len(strategy._cumulative_assignments), 0)

    def test_release_on_unknown_resource_is_noop(self):
        """Releasing without prior assignment is a no-op."""
        strategy = ShortestQueueStrategy()
        strategy.notify_release("R1")  # Should not raise
        self.assertEqual(strategy._cumulative_assignments.get("R1", 0), 0)


class TestCreateStrategy(unittest.TestCase):
    """Tests for the factory function and registry."""

    def test_create_random(self):
        s = create_strategy("random")
        self.assertIsInstance(s, RandomStrategy)

    def test_create_round_robin(self):
        s = create_strategy("round_robin")
        self.assertIsInstance(s, RoundRobinStrategy)

    def test_create_shortest_queue(self):
        s = create_strategy("shortest_queue")
        self.assertIsInstance(s, ShortestQueueStrategy)

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            create_strategy("nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("Valid options", str(ctx.exception))

    def test_registry_has_three_entries(self):
        self.assertEqual(len(STRATEGY_REGISTRY), 3)


if __name__ == "__main__":
    unittest.main()

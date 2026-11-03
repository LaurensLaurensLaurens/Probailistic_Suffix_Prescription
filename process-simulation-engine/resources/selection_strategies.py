"""
Resource selection strategies for the simulation engine.

Push-pattern heuristics for selecting a resource from the pool of candidates that passed the
filtering (permissions -> availability -> busy state), for every dataset: in the engine
(resources/case_allocation.py, the allocations random, round_robin, shortest_queue) and on the pool of a
dataset (resources/datasets.py, Resources.select).

- R-RMA (Random Allocation, Pattern 15): uniform random selection (the default of the base allocation)
- R-RRA (Round Robin Allocation, Pattern 16): cyclic per-activity selection
- R-SHQ (Shortest Queue Allocation, Pattern 17): least-assigned resource, with R-RMA tiebreaker
"""

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List


class ResourceSelectionStrategy(ABC):
    """Abstract base class for resource selection strategies."""

    @abstractmethod
    def select(self, available_resources: List[str], activity: str) -> str:
        """
        Select one resource from the available candidates.

        Args:
            available_resources: Non-empty list of resources that passed
                all 3 tiers of filtering (eligible, on-duty, not busy).
            activity: The activity being assigned (used as allocation group key).

        Returns:
            The selected resource identifier.
        """
        ...

    def notify_assignment(self, resource: str, activity: str) -> None:
        """Called after a resource is assigned. Override in stateful strategies."""
        pass

    def notify_release(self, resource: str) -> None:
        """Called when a resource is released. Override in stateful strategies."""
        pass

    def reset(self) -> None:
        """Reset internal state for a new simulation run."""
        pass


class RandomStrategy(ResourceSelectionStrategy):
    """
    R-RMA: Random Allocation (Pattern 15).

    Selects a resource uniformly at random from the available pool.
    """

    def select(self, available_resources: List[str], activity: str) -> str:
        return random.choice(available_resources)


class RoundRobinStrategy(ResourceSelectionStrategy):
    """
    R-RRA: Round Robin Allocation (Pattern 16).

    Cycles through resources on a per-activity basis. Resources are sorted
    alphabetically to ensure stable ordering across calls where the available
    set may vary due to availability/busy fluctuations.

    State: maintains LastAssignedIndex per activity (allocation group).
    """

    def __init__(self):
        # activity -> last assigned index into sorted available list
        self._last_index: Dict[str, int] = defaultdict(lambda: -1)

    def select(self, available_resources: List[str], activity: str) -> str:
        sorted_resources = sorted(available_resources)
        last = self._last_index[activity]
        next_idx = (last + 1) % len(sorted_resources)
        self._last_index[activity] = next_idx
        return sorted_resources[next_idx]

    def reset(self) -> None:
        self._last_index.clear()


class ShortestQueueStrategy(ResourceSelectionStrategy):
    """
    R-SHQ: Shortest Queue Allocation (Pattern 17).

    Adapted for push-model DES: since the 3-tier filtering pipeline
    (permissions → availability → busy state) guarantees all candidates
    are idle, instantaneous queue depth is always zero.  Instead, this
    strategy tracks **cumulative** assignments per resource and selects
    the least-loaded resource, with R-RMA tiebreaker.

    This provides meaningful load balancing: resources who have received
    fewer assignments overall are preferred.  The concern about steering
    work toward low-availability resources is mitigated by Tier 2
    (availability model), which already excludes off-duty resources.
    """

    def __init__(self):
        self._cumulative_assignments: Dict[str, int] = defaultdict(int)

    def select(self, available_resources: List[str], activity: str) -> str:
        min_count = min(self._cumulative_assignments.get(r, 0) for r in available_resources)
        candidates = [r for r in available_resources
                      if self._cumulative_assignments.get(r, 0) == min_count]
        return random.choice(candidates)

    def notify_assignment(self, resource: str, activity: str) -> None:
        self._cumulative_assignments[resource] += 1

    def notify_release(self, resource: str) -> None:
        pass  # Cumulative — no decrement

    def reset(self) -> None:
        self._cumulative_assignments.clear()


# ---------------------------------------------------------------------------
# Registry & Factory
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, type] = {
    "random": RandomStrategy,
    "round_robin": RoundRobinStrategy,
    "shortest_queue": ShortestQueueStrategy,
}


def create_strategy(name: str) -> ResourceSelectionStrategy:
    """
    Create a selection strategy by name.

    Args:
        name: One of "random", "round_robin", "shortest_queue".

    Raises:
        ValueError: If the name is not in the registry.
    """
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        valid = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown resource selection strategy: '{name}'. "
            f"Valid options: {valid}"
        )
    return cls()

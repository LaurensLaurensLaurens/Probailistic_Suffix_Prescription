"""
Batch allocation policies for the simulation engine.

Implements the 1-Batch-1 MSA (Minimum Service-time Allocation) policy
from Zeng et al. (2005).  Instead of greedily assigning the first
eligible FIFO-head task when a resource becomes free, the policy solves
a global MILP over ALL waiting tasks and ALL eligible workers to
minimize the maximum worker completion time (makespan).

Only the assignment for the just-freed worker is committed; all other
tasks stay queued for re-optimization at the next trigger.

Solver: scipy.optimize.milp (HiGHS backend, bundled with scipy).
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class TaskInfo:
    """A waiting task as seen by the batch policy."""
    task_id: str  # f"{case_id}::{allocation_activity}"
    case_id: str
    activity: str  # original label (for logging)
    allocation_activity: str  # normalized label (for permissions)
    hours_waited: float  # hours since arrival


@dataclass
class WorkerInfo:
    """A worker (resource) as seen by the batch policy."""
    worker_id: str
    remaining_busy_seconds: float  # a_i: 0 if idle


@dataclass
class BatchDecision:
    """The output of a batch policy: which task to assign to the freed worker."""
    task_id: str  # matches TaskInfo.task_id
    worker_id: str  # should equal the freed_resource


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------

class BatchAllocationPolicy(ABC):
    """Interface for batch (holistic) allocation policies."""

    @abstractmethod
    def decide(
        self,
        freed_resource: str,
        current_time_s: float,
        tasks: List[TaskInfo],
        workers: List[WorkerInfo],
        eligible_map: Dict[str, Set[str]],
        processing_time_fn: Callable[[str, str], float],
    ) -> Optional[BatchDecision]:
        """
        Decide which task to assign to *freed_resource*.

        Args:
            freed_resource: The resource that just became free.
            current_time_s: Current simulation time as epoch seconds (unused
                by the optimizer but available for logging).
            tasks: All currently waiting tasks.
            workers: All eligible workers (including busy ones).
            eligible_map: allocation_activity -> set of eligible worker IDs.
            processing_time_fn: (worker_id, activity) -> seconds estimate.

        Returns:
            A BatchDecision or None if no feasible assignment exists.
        """
        ...

    def reset(self) -> None:
        """Reset internal state for a new simulation run."""
        pass


# ---------------------------------------------------------------------------
# 1-Batch-1 Policy (Zeng et al. 2005)
# ---------------------------------------------------------------------------

class OneBatchOnePolicy(BatchAllocationPolicy):
    """
    1-Batch-1 MSA: solve a makespan MILP over all waiting tasks and all
    eligible workers.  Only the assignment for the freed worker is committed.

    Scalability guards:
        - max_tasks: if the number of waiting tasks exceeds this, fall back
          to a greedy (FIFO) assignment.
        - timeout_s: HiGHS solver time limit.
    """

    def __init__(self, max_tasks: int = 200, timeout_s: float = 2.0):
        self.max_tasks = max_tasks
        self.timeout_s = timeout_s

        # --- Diagnostics ---
        self._diag_trigger_count = 0
        self._diag_early_exit_empty = 0
        self._diag_early_exit_ineligible = 0
        self._diag_milp_success_count = 0
        self._diag_greedy_max_tasks_count = 0
        self._diag_greedy_neighborhood_too_large = 0
        self._diag_greedy_solver_fail_count = 0

        self._diag_queue_sizes: List[int] = []
        self._diag_worker_sizes: List[int] = []
        self._diag_problem_dims: List[tuple] = []  # (n_w, n_t, n_w*n_t)
        self._diag_milp_times: List[float] = []
        self._diag_growth_log: List[tuple] = []  # (trigger#, queue_size, worker_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        freed_resource: str,
        current_time_s: float,
        tasks: List[TaskInfo],
        workers: List[WorkerInfo],
        eligible_map: Dict[str, Set[str]],
        processing_time_fn: Callable[[str, str], float],
    ) -> Optional[BatchDecision]:

        self._diag_trigger_count += 1

        if not tasks:
            self._diag_early_exit_empty += 1
            return None

        # Ensure freed_resource is in the worker list
        freed_ids = {w.worker_id for w in workers}
        if freed_resource not in freed_ids:
            workers = list(workers) + [WorkerInfo(freed_resource, 0.0)]

        # Check if freed worker is eligible for ANY waiting task
        freed_eligible = False
        for t in tasks:
            if freed_resource in eligible_map.get(t.allocation_activity, set()):
                freed_eligible = True
                break
        if not freed_eligible:
            self._diag_early_exit_ineligible += 1
            return None

        # Record sizes for non-trivial calls
        n_t = len(tasks)
        n_w = len(workers)
        self._diag_queue_sizes.append(n_t)
        self._diag_worker_sizes.append(n_w)
        self._diag_problem_dims.append((n_w, n_t, n_w * n_t))

        # Log growth every 1000th trigger
        if self._diag_trigger_count % 1000 == 0:
            self._diag_growth_log.append(
                (self._diag_trigger_count, n_t, n_w)
            )

        # Guard: fall back if problem too large
        if len(tasks) > self.max_tasks:
            logger.info(
                "1-Batch-1: %d tasks exceed max_tasks=%d, using greedy fallback",
                len(tasks), self.max_tasks,
            )
            self._diag_greedy_max_tasks_count += 1
            return self._greedy_fallback(freed_resource, tasks, eligible_map, processing_time_fn)

        # Build and solve MILP
        decision = self._solve_msa(
            freed_resource, tasks, workers, eligible_map, processing_time_fn,
        )

        if decision is not None:
            self._diag_milp_success_count += 1
            return decision

        # Solver failed — greedy fallback
        logger.debug("1-Batch-1: MILP infeasible or failed, using greedy fallback")
        self._diag_greedy_solver_fail_count += 1
        return self._greedy_fallback(freed_resource, tasks, eligible_map, processing_time_fn)

    # ------------------------------------------------------------------
    # MILP Solver
    # ------------------------------------------------------------------

    def _solve_msa(
        self,
        freed_resource: str,
        tasks: List[TaskInfo],
        workers: List[WorkerInfo],
        eligible_map: Dict[str, Set[str]],
        processing_time_fn: Callable[[str, str], float],
    ) -> Optional[BatchDecision]:
        """
        Formulate and solve the MSA (Minimum Service-time Allocation) MILP.

        Variables
        ---------
        x_{ij} : binary, 1 if worker i is assigned task j   (n_w * n_t vars)
        z      : continuous, makespan                         (1 var)

        Objective:  minimize z

        Constraints
        -----------
        (1) Each task assigned to exactly one worker:
            sum_i x_{ij} = 1   for all j

        (2) Makespan >= each worker's completion:
            a_i + sum_j (x_{ij} * p'_{ij}) <= z   for all i
            (rewritten as: sum_j(p'_{ij} * x_{ij}) - z <= -a_i)

        Ineligibility is enforced by setting upper bound x_{ij} = 0.
        Aging discount:  p'_{ij} = p_{ij} * 0.95^(hours_waited_j)
        """
        n_w = len(workers)
        n_t = len(tasks)
        n_x = n_w * n_t  # binary variables
        n_vars = n_x + 1  # + z

        worker_idx = {w.worker_id: i for i, w in enumerate(workers)}
        task_idx = {t.task_id: j for j, t in enumerate(tasks)}

        # Build cost matrix p'_{ij} and eligibility mask
        p = np.zeros((n_w, n_t))
        eligible_mask = np.zeros((n_w, n_t), dtype=bool)

        for j, t in enumerate(tasks):
            elig_set = eligible_map.get(t.allocation_activity, set())
            aging = 0.95 ** t.hours_waited
            for i, w in enumerate(workers):
                if w.worker_id in elig_set:
                    eligible_mask[i, j] = True
                    raw_p = processing_time_fn(w.worker_id, t.allocation_activity)
                    p[i, j] = raw_p * aging

        # Check feasibility: every task must have at least one eligible worker
        for j in range(n_t):
            if not eligible_mask[:, j].any():
                # Infeasible — cannot assign this task
                logger.debug(
                    "1-Batch-1: task %s has no eligible worker, MILP infeasible",
                    tasks[j].task_id,
                )
                return None

        # ---- Objective: minimize z ----
        # c = [0, 0, ..., 0, 1]  (only z has coefficient 1)
        c = np.zeros(n_vars)
        c[-1] = 1.0

        # ---- Variable bounds ----
        # x_{ij} in {0, 1}, but ineligible pairs forced to 0
        # z >= 0
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        ub[-1] = np.inf  # z is unbounded above

        # Force ineligible pairs to 0
        for i in range(n_w):
            for j in range(n_t):
                if not eligible_mask[i, j]:
                    ub[i * n_t + j] = 0.0

        # ---- Integrality ----
        # x_{ij} = 1 (integer), z = 0 (continuous)
        integrality = np.ones(n_vars, dtype=int)
        integrality[-1] = 0

        # ---- Constraint (1): each task assigned to exactly one worker ----
        # sum_i x_{ij} = 1  for each j
        # n_t equality constraints
        A_task = lil_matrix((n_t, n_vars))
        for j in range(n_t):
            for i in range(n_w):
                A_task[j, i * n_t + j] = 1.0

        task_lb = np.ones(n_t)
        task_ub = np.ones(n_t)

        # ---- Constraint (2): makespan >= worker completion ----
        # sum_j(p'_{ij} * x_{ij}) - z <= -a_i   for each i
        # n_w inequality constraints (upper bound)
        A_make = lil_matrix((n_w, n_vars))
        a_vals = np.array([w.remaining_busy_seconds for w in workers])

        for i in range(n_w):
            for j in range(n_t):
                A_make[i, i * n_t + j] = p[i, j]
            A_make[i, -1] = -1.0  # -z

        make_lb = np.full(n_w, -np.inf)
        make_ub = -a_vals  # sum_j(p * x) - z <= -a_i

        # Stack constraints
        from scipy.sparse import vstack as sp_vstack
        A = sp_vstack([A_task, A_make], format="csc")
        con_lb = np.concatenate([task_lb, make_lb])
        con_ub = np.concatenate([task_ub, make_ub])

        constraints = LinearConstraint(A, con_lb, con_ub)

        # ---- Solve ----
        options = {"time_limit": self.timeout_s, "presolve": True}
        t0 = time.perf_counter()
        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb=lb, ub=ub),
            options=options,
        )
        self._diag_milp_times.append(time.perf_counter() - t0)

        if not result.success:
            logger.debug("1-Batch-1 MILP failed: %s", result.message)
            return None

        x = result.x

        # Extract freed worker's assignment
        freed_idx = worker_idx.get(freed_resource)
        if freed_idx is None:
            return None

        best_j = None
        best_val = -1.0
        for j in range(n_t):
            val = x[freed_idx * n_t + j]
            if val > best_val:
                best_val = val
                best_j = j

        if best_j is None or best_val < 0.5:
            # Freed worker was not assigned any task (rare but possible)
            return None

        return BatchDecision(
            task_id=tasks[best_j].task_id,
            worker_id=freed_resource,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_diagnostics_summary(self) -> None:
        """Print a structured summary of solver diagnostics."""
        total = self._diag_trigger_count
        early_empty = self._diag_early_exit_empty
        early_inelig = self._diag_early_exit_ineligible
        non_trivial = total - early_empty - early_inelig

        print("\n" + "=" * 60)
        print("1-BATCH-1 DIAGNOSTICS")
        print("=" * 60)

        print(f"\n--- Trigger counts ---")
        print(f"  Total decide() calls:     {total:,}")
        print(f"  Early exit (empty queue):  {early_empty:,}")
        print(f"  Early exit (ineligible):   {early_inelig:,}")
        print(f"  Non-trivial (solver path): {non_trivial:,}")

        print(f"\n--- Fallback breakdown (non-trivial) ---")
        print(f"  MILP success:              {self._diag_milp_success_count:,}")
        print(f"  Greedy (max_tasks cap):    {self._diag_greedy_max_tasks_count:,}")
        print(f"  Greedy (neighborhood cap): {self._diag_greedy_neighborhood_too_large:,}")
        print(f"  Greedy (solver failure):   {self._diag_greedy_solver_fail_count:,}")

        def _percentiles(data, label):
            if not data:
                print(f"  {label}: (no data)")
                return
            arr = np.array(data)
            pcts = np.percentile(arr, [0, 25, 50, 75, 95, 100])
            print(f"  {label}:")
            print(f"    min={pcts[0]:.1f}  p25={pcts[1]:.1f}  median={pcts[2]:.1f}"
                  f"  p75={pcts[3]:.1f}  p95={pcts[4]:.1f}  max={pcts[5]:.1f}")

        print(f"\n--- Queue size distribution ---")
        _percentiles(self._diag_queue_sizes, "Waiting tasks")

        print(f"\n--- Worker set size distribution ---")
        _percentiles(self._diag_worker_sizes, "Workers")

        print(f"\n--- Top 5 largest problem dimensions (workers x tasks) ---")
        if self._diag_problem_dims:
            sorted_dims = sorted(self._diag_problem_dims, key=lambda x: x[2], reverse=True)
            for i, (nw, nt, prod) in enumerate(sorted_dims[:5]):
                print(f"  #{i+1}: {nw} workers x {nt} tasks = {prod:,} binary vars")
        else:
            print("  (no data)")

        print(f"\n--- MILP solve time (seconds) ---")
        if self._diag_milp_times:
            arr = np.array(self._diag_milp_times)
            pcts = np.percentile(arr, [0, 50, 95, 100])
            print(f"  min={pcts[0]:.4f}  median={pcts[1]:.4f}"
                  f"  p95={pcts[2]:.4f}  max={pcts[3]:.4f}")
            print(f"  total={arr.sum():.2f}s across {len(arr):,} solves")
        else:
            print("  (no solves)")

        print(f"\n--- Queue growth pattern (every 1000th trigger) ---")
        if self._diag_growth_log:
            print(f"  {'Trigger':>10s}  {'Queue':>8s}  {'Workers':>8s}")
            for trigger, qs, ws in self._diag_growth_log:
                print(f"  {trigger:>10,d}  {qs:>8,d}  {ws:>8,d}")
        else:
            print("  (fewer than 1000 triggers)")

        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Greedy Fallback (FIFO)
    # ------------------------------------------------------------------

    def _greedy_fallback(
        self,
        freed_resource: str,
        tasks: List[TaskInfo],
        eligible_map: Dict[str, Set[str]],
        processing_time_fn: Callable[[str, str], float],
    ) -> Optional[BatchDecision]:
        """Pick the oldest waiting task the freed worker is eligible for."""
        # Tasks are not guaranteed to be sorted — sort by hours_waited descending
        # (longest wait = oldest)
        sorted_tasks = sorted(tasks, key=lambda t: t.hours_waited, reverse=True)

        for t in sorted_tasks:
            elig = eligible_map.get(t.allocation_activity, set())
            if freed_resource in elig:
                return BatchDecision(task_id=t.task_id, worker_id=freed_resource)
        return None


# ---------------------------------------------------------------------------
# Registry & Factory
# ---------------------------------------------------------------------------

BATCH_POLICY_REGISTRY: Dict[str, type] = {
    "1_batch_1": OneBatchOnePolicy,
}


def create_batch_policy(name: str, **kwargs) -> BatchAllocationPolicy:
    """
    Create a batch allocation policy by name.

    Args:
        name: One of the keys in BATCH_POLICY_REGISTRY.
        **kwargs: Forwarded to the policy constructor.

    Raises:
        ValueError: If the name is not in the registry.
    """
    cls = BATCH_POLICY_REGISTRY.get(name)
    if cls is None:
        valid = ", ".join(sorted(BATCH_POLICY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown batch allocation policy: '{name}'. "
            f"Valid options: {valid}"
        )
    return cls(**kwargs)

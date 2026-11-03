"""
PMSP-Zuteilung (resources/case_allocation.py, PMSPAllocation): alle wartenden Aufgaben zusammen den Mitarbeitern zuteilen,
als Parallel Machine Scheduling Problem (CP-SAT, bei Bedarf Jonker-Volgenant). Ziel: späteste Fertigstellung, Ausgleich
zwischen den Mitarbeitern und Dummy-Kosten.

Dummy-Regel:
  c_d(t) = delta * avg(c(t,r) for r in R_predicted(t))
  Ist der beste verfügbare Mitarbeiter schlechter als die Dummy-Kosten, wartet die Aufgabe.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# Tie-breaking constant (ms) added to the dummy/unassigned cost so that a real
# resource is always preferred when its cost exactly equals the dummy threshold.
# With delta=1 and a single eligible resource the dummy cost equals that
# resource's cost exactly, which previously caused systematic "postpone" decisions.
_DUMMY_TIEBREAK_MS = 1


def solve_assignment_jv(
    *,
    tasks: List[str],
    resources: List[str],
    costs: Dict[Tuple[str, str], int],
    dummy_cost: Dict[str, int],
    remaining: Dict[str, int],
) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    """
    Solve assignment adaptation with Jonker-Volgenant (scipy linear_sum_assignment).
    Worst-case O(N^3), where N = max(n_tasks, n_resources + n_dummy_cols).
    """
    T = list(tasks)
    R = list(resources)
    nT = len(T)
    nR = len(R)
    assignment: Dict[str, Optional[str]] = {t: None for t in T}
    if nT == 0:
        return assignment, {"solver": "scipy_jv", "nT": 0, "nR": nR, "objective": 0}

    n_cols = nR + nT
    big_m = float(10**9)
    C = np.full((nT, n_cols), big_m, dtype=np.float64)

    for i, t in enumerate(T):
        for j, r in enumerate(R):
            if (t, r) in costs:
                C[i, j] = float(remaining.get(r, 0) + costs[(t, r)])
        # +_DUMMY_TIEBREAK_MS: real resources win ties against the dummy column
        C[i, nR + i] = float(dummy_cost.get(t, 10**9)) + _DUMMY_TIEBREAK_MS

    row_ind, col_ind = linear_sum_assignment(C)
    objective = 0.0
    for i, j in zip(row_ind, col_ind):
        t = T[i]
        objective += float(C[i, j])
        if j < nR and C[i, j] < big_m:
            assignment[t] = R[j]
        else:
            assignment[t] = None

    return assignment, {
        "solver": "scipy_jv",
        "nT": nT,
        "nR": nR,
        "objective": int(round(objective)),
    }


def solve_pmsp_ilp(
    *,
    delta: float,
    tasks: List[str],
    authorized_resources_by_waiting_task: Dict[str, List[str]],
    authorized_and_timeSlotOperating_resources_by_task: Dict[str, List[str]],
    costs_authorized_and_timeslotoperating_resources_per_task: Dict[str, Dict[str, int]],
    dummy_costs: Dict[str, int],
    predicted_remaining_times: Dict[str, float],
    costs_authorized_resource_task_assignment: Dict[str, Dict[str, int]],
    solver_time_limit_seconds: Optional[float] = None,
) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    """
    Solve PMSP adaptation according to Eq. 4.

    Objective: min C = k_m + k_f + sum(y_t * c_d(t))
    where:
    - k_m = max(c_r(r) + sum(x_tr * c(t,r)) for r in R_P)
    - k_f = (1/|R_P|) * sum(k_m - (c_r(r) + sum(x_tr * c(t,r))) for r in R_P)

    Primary solver: OR-Tools CP-SAT (time-limited, default 2s).
    Backup solver: SciPy Jonker-Volgenant assignment adaptation.
    """
    T = list(tasks)
    R_P_set = set()
    for resources in authorized_and_timeSlotOperating_resources_by_task.values():
        R_P_set.update(resources)
    R_P = list(R_P_set)
    nT = len(T)
    nR_P = len(R_P)

    if nT == 0 or nR_P == 0:
        return {t: None for t in T}, {"status": 0, "nT": nT, "nR_P": nR_P, "solver": "empty"}

    # Intentionally do NOT use a greedy fast-path for nT == 1.
    # The CP-SAT objective (makespan + fairness + dummy decision)
    # can differ meaningfully from "min remaining[r] + c(t,r)".
    # Keeping CP-SAT ensures we don't systematically bias assignments
    # toward `User_1` when only a single task is optimized.

    def _run_jv_fallback() -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
        """
        Lazily run JV fallback only when CP-SAT is unavailable or clearly worse.
        Avoids always paying O(N^3) JV cost when CP-SAT is fast and optimal.
        """
        costs_for_jv: Dict[Tuple[str, str], int] = {}
        dummy_cost_for_jv: Dict[str, int] = {}
        remaining_for_jv: Dict[str, int] = {}

        for task_id in T:
            if task_id in costs_authorized_and_timeslotoperating_resources_per_task:
                for r, cost in costs_authorized_and_timeslotoperating_resources_per_task[task_id].items():
                    costs_for_jv[(task_id, r)] = cost
            dummy_cost_for_jv[task_id] = dummy_costs.get(task_id, 10**9)

        for r in R_P:
            remaining_for_jv[r] = int(predicted_remaining_times.get(r, 0.0) * 1000)

        return solve_assignment_jv(
            tasks=T,
            resources=R_P,
            costs=costs_for_jv,
            dummy_cost=dummy_cost_for_jv,
            remaining=remaining_for_jv,
        )

    effective_time_limit = 2.0 if solver_time_limit_seconds is None else float(solver_time_limit_seconds)
    effective_time_limit = max(0.01, effective_time_limit)

    try:
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        x: Dict[Tuple[str, str], Any] = {}
        y: Dict[str, Any] = {}

        for task_id in T:
            task_costs = costs_authorized_and_timeslotoperating_resources_per_task.get(task_id, {})
            if task_id in authorized_and_timeSlotOperating_resources_by_task:
                for r in authorized_and_timeSlotOperating_resources_by_task[task_id]:
                    if r in R_P and r in task_costs:
                        x[(task_id, r)] = model.NewBoolVar(f"x__{task_id}__{r}")

        for t in T:
            y[t] = model.NewBoolVar(f"y__{t}")

        # Upper bound for makespan
        max_remaining = max((int(predicted_remaining_times.get(r, 0.0) * 1000) for r in R_P), default=0)
        max_task_cost = max(
            (cost for task_costs in costs_authorized_and_timeslotoperating_resources_per_task.values()
             for cost in task_costs.values()),
            default=0
        )
        ub_km = max_remaining + max_task_cost * nT
        k_m = model.NewIntVar(0, ub_km, "k_m")

        # Constraint (4e): Each task -> exactly one resource or dummy
        for t in T:
            x_t = [x[(t, r)] for r in R_P if (t, r) in x]
            model.Add(sum(x_t) + y[t] == 1)

        # Makespan constraints: k_m >= c_r(r) + sum(x_tr * c(t,r)) for all r in R_P
        for r in R_P:
            c_r = int(predicted_remaining_times.get(r, 0.0) * 1000)
            terms = []
            for t in T:
                if (t, r) in x:
                    if t in costs_authorized_and_timeslotoperating_resources_per_task:
                        if r in costs_authorized_and_timeslotoperating_resources_per_task[t]:
                            cost = costs_authorized_and_timeslotoperating_resources_per_task[t][r]
                            terms.append(cost * x[(t, r)])
            model.Add(c_r + sum(terms) <= k_m)

        # Objective: scaled by |R_P| to avoid fractions
        # = |R_P| * 2 * k_m - sum_workload + |R_P| * sum(y_t * c_d(t))
        sum_workload_terms = []
        sum_c_r_constant = 0

        for r in R_P:
            c_r = int(predicted_remaining_times.get(r, 0.0) * 1000)
            sum_c_r_constant += c_r
            for t in T:
                if (t, r) in x:
                    if t in costs_authorized_and_timeslotoperating_resources_per_task:
                        if r in costs_authorized_and_timeslotoperating_resources_per_task[t]:
                            cost = costs_authorized_and_timeslotoperating_resources_per_task[t][r]
                            sum_workload_terms.append(cost * x[(t, r)])

        max_sum_workload = max_remaining * nR_P + max_task_cost * nT * nR_P
        sum_workload_var = model.NewIntVar(0, max_sum_workload, "sum_workload")
        model.Add(sum_workload_var == sum_c_r_constant + sum(sum_workload_terms))

        dummy_sum_terms = []
        for t in T:
            if t in dummy_costs:
                # +_DUMMY_TIEBREAK_MS per dummy assignment so that a real resource
                # is strictly preferred when its cost equals the dummy threshold.
                dummy_sum_terms.append((dummy_costs[t] + _DUMMY_TIEBREAK_MS) * y[t])

        if nR_P > 0:
            objective = nR_P * 2 * k_m - sum_workload_var + nR_P * sum(dummy_sum_terms)
        else:
            objective = sum(dummy_sum_terms)

        model.Minimize(objective)

        validation = model.Validate()
        if validation:
            raise ValueError(f"CP-SAT model validation failed: {validation}")

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = effective_time_limit
        solver.parameters.num_search_workers = 30

        logger.info(
            "PMSP [Solver]: Starting CP-SAT solver for %d tasks, %d resources (time limit: %.2fs)",
            nT,
            nR_P,
            effective_time_limit,
        )
        logger.info(
            "PMSP [Solver]: Problem size - Tasks: %d, Resources: %d, Variables: %d",
            nT,
            nR_P,
            len(x) + len(y),
        )
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
        logger.info(
            "PMSP [Solver]: CP-SAT solver finished with status: %s (wall_time: %.3fs)",
            status_name,
            solver.WallTime(),
        )

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            assignment_cp: Dict[str, Optional[str]] = {t: None for t in T}
            for t in T:
                if solver.Value(y[t]) >= 1:
                    assignment_cp[t] = None
                    continue
                chosen = None
                for r in R_P:
                    if (t, r) in x and solver.Value(x[(t, r)]) >= 1:
                        chosen = r
                        break
                assignment_cp[t] = chosen

            k_m_val = solver.Value(k_m)
            sum_workload_val = solver.Value(sum_workload_var)
            k_f_val = k_m_val - (sum_workload_val / nR_P) if nR_P > 0 else 0.0
            dummy_sum = sum(int(dummy_costs.get(t, 0)) * solver.Value(y[t]) for t in T)
            dummy_count = sum(1 for t in T if solver.Value(y[t]) >= 1)
            cp_obj = int(round(k_m_val + k_f_val + dummy_sum))
            # JV fallback is only beneficial when CP-SAT did NOT find any feasible
            # solution at all (status UNKNOWN). When CP-SAT returns FEASIBLE the
            # solution is valid — running the O(N³) JV on top would waste time.
            # (Previously JV ran on every non-OPTIMAL result, which fired almost
            # every cycle at the 2 s time limit and added 10-50 s of overhead.)
            if status == cp_model.UNKNOWN:
                assignment_jv, debug_jv = _run_jv_fallback()
                jv_obj = int(debug_jv.get("objective", 10**18))
                if jv_obj <= cp_obj:
                    dbg = dict(debug_jv)
                    dbg.update({"status": int(status), "status_name": status_name, "fallback_from": "cp_sat"})
                    return assignment_jv, dbg

            assigned_count = sum(1 for v in assignment_cp.values() if v is not None)
            logger.info(
                "PMSP [Solver]: Optimization complete - %d/%d tasks assigned (objective: %d, k_m: %d, k_f: %d)",
                assigned_count,
                nT,
                cp_obj,
                int(k_m_val),
                int(round(k_f_val)),
            )
            logger.info(
                "PMSP [Solver]: Makespan (k_m): %d ms, Fairness (k_f): %d ms, Dummy assignments: %d/%d tasks (cost: %d ms)",
                int(k_m_val),
                int(round(k_f_val)),
                dummy_count,
                nT,
                dummy_sum,
            )
            return assignment_cp, {
                "status": int(status),
                "status_name": status_name,
                "nT": nT,
                "nR_P": nR_P,
                "solver": "cp_sat",
                "objective": cp_obj,
                "k_m": int(k_m_val),
                "k_f": int(round(k_f_val)),
            }

        # CP-SAT finished but without a feasible/optimal solution: use JV fallback.
        assignment_jv, debug_jv = _run_jv_fallback()
        dbg = dict(debug_jv)
        dbg.update({"status": int(status), "status_name": status_name, "fallback_from": "cp_sat"})
        return assignment_jv, dbg

    except Exception as e:
        logger.warning("PMSP [Solver]: CP-SAT solver failed, using JV fallback: %s", e)
        assignment_jv, debug_jv = _run_jv_fallback()
        assigned_count = sum(1 for v in assignment_jv.values() if v is not None)
        logger.info(
            "PMSP [Solver]: Using JV fallback - %d/%d tasks assigned",
            assigned_count,
            nT,
        )
        dbg = dict(debug_jv)
        dbg.update({"fallback_from": "cp_sat_import_or_runtime_error", "error": str(e)})
        return assignment_jv, dbg

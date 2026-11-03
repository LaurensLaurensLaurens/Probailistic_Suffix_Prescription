"""
BPIC17: die Zuteilungen der Engine außer der Grundklasse (resources/case_allocation.py, create_allocation): bank, batch,
drl, pmsp; learned in learned_allocation.py (random, round_robin, shortest_queue: resources/selection_strategies.py).
"""
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from resources.case_allocation import Allocation, Task


def _datetime(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)


class _TaskChoosingAllocation(Allocation):
    """Policies that let a free employee choose among the waiting tasks."""

    def __init__(self, estimator, **kwargs):
        super().__init__(**kwargs)
        # (employee, activity) -> expected seconds of work, mined from the log
        self.estimator = estimator

    def task_for_resource(self, resource: str, tasks: List[Task], time: float) -> Optional[Task]:
        raise NotImplementedError

    def assign(self, tasks, time, new_task=None, freed=None):
        if new_task is not None:
            return super().assign(tasks, time, new_task=new_task)
        resources = [freed] if freed is not None else sorted(self.engine.free_on_duty(time))
        remaining, pairs = list(tasks), []
        for resource in resources:
            if not any(resource in self.engine.on_duty_eligible(task.activity, time) for task in remaining):
                continue
            task = self.task_for_resource(resource, remaining, time)
            if task is not None:
                pairs.append((task, resource))
                remaining.remove(task)
        return pairs

    def _remaining_seconds(self, resource: str, time: float) -> float:
        """Expected time until a busy employee is free again."""
        if resource not in self.engine.busy:
            return 0.0
        _, activity, since = self.engine.busy[resource]
        return max(0.0, self.estimator.estimate(resource, activity) - (time - since))


class BatchAllocation(_TaskChoosingAllocation):
    """1-Batch-1 (Zeng et al. 2005): the free employee gets the task that minimises the latest finishing time over all employees."""

    MAX_NEIGHBOURHOOD = 50  # above this the optimisation is skipped and the longest-waiting task is taken

    def __init__(self, estimator, **kwargs):
        from resources.BPIC2017.allocation.batch_policies import OneBatchOnePolicy

        super().__init__(estimator, **kwargs)
        self.policy = OneBatchOnePolicy()

    def reset(self) -> None:
        super().reset()
        self.policy.reset()

    def task_for_resource(self, resource, tasks, time):
        from resources.BPIC2017.allocation.batch_policies import TaskInfo, WorkerInfo

        eligible = {activity: set(self.engine.on_duty_eligible(activity, time)) for activity in {task.activity for task in tasks}}
        # Only the tasks this employee competes for: the ones they may do and those sharing employees with them
        activities = {activity for activity, workers in eligible.items() if resource in workers}
        neighbours = set().union(*(eligible[activity] for activity in activities)) if activities else set()
        activities |= {activity for activity, workers in eligible.items() if workers & neighbours}
        tasks = [task for task in tasks if task.activity in activities]
        mine = [task for task in tasks if resource in eligible[task.activity]]
        if not mine:
            return None
        if len(tasks) > self.MAX_NEIGHBOURHOOD:
            return mine[0]
        by_id = {f"{task.case_id}::{task.activity}": task for task in tasks}
        decision = self.policy.decide(
            freed_resource=resource,
            current_time_s=time,
            tasks=[TaskInfo(task_id, task.case_id, task.activity, task.activity, (time - task.since) / 3600.0) for task_id, task in by_id.items()],
            workers=[WorkerInfo(worker, self._remaining_seconds(worker, time)) for worker in set().union(*(eligible[a] for a in activities))],
            eligible_map={activity: eligible[activity] for activity in activities},
            processing_time_fn=self.estimator.estimate,
        )
        return by_id.get(decision.task_id) if decision is not None else None


class DRLAllocation(_TaskChoosingAllocation):
    """
    The trained PPO policy of the original engine chooses for a free employee which activity to
    serve, or to serve none. It was trained in the original engine, where an employee is bound for
    one event at a time; here it runs outside the conditions it was trained in.
    """

    def __init__(self, estimator, model_path: str, max_postpone_wait_hours: float = 4.0, **kwargs):
        import os
        import pickle

        from sb3_contrib import MaskablePPO

        from resources.BPIC2017.allocation.drl_allocation.state import DRLStateBuilder

        super().__init__(estimator, **kwargs)
        self.model = MaskablePPO.load(model_path)
        with open(os.path.join(os.path.dirname(model_path), "state_builder_config.pkl"), "rb") as f:
            config = pickle.load(f)
        self.state = DRLStateBuilder(activity_list=config["activity_list"], role_groups=config["role_groups"],
                                     resource_to_role=config["resource_to_role"], activity_to_roles=config["activity_to_roles"])
        self.max_postpone_wait_hours = max_postpone_wait_hours

    def task_for_resource(self, resource, tasks, time):
        now = _datetime(time)
        engine = self.engine
        snapshot = {
            "busy_resources": {
                worker: {"busy_until": _datetime(time + self._remaining_seconds(worker, time)), "case_id": case_id, "activity": activity}
                for worker, (case_id, activity, _) in engine.busy.items()
            },
            "waiting_queues": {},
        }
        for task in tasks:
            snapshot["waiting_queues"].setdefault(task.activity, []).append(
                {"case_id": task.case_id, "activity": task.activity, "allocation_activity": task.activity, "arrival_time": _datetime(task.since)})
        observation = self.state.build_from_snapshot(resource, now, snapshot, len(engine.cases))
        eligible = {activity: set(engine.on_duty_eligible(activity, time)) for activity in snapshot["waiting_queues"]}
        mask = self.state.build_action_mask(resource, eligible, set(eligible))
        postpone = self.state.num_activities
        # Holding back is not allowed once a task has waited too long
        if any(time - task.since > self.max_postpone_wait_hours * 3600.0 for task in tasks) and mask[:postpone].any():
            mask[postpone] = False
        action, _ = self.model.predict(observation, deterministic=True, action_masks=mask)
        if int(action) >= postpone:
            return None
        activity = self.state.activity_list[int(action)]
        return next((task for task in tasks if task.activity == activity), None)


class PMSPAllocation(Allocation):
    """
    All waiting tasks are assigned together (parallel machine scheduling, solved with CP-SAT): it
    minimises the latest finishing time and the imbalance between employees. A task is held back
    if no employee on duty is better than the average employee who may do it.
    """

    MAX_TASKS = 40  # the longest-waiting tasks are optimised, the rest wait for the next round

    def __init__(self, estimator, delta: float = 1.5, solver_time_limit_seconds: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator
        self.delta = delta
        self.solver_time_limit_seconds = solver_time_limit_seconds

    def assign(self, tasks, time, new_task=None, freed=None):
        from resources.BPIC2017.resource_optimization.resource_optimization import _DUMMY_TIEBREAK_MS, solve_pmsp_ilp

        engine = self.engine
        tasks = tasks[: self.MAX_TASKS]
        ids = {f"{task.case_id}_{task.activity}": task for task in tasks}
        authorized = {task_id: engine.eligible_for(task.activity) for task_id, task in ids.items()}
        operating = {task_id: engine.on_duty_eligible(task.activity, time) for task_id, task in ids.items()}
        costs = {task_id: {r: int(self.estimator.estimate(r, task.activity) * 1000) for r in authorized[task_id]} for task_id, task in ids.items()}
        dummy = {task_id: int(self.delta * sum(c.values()) / len(c)) + _DUMMY_TIEBREAK_MS for task_id, c in costs.items() if c}
        workers = {r for resources in operating.values() for r in resources}
        if not workers:
            return []
        remaining = {r: (self.estimator.estimate(r, engine.busy[r][1]) - (time - engine.busy[r][2]) if r in engine.busy else 0.0) for r in workers}
        assignment, _ = solve_pmsp_ilp(
            delta=self.delta,
            tasks=list(ids),
            authorized_resources_by_waiting_task=authorized,
            authorized_and_timeSlotOperating_resources_by_task=operating,
            costs_authorized_and_timeslotoperating_resources_per_task={t: {r: costs[t][r] for r in operating[t]} for t in ids},
            dummy_costs=dummy,
            predicted_remaining_times={r: max(0.0, seconds) for r, seconds in remaining.items()},
            costs_authorized_resource_task_assignment=costs,
            solver_time_limit_seconds=self.solver_time_limit_seconds,
        )
        # The optimisation may line up several tasks for one employee; only the first one of an
        # employee who is free now is committed, the others are optimised again in the next round.
        pairs, taken = [], set()
        for task_id, task in ids.items():
            resource = assignment.get(task_id)
            if resource is not None and resource not in engine.busy and resource not in taken:
                pairs.append((task, resource))
                taken.add(resource)
        return pairs


class BankAllocation(Allocation):
    """Like random, but each free employee is chosen with a weight of how often they did the activity in the real log."""

    def __init__(self, counts: Dict[Tuple[str, str], int], keep_last_employee: bool = True):
        super().__init__(keep_last_employee=keep_last_employee)
        self.counts = counts

    def resource_for_task(self, task: Task, free: List[str], time: float) -> str:
        if self.keep_last_employee and task.opening == "start" and task.last_resource in free:
            return task.last_resource
        # Python's random, seeded by the engine like the other selection strategies
        return random.choices(free, weights=[self.counts.get((r, task.activity), 0) + 1 for r in free])[0]

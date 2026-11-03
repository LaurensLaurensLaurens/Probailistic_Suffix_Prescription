"""
Who works on which case: the allocation policies of the case engine (simulation/case_engine.py).

A task is a work item of a case that needs an employee (its "start" or "resume"). The engine asks
the policy whenever a new task comes up, an employee becomes free, or a new hour begins and other
employees come on duty. The policy answers with the pairs (task, employee) to commit now; all other
tasks keep waiting.

Here: the base policy (Allocation) with the selection strategies (resources/selection_strategies.py), which work on
the pool of every dataset, and create_allocation. The other policies are used only by BPIC17 (the only dataset that
runs in the engine with employees) and live in BPIC2017/allocation/ (allocations.py and the modules they use):
- random, round_robin, shortest_queue (here): a new task takes a free employee chosen by the selection
  strategy; a free employee takes the longest-waiting task.
- bank: like random, but an employee is chosen as often as they did the activity in the real log. In the
  real log a few employees do most of an activity (e.g. 20 employees 76 % of W_Validate application).
- learned: chosen with the probabilities of a model learned from the real log (experience, continuity,
  four-eyes principle; BPIC2017/allocation/learned_allocation.py).
- batch: a free employee takes the task that the 1-Batch-1 optimisation gives them
  (BPIC2017/allocation/batch_policies.py), which looks at all waiting tasks and all employees.
- drl: a free employee takes a task of the activity the trained policy chooses, or none (the model was trained for BPIC17).
- pmsp: all waiting tasks are assigned together by the PMSP optimisation
  (BPIC2017/resource_optimization); a task may be held back for a better employee.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from resources.selection_strategies import RandomStrategy, ResourceSelectionStrategy, create_strategy


@dataclass(eq=False)
class Task:
    case_id: str
    activity: str
    opening: str  # "start" or "resume"
    since: float  # when the task came up, seconds since 1970
    last_resource: Optional[str]  # who handled the case last
    reason: str = ""  # why it had to wait, set by the engine


class Allocation:
    """
    Base policy: a new task takes a free employee chosen by the selection strategy, a free employee
    takes the longest-waiting task they may do.
    """

    def __init__(self, strategy: Optional[ResourceSelectionStrategy] = None, keep_last_employee: bool = True):
        """strategy: chooses among the free employees (resources/selection_strategies.py; default: random)."""
        self.strategy = strategy or RandomStrategy()
        # A work item is mostly started by whoever handled the case last (85 % in the real log).
        # With this set, that employee gets the task if free; the strategy decides otherwise.
        self.keep_last_employee = keep_last_employee
        self.engine = None

    def bind(self, engine) -> None:
        self.engine = engine

    def reset(self) -> None:
        self.strategy.reset()

    def committed(self, resource: str, activity: str) -> None:
        self.strategy.notify_assignment(resource, activity)

    def released(self, resource: str) -> None:
        self.strategy.notify_release(resource)

    # Hier wird dann, nachdem wir wissen, wer alles bearbeiten darf, je nach Strategie der Bearbeiter zugewiesen
    def resource_for_task(self, task: Task, free: List[str], time: float) -> str:
        # Einfach die letzte Resource verwenden als Strategie
        if self.keep_last_employee and task.opening == "start" and task.last_resource in free:
            return task.last_resource
        # Die jeweilge Ressourcenstrategie anwenden
        return self.strategy.select(free, task.activity)

    # Allgemein: Assignen. Die Strategie legt fest wie genau
    def assign(self, tasks: List[Task], time: float, new_task: Optional[Task] = None,
               freed: Optional[str] = None) -> List[Tuple[Task, str]]:
        """
        tasks: all waiting tasks, longest-waiting first (including new_task).
        new_task: set if the engine asks because this task just came up.
        freed: set if the engine asks because this employee just became free.
        Neither set: a new hour began.
        """
        # Neuer Task
        if new_task is not None:
            # Kandidaten, die diese Aktivität ausführen können und derzet nicht zu tun haben.
            free = self.engine.free_candidates(new_task.activity, time)
            # Zuweisen der Ressource nach Strategie
            return [(new_task, self.resource_for_task(new_task, free, time))] if free else []

        # Bestehende Tasks in Warteschlange
        # Her wird bei der Gelegenheit direkt geguckt, ob wir etwas in der Warteschlange bearbeten können
        taken, pairs, free_for = set(), [], {}
        for task in tasks:
            # Je Tätigkeit nur einmal: bei langer Warteschlange wäre das der teuerste Teil der Simulation
            if task.activity not in free_for:
                free_for[task.activity] = [r for r in self.engine.free_candidates(task.activity, time) if freed in (None, r)]
            free = [r for r in free_for[task.activity] if r not in taken]
            if free:
                resource = self.resource_for_task(task, free, time)
                pairs.append((task, resource))
                taken.add(resource)
                if freed is not None:
                    break  # der eine frei gewordene Mitarbeiter ist vergeben
        return pairs


def work_counts(event_log, prefix: str) -> Dict[Tuple[str, str], int]:
    """
    How often each employee started or resumed each work item (activities with prefix, next_event_prediction/datasets.py
    work_item_prefix; BPIC17: W_...) in the real log: (employee, activity) -> count.
    """
    rows = event_log[event_log["concept:name"].astype(str).str.startswith(prefix) & event_log["lifecycle:transition"].isin(["start", "resume"])]
    return dict(Counter(zip(rows["org:resource"], rows["concept:name"])))


ALLOCATIONS = ("bank", "learned", "random", "round_robin", "shortest_queue", "batch", "drl", "pmsp")


def create_allocation(name: str, event_log, dataset: str, keep_last_employee: bool = True) -> Allocation:
    """
    event_log: the real log (after the preprocessing); batch, drl and pmsp mine the expected working times from it.
    dataset: the saved models (learned, drl) and the system user (resources/datasets.py), the work items
    (next_event_prediction/datasets.py).
    """
    from next_event_prediction.datasets import model_spec
    from resources.datasets import resource_spec, system_users

    spec, prefix = resource_spec(dataset), model_spec(dataset)["work_item_prefix"]
    saved = {"learned": "learned_allocation", "drl": "drl_model"}.get(name)
    if saved and saved not in spec:
        raise ValueError(f"Für {dataset} gibt es kein Modell der Zuteilung {name} ({saved} in RESOURCES, resources/datasets.py)")
    if name in ("random", "round_robin", "shortest_queue"):
        return Allocation(create_strategy(name), keep_last_employee=keep_last_employee)
    from resources.BPIC2017.allocation import allocations
    if name == "bank":
        return allocations.BankAllocation(work_counts(event_log, prefix), keep_last_employee=keep_last_employee)
    if name == "learned":
        # Gelernt aus dem echten Log, wen die Bank für einen Arbeitsschritt wählt (BPIC2017/allocation/learned_allocation.py)
        from resources.BPIC2017.allocation.learned_allocation import load_learned_allocation
        # Mit Arbeitsabschnitten (ohne resume und suspend im Log) eine eigene Datei: gelernt aus dem bereinigten Log
        path = spec["learned_allocation"]
        if "lifecycle:transition" in event_log and not event_log["lifecycle:transition"].isin(["resume", "suspend"]).any():
            path = path.replace(".pkl", "_sessions.pkl")
        return load_learned_allocation(event_log, path, system_users(dataset, event_log), prefix)
    from resources.BPIC2017.allocation.processing_time_estimator import ProcessingTimeEstimator

    estimator = ProcessingTimeEstimator(df=event_log)
    if name == "batch":
        return allocations.BatchAllocation(estimator, keep_last_employee=keep_last_employee)
    if name == "drl":
        return allocations.DRLAllocation(estimator, spec["drl_model"], keep_last_employee=keep_last_employee)
    if name == "pmsp":
        return allocations.PMSPAllocation(estimator, keep_last_employee=keep_last_employee)
    raise ValueError(f"Unknown allocation '{name}'. Valid options: {', '.join(ALLOCATIONS)}")

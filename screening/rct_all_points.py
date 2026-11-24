"""Sequential randomized BPIC17 study, including offer attributes and response deadlines.

The production simulator is unchanged. This study uses its own engine subclass and
catalog. All assignments, available actions, probabilities and event positions are
recorded before the assigned action happens. Response time is analysed by assigned
deadline (intention to treat), not by the subsequently observed waiting time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from helpers.datasets import ACTIVITY, CASE, LIFECYCLE, TIMESTAMP
from simulation.case_engine_lifecycle_aware import CaseEngine, EVENT


def study_design(log, tests):
    offers = log[(log[ACTIVITY] == "O_Create Offer") & (log[LIFECYCLE] == "complete")]
    design = {"attributes": {}}
    for name, test in tests.items():
        if test.get("kind") != "attribute_bins":
            continue
        values = pd.to_numeric(offers[test["attribute"]], errors="coerce")
        if test.get("ratio_to"):
            denominator = pd.to_numeric(offers[test["ratio_to"]], errors="coerce")
            values = values / denominator.where(denominator > 0)
        values = values[np.isfinite(values) & (values >= 0)]
        bins = np.digitize(values, test["bins"])
        samples = {label: values.to_numpy()[bins == j].tolist() for j, label in enumerate(test["labels"])}
        if any(not sample for sample in samples.values()):
            raise ValueError(f"{name}: empty empirical treatment bin")
        design["attributes"][name] = {**test, "samples": samples}
    # Same observed lag definition as the historical screening, fixed before RCT generation.
    from screening.events import actions
    from screening.candidates import prepared_log
    from helpers.datasets import dataset_info
    _, _, lag, _, _ = actions(prepared_log(log), tests["reaktionszeit_antrag"], dataset_info("bpic17")["screening"])
    seconds = lag.to_numpy(dtype=float) * 3600
    seconds = seconds[np.isfinite(seconds) & (seconds >= 0)]
    median = float(np.median(seconds))
    design["reaction"] = {"threshold_seconds": median,
                          "samples": {"schnell": seconds[seconds < median].tolist(),
                                      "langsam": seconds[seconds >= median].tolist()},
                          "estimand": "assigned response deadline, intention to treat; staffing may delay actual start"}
    if any(not sample for sample in design["reaction"]["samples"].values()):
        raise ValueError("Response deadline needs two nonempty empirical bins")
    return design


class RecordingRCT:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.records = []

    def record(self, name, kind, case, options, time, event_pos, value=None):
        if not options:
            raise ValueError(f"{name}: cannot randomize an empty action set")
        choice = options[int(self.rng.integers(len(options)))]
        self.records.append({"case": case.case_id, "point": name, "kind": kind,
                             "options": tuple(options), "choice": choice, "probability": 1 / len(options),
                             "time": time, "event_pos": event_pos, "assigned_value": value})
        return choice

    def choose(self, point, case, options, time, engine):
        pos = case.n_events - 1 if point.kind == "control_flow" else case.n_events
        return self.record(point.name, point.kind, case, options, time, pos)


class RandomizedOffers:
    def __init__(self, original, engine):
        self.original, self.engine = original, engine

    def __getattr__(self, name):
        return getattr(self.original, name)

    def dynamic_attributes(self, state):
        values, event = self.original.dynamic_attributes(state)
        engine = self.engine
        case, policy = engine.cases[state.case_id], engine.policy
        for name, test in engine.study_design["attributes"].items():
            # A ratio to an absent/zero requested amount is undefined, not another arm.
            denominator = case.attributes.get(test.get("ratio_to"), values.get(test.get("ratio_to"))) if test.get("ratio_to") else 1.0
            if denominator is None or not np.isfinite(denominator) or denominator <= 0:
                continue
            choice = policy.record(name, "attribute", case, test["labels"], case.pending[1], case.n_events)
            samples = test["samples"][choice]
            sampled = float(samples[int(policy.rng.integers(len(samples)))])
            value = sampled * denominator
            if test["attribute"] == "NumberOfTerms":
                value = int(round(value))
            values[test["attribute"]] = value
            policy.records[-1]["assigned_value"] = value
        # Keep dependent monthly payments consistent with the randomized amount and term.
        generator = self.original.generator
        values["MonthlyCost"] = float(generator.registry.monthly_cost.predict(
            offered_amount=values["OfferedAmount"], number_of_terms=int(values["NumberOfTerms"]),
            credit_score=state.credit_score, application_type=state.application_type))
        for column, field in self.original.spec["dynamic"].items():
            if column in values:
                setattr(state, field, values[column])
        return values, {**event, **values, **self.original.spec["open"]}


class AllPointsRCTEngine(CaseEngine):
    @classmethod
    def from_engine(cls, engine, design):
        study = cls.__new__(cls)
        study.__dict__.update(engine.__dict__)
        study.allocation.bind(study)
        study.study_design = design
        study.deadlines, study.deadline_applied = {}, set()
        study.attribute_generator = RandomizedOffers(study.attribute_generator, study)
        return study

    def _happen(self, case, token, time):
        activity, lifecycle = self._tokens[token]
        if activity == "A_Create Application" and lifecycle == "complete" and case.case_id not in self.deadlines:
            choice = self.policy.record("reaktionszeit_antrag", "deadline", case,
                                        ["schnell", "langsam"], time, case.n_events)
            samples = self.study_design["reaction"]["samples"][choice]
            seconds = float(samples[int(self.policy.rng.integers(len(samples)))])
            self.deadlines[case.case_id] = time + seconds
            self.policy.records[-1]["assigned_value"] = seconds
        super()._happen(case, token, time)

    def _schedule(self, time, kind, case_id):
        # Override the first scheduled application-work start with the assigned
        # deadline. Other events and resource availability continue normally.
        if kind == EVENT and case_id in self.deadlines and case_id not in self.deadline_applied:
            case = self.cases[case_id]
            if case.pending is not None:
                token, _ = case.pending
                if self._tokens[token] == ("W_Complete application", "start"):
                    time = max(case.last_time, self.deadlines[case_id])
                    case.pending = (token, time)
                    self.deadline_applied.add(case_id)
        super()._schedule(time, kind, case_id)


def simulate_run(log, design, n_cases, seed, run):
    import interventions
    from helpers.simulator_check import prepare_simulated
    from preprocessing import chosen_outcome_spec
    from simulation.case_engine_lifecycle_aware import build_engine
    from simulation.dataset_simulator import model_folder

    base = build_engine(log, n_cases, seed, model_folder("bpic17"), dataset="bpic17",
                        allocation="learned", permissions=interventions.run_permissions("bpic17"))
    engine = AllPointsRCTEngine.from_engine(base, design)
    policy = RecordingRCT([seed, 321])
    engine.set_policy(policy, interventions.choice_points("bpic17"))
    events = pd.DataFrame(engine.run(n_cases))
    events = prepare_simulated(events, chosen_outcome_spec("bpic17"))
    records = pd.DataFrame(policy.records)
    validate_assignments(events, records, {row["name"]: row["test"] for row in interventions.candidates("bpic17")})
    prefix = f"R{run:02d}_"
    events[CASE] = prefix + events[CASE].astype(str)
    records["case"] = prefix + records["case"].astype(str)
    events["simulation_run"], records["simulation_run"] = run, run
    events["rct"] = True
    summary = {"run": run, "seed": seed, "cases": int(events[CASE].nunique()), "events": len(events),
               "assignments": len(records), "forced_assignments": engine.forced_assignments,
               "truncated_cases": len(engine.truncated_cases), "deadlines_applied": len(engine.deadline_applied)}
    if summary["cases"] != n_cases:
        raise ValueError(f"Expected {n_cases} cases, got {summary['cases']}")
    return events, records, summary


def validate_assignments(events, records, tests):
    """Check assignments against actual logged events, not only the policy output."""
    from helpers.datasets import RESOURCE
    from interventions import option_token
    ordered = events.sort_values([CASE, TIMESTAMP], kind="stable").copy()
    ordered["_pos"] = ordered.groupby(CASE).cumcount()
    at = ordered.set_index([CASE, "_pos"])
    for record in records.itertuples(index=False):
        event = at.loc[(record.case, record.event_pos)]
        if record.choice not in record.options or not np.isclose(record.probability, 1 / len(record.options)):
            raise ValueError("Assignment does not match recorded probability/action set")
        if record.kind == "employee" and str(event[RESOURCE]) != str(record.choice):
            raise ValueError("Recorded employee differs from actual assigned employee")
        if record.kind == "control_flow":
            following = at.loc[(record.case, record.event_pos + 1)]
            if option_token(record.choice) != f"{following[ACTIVITY]}|{following[LIFECYCLE]}":
                raise ValueError("Recorded CF choice differs from actual next event")
        if record.kind == "attribute":
            test = tests[record.point]
            value = event[test["attribute"]]
            if not np.isclose(value, record.assigned_value):
                raise ValueError("Attribute assignment did not reach the event log")
            if test.get("ratio_to"):
                value /= event[test["ratio_to"]]
            label = test["labels"][np.digitize(value, test["bins"])]
            if label != record.choice:
                raise ValueError("Actual offer value is outside its randomized treatment bin")

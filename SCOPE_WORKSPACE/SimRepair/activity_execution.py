import numpy as np
import random
from datetime import timedelta
from SimRepair.workshop_policy import WorkshopPolicy


STATE_FIELDS = [
    "bike_value",
    "repair_severity",
    "quality_uncertainty",
    "customer_patience",
    "customer_friendliness",
    "process_type",
    "employee_competence",
    "material_quality",
]


class ActivityExecutioner():
    def __init__(self, random_obj=random.Random()):
        self.times_dic = {
            "initiate_case": 1,
            "start_priority": 1,
            "start_standard": 2,
            "wait_for_employee": 0.8,
            "use_spare_parts": 0.45,
            "choose_employee": 1.5,
            "choose_material": 1.25,
            "repair_priority": 25 / 6,
            "repair_standard": 25 / 6,
            "prepare_rerepair_priority": 0.25,
            "prepare_rerepair_standard": 0.25,
            "conduct_qc": 2,
            "improve_rework": 3,
            "shipping": 1,
            "customer_acceptance": 0,
            "customer_refusal": 100,
        }
        self.times_dic = {key: value * 86400 for key, value in self.times_dic.items()}
        self.random_obj = random_obj
        self.workshop_policy = WorkshopPolicy(random_obj)

    def set_state(self, random_state):
        self.random_obj.setstate(random_state)

    def set_time_choose_employee(self, employee_competence):
        self.times_dic["choose_employee"] = (0.5 + 0.20 * employee_competence) * 86400

    def set_time_choose_material(self, material_quality):
        self.times_dic["choose_material"] = (0.5 + 0.15 * material_quality) * 86400

    def set_time_repair(self, repair_activity, repair_severity, employee_competence):
        self.times_dic[repair_activity] = (2 + 0.6 * repair_severity) / (0.8 + 0.08 * employee_competence) * 86400

    def calc_outcome(self, current_event):
        if current_event["activity"] == "customer_acceptance":
            return 0
        if current_event["activity"] == "customer_refusal":
            return -100
        return np.nan

    def sample_clipped(self, mean, std_dev, lo, hi, as_int=False):
        while True:
            sample = self.random_obj.gauss(mean, std_dev)
            if as_int:
                sample = int(sample)
            if lo <= sample <= hi:
                return sample

    def sample_bike_value(self):
        return self.sample_clipped(5, 2, 1, 10, as_int=True)

    def sample_repair_severity(self):
        return self.sample_clipped(5, 2, 1, 10, as_int=True)

    def sample_quality_uncertainty(self):
        return self.sample_clipped(0.5, 0.25, 0, 1, as_int=False)

    def sample_customer_patience(self):
        return self.sample_clipped(5, 2, 1, 10, as_int=True)

    def sample_customer_friendliness(self):
        return self.sample_clipped(5, 2, 1, 10, as_int=True)

    def sample_employee_competence(self):
        return self.sample_clipped(5, 2, 1, 10, as_int=True)

    def sample_material_quality(self):
        return self.sample_clipped(5, 2, 1, 10, as_int=True)

    def set_simulation_end_and_start(self, simulation_start, last_event):
        last_duration = last_event.get("duration", self.times_dic[last_event["activity"]])
        simulation_end = last_event["timestamp"] + timedelta(seconds=last_duration) + timedelta(days=1)
        simulation_start = simulation_end
        return simulation_start, simulation_end

    def set_event_timestamp(self, current_activity, prev_event, env, parallel_executions, parallel_timestamps, simulation_start):
        current_duration = self.times_dic[current_activity]
        if prev_event is None:
            timestamp = simulation_start + timedelta(seconds=env.now)
        else:
            prev_duration = prev_event.get("duration", self.times_dic[prev_event["activity"]])
            timestamp = prev_event["timestamp"] + timedelta(seconds=prev_duration)
        timeout = current_duration
        return timestamp, parallel_executions, parallel_timestamps, timeout

    def copy_event_state(self, current_event, prev_event):
        for field in STATE_FIELDS:
            if field in prev_event:
                current_event[field] = prev_event[field]
        current_event["outcome"] = np.nan

    def set_event_variables(self, current_event, prev_event, action_to_be_taken=None, intervention_info=None):
        current_activity = current_event["activity"]

        if current_activity == "initiate_case":
            current_event["bike_value"] = self.sample_bike_value()
            current_event["repair_severity"] = self.sample_repair_severity()
            current_event["quality_uncertainty"] = self.sample_quality_uncertainty()
            current_event["customer_patience"] = self.sample_customer_patience()
            current_event["customer_friendliness"] = self.sample_customer_friendliness()
            current_event["outcome"] = np.nan
        else:
            self.copy_event_state(current_event, prev_event)

            if current_activity == "start_priority" or current_activity == "start_standard":
                current_event["process_type"] = "priority" if current_activity == "start_priority" else "standard"

            elif current_activity == "wait_for_employee":
                current_event["employee_competence"] = self.sample_employee_competence()

            elif current_activity == "choose_employee":
                current_event["employee_competence"] = self.workshop_policy.choose_employee(prev_event)
                self.set_time_choose_employee(current_event["employee_competence"])

            elif current_activity == "use_spare_parts":
                current_event["material_quality"] = self.sample_material_quality()

            elif current_activity == "choose_material":
                current_event["material_quality"] = self.workshop_policy.choose_material(prev_event)
                self.set_time_choose_material(current_event["material_quality"])

            elif current_activity == "repair_priority" or current_activity == "repair_standard":
                self.set_time_repair(current_activity, current_event["repair_severity"], current_event["employee_competence"])

            elif current_activity == "conduct_qc":
                current_event["quality_uncertainty"] = max(0.05, 0.4 * prev_event["quality_uncertainty"])

            elif current_activity == "customer_acceptance" or current_activity == "customer_refusal":
                current_event["outcome"] = self.calc_outcome(current_event)

        current_event["duration"] = self.times_dic[current_activity]

        if action_to_be_taken is not None and intervention_info is not None:
            for activity_index, activities in enumerate(intervention_info["activities"]):
                if current_activity in activities:
                    current_event[intervention_info["column"][activity_index]] = action_to_be_taken
                    break

        return current_event

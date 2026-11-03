import math
from pm4py.objects.petri_net.utils.petri_utils import get_transition_by_name


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def normalize(z):
    return (z - 1) / 9


def completed_time(prev_event):
    return prev_event["elapsed_time"] + prev_event.get("duration", 0) / 86400


def required_quality(event):
    # quality the repair needs, in [0, 1]
    return 0.6 * normalize(event["repair_severity"]) + 0.4 * normalize(event["bike_value"])


def produced_quality(event):
    # quality the chosen employee and material deliver, in [0, 1]
    return 0.6 * normalize(event["employee_competence"]) + 0.4 * normalize(event["material_quality"])


class WorkshopPolicy():
    def __init__(self, random_obj):
        self.random_obj = random_obj
        self.take_employee_tolerance = 0.1
        self.qc_uncertainty_threshold = 0.5

    def procedure_standard_probability(self, prev_event):
        V_bar = normalize(prev_event["bike_value"])
        S_bar = normalize(prev_event["repair_severity"])
        U = prev_event["quality_uncertainty"]
        return sigmoid(-2 + 1.2 * V_bar + 1.8 * S_bar + 1.0 * U)

    def choose_procedure(self, net, prev_event):
        standard_prob = self.procedure_standard_probability(prev_event)
        return self.random_obj.choices(
            [get_transition_by_name(net, "start_priority"), get_transition_by_name(net, "start_standard")],
            weights=[standard_prob, 1 - standard_prob], k=1)[0]

    def take_employee_probability(self, prev_event):
        # Deterministic workshop rule (as the threshold policies in SimBank): take the available employee
        # if the competence is at most take_employee_tolerance below what the repair requires
        E_bar = normalize(prev_event["employee_competence"])
        return 1.0 if E_bar >= required_quality(prev_event) - self.take_employee_tolerance else 0.0

    def wait_or_take_employee(self, net, prev_event):
        take_prob = self.take_employee_probability(prev_event)
        return self.random_obj.choices(
            [get_transition_by_name(net, "ghost_reject_employee"), get_transition_by_name(net, "ghost_take_employee")],
            weights=[take_prob, 1 - take_prob], k=1)[0]

    def choose_employee(self, prev_event):
        # Standard procedure: the employee matches what the repair requires
        return 1 + round(9 * required_quality(prev_event))

    def choose_material(self, prev_event):
        # Standard procedure: the material matches what the repair requires
        return 1 + round(9 * required_quality(prev_event))

    def conduct_qc_probability(self, prev_event):
        # Deterministic workshop rule: quality control as long as the uncertainty is above qc_uncertainty_threshold
        return 1.0 if prev_event["quality_uncertainty"] > self.qc_uncertainty_threshold else 0.0

    def conduct_qc_or_ship(self, net, prev_event):
        qc_prob = self.conduct_qc_probability(prev_event)
        return self.random_obj.choices(
            [get_transition_by_name(net, "shipping"), get_transition_by_name(net, "conduct_qc")],
            weights=[qc_prob, 1 - qc_prob], k=1)[0]

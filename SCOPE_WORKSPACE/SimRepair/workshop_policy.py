import math
from pm4py.objects.petri_net.utils.petri_utils import get_transition_by_name


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def normalize(z):
    return (z - 1) / 9


def completed_time(prev_event):
    return prev_event["elapsed_time"] + prev_event.get("duration", 0) / 86400


class WorkshopPolicy():
    def __init__(self, random_obj):
        self.random_obj = random_obj

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
        E_bar = normalize(prev_event["employee_competence"])
        S_bar = normalize(prev_event["repair_severity"])
        V_bar = normalize(prev_event["bike_value"])
        return sigmoid(-0.5 + 2.5 * E_bar - 1.2 * S_bar - 0.8 * V_bar)

    def wait_or_take_employee(self, net, prev_event):
        take_prob = self.take_employee_probability(prev_event)
        return self.random_obj.choices(
            [get_transition_by_name(net, "ghost_reject_employee"), get_transition_by_name(net, "ghost_take_employee")],
            weights=[take_prob, 1 - take_prob], k=1)[0]

    def choose_employee(self, prev_event):
        S_bar = normalize(prev_event["repair_severity"])
        V_bar = normalize(prev_event["bike_value"])
        U = prev_event["quality_uncertainty"]
        p_E = sigmoid(-1.5 + 1.8 * S_bar + 1.2 * V_bar + 0.8 * U)
        return 1 + round(9 * p_E)

    def choose_material(self, prev_event):
        S_bar = normalize(prev_event["repair_severity"])
        V_bar = normalize(prev_event["bike_value"])
        E_bar = normalize(prev_event["employee_competence"])
        p_M = sigmoid(-1.2 + 1.5 * S_bar + 1.1 * V_bar - 0.7 * E_bar)
        return 1 + round(9 * p_M)

    def conduct_qc_probability(self, prev_event):
        U = prev_event["quality_uncertainty"]
        S_bar = normalize(prev_event["repair_severity"])
        V_bar = normalize(prev_event["bike_value"])
        T = completed_time(prev_event)
        return sigmoid(-1.5 + 2.5 * U + 0.7 * S_bar + 0.4 * V_bar - 0.05 * T)

    def conduct_qc_or_ship(self, net, prev_event):
        qc_prob = self.conduct_qc_probability(prev_event)
        return self.random_obj.choices(
            [get_transition_by_name(net, "shipping"), get_transition_by_name(net, "conduct_qc")],
            weights=[qc_prob, 1 - qc_prob], k=1)[0]

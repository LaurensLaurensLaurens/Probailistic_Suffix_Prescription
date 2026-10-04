import math
from pm4py.objects.petri_net.utils.petri_utils import get_transition_by_name


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def normalize(z):
    return (z - 1) / 9


def completed_time(prev_event):
    return prev_event["elapsed_time"] + prev_event.get("duration", 0) / 86400


class Environment():
    def __init__(self, random_obj):
        self.random_obj = random_obj

    def bad_repair(self, net, prev_event, good_activity, bad_activity):
        S_bar = normalize(prev_event["repair_severity"])
        V_bar = normalize(prev_event["bike_value"])
        E_bar = normalize(prev_event["employee_competence"])
        M_bar = normalize(prev_event["material_quality"])
        q_required = 0.6 * S_bar + 0.4 * V_bar
        q_produced = 0.6 * E_bar + 0.4 * M_bar
        bad_repair_prob = sigmoid(-1 + 5 * (q_required - q_produced))
        return self.random_obj.choices(
            [get_transition_by_name(net, bad_activity), get_transition_by_name(net, good_activity)],
            weights=[1 - bad_repair_prob, bad_repair_prob], k=1)[0]

    def material_forgotten(self, net, prev_event):
        S_bar = normalize(prev_event["repair_severity"])
        E_bar = normalize(prev_event["employee_competence"])
        forgotten_prob = sigmoid(-1 + 2.5 * S_bar - 3 * E_bar)
        return self.random_obj.choices(
            [get_transition_by_name(net, "ghost_material_ordered"), get_transition_by_name(net, "ghost_material_forgotten")],
            weights=[forgotten_prob, 1 - forgotten_prob], k=1)[0]

    def customer_accepts(self, net, prev_event):
        P_bar = normalize(prev_event["customer_patience"])
        F_bar = normalize(prev_event["customer_friendliness"])
        T = completed_time(prev_event)
        U_final = prev_event["quality_uncertainty"]
        accept_prob = sigmoid(1.5 + 2 * P_bar - 0.12 * T - 2.5 * U_final + 0.5 * F_bar)
        return self.random_obj.choices(
            [get_transition_by_name(net, "customer_refusal"), get_transition_by_name(net, "customer_acceptance")],
            weights=[accept_prob, 1 - accept_prob], k=1)[0]

from pm4py.objects.petri_net.utils.petri_utils import get_transition_by_name
from SimRepair.workshop_policy import sigmoid, normalize, completed_time, required_quality, produced_quality


class Environment():
    def __init__(self, random_obj):
        self.random_obj = random_obj
        #CUSTOMER params (days)
        self.min_time = 6 # fastest possible repair
        self.deadline_base = 10
        self.deadline_per_patience = 2
        self.unfavorable_scale = 0.3 # unfavorable conditions: probability in [0, 0.5 * unfavorable_scale)
        self.favorable_base = 0.85 # favorable conditions: probability in [favorable_base, 1]

    def bad_repair(self, net, prev_event, good_activity, bad_activity):
        # Steep in the quality gap: rare (12%) if employee and material match the repair, almost sure if they are clearly too weak
        bad_repair_prob = sigmoid(-2 + 12 * (required_quality(prev_event) - produced_quality(prev_event)))
        return self.random_obj.choices(
            [get_transition_by_name(net, bad_activity), get_transition_by_name(net, good_activity)],
            weights=[1 - bad_repair_prob, bad_repair_prob], k=1)[0]

    def material_forgotten(self, net, prev_event):
        S_bar = normalize(prev_event["repair_severity"])
        E_bar = normalize(prev_event["employee_competence"])
        forgotten_prob = sigmoid(-2 + 6 * (S_bar - E_bar))
        return self.random_obj.choices(
            [get_transition_by_name(net, "ghost_material_ordered"), get_transition_by_name(net, "ghost_material_forgotten")],
            weights=[forgotten_prob, 1 - forgotten_prob], k=1)[0]

    def customer_acceptance_probability(self, prev_event):
        # Same structure as the customer decision in SimBank: adjustments around 0.5, then mapped to
        # a low range for unfavorable and a high range for favorable conditions, and a hard refusal after the deadline
        F_bar = normalize(prev_event["customer_friendliness"])
        T = completed_time(prev_event)
        U_final = prev_event["quality_uncertainty"]
        # The patience of the customer sets how many days the repair may take
        deadline = self.deadline_base + self.deadline_per_patience * prev_event["customer_patience"]
        if T >= deadline:
            return 0.0001

        time_factor = 1 - 2 * (T - self.min_time) / (deadline - self.min_time) # 1 = fastest possible, -1 = at the deadline
        time_factor = min(1, time_factor)
        quality_factor = 1 - 2 * U_final
        friendliness_factor = 2 * F_bar - 1
        prob = 0.5 + 0.25 * time_factor + 0.2 * quality_factor + 0.05 * friendliness_factor

        if prob < 0.5:
            final_prob = prob * self.unfavorable_scale
        else:
            final_prob = self.favorable_base + (prob - 0.5) * 2 * (1 - self.favorable_base)
        return min(0.9999, max(0.0001, final_prob))

    def customer_accepts(self, net, prev_event):
        accept_prob = self.customer_acceptance_probability(prev_event)
        return self.random_obj.choices(
            [get_transition_by_name(net, "customer_refusal"), get_transition_by_name(net, "customer_acceptance")],
            weights=[accept_prob, 1 - accept_prob], k=1)[0]

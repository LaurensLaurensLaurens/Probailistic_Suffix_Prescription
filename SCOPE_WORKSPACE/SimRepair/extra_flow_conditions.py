import random
from pm4py.objects.petri_net.utils.petri_utils import get_transition_by_name
from SimRepair.workshop_policy import WorkshopPolicy
from SimRepair.environment import Environment


class ExtraFlowConditioner():
    def __init__(self, random_obj=random.Random()):
        self.random_obj = random_obj
        self.workshop_policy = WorkshopPolicy(random_obj)
        self.environment = Environment(random_obj)

    def filter_enabled_trans(self, net, marking, control_flow_enabled_trans, trace, policies_info, intervention_info, action_to_be_taken=None, ignore_intervention_policy=False):
        prev_event = trace[-1] if len(trace) > 0 else None
        marking = str(marking)

        if intervention_info["RCT"] or ignore_intervention_policy:
            policies_to_ignore = intervention_info["name"]
        else:
            policies_to_ignore = []

        if marking == "['source:1']":
            all_enabled_trans = control_flow_enabled_trans

        elif marking == "['p_proc:1']":
            activity_to_delete = self.workshop_policy.choose_procedure(net, prev_event)
            all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [activity_to_delete]]
            if "choose_procedure" in policies_to_ignore:
                all_enabled_trans = control_flow_enabled_trans

        elif marking == "['p_emp_accept:1']":
            activity_to_delete = self.workshop_policy.wait_or_take_employee(net, prev_event)
            all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [activity_to_delete]]
            if "take_employee" in policies_to_ignore:
                all_enabled_trans = control_flow_enabled_trans

            employee_wait_count = 0
            for event in reversed(trace):
                if event["activity"] == "wait_for_employee":
                    employee_wait_count += 1
                elif event["activity"] in ["start_priority", "prepare_rerepair_priority"]:
                    break
            if employee_wait_count >= policies_info["take_employee"]["max_employee_waits"]:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "ghost_reject_employee")]]

        elif marking == "['p_x1:1']":
            activity_to_delete = self.environment.bad_repair(net, prev_event, "ghost_prior_good", "prepare_rerepair_priority")
            all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [activity_to_delete]]
            rerepair_count = sum(1 for event in trace if event["activity"] == "prepare_rerepair_priority")
            if rerepair_count >= policies_info["max_rerepairs"]:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "prepare_rerepair_priority")]]

        elif marking == "['p_x2:1']":
            activity_to_delete = self.environment.material_forgotten(net, prev_event)
            all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [activity_to_delete]]

        elif marking == "['p_x3:1']":
            activity_to_delete = self.environment.bad_repair(net, prev_event, "ghost_stan_good", "prepare_rerepair_standard")
            all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [activity_to_delete]]
            rerepair_count = sum(1 for event in trace if event["activity"] == "prepare_rerepair_standard")
            if rerepair_count >= policies_info["max_rerepairs"]:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "prepare_rerepair_standard")]]

        elif marking == "['p_qc_decision:1']":
            activity_to_delete = self.workshop_policy.conduct_qc_or_ship(net, prev_event)
            all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [activity_to_delete]]
            if "conduct_qc" in policies_to_ignore:
                all_enabled_trans = control_flow_enabled_trans

            qc_count = sum(1 for event in trace if event["activity"] == "conduct_qc")
            if qc_count >= policies_info["conduct_qc"]["max_qc"]:
                all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [get_transition_by_name(net, "conduct_qc")]]

        elif marking == "['p_accept:1']":
            activity_to_delete = self.environment.customer_accepts(net, prev_event)
            all_enabled_trans = [act for act in control_flow_enabled_trans if act not in [activity_to_delete]]

        else:
            all_enabled_trans = control_flow_enabled_trans

        if action_to_be_taken is not None and action_to_be_taken != "do_nothing":
            all_enabled_trans = [get_transition_by_name(net, action_to_be_taken)]

        return all_enabled_trans

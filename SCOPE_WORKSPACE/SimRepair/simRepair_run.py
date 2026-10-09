from itertools import product
import math
from copy import deepcopy
from datetime import datetime
import os
import sys
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SimRepair.simulation import PresProcessGenerator
from SimRepair.confounding_level import set_delta
from SimRepair.workshop_policy import WorkshopPolicy
from src.utils.mini_tools import save_data, load_data
import pandas as pd

# Pro Intervention die Einstellungen, die in SimBPIC17 fuer alle Stages gleich sind.
# In SimRepair ist jede Stage eine andere Intervention, deshalb stehen sie hier einzeln.
INTERVENTIONS = {
    "choose_procedure": {
        "data_impact": "direct",
        "actions": ["start_standard", "start_priority"], #If binary, last action is the 'treatment' action
        "action_width": 2,
        "action_depth": 1,
        "activities": ["start_standard", "start_priority"],
        "column": "activity",
        "start_control_activity": ["initiate_case"],
        "end_control_activity": ["initiate_case"],
    },
    "conduct_qc": {
        "data_impact": "direct",
        "actions": ["shipping", "conduct_qc"], #If binary, last action is the 'treatment' action
        "action_width": 2,
        "action_depth": 4, # max_qc + 1 decision points
        "activities": ["shipping", "conduct_qc"],
        "column": "activity",
        "start_control_activity": ["repair_priority", "repair_standard", "improve_rework"],
        "end_control_activity": ["repair_priority", "repair_standard", "improve_rework"],
    },
    "choose_employee": {
        "data_impact": "indirect",
        "actions": list(range(1, 11)),
        "action_width": 10,
        "action_depth": 1,
        "activities": ["choose_employee"],
        "column": "employee_competence",
        "start_control_activity": [],
        "end_control_activity": [],
    },
}
# Schluessel, die in dataset_params pro Stage als Liste vorliegen
STAGE_KEYS = ["name", "data_impact", "actions", "action_width", "action_depth", "activities", "column", "start_control_activity", "end_control_activity", "len"]


# Diese Funktion simuliert SimRepair-Trainings- und Tuningdaten.
def generate_training_and_tuning_simrepair(train_size, delta, n_stages=1, confounding_type='case', max_attempts=20, test_size=10000, intervention_names=("choose_procedure",)):
    # Gucken, ob Stages mit Anzahl der Interventionen übereinstimmt.
    intervention_names = list(intervention_names)
    if len(intervention_names) != n_stages:
        raise ValueError(f"n_stages={n_stages}, aber {len(intervention_names)} Interventionen angegeben: {intervention_names}")
    if confounding_type != 'case':
        # Der Petri-Netz-Simulator kennt nur komplett normale oder komplett randomisierte Laeufe.
        raise NotImplementedError("SimRepair unterstuetzt nur confounding_type='case'.")

    dataset_params = {}
    #general
    dataset_params["train_size"] = train_size
    dataset_params["test_size"] = test_size
    dataset_params["val_share"] = .5
    dataset_params["train_val_size"] = test_size
    dataset_params["test_val_size"] = min(int(dataset_params["val_share"] * dataset_params["test_size"]), 1000)
    dataset_params["simulation_start"] = datetime(2024, 3, 20, 8, 0)
    dataset_params["random_seed_train"] = 82*82
    dataset_params["random_seed_test"] = 130*130

    #process
    dataset_params["log_cols"] = ["case_nr", "activity", "timestamp", "elapsed_time", "duration", "bike_value", "repair_severity", "quality_uncertainty", "process_type", "employee_competence", "material_quality", "customer_patience", "customer_friendliness", "outcome"]
    dataset_params["scale_cols"] = ["elapsed_time", "bike_value", "repair_severity", "quality_uncertainty", "employee_competence", "material_quality", "outcome"]
    dataset_params["case_cols"] = ["bike_value", "repair_severity"]
    dataset_params["event_cols"] = ["activity", "elapsed_time", "quality_uncertainty", "process_type", "employee_competence", "material_quality"]
    dataset_params["last_state_cols"] = ["elapsed_time"]
    dataset_params["cat_cols"] = ["activity", "process_type"]

    dataset_params["intervention_info"] = {}
    dataset_params["intervention_info"]["name"] = intervention_names
    for key in ["data_impact", "actions", "action_width", "action_depth", "activities", "column", "start_control_activity", "end_control_activity"]:
        dataset_params["intervention_info"][key] = [deepcopy(INTERVENTIONS[name][key]) for name in intervention_names]
    dataset_params["intervention_info"]["retain_method"] = "precise"

    # Combinations
    dataset_params["intervention_info"]["action_combinations"] = list(product(*dataset_params["intervention_info"]["actions"]))
    dataset_params["intervention_info"]["action_width_combinations"] = math.prod(dataset_params["intervention_info"]["action_width"])
    dataset_params["intervention_info"]["action_depth_combinations"] = math.prod(dataset_params["intervention_info"]["action_depth"])

    dataset_params["intervention_info"]["len"] = [action_width if action_width > 2 else 1 for action_width in dataset_params["intervention_info"]["action_width"]]
    dataset_params["intervention_info"]["RCT"] = False
    dataset_params["filename"] = "repair_log_" +  str(dataset_params["intervention_info"]["name"])

    #policy
    dataset_params["policies_info"] = {}
    dataset_params["policies_info"]["general"] = "real"
    dataset_params["policies_info"]["take_employee"] = {"max_employee_waits": 3}
    dataset_params["policies_info"]["conduct_qc"] = {"max_qc": 3}
    dataset_params["policies_info"]["max_rerepairs"] = 3

    # Generate fully workshop-policy data (delta=1) and fully RCT data (delta=0),
    # then combine them using set_delta — same approach as the 'case' branch of SimBPIC17.
    # Random
    gen_bank = PresProcessGenerator(dataset_params, dataset_params["random_seed_train"])
    data_bank = gen_bank.run_simulation_normal(dataset_params["train_size"])

    # RCT 
    dataset_params_RCT = deepcopy(dataset_params)
    dataset_params_RCT["intervention_info"]["RCT"] = True
    dataset_params_RCT["random_seed_train"] = dataset_params["random_seed_train"]*10
    dataset_params_RCT["simulation_start"] = deepcopy(gen_bank.simulation_end)
    gen_rct = PresProcessGenerator(dataset_params_RCT, dataset_params_RCT["random_seed_train"])
    # Der normal modus simuliert viele Fälle und trifft entscheidngen selbst
    # Der Inferenz-Modus simuliert einen einzigen Fall, hält an jedem Eintscheidungspunkt and und läasst aurufer aktionen wählen
    data_rct = gen_rct.run_simulation_normal(dataset_params_RCT["train_size"])

    # Die Hilfsfunktion prüft, ob jede mögliche Intervention in mindestens zwei verschiedenen Fällen vorkommt.
    def all_actions_present(train_df, intervention_actions, intervention_columns):
        for actions, col in zip(intervention_actions, intervention_columns):
            for action in actions:
                unique_cases = train_df.loc[train_df[col] == action, "case_nr"].nunique()
                if unique_cases < 2:
                    return False
        return True

    # Dieser Block mischt Werkstatt- und RCT-Fälle wiederholt neu, bis alle Aktionen ausreichend vertreten sind.
    attempt = 0
    seed_offset = 0
    while attempt < max_attempts:
        current_seed = dataset_params["random_seed_train"] + seed_offset
        data = set_delta(data=data_bank, data_RCT=data_rct, delta=delta, seed=current_seed)
        if all_actions_present(data, dataset_params["intervention_info"]["actions"], dataset_params["intervention_info"]["column"]):
            break
        else:
            attempt += 1
            seed_offset += 1
            print(f"Attempt {attempt}: Some actions missing, retrying with new seed {current_seed}")

    if attempt == max_attempts:
        print("Warning: Could not ensure all actions appear at least twice after max attempts.")

    # Dieser Block zerlegt die gemeinsame Konfiguration in eine Konfiguration pro Entscheidungsstufe.
    # Anders als in SimBPIC17 werden nur die Stage-Schluessel zerlegt (nicht jede Liste),
    # und die Kombinationen werden pro Stage neu gesetzt.
    dataset_params_list = []
    for intervention in range(len(dataset_params["intervention_info"]["action_width"])):
        params = deepcopy(dataset_params)
        for key in STAGE_KEYS:
            params["intervention_info"][key] = params["intervention_info"][key][intervention]
        stage_actions = params["intervention_info"]["actions"]
        params["intervention_info"]["action_combinations"] = [(action,) for action in stage_actions]
        params["intervention_info"]["action_width_combinations"] = params["intervention_info"]["action_width"]
        params["intervention_info"]["action_depth_combinations"] = params["intervention_info"]["action_depth"]
        params["intervention_info"]["flat_activities"] = params["intervention_info"]["activities"]
        dataset_params_list.append(params)

    return dataset_params, dataset_params_list, data

def generate_eval_simrepair(args, dataset_params):
    eval_dfs = {}
    # Alles nur bzgl Location der Daten
    eval_dir = os.path.join(os.getcwd(), "data", "SimRepair", str(args.train_size), str(int(100 * args.delta)), "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # us "action_width": is for example [2, 3], so you have 6 combinations of actions, and for each of them you need to evaluate the policy
    action_combos = list(product(*[range(width) for width in dataset_params["intervention_info"]["action_width"]]))
    for action_combo in action_combos:
        if (args.already_eval_generated or ('fixed' in args.already_eval_generated_list)) and os.path.exists(os.path.join(eval_dir, "fixed_" + str(action_combo) + "_performance.pkl")):
            # just load the data
            performance = load_data(os.path.join(eval_dir, "fixed_" + str(action_combo) + "_performance"))
            outcome_df = load_data(os.path.join(eval_dir, "fixed_" + str(action_combo) + "_outcome_df"))
            test_df = load_data(os.path.join(eval_dir, "fixed_" + str(action_combo) + "_test_df"))
            print('Fixed performance, combo: ', performance, action_combo)
        else:
            print("Evaluating action combo: ", action_combo)
            performance, outcome_df, test_df = generate_one_eval_simrepair(policy="fixed", args=args, dataset_params=dataset_params, action_combo=action_combo)
            print('Fixed performance, combo: ', performance, action_combo)

            save_data(performance, os.path.join(eval_dir, "fixed_" + str(action_combo) + "_performance"))
            save_data(outcome_df, os.path.join(eval_dir, "fixed_" + str(action_combo) + "_outcome_df"))
            save_data(test_df, os.path.join(eval_dir, "fixed_" + str(action_combo) + "_test_df"))

        eval_dfs[str(action_combo)] = {
            "performance": performance,
            "outcome_df": outcome_df,
            "test_df": test_df
        }

    # Dieser Block lädt oder erzeugt die Evaluation der historischen Werkstatt-Policy (Schluessel bleibt "bank").
    if (args.already_eval_generated or ('bank' in args.already_eval_generated_list)) and os.path.exists(os.path.join(eval_dir, "bank_performance.pkl")):
        bank_performance = load_data(os.path.join(eval_dir, "bank_performance"))
        bank_outcome_df = load_data(os.path.join(eval_dir, "bank_outcome_df"))
        bank_test_df = load_data(os.path.join(eval_dir, "bank_test_df"))
        print('Bank performance: ', bank_performance)
    else:
        bank_performance, bank_outcome_df, bank_test_df = generate_one_eval_simrepair(policy="bank", args=args, dataset_params=dataset_params)
        print('Bank performance: ', bank_performance)

        save_data(bank_performance, os.path.join(eval_dir, "bank_performance"))
        save_data(bank_outcome_df, os.path.join(eval_dir, "bank_outcome_df"))
        save_data(bank_test_df, os.path.join(eval_dir, "bank_test_df"))

    eval_dfs["bank"] = {
        "performance": bank_performance,
        "outcome_df": bank_outcome_df,
        "test_df": bank_test_df
    }

    # Generate random policy evaluations
    for iter in range(args.num_iterations):
        if (args.already_eval_generated or ('random' in args.already_eval_generated_list)) and os.path.exists(os.path.join(eval_dir, "random_" + str(iter) + "_performance.pkl")):
            random_performance = load_data(os.path.join(eval_dir, "random_" + str(iter) + "_performance"))
            random_outcome_df = load_data(os.path.join(eval_dir, "random_" + str(iter) + "_outcome_df"))
            random_test_df = load_data(os.path.join(eval_dir, "random_" + str(iter) + "_test_df"))
            print('Random performance: ', random_performance)
        else:
            random_object_for_random_policy = random.Random(dataset_params["random_seed_test"] + 5*iter)
            random_performance, random_outcome_df, random_test_df = generate_one_eval_simrepair(policy="random", args=args, dataset_params=dataset_params, random_object_random_policy=random_object_for_random_policy)
            print('Random performance: ', random_performance)

            save_data(random_performance, os.path.join(eval_dir, "random_" + str(iter) + "_performance"))
            save_data(random_outcome_df, os.path.join(eval_dir, "random_" + str(iter) + "_outcome_df"))
            save_data(random_test_df, os.path.join(eval_dir, "random_" + str(iter) + "_test_df"))

        eval_dfs["random_" + str(iter)] = {
                "performance": random_performance,
                "outcome_df": random_outcome_df,
                "test_df": random_test_df
        }

    # Generate the optimal policy:
    if (args.already_eval_generated or ('optimal' in args.already_eval_generated_list)) and os.path.exists(os.path.join(eval_dir, "optimal_performance.pkl")):
        optimal_performance = load_data(os.path.join(eval_dir, "optimal_performance"))
        optimal_outcome_df = load_data(os.path.join(eval_dir, "optimal_outcome_df"))
        optimal_test_df = load_data(os.path.join(eval_dir, "optimal_test_df"))
        print('Optimal performance: ', optimal_performance)
    else:
        # for every case in the test dfs of all action combo's, grab the case that has the max outcome over the action combo's
        for case_nr in range(args.test_size):
            best_outcome = -float('inf')
            best_case = None
            for action_combo in eval_dfs.keys():
                if action_combo == 'bank' or ('random') in action_combo: continue
                current_case = eval_dfs[action_combo]["test_df"][eval_dfs[action_combo]["test_df"]["case_nr"] == case_nr]
                current_outcome = current_case["outcome"].iloc[-1]
                if current_outcome > best_outcome:
                    best_outcome = current_outcome
                    best_case = current_case
            if case_nr == 0:
                optimal_test_df = best_case
                optimal_performance = best_outcome
                optimal_outcome_df = pd.DataFrame([{"case_nr": case_nr, "outcome": best_outcome}])
            else:
                optimal_test_df = pd.concat([optimal_test_df, best_case], axis=0)
                optimal_performance += best_outcome
                optimal_outcome_df = pd.concat([optimal_outcome_df, pd.DataFrame([{"case_nr": case_nr, "outcome": best_outcome}])], axis=0)

        print('Optimal performance: ', optimal_performance)
        save_data(optimal_performance, os.path.join(eval_dir, "optimal_performance"))
        save_data(optimal_outcome_df, os.path.join(eval_dir, "optimal_outcome_df"))
        save_data(optimal_test_df, os.path.join(eval_dir, "optimal_test_df"))

    eval_dfs["optimal"] = {
        "performance": optimal_performance,
        "outcome_df": optimal_outcome_df,
        "test_df": optimal_test_df
    }

    return eval_dfs

# Die Funktion bewertet genau eine Policy auf neu simulierten Testfällen.
def generate_one_eval_simrepair(policy, args, dataset_params, action_combo=None, random_object_random_policy=None):
    print("Calculate True Performance for policy ", policy)

    # SimBPIC17 simuliert alle Testfaelle in einem Aufruf von run_simulation.
    # Der Petri-Netz-Simulator laeuft dagegen fallweise und haelt an jedem Entscheidungspunkt an.
    case_gen = PresProcessGenerator(dataset_params, seed=dataset_params["random_seed_test"])
    cases = []
    for case_nr in range(args.test_size):
        best_action = 0
        int_index = 0
        prefix_list = case_gen.start_simulation_inference(seed_to_add=case_nr)
        while case_gen.int_points_available:
            if policy == "bank":
                best_action = get_bank_best_action(prefix_list, int_index, dataset_params, case_gen.random_obj)
            elif policy == "random":
                best_action = get_random_best_action(dataset_params, int_index, random_object_random_policy)
            elif policy == "fixed":
                best_action = action_combo[int_index]
                int_index += 1
            prefix_list = case_gen.continue_simulation_inference(best_action)

        full_case = pd.DataFrame(case_gen.end_simulation_inference())
        full_case["case_nr"] = case_nr
        cases.append(full_case)

    # test_df: vollständiges Event-Log der Simulation
    test_df = pd.concat(cases, axis=0)
    # performance: Gesamtwert über alle Fälle (das Outcome steht auf jedem Event des Falls)
    performance = test_df.groupby('case_nr')['outcome'].last().sum()
    # outcome_df: ein Outcome pro Fall
    outcome_df = test_df.groupby('case_nr', as_index=False)['outcome'].last()

    return performance, outcome_df, test_df

def get_bank_best_action(prefix_list, current_int_index, dataset_params, random_obj=None):
    if random_obj is None:
        random_obj = random.Random()
    workshop_policy = WorkshopPolicy(random_obj)
    prefix_without_int = prefix_list[0][0:-1]
    prev_event = prefix_without_int[-1]
    intervention_name = dataset_params["intervention_info"]["name"][current_int_index]

    if intervention_name == "choose_procedure":
        standard_prob = workshop_policy.procedure_standard_probability(prev_event)
        return 0 if random_obj.random() < standard_prob else 1

    if intervention_name == "conduct_qc":
        qc_count = sum(1 for event in prefix_without_int if event["activity"] == "conduct_qc")
        if qc_count >= dataset_params["policies_info"]["conduct_qc"]["max_qc"]:
            return 0
        qc_prob = workshop_policy.conduct_qc_probability(prev_event)
        return 1 if random_obj.random() < qc_prob else 0

    return 0

def get_random_best_action(dataset_params, current_int_index, random_object_random_policy):
    return random_object_random_policy.choice(range(dataset_params["intervention_info"]["action_width"][current_int_index]))

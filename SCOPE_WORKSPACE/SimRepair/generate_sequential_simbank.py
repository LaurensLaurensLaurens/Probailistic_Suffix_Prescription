import math
import pandas as pd
from datetime import datetime
from copy import deepcopy
from itertools import product
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import SimRepair.simulation as simulation
import SimRepair.confounding_level as confounding_level
from src.utils.mini_tools import save_data, load_data
from SimRepair.workshop_policy import WorkshopPolicy


def generate_training_and_tuning(size, delta, max_attempts=20, test_size=10000):
    """
    Generate a sequential SimRepair dataset based on the provided parameters.
    GENERIERT DEN DATENSATZ. Muss über scripts ausgeführt werden
    
    
    Args:
        dataset_params (dict): Parameters for dataset generation.
        path (str): Path to save the generated dataset.
    """

    #DATASET parameters
    dataset_params = {}
    #general
    dataset_params["train_size"] = size
    dataset_params["test_size"] = test_size
    dataset_params["val_share"] = .5
    dataset_params["train_val_size"] = test_size
    dataset_params["test_val_size"] = min(int(dataset_params["val_share"] * dataset_params["test_size"]), 1000)
    dataset_params["simulation_start"] = datetime(2024, 3, 20, 8, 0)
    dataset_params["random_seed_train"] = 82*82
    dataset_params["random_seed_test"] = 130*130
    #process
    dataset_params["log_cols"] = ["case_nr", "activity", "timestamp", "elapsed_time", "duration", "bike_value", "repair_severity", "quality_uncertainty", "process_type", "employee_competence", "material_quality", "customer_patience", "customer_friendliness", "outcome"]
    dataset_params["case_cols"] = ["bike_value", "repair_severity"]
    dataset_params["event_cols"] = ["activity", "elapsed_time", "quality_uncertainty", "process_type", "employee_competence", "material_quality"]
    dataset_params["cat_cols"] = ["activity", "process_type"]
    dataset_params["scale_cols"] = ["elapsed_time", "bike_value", "repair_severity", "quality_uncertainty", "employee_competence", "material_quality", "outcome"]
    dataset_params["last_state_cols"] = ["elapsed_time"]

    #intervention
    dataset_params["intervention_info"] = {}

    dataset_params["intervention_info"]["name"] = ["choose_procedure"]
    
    if dataset_params["intervention_info"]["name"] == ["choose_procedure"]:
        dataset_params["intervention_info"]["data_impact"] = ["direct"]
        dataset_params["intervention_info"]["actions"] = [["start_standard", "start_priority"]] #If binary, last action is the 'treatment' action
        dataset_params["intervention_info"]["action_width"] = [2]
        dataset_params["intervention_info"]["action_depth"] = [1]
        dataset_params["intervention_info"]["activities"] = [["start_standard", "start_priority"]]
        dataset_params["intervention_info"]["column"] = ["activity"]
        dataset_params["intervention_info"]["start_control_activity"] = [["initiate_case"]]
        dataset_params["intervention_info"]["end_control_activity"] = [["initiate_case"]]
    elif dataset_params["intervention_info"]["name"] == ["conduct_qc"]:
        dataset_params["intervention_info"]["data_impact"] = ["direct"]
        dataset_params["intervention_info"]["actions"] = [["shipping", "conduct_qc"]] #If binary, last action is the 'treatment' action
        dataset_params["intervention_info"]["action_width"] = [2]
        dataset_params["intervention_info"]["action_depth"] = [4] # max_qc + 1 decision points
        dataset_params["intervention_info"]["activities"] = [["shipping", "conduct_qc"]]
        dataset_params["intervention_info"]["column"] = ["activity"]
        dataset_params["intervention_info"]["start_control_activity"] = [["repair_priority", "repair_standard", "improve_rework"]]
        dataset_params["intervention_info"]["end_control_activity"] = [["repair_priority", "repair_standard", "improve_rework"]]
    elif dataset_params["intervention_info"]["name"] == ["choose_procedure", "conduct_qc"]:
        dataset_params["intervention_info"]["data_impact"] = ["direct", "direct"]
        dataset_params["intervention_info"]["actions"] = [["start_standard", "start_priority"], ["shipping", "conduct_qc"]]
        dataset_params["intervention_info"]["action_width"] = [2, 2]
        dataset_params["intervention_info"]["action_depth"] = [1, 4]
        dataset_params["intervention_info"]["activities"] = [["start_standard", "start_priority"], ["shipping", "conduct_qc"]]
        dataset_params["intervention_info"]["column"] = ["activity", "activity"]
        dataset_params["intervention_info"]["start_control_activity"] = [["initiate_case"], ["repair_priority", "repair_standard", "improve_rework"]]
        dataset_params["intervention_info"]["end_control_activity"] = [["initiate_case"], ["repair_priority", "repair_standard", "improve_rework"]]
    elif dataset_params["intervention_info"]["name"] == ["choose_procedure", "choose_employee"]:
        dataset_params["intervention_info"]["data_impact"] = ["direct", "indirect"]
        dataset_params["intervention_info"]["actions"] = [["start_standard", "start_priority"], list(range(1, 11))]
        dataset_params["intervention_info"]["action_width"] = [2, 10]
        dataset_params["intervention_info"]["action_depth"] = [1, 1]
        dataset_params["intervention_info"]["activities"] = [["start_standard", "start_priority"], ["choose_employee"]]
        dataset_params["intervention_info"]["column"] = ["activity", "employee_competence"]
        dataset_params["intervention_info"]["start_control_activity"] = [["initiate_case"], []]
        dataset_params["intervention_info"]["end_control_activity"] = [["initiate_case"], []]

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

    # Initiate simulation
    offline_gen_normal = simulation.PresProcessGenerator(dataset_params, dataset_params["random_seed_train"])

    # Generate training data (bank policy)
    train_normal = offline_gen_normal.run_simulation_normal(dataset_params["train_size"])

    # Generate RCT data (randomly chosen intervention actions)
    dataset_params_RCT = deepcopy(dataset_params)
    dataset_params_RCT["intervention_info"]["RCT"] = True
    dataset_params_RCT["random_seed_train"] = dataset_params["random_seed_train"]*10
    dataset_params_RCT["simulation_start"] = deepcopy(offline_gen_normal.simulation_end)

    # Initiate simulation
    offline_gen_RCT = simulation.PresProcessGenerator(dataset_params_RCT, dataset_params_RCT["random_seed_train"])

    # Generate training data
    train_RCT = offline_gen_RCT.run_simulation_normal(dataset_params_RCT["train_size"])

    attempt = 0
    seed_offset = 0

    def all_actions_present(train_df, intervention_actions, intervention_columns):
        """
        Check if each action in intervention_actions appears in at least 2 unique cases.
        """
        for actions, col in zip(intervention_actions, intervention_columns):
            for action in actions:
                # Count unique case_nr where this action occurs
                unique_cases = train_df.loc[train_df[col] == action, "case_nr"].nunique()
                if unique_cases < 2:
                    return False
        return True

    while attempt < max_attempts:
        # Pass a new seed each attempt to set_delta
        current_seed = dataset_params["random_seed_train"] + seed_offset
        train = confounding_level.set_delta(data=train_normal, data_RCT=train_RCT, delta=delta, seed=current_seed)

        if all_actions_present(train, dataset_params["intervention_info"]["actions"], dataset_params["intervention_info"]["column"]):
            break  # all actions appear at least twice
        else:
            attempt += 1
            seed_offset += 1
            print(f"Attempt {attempt}: Some actions missing, retrying with new seed {current_seed}")

    if attempt == max_attempts:
        print("Warning: Could not ensure all actions appear at least twice after max attempts.")

    dataset_params_list = []
    for intervention in range(len(dataset_params["intervention_info"]["action_width"])):
        params = deepcopy(dataset_params)
        for key, value in params["intervention_info"].items():
            if key in {
                "name", "data_impact", "actions", "action_width", "action_depth",
                "activities", "column", "start_control_activity",
                "end_control_activity", "len"
            }:
                params["intervention_info"][key] = value[intervention]
        stage_actions = params["intervention_info"]["actions"]
        params["intervention_info"]["action_combinations"] = [(action,) for action in stage_actions]
        params["intervention_info"]["action_width_combinations"] = params["intervention_info"]["action_width"]
        params["intervention_info"]["action_depth_combinations"] = params["intervention_info"]["action_depth"]
        params["intervention_info"]["flat_activities"] = params["intervention_info"]["activities"]
        dataset_params_list.append(params)

    return dataset_params, dataset_params_list, train

def generate_eval(args, dataset_params):
    eval_dfs = {}
    eval_data_folder = os.path.join(
        os.getcwd(), "data", "SimRepair", str(args.train_size),
        str(int(100 * args.delta)), "eval"
    )
    os.makedirs(eval_data_folder, exist_ok=True)

    # us "action_width": is for example [2, 3], so you have 6 combinations of actions, and for each of them you need to evaluate the policy
    action_combos = list(product(*[range(width) for width in dataset_params["intervention_info"]["action_width"]]))
    for action_combo in action_combos:
        if args.already_eval_generated and os.path.exists(os.path.join(eval_data_folder, "fixed_" + str(action_combo) + "_performance.pkl")):
            # just load the data
            performance = load_data(os.path.join(eval_data_folder, "fixed_" + str(action_combo) + "_performance"))
            outcome_df = load_data(os.path.join(eval_data_folder, "fixed_" + str(action_combo) + "_outcome_df"))
            test_df = load_data(os.path.join(eval_data_folder, "fixed_" + str(action_combo) + "_test_df"))
        else:
            print("Evaluating action combo: ", action_combo)
            performance, outcome_df, test_df = generate_one_eval(policy="fixed", args=args, dataset_params=dataset_params, action_combo=action_combo)
            eval_dfs[str(action_combo)] = {
                "performance": performance,
                "outcome_df": outcome_df,
                "test_df": test_df
            }

            save_data(performance, os.path.join(eval_data_folder, "fixed_" + str(action_combo) + "_performance"))
            save_data(outcome_df, os.path.join(eval_data_folder, "fixed_" + str(action_combo) + "_outcome_df"))
            save_data(test_df, os.path.join(eval_data_folder, "fixed_" + str(action_combo) + "_test_df"))

        eval_dfs[str(action_combo)] = {
            "performance": performance,
            "outcome_df": outcome_df,
            "test_df": test_df
        }

    if args.already_eval_generated and os.path.exists(os.path.join(eval_data_folder, "bank_performance.pkl")):
        # Load bank policy evaluation
        bank_performance = load_data(os.path.join(eval_data_folder, "bank_performance"))
        bank_outcome_df = load_data(os.path.join(eval_data_folder, "bank_outcome_df"))
        bank_test_df = load_data(os.path.join(eval_data_folder, "bank_test_df"))
    else:
        bank_performance, bank_outcome_df, bank_test_df = generate_one_eval(policy="bank", args=args, dataset_params=dataset_params)
        eval_dfs["bank"] = {
            "performance": bank_performance,
            "outcome_df": bank_outcome_df,
            "test_df": bank_test_df
        }

        save_data(bank_performance, os.path.join(eval_data_folder, "bank_performance"))
        save_data(bank_outcome_df, os.path.join(eval_data_folder, "bank_outcome_df"))
        save_data(bank_test_df, os.path.join(eval_data_folder, "bank_test_df"))

    eval_dfs["bank"] = {
        "performance": bank_performance,
        "outcome_df": bank_outcome_df,
        "test_df": bank_test_df
    }

    # Generate random policy evaluations
    for iter in range(args.num_iterations):
        if args.already_eval_generated and os.path.exists(os.path.join(eval_data_folder, "random_" + str(iter) + "_performance.pkl")):
            random_performance = load_data(os.path.join(eval_data_folder, "random_" + str(iter) + "_performance"))
            random_outcome_df = load_data(os.path.join(eval_data_folder, "random_" + str(iter) + "_outcome_df"))
            random_test_df = load_data(os.path.join(eval_data_folder, "random_" + str(iter) + "_test_df"))
        else:
            random_object_for_random_policy = random.Random(dataset_params["random_seed_test"] + 5*iter)
            random_performance, random_outcome_df, random_test_df = generate_one_eval(policy="random", args=args, dataset_params=dataset_params, random_object_for_random_policy=random_object_for_random_policy)

            save_data(random_performance, os.path.join(eval_data_folder, "random_" + str(iter) + "_performance"))
            save_data(random_outcome_df, os.path.join(eval_data_folder, "random_" + str(iter) + "_outcome_df"))
            save_data(random_test_df, os.path.join(eval_data_folder, "random_" + str(iter) + "_test_df"))
        
        eval_dfs["random_" + str(iter)] = {
                "performance": random_performance,
                "outcome_df": random_outcome_df,
                "test_df": random_test_df
        }

    # Generate the optimal policy:
    if args.already_eval_generated and os.path.exists(os.path.join(eval_data_folder, "optimal_performance.pkl")):
        optimal_performance = load_data(os.path.join(eval_data_folder, "optimal_performance"))
        optimal_outcome_df = load_data(os.path.join(eval_data_folder, "optimal_outcome_df"))
        optimal_test_df = load_data(os.path.join(eval_data_folder, "optimal_test_df"))
    else:
        # for every case in the test dfs of all action combo's, grab the case that has the max outcome over the action combo's
        for case_nr in range(args.test_size):
            if case_nr % 500 == 0 and case_nr != 0:
                print("Case nr: ", case_nr)
                print("Current optimal performance", optimal_performance)
                print('\n')
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

        eval_dfs["optimal"] = {
            "performance": optimal_performance,
            "outcome_df": optimal_outcome_df,
            "test_df": optimal_test_df
        }
        save_data(optimal_performance, os.path.join(eval_data_folder, "optimal_performance"))
        save_data(optimal_outcome_df, os.path.join(eval_data_folder, "optimal_outcome_df"))
        save_data(optimal_test_df, os.path.join(eval_data_folder, "optimal_test_df"))
    
    eval_dfs["optimal"] = {
        "performance": optimal_performance,
        "outcome_df": optimal_outcome_df,
        "test_df": optimal_test_df
    }

    return eval_dfs

def generate_one_eval(policy, args, dataset_params, action_combo=(0, 0), random_object_for_random_policy=None):
    print("Calculate True Performance for policy ", policy)

    #Init performance metrics
    performance = 0
    outcome_df = pd.DataFrame()
    test_df = pd.DataFrame()

    #Init data generator
    case_gen = simulation.PresProcessGenerator(dataset_params, seed=dataset_params["random_seed_test"])

    #Run
    for case_nr in range(args.test_size):
        if case_nr % 500 == 0 and case_nr != 0:
                print("Case nr: ", case_nr)
                print("Current performance", performance)
                print('\n')
        current_case_outcomes = []
        best_action = 0
        seed_to_add = case_nr
        prefix_list = []
        prefix_list = case_gen.start_simulation_inference(seed_to_add=seed_to_add)
        int_index = 0
        current_timing = 0
        while case_gen.int_points_available:
            if current_timing % 2 == 0:
                if policy == "bank":
                    best_action = get_bank_best_action(prefix_list, int_index, dataset_params, case_gen.random_obj)
                elif policy == "random":
                    best_action = get_random_best_action(dataset_params, int_index, random_object_for_random_policy=random_object_for_random_policy)
                elif policy == "fixed":
                    best_action = action_combo[int_index]
                    if dataset_params["intervention_info"]["name"] != ["time_contact_HQ"]:
                        int_index += 1
                    else:
                        if current_timing > 8:
                            int_index += 1

            # Break if intervention done or in last timing
            prefix_list = case_gen.continue_simulation_inference(best_action)
            if dataset_params["intervention_info"]["name"] == ["time_contact_HQ"]:
                current_timing += 1

        full_case = case_gen.end_simulation_inference()
        full_case = pd.DataFrame(full_case)
        current_case_outcomes.append(full_case["outcome"].iloc[-1])
        
        performance += full_case["outcome"].iloc[-1]
        full_case["case_nr"] = case_nr
        test_df = pd.concat([test_df, full_case], axis=0)

        # add to outcome_df with corresponding case_nr
        current_case_outcomes = pd.DataFrame(current_case_outcomes, columns=["outcome"])
        current_case_outcomes["case_nr"] = case_nr
        outcome_df = pd.concat([outcome_df, current_case_outcomes], axis=0, ignore_index=True)
    
    return performance, outcome_df, test_df

def get_bank_best_action(prefix_list, current_int_index, DATASET_PARAMS, random_obj=None):
    if random_obj is None:
        random_obj = random.Random()
    workshop_policy = WorkshopPolicy(random_obj)
    prefix_without_int = prefix_list[0][0:-1]
    prev_event = prefix_without_int[-1]
    intervention_name = DATASET_PARAMS["intervention_info"]["name"][current_int_index]

    if intervention_name == "choose_procedure":
        standard_prob = workshop_policy.procedure_standard_probability(prev_event)
        return 0 if random_obj.random() < standard_prob else 1

    if intervention_name == "conduct_qc":
        qc_count = sum(1 for event in prefix_without_int if event["activity"] == "conduct_qc")
        if qc_count >= DATASET_PARAMS["policies_info"]["conduct_qc"]["max_qc"]:
            return 0
        qc_prob = workshop_policy.conduct_qc_probability(prev_event)
        return 1 if random_obj.random() < qc_prob else 0

    return 0

def get_random_best_action(DATASET_PARAMS, current_int_index, random_object_for_random_policy):
    random_best_action = random_object_for_random_policy.choice(range(DATASET_PARAMS["intervention_info"]["action_width"][current_int_index]))
    return random_best_action
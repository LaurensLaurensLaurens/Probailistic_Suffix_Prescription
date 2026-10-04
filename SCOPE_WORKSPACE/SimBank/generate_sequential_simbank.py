path = "c:\\Users\\u0166838\\OneDrive - KU Leuven\\Documents\\Doc\\Code\\SCOPE"

import math
import pandas as pd
from datetime import datetime
from copy import deepcopy
from itertools import product
import random

import sys
sys.path.append(path)
import SimBank.simulation as simulation
import SimBank.confounding_level as confounding_level
from src.utils.mini_tools import save_data, load_data
from SimBank.activity_execution import ActivityExecutioner
import os


def generate_training_and_tuning(size, delta, max_attempts=20):
    """
    Generate a sequential SimBank dataset based on the provided parameters.
    
    Args:
        dataset_params (dict): Parameters for dataset generation.
        path (str): Path to save the generated dataset.
    """

    #DATASET parameters
    dataset_params = {}
    #general
    dataset_params["train_size"] = size
    dataset_params["test_size"] = 10000
    dataset_params["val_share"] = .5
    dataset_params["train_val_size"] = 10000
    dataset_params["test_val_size"] = min(int(dataset_params["val_share"] * dataset_params["test_size"]), 1000)
    dataset_params["simulation_start"] = datetime(2024, 3, 20, 8, 0)
    dataset_params["random_seed_train"] = 82*82
    dataset_params["random_seed_test"] = 130*130
    #process
    dataset_params["log_cols"] = ["case_nr", "activity", "timestamp", "elapsed_time", "cum_cost", "est_quality", "unc_quality", "amount", "interest_rate", "discount_factor", "outcome", "quality", "noc", "nor", "min_interest_rate"]
    dataset_params["case_cols"] = ["amount"]
    dataset_params["event_cols"] = ["activity", "elapsed_time", "cum_cost", "est_quality", "unc_quality", "interest_rate", "discount_factor"]
    dataset_params["cat_cols"] = ["activity"]
    dataset_params["scale_cols"] = ["amount", "elapsed_time", "cum_cost", "est_quality", "unc_quality", "interest_rate", "discount_factor", "outcome"]
    dataset_params["last_state_cols"] = ["elapsed_time", "cum_cost"]

    #intervention
    dataset_params["intervention_info"] = {}

    dataset_params["intervention_info"]["name"] = ["choose_procedure", "set_ir_3_levels"]
    
    if dataset_params["intervention_info"]["name"] == ["choose_procedure"]:
        dataset_params["intervention_info"]["data_impact"] = ["direct"]
        dataset_params["intervention_info"]["actions"] = [["start_standard", "start_priority"]] #If binary, last action is the 'treatment' action
        dataset_params["intervention_info"]["action_width"] = [2]
        dataset_params["intervention_info"]["action_depth"] = [1]
        dataset_params["intervention_info"]["activities"] = [["start_standard", "start_priority"]]
        dataset_params["intervention_info"]["column"] = ["activity"]
        dataset_params["intervention_info"]["start_control_activity"] = [["initiate_application"]]
        dataset_params["intervention_info"]["end_control_activity"] = [["initiate_application"]]
    elif dataset_params["intervention_info"]["name"] == ["set_ir_3_levels"]:
        dataset_params["intervention_info"]["data_impact"] = ["indirect"]
        dataset_params["intervention_info"]["actions"] = [[0.07, 0.08, 0.09]]
        dataset_params["intervention_info"]["action_width"] = [3]
        dataset_params["intervention_info"]["action_depth"] = [1]
        dataset_params["intervention_info"]["activities"] = [["calculate_offer"]]
        dataset_params["intervention_info"]["column"] = ["interest_rate"]
        dataset_params["intervention_info"]["start_control_activity"] = [[]]
        dataset_params["intervention_info"]["end_control_activity"] = [[]]
    elif dataset_params["intervention_info"]["name"] == ["time_contact_HQ"]:
        dataset_params["intervention_info"]["data_impact"] = ["direct"]
        dataset_params["intervention_info"]["actions"] = [["do_nothing","contact_headquarters"]] #If binary, last action is the 'treatment' action
        dataset_params["intervention_info"]["action_width"] = [2]
        dataset_params["intervention_info"]["action_depth"] = [4] 
        dataset_params["intervention_info"]["activities"] = [["do_nothing", "contact_headquarters"]]
        dataset_params["intervention_info"]["column"] = ["activity"]
        dataset_params["intervention_info"]["start_control_activity"] = [["start_standard"]]
        dataset_params["intervention_info"]["end_control_activity"] = [["start_standard", "email_customer", "call_customer"]]
    elif dataset_params["intervention_info"]["name"] == ["choose_procedure", "set_ir_3_levels"]:
        dataset_params["intervention_info"]["data_impact"] = ["direct", "indirect"]
        dataset_params["intervention_info"]["actions"] = [["start_standard", "start_priority"], [0.07, 0.08, 0.09]]
        dataset_params["intervention_info"]["action_width"] = [2, 3] 
        dataset_params["intervention_info"]["action_depth"] = [1, 1] 
        dataset_params["intervention_info"]["activities"] = [["start_standard", "start_priority"], ["calculate_offer"]]
        dataset_params["intervention_info"]["column"] = ["activity", "interest_rate"]
        dataset_params["intervention_info"]["start_control_activity"] = [["initiate_application"], []]
        dataset_params["intervention_info"]["end_control_activity"] = [["initiate_application"], []]

    dataset_params["intervention_info"]["retain_method"] = "precise"

    # Combinations
    dataset_params["intervention_info"]["action_combinations"] = list(product(*dataset_params["intervention_info"]["actions"]))
    dataset_params["intervention_info"]["action_width_combinations"] = math.prod(dataset_params["intervention_info"]["action_width"])
    dataset_params["intervention_info"]["action_depth_combinations"] = math.prod(dataset_params["intervention_info"]["action_depth"])

    dataset_params["intervention_info"]["len"] = [action_width if action_width > 2 else 1 for action_width in dataset_params["intervention_info"]["action_width"]]
    dataset_params["intervention_info"]["RCT"] = False
    dataset_params["filename"] = "loan_log_" +  str(dataset_params["intervention_info"]["name"])
    #policy
    dataset_params["policies_info"] = {}
    dataset_params["policies_info"]["general"] = "real"
    dataset_params["policies_info"]["choose_procedure"] = {"amount": 50000, "est_quality": 5}
    dataset_params["policies_info"]["time_contact_HQ"] = "real"
    dataset_params["policies_info"]["min_quality"] = 2
    dataset_params["policies_info"]["max_noc"] = 3
    dataset_params["policies_info"]["max_nor"] = 1
    dataset_params["policies_info"]["min_amount_contact_cust"] = 50000

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
            if isinstance(value, list):
                params["intervention_info"][key] = value[intervention]
        dataset_params_list.append(params)

    return dataset_params, dataset_params_list, train

def generate_eval(args, dataset_params):
    eval_dfs = {}

    # us "action_width": is for example [2, 3], so you have 6 combinations of actions, and for each of them you need to evaluate the policy
    action_combos = list(product(*[range(width) for width in dataset_params["intervention_info"]["action_width"]]))
    for action_combo in action_combos:
        if args.already_eval_generated and os.path.exists(os.path.join(os.getcwd(), "data", "eval", "fixed_" + str(action_combo) + "_performance.pkl")):
            # just load the data
            performance = load_data(os.path.join(os.getcwd(), "data", "eval", "fixed_" + str(action_combo) + "_performance"))
            outcome_df = load_data(os.path.join(os.getcwd(), "data", "eval", "fixed_" + str(action_combo) + "_outcome_df"))
            test_df = load_data(os.path.join(os.getcwd(), "data", "eval", "fixed_" + str(action_combo) + "_test_df"))
        else:
            print("Evaluating action combo: ", action_combo)
            performance, outcome_df, test_df = generate_one_eval(policy="fixed", args=args, dataset_params=dataset_params, action_combo=action_combo)
            eval_dfs[str(action_combo)] = {
                "performance": performance,
                "outcome_df": outcome_df,
                "test_df": test_df
            }

            save_data(performance, os.path.join(os.getcwd(), "data", "eval", "fixed_" + str(action_combo) + "_performance"))
            save_data(outcome_df, os.path.join(os.getcwd(), "data", "eval", "fixed_" + str(action_combo) + "_outcome_df"))
            save_data(test_df, os.path.join(os.getcwd(), "data", "eval", "fixed_" + str(action_combo) + "_test_df"))

        eval_dfs[str(action_combo)] = {
            "performance": performance,
            "outcome_df": outcome_df,
            "test_df": test_df
        }

    if args.already_eval_generated and os.path.exists(os.path.join(os.getcwd(), "data", "eval", "bank_performance.pkl")):
        # Load bank policy evaluation
        bank_performance = load_data(os.path.join(os.getcwd(), "data", "eval", "bank_performance"))
        bank_outcome_df = load_data(os.path.join(os.getcwd(), "data", "eval", "bank_outcome_df"))
        bank_test_df = load_data(os.path.join(os.getcwd(), "data", "eval", "bank_test_df"))
    else:
        bank_performance, bank_outcome_df, bank_test_df = generate_one_eval(policy="bank", args=args, dataset_params=dataset_params)
        eval_dfs["bank"] = {
            "performance": bank_performance,
            "outcome_df": bank_outcome_df,
            "test_df": bank_test_df
        }

        save_data(bank_performance, os.path.join(os.getcwd(), "data", "eval", "bank_performance"))
        save_data(bank_outcome_df, os.path.join(os.getcwd(), "data", "eval", "bank_outcome_df"))
        save_data(bank_test_df, os.path.join(os.getcwd(), "data", "eval", "bank_test_df"))

    eval_dfs["bank"] = {
        "performance": bank_performance,
        "outcome_df": bank_outcome_df,
        "test_df": bank_test_df
    }

    # Generate random policy evaluations
    for iter in range(args.num_iterations):
        if args.already_eval_generated and os.path.exists(os.path.join(os.getcwd(), "data", "eval", "random_" + str(iter) + "_performance.pkl")):
            random_performance = load_data(os.path.join(os.getcwd(), "data", "eval", "random_" + str(iter) + "_performance"))
            random_outcome_df = load_data(os.path.join(os.getcwd(), "data", "eval", "random_" + str(iter) + "_outcome_df"))
            random_test_df = load_data(os.path.join(os.getcwd(), "data", "eval", "random_" + str(iter) + "_test_df"))
        else:
            random_object_for_random_policy = random.Random(dataset_params["random_seed_test"] + 5*iter)
            random_performance, random_outcome_df, random_test_df = generate_one_eval(policy="random", args=args, dataset_params=dataset_params, random_object_for_random_policy=random_object_for_random_policy)

            save_data(random_performance, os.path.join(os.getcwd(), "data", "eval", "random_" + str(iter) + "_performance"))
            save_data(random_outcome_df, os.path.join(os.getcwd(), "data", "eval", "random_" + str(iter) + "_outcome_df"))
            save_data(random_test_df, os.path.join(os.getcwd(), "data", "eval", "random_" + str(iter) + "_test_df"))
        
        eval_dfs["random_" + str(iter)] = {
                "performance": random_performance,
                "outcome_df": random_outcome_df,
                "test_df": random_test_df
        }

    # Generate the optimal policy:
    if args.already_eval_generated and os.path.exists(os.path.join(os.getcwd(), "data", "eval", "optimal_performance.pkl")):
        optimal_performance = load_data(os.path.join(os.getcwd(), "data", "eval", "optimal_performance"))
        optimal_outcome_df = load_data(os.path.join(os.getcwd(), "data", "eval", "optimal_outcome_df"))
        optimal_test_df = load_data(os.path.join(os.getcwd(), "data", "eval", "optimal_test_df"))
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
        save_data(optimal_performance, os.path.join(os.getcwd(), "data", "eval", "optimal_performance"))
        save_data(optimal_outcome_df, os.path.join(os.getcwd(), "data", "eval", "optimal_outcome_df"))
        save_data(optimal_test_df, os.path.join(os.getcwd(), "data", "eval", "optimal_test_df"))
    
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
                    best_action = get_bank_best_action(prefix_list, 0, dataset_params)
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

def get_bank_best_action(prefix_list, current_int_index, DATASET_PARAMS):
    prefix_without_int = prefix_list[0][0:-1]
    prev_event = prefix_without_int[-1]
    action_index = 0
    
    if DATASET_PARAMS["intervention_info"]["name"][current_int_index] == "choose_procedure":
        priority_condition = (prev_event["amount"] > DATASET_PARAMS["policies_info"]["choose_procedure"]["amount"] and prev_event["est_quality"] >= DATASET_PARAMS["policies_info"]["choose_procedure"]["est_quality"])

        if priority_condition:
            action_index = 1
        else:
            action_index = 0
    
    elif DATASET_PARAMS["intervention_info"]["name"][current_int_index] == "set_ir_3_levels":
        activity_executioner = ActivityExecutioner()
        ir, _, _ = activity_executioner.calculate_offer(prev_event=prev_event, intervention_info=DATASET_PARAMS["intervention_info"])
        action_index = DATASET_PARAMS["intervention_info"]["actions"][current_int_index].index(ir)
    
    return action_index

def get_random_best_action(DATASET_PARAMS, current_int_index, random_object_for_random_policy):
    random_best_action = random_object_for_random_policy.choice(range(DATASET_PARAMS["intervention_info"]["action_width"][current_int_index]))
    return random_best_action
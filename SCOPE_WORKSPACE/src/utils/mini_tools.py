import numpy as np
import pickle
import argparse
import json
import os
import torch
import pandas as pd
from config.config import model_params
from copy import deepcopy
from collections import OrderedDict
from torch.nn.modules.module import _addindent
from src.utils.model_tools.model_functions import MLCausalRegressor, DLCausalRegressor, KMeans_QLearning
from src.utils.model_tools.psp_model_functions import PSPCausalRegressor

def make_dirs(args, DATA_FOLDER, RESULTS_FOLDER):
    if not os.path.exists(os.path.join(os.getcwd(), DATA_FOLDER)):
        os.makedirs(os.path.join(os.getcwd(), DATA_FOLDER))
        # also make the folders in this folder: training_and_tuning and eval
    if not os.path.exists(os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning")):
        os.makedirs(os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning"))
    if not os.path.exists(os.path.join(os.getcwd(), DATA_FOLDER, "eval")):
        os.makedirs(os.path.join(os.getcwd(), DATA_FOLDER, "eval"))
    
    if not os.path.exists(os.path.join(os.getcwd(), RESULTS_FOLDER)):
        os.makedirs(os.path.join(os.getcwd(), RESULTS_FOLDER))
        # Create for each method a folder as well, and inside that folder a folder for tuning, training and eval
    
    for method in args.methods:
        if not os.path.exists(os.path.join(os.getcwd(), RESULTS_FOLDER, method)):
            print("Creating folder for method: ", method)
            os.makedirs(os.path.join(os.getcwd(), RESULTS_FOLDER, method))
        if not os.path.exists(os.path.join(os.getcwd(), RESULTS_FOLDER, method, "tuning")):
            os.makedirs(os.path.join(os.getcwd(), RESULTS_FOLDER, method, "tuning"))
        if not os.path.exists(os.path.join(os.getcwd(), RESULTS_FOLDER, method, "training")):
            os.makedirs(os.path.join(os.getcwd(), RESULTS_FOLDER, method, "training"))
        if not os.path.exists(os.path.join(os.getcwd(), RESULTS_FOLDER, method, "eval")):
            os.makedirs(os.path.join(os.getcwd(), RESULTS_FOLDER, method, "eval"))
    # Also make a folder for bank, optimal, and random
    if not os.path.exists(os.path.join(os.getcwd(), RESULTS_FOLDER, "bank")):
        os.makedirs(os.path.join(os.getcwd(), RESULTS_FOLDER, "bank"))
    if not os.path.exists(os.path.join(os.getcwd(), RESULTS_FOLDER, "random")):
        os.makedirs(os.path.join(os.getcwd(), RESULTS_FOLDER, "random"))
    if not os.path.exists(os.path.join(os.getcwd(), RESULTS_FOLDER, "optimal")):
        os.makedirs(os.path.join(os.getcwd(), RESULTS_FOLDER, "optimal"))

def save_data(data, path):
    # Check if data looks like a PyTorch state_dict:
    # Typically, it's an OrderedDict of tensors
    is_state_dict = isinstance(data, OrderedDict) and all(
        isinstance(v, torch.Tensor) for v in data.values()
    )
    
    if is_state_dict:
        # add .pth extension for PyTorch state_dicts
        if not path.endswith('.pth'):
            path += '.pth'
        torch.save(data, path)  # Use torch.save for PyTorch state_dict
    else:
        # add .pkl extension for other data types
        if not path.endswith('.pkl'):
            path += '.pkl'
        with open(path, 'wb') as f:
            pickle.dump(data, f)  # Fallback to pickle for everything else

def load_data(path, is_state_dict=False):
    # Check if the file is a PyTorch state_dict
    if is_state_dict:
        if not path.endswith('.pth'):
            path += '.pth'
        return torch.load(path)
    # Otherwise, assume it's a pickle file
    else:
        if not path.endswith('.pkl'):
            path += '.pkl'
        with open(path, 'rb') as f:
            return pickle.load(f)

def generate_dash_patterns(num_patterns, max_segments=2, max_length=10):
    dash_patterns = []
    for _ in range(num_patterns):
        num_segments = np.random.randint(1, max_segments + 1)
        dash_pattern = np.random.randint(1, max_length + 1, size=num_segments * 2)
        dash_patterns.append(tuple(dash_pattern))
    return dash_patterns

# Erstellt das Modell, dass dann weitergegeben wird für die Prediction
# Die modelle und deren functions befinden sich in model_functions.py
# Das wird auch von calc_target_outcomes im DTR aufgerufen, um eine Vorhersage über den outcome zu machen
def get_model_functions(model_params, model_to_load=None):
    if model_params["target"] == "ps":
        pass
    elif model_params["target"] == "outcome":
        if model_params["model_category"] == "dl":
            # Das PSP hat eigene Model Functions (siehe psp_model_functions.py)
            # WICHTIGE NEUEURUNG!!
            if model_params["model_specific"] == "psp":
                model_functions = PSPCausalRegressor(model_params=model_params)
            else:
                model_functions = DLCausalRegressor(model_params=model_params)

        elif model_params["model_category"] == "ml":
            model_functions = MLCausalRegressor(model_params=model_params)

        elif model_params["model_category"] == "rl":
            model_functions = KMeans_QLearning(model_params=model_params)

    elif model_params["target"] == "effect" and "reg" in model_params["method"]:
        if model_params["model_category"] == "ml":
            model_functions = MLCausalRegressor(model_params=model_params)
        else:
            model_functions = DLCausalRegressor(model_params=model_params)

    # Das lädt das Modell, das bereits vortrainiert sein sollte. Lädt dann die gespiechetren Gewichte rein
    # Jedes Modell hat eine load Methode, die es ermöglicht, das Modell zu laden
    # load_state_dict
    if model_to_load is not None:
        if model_params["model_category"] == "dl":
            if isinstance(model_to_load, list):
                # T-learner: multiple models
                assert isinstance(model_functions.model, list), \
                    "Expected model_functions.model to be a list for T-learner setup."
                states_to_load = model_to_load
                if model_params["target"] == "effect" and len(model_to_load) == len(model_functions.model) + 1:
                    states_to_load = model_to_load[1:]
                for m, state in zip(model_functions.model, states_to_load):
                    m.load_state_dict(state)
            else:
                # S-learner: single model
                model_functions.model.load_state_dict(model_to_load)
        else:
            if (
                model_params["target"] == "effect"
                and isinstance(model_to_load, list)
                and isinstance(model_functions.model, list)
                and len(model_to_load) == len(model_functions.model) + 1
            ):
                model_to_load = model_to_load[1:]
            model_functions.model = model_to_load
    
    return model_functions

# Bildet model_params_list_of_dicts
def get_model_params_list_of_dicts(method, args, prep_utils):
    dataset = args.dataset

    if method == "kmeans_q":
        n_stages = 1
    else:
        n_stages = args.n_stages

    model_params_list_of_dicts = []
    for stage in range(n_stages):
        model_params_dict = {}
        
        # Lädt standardeinstellungen über config
        init_model_params = deepcopy(model_params) # imported from config
        init_model_params["method"] = method
        # z.B. 1
        init_model_params["stage"] = stage
        init_model_params["dataset"] = dataset
        # Legt fest, ob der KMeans-Q-Ansatz den ermittelten Reward positiv verwendet. Im Code: True, wenn kmeans_config[1] == "pos_rewards"; sonst False.
        init_model_params["pos_rewards"] = True if args.kmeans_config[1] == "pos_rewards" else False
        # Schaltet die Min-Max-Normalisierung der Rewards ein. Aktiv ist sie, wenn kmeans_config[3] == "norm"
        init_model_params["normalize_reward"] = True if args.kmeans_config[3] == "norm" else False
        # Bestimmt, ob Null-Rewards in die Anpassung des Normalisierers einbezogen werden
        init_model_params["change_zero_reward"] = True if args.kmeans_config[5] == "change_zero" else False
        # Bestimmt den Zeitpunkt der Reward-Normalisierung im KMeans-Q-Ablauf:
        init_model_params["norm_mdp"] = True if args.kmeans_config[4] == "norm_mdp" else False
        if "dtr" in method or "separate" in method:
            #  Wird bei Namen wie dtr-S-reg-R aus dem zweiten Namensbestandteil gelesen: hier also S. Das bezeichnet den Learner-Ansatz, zum Beispiel S, T, RA oder AIPWE.
            init_model_params["learner_method"] = method.split("-")[1]
            # bei dtr-S-reg-R ist es reg. Das gibt an, wie das Modell eine Aktion empfiehlt, zum Beispiel regressionsbasiert (reg) oder klassifikationsbasiert (class)
            init_model_params["action_recomm_method"] = method.split("-")[2]
            # G verwendet den Q-Wert der beobachteten Aktion, R passt das Outcome um die geschätzte Differenz zwischen optimaler und beobachteter Aktion an
            init_model_params["value_function_method"] = method.split("-")[-1]
        init_model_params["cross_fitting"] = args.cross_fitting

        # ps       → Propensity-Score-Modell
        # outcome  → Outcome-Modell
        # effect   → Effekt-/Entscheidungsmodell
        model_params_dict["ps"] = "nope"
        model_params_dict["outcome"] = "nope"
        model_params_dict["effect"] = "nope"

        if "dtr" in method or "separate" in method:
            if init_model_params["learner_method"] == "AIPWE":
                init_model_params["target"] = "ps"
                init_model_params["encoding"] = "agg"
                init_model_params["model_category"] = args.model_categories[0]
                if args.model_categories[0] == "dl":
                    if stage == 0 and dataset == "SimBank":
                        init_model_params["model_specific"] = "vanilla_nn"
                    else:
                        init_model_params["model_specific"] = "lstm"
                        init_model_params["encoding"] = "tensor"
                elif args.model_categories[0] == "ml":
                    init_model_params["model_specific"] = args.model_specifics[0]
                model_params_dict["ps"] = deepcopy(init_model_params)

            init_model_params["target"] = "outcome"
            init_model_params["encoding"] = "agg"
            init_model_params["model_category"] = args.model_categories[2]
            if args.model_categories[1] == "dl":
                # PSP wird über --model_specifics gewählt (zweiter Eintrag = Outcome-Modell)
                if args.model_specifics[1] == "psp":
                    init_model_params["model_specific"] = "psp"
                    init_model_params["encoding"] = "tensor"
                elif stage == 0 and dataset == "SimBank":
                    init_model_params["model_specific"] = "vanilla_nn"
                else:
                    init_model_params["model_specific"] = "lstm"
                    init_model_params["encoding"] = "tensor"
            else:
                init_model_params["model_specific"] = args.model_specifics[1]
            model_params_dict["outcome"] = deepcopy(init_model_params)

            if "AIPWE" in init_model_params["learner_method"] or "DR" in init_model_params["learner_method"] or "RA" in init_model_params["learner_method"]:
                init_model_params["target"] = "effect"
                init_model_params["encoding"] = "agg"
                init_model_params["model_category"] = args.model_categories[2]
                if args.model_categories[2] == "dl":
                    if stage == 0 and dataset == "SimBank":
                        init_model_params["model_specific"] = "vanilla_nn"
                    else:
                        init_model_params["model_specific"] = "lstm"
                        init_model_params["encoding"] = "tensor"
                elif args.model_categories[2] == "ml":
                    init_model_params["model_specific"] = args.model_specifics[2]
                model_params_dict["effect"] = deepcopy(init_model_params)

        elif method == "kmeans_q":
            init_model_params["train_size"] = args.train_size
            init_model_params["target"] = "outcome"
            init_model_params["encoding"] = "kmeans"
            init_model_params["model_category"] = "rl"
            init_model_params["model_specific"] = "kmeans_q"
            model_params_dict["outcome"] = deepcopy(init_model_params)

        for target, prms in model_params_dict.items():
            if prms != "nope":
                if prms["model_category"] == "dl" or prms["model_category"] == "ml":
                    prms["dim_x_case"] = prep_utils[prms["encoding"]][stage]["dim_x_case"]
                    prms["dim_x_event"] = prep_utils[prms["encoding"]][stage]["dim_x_event"]
                    prms["dim_t"] = prep_utils[prms["encoding"]][stage]["dim_t"]
                    prms["dim_output"] = prep_utils[prms["encoding"]][stage]["dim_output"]
                    if prms["model_specific"] == "psp":
                        # Das PSP braucht die Spaltennamen, um die Aktivität einzubetten
                        prms["event_cols_encoded"] = prep_utils[prms["encoding"]][stage]["event_cols_encoded"]
                        # und die Spalten des Suffixes für die Decoder-Köpfe
                        prms["suffix_cols_encoded"] = prep_utils[prms["encoding"]][stage].get("suffix_cols_encoded")
        
        model_params_list_of_dicts.append(model_params_dict)
    
    if "dtr" in method:
        if model_params_list_of_dicts[0]["outcome"] != "nope":
            model_params_list_of_dicts[0]["outcome"]["prev_ps_model_params"] = deepcopy(model_params_list_of_dicts[1]["ps"])
            model_params_list_of_dicts[0]["outcome"]["prev_outcome_model_params"] = deepcopy(model_params_list_of_dicts[1]["outcome"])
        if model_params_list_of_dicts[0]["effect"] != "nope":
            model_params_list_of_dicts[0]["effect"]["prev_ps_model_params"] = deepcopy(model_params_list_of_dicts[1]["ps"])
            model_params_list_of_dicts[0]["effect"]["prev_outcome_model_params"] = deepcopy(model_params_list_of_dicts[1]["outcome"])

    return model_params_list_of_dicts
        
def parse():
    # Parse
    parser = argparse.ArgumentParser(description='SCOPE')
    parser.add_argument('--config', type=str, help='Path to config file')
    args, unknown = parser.parse_known_args()
    config_args = {}
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config_args = json.load(f)

    parser.add_argument('--methods', nargs='+', type=str, default=config_args.get('methods', ["dtr-S-reg-R", "separate-S-reg-none", "kmeans_q"]), help='Methods to run')
    parser.add_argument('--dataset', type=str, default=config_args.get('dataset', "SimBank"), help='Dataset to use')
    parser.add_argument('--n_stages', type=int, default=config_args.get('n_stages', 2), help='Number of stages')
    parser.add_argument('--encodings', nargs='+', type=str, default=config_args.get('encodings', ["tensor", "kmeans"]), help='Encodings to use')
    parser.add_argument('--model_categories', nargs='+', type=str, default=config_args.get('model_categories', ["ml","ml","ml", "ml"]), help='Model categories to use (outcome, effect): ml, dl, or rl')
    parser.add_argument('--model_specifics', nargs='+', type=str, default=config_args.get('model_specifics', ["xgb","xgb","xgb"]), help='Models to use (outcome, effect): xgb, rf, lstm...')
    parser.add_argument('--cross_fitting', type=lambda x: x.lower() == 'true', default=config_args.get('cross_fitting', False), help='Cross-fitting (True or False)')
    parser.add_argument('--wandb', type=lambda x: x.lower() == 'true', default=config_args.get('wandb', False), help='Connect with wandb or not')
    parser.add_argument('--kmeans_config', nargs='+', type=str, default=config_args.get('kmeans_config', ["big", "neg_rewards", "prep_outcome", "norm", "norm_mdp", "change_zero"]), help='max_scale (small, mid, big), pos_rewards(bool), prep_outcome(bool), norm(yes/no), norm_mdp(bool), change_zero(bool)')

    parser.add_argument('--train_size', type=int, default=config_args.get('train_size', 10000), help='Train size')
    parser.add_argument('--test_size', type=int, default=config_args.get('test_size', 10000), help='Test size')
    parser.add_argument('--big_data', type=lambda x: x.lower() == 'true', default=config_args.get('big_data', True), help='Big data (True or False)')
    parser.add_argument('--big_eval', type=lambda x: x.lower() == 'true', default=config_args.get('big_eval', True), help='Big eval (True or False)')
    parser.add_argument('--big_tuning', type=lambda x: x.lower() == 'true', default=config_args.get('big_tuning', True), help='Big tuning (True or False)')

    parser.add_argument('--already_trained_list', nargs='+', type=str, default=config_args.get('already_trained_list', []), help='Are there any models already trained?')
    parser.add_argument('--already_trained', type=lambda x: x.lower() == 'true', default=config_args.get('already_trained', False), help='Already trained (True or False)')
    parser.add_argument('--already_tuned_list', nargs='+', type=str, default=config_args.get('already_tuned_list', []), help='Are there any models already tuned?')
    parser.add_argument('--already_tuned', type=lambda x: x.lower() == 'true', default=config_args.get('already_tuned', False), help='Already tuned (True or False)')
    parser.add_argument('--already_train_tune_generated', type=lambda x: x.lower() == 'true', default=config_args.get('already_train_tune_generated', False), help='Already generated train and tune dfs (True or False)')
    parser.add_argument('--already_train_tune_preprocessed', type=lambda x: x.lower() == 'true', default=config_args.get('already_train_tune_preprocessed', False), help='Already preprocessed train and tune dfs (True or False)')

    parser.add_argument('--already_eval_generated', type=lambda x: x.lower() == 'true', default=config_args.get('already_eval_generated', False), help='Already generated eval dfs (True or False)')
    parser.add_argument('--already_eval_generated_list', nargs='+', type=str, default=config_args.get('already_eval_generated_list', []), help='Already evaluation generated?')
    parser.add_argument('--already_eval_preprocessed', type=lambda x: x.lower() == 'true', default=config_args.get('already_eval_preprocessed', False), help='Already preprocessed eval dfs (True or False)')
    parser.add_argument('--already_evaluated_list', nargs='+', type=str, default=config_args.get('already_evaluated_list', []), help='Already evaluated? (give the full method, e.g., dtr-S-reg-R etc.)')
    parser.add_argument('--already_evaluated', type=lambda x: x.lower() == 'true', default=config_args.get('already_evaluated', False), help='Already evaluated (True or False)')

    parser.add_argument('--iterations_to_skip', nargs='+', type=int, default=config_args.get('iterations_to_skip', []), help='Iterations to skip')
    parser.add_argument('--num_iterations', type=int, default=config_args.get('num_iterations', 10), help='Num iterations')
    parser.add_argument('--delta', type=float, default=config_args.get('delta', 0.95), help='Delta')
    parser.add_argument('--confounding_type', type=str, default=config_args.get('confounding_type', 'point'), help='Confounding type for bpic17: "point" (inline per-decision-point random) or "case" (generate full bank-policy + full RCT datasets, then combine via set_delta)')

    args = parser.parse_args()

    return parser, args

def make_label(settings):
    label = ""
    for key, value in settings.items():
        label += ", {}:{}".format(key, value)
    return label[2:]

# Splitten der Rohdaten in Trainings- und Inferenzdaten
def split_raw_data(data, infer_prop=0.2):
    # to change when using real data
    # split (per case though, and in the order of the data)
    case_nrs = data["case_nr"].unique()
    case_nrs_train = case_nrs[:int(len(case_nrs) * (1 - infer_prop))]
    case_nrs_infer = case_nrs[int(len(case_nrs) * (1 - infer_prop)):]

    data_train = data[data["case_nr"].isin(case_nrs_train)]
    data_infer = data[data["case_nr"].isin(case_nrs_infer)]
    
    return data_train, data_infer

# data_train_list sind die prefixe mit, Fallnummer, Prefix
def create_splits(data_train_list, data_infer_list, model_params):
    data_train_list = deepcopy(data_train_list)
    data_infer_list = deepcopy(data_infer_list)

    final_data_train_list = []
    final_data_infer_list = []

    method = model_params["method"]
    model_category = model_params["model_category"]
    target = model_params["target"]
    model_specific = model_params["model_specific"]

    # === Stage 0 ===
    i = 0
    data_train = data_train_list[i] # Default
    data_infer = data_infer_list[i] # Default
    split = "split_train_infer"  # Default split
    if "dtr" in method:
        # check which target
        if target == "ps":
            if model_category == "ml" and model_specific == "logreg":
                # Use data_train + data_infer
                split = "merge_train_infer"
            elif model_category == "ml" and model_specific == "xgb":
                # Use data_train + data_infer_1
                split = "merge_train_infer"
            elif model_category == "dl":
                # Use data_train, data_infer_1 separately
                split = "split_train_infer_1"
            else:
                # Use data_train + data_infer_1
                split = "merge_train_infer_1"
        elif model_category != "dl":
            # Use data_train + data_infer
            split = "merge_train_infer"
    
    elif model_category != "dl":
        # Use data_train + data_infer
        split = "merge_train_infer"

    if method == "kmeans_q":
        split = "train"

    if split == "merge_train_infer":
        final_data_train = merge_prepped_data(data_train, data_infer)
        final_data_infer = None
    elif split == "split_train_infer":
        final_data_train, final_data_infer = data_train, data_infer
    elif "1" in split or "2" in split:
        data_infer_1, data_infer_2 = split_prepped_data(data_infer, 0.75)
        if split == "merge_train_infer_1":
            final_data_train = merge_prepped_data(data_train, data_infer_1)
            final_data_infer = None
        elif split == "split_train_infer_1":
            final_data_train = data_train
            final_data_infer = data_infer_1
        elif split == "data_infer_2":
            final_data_train = data_infer_2
            final_data_infer = None
    elif split == "train":
        final_data_train = merge_prepped_data(data_train, data_infer) if data_infer is not None else data_train
        final_data_infer = None
    
    final_data_train_list.append(final_data_train)
    final_data_infer_list.append(final_data_infer)

    case_nrs_train = final_data_train["case_nr"]
    case_nrs_infer = final_data_infer["case_nr"] if final_data_infer is not None else None

    # === Stage > 0 === it could be that there are cases that do not reach stage > 0, so the elements in data_train_list do not necessarily have the same length (less cases in stage > 0 than stage 0)
    # therefore, if we split the same way as above, we could have different cases than those retained for stage 0, so we make sure we only retain cases in stage > 0 that are definitely in the split data of stage 0
    if method != "kmeans_q":
        for i in range(1, len(data_train_list)):
            data_train = data_train_list[i]  # Default
            data_infer = data_infer_list[i]  # Default
            total_data = merge_prepped_data(data_train, data_infer) if data_infer is not None else data_train

            # only get case_nrs that are in case_nrs_train
            mask_train = torch.isin(total_data["case_nr"], case_nrs_train)
            filtered_data_train = {key: tensor[mask_train] if tensor is not None else None for key, tensor in total_data.items()}

            filtered_data_infer = None
            if case_nrs_infer is not None:
                # only get case_nrs that are in case_nrs_infer
                mask_infer = torch.isin(total_data["case_nr"], case_nrs_infer)
                filtered_data_infer = {key: tensor[mask_infer] if tensor is not None else None for key, tensor in total_data.items()}

            final_data_train_list.append(filtered_data_train)
            final_data_infer_list.append(filtered_data_infer)

    # Set both final_data_train_list and final_data_infer_list to None, if they have only None elements
    final_data_train_list = None if all(data is None for data in final_data_train_list) else final_data_train_list
    final_data_infer_list = None if all(data is None for data in final_data_infer_list) else final_data_infer_list

    return final_data_train_list, final_data_infer_list

def filter_by_case_range(data_dict, min_case, max_case):
    """Filter a dict of tensors by case_nr range."""
    case_nrs = data_dict["case_nr"]
    mask = (case_nrs >= min_case) & (case_nrs <= max_case)
    return {
        k: (v[mask] if v is not None else None)
        for k, v in data_dict.items()
    }

def merge_prepped_data(data_train, data_infer):
    if isinstance(data_train, pd.DataFrame) and isinstance(data_infer, pd.DataFrame):
        return pd.concat([data_train, data_infer], axis=0, ignore_index=True)
    else:
        merged_data = {}

        for key in data_train.keys():
            val1 = data_train[key]
            val2 = data_infer[key]

            # Handle None values consistently
            if val1 is None and val2 is None:
                merged_data[key] = None
            elif isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                merged_data[key] = torch.cat([val1, val2], dim=0)
            else:
                raise ValueError(f"Inconsistent or unsupported type for key '{key}': {type(val1)}, {type(val2)}")
        return merged_data

def split_prepped_data(data, proportion):
    # Get number of rows from one of the tensors (e.g., "Y")
    total_len = data["Y"].shape[0]
    split_idx = int(total_len * proportion)

    first_part = {}
    second_part = {}

    for key, val in data.items():
        if val is None:
            first_part[key] = None
            second_part[key] = None
        elif isinstance(val, torch.Tensor):
            first_part[key] = val[:split_idx]
            second_part[key] = val[split_idx:]
        else:
            first_part[key] = None
            second_part[key] = None

    return first_part, second_part

class Incremental():
    '''
    incrementally computes mean and variance
    purpose is to avoid memory problems
    '''
    def __init__(self):
        self.mu = 0
        self.previous_mu = 0
        self.n_var = 0
        self.counter = 0

    def update(self, x):
        self.counter += 1
        self.previous_mu = self.mu
        # self.previous_var = self.var
        self.mu = self.previous_mu + (x - self.previous_mu) / self.counter
        self.n_var = self.n_var + (x - self.previous_mu) * (x - self.mu)

        return self.mu, self.n_var / self.counter

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """From: https://github.com/Bjarten/early-stopping-pytorch"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, epoch=None):
        if isinstance(model, list):
            self.is_list = True
        else:
            self.is_list = False

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.is_list:
                self.save_checkpoint(val_loss, model[0], suffix="_T")
                self.save_checkpoint(val_loss, model[1], suffix="_C")
            else:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            if self.is_list:
                self.save_checkpoint(val_loss, model[0], suffix="_T")
                self.save_checkpoint(val_loss, model[1], suffix="_C")
            else:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, suffix=""):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if suffix == "":
            torch.save(model.state_dict(), self.path, _use_new_zipfile_serialization=False)
        else:
            torch.save(model.state_dict(), self.path+suffix, _use_new_zipfile_serialization=False)
        self.val_loss_min = val_loss

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr

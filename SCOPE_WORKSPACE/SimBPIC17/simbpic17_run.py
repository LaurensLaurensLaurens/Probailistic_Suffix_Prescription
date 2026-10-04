path = "c:\\Users\\u0166838\\OneDrive - KU Leuven\\Documents\\Doc\\Code\\SCOPE"

from itertools import product
import math
from copy import deepcopy
import os
import sys
import random
sys.path.append(path)
from SimBPIC17.simulation import run_simulation
from SimBank.confounding_level import set_delta
from src.utils.mini_tools import save_data, load_data
import pandas as pd

# Diese Zeile definiert eine Funktion, die BPIC17-Trainings- und Tuningdaten simuliert.
def generate_training_and_tuning_bpic17(train_size, delta, n_stages=6, confounding_type='old', max_attempts=20):
    random_seed = 42
    dataset_params = {}

    dataset_params["scale_cols"] = ["duration", "monthlycost", "outcome"]
    dataset_params["case_cols"] = ["loangoal"]
    dataset_params["event_cols"] = ["activity", "duration", "monthlycost"]
    dataset_params["last_state_cols"] = ["monthlycost", "loangoal"]
    dataset_params["cat_cols"] = ["activity", "loangoal"]

    dataset_params["intervention_info"] = {}
    # n_stages gibt an, wie oft im Prozess nacheinander entschieden wird:
    # Dateien anrufen: call_incomplete_files
    # warten: wait_incomplete_files
    # Bei n_stages=6 gibt es sechs solcher Entscheidungspunkte pro Fall
    dataset_params["intervention_info"]["name"] = ["call_or_not"] * n_stages
    dataset_params["intervention_info"]["data_impact"] = ["direct"] * n_stages
    dataset_params["intervention_info"]["actions"] = [["call_incomplete_files", "wait_incomplete_files"]] * n_stages
    dataset_params["intervention_info"]["action_width"] = [2] * n_stages
    dataset_params["intervention_info"]["action_depth"] = [1] * n_stages
    dataset_params["intervention_info"]["activities"] = [["call_incomplete_files", "wait_incomplete_files"]] * n_stages
    dataset_params["intervention_info"]["column"] = ["activity"] * n_stages
    dataset_params["intervention_info"]["start_control_activity"] = [["validate_application"]] * n_stages
    dataset_params["intervention_info"]["end_control_activity"] = [["validate_application"]] * n_stages
    dataset_params["intervention_info"]["retain_method"] = "precise"

    # Combinations
    dataset_params["intervention_info"]["action_combinations"] = list(product(*dataset_params["intervention_info"]["actions"]))
    dataset_params["intervention_info"]["action_width_combinations"] = math.prod(dataset_params["intervention_info"]["action_width"])
    dataset_params["intervention_info"]["action_depth_combinations"] = math.prod(dataset_params["intervention_info"]["action_depth"])

    dataset_params["intervention_info"]["len"] = [action_width if action_width > 2 else 1 for action_width in dataset_params["intervention_info"]["action_width"]]
    dataset_params["intervention_info"]["RCT"] = False
    dataset_params["filename"] = "bpic17_loan_log_" +  str(dataset_params["intervention_info"]["name"])

    # Beispielsweise bedeutet delta=0.8: ungefähr 80 % Bankstrategie-Fälle und 20 % randomisierte Fälle.
    if confounding_type == 'case':
        # Generate fully bank-policy data (delta=1) and fully RCT data (delta=0),
        # then combine them using set_delta — same approach as SimBank's generate_training_and_tuning.
        data_bank = run_simulation(random_seed=random_seed, n_cases=train_size, n_stages=n_stages, delta=1)
        data_rct  = run_simulation(random_seed=random_seed * 10, n_cases=train_size, n_stages=n_stages, delta=0)

        # drop unimportant columns from both
        # Spalten werden also zuerst zur Datengenerierung verwendet und danach entfernt, damit das Modell nur realistisch verfügbare Informationen erhält.
        columns_to_drop = ['avg_duration', 'length', 'prev_treatment', 'prev_treatment_effect', 'elapsed_time', 'treatment_effect', 'total_nr_treatments']
        data_bank = data_bank.drop(columns=columns_to_drop)
        data_rct  = data_rct.drop(columns=columns_to_drop)

        # Die Hilfsfunktion prüft, ob jede mögliche Intervention in mindestens zwei verschiedenen Fällen vorkommt.
        # Das verhindert einen Trainingsdatensatz, in dem eine Behandlungsoption fehlt oder nur einmal vorkommt. Ohne beide Optionen kann ein kausales Modell ihre Wirkungen nicht sinnvoll vergleichen.
        def all_actions_present(train_df, intervention_actions, intervention_columns):
            for actions, col in zip(intervention_actions, intervention_columns):
                for action in actions:
                    unique_cases = train_df.loc[train_df[col] == action, "case_nr"].nunique()
                    if unique_cases < 2:
                        return False
            return True

        # Dieser Block mischt Bank- und RCT-Fälle wiederholt neu, bis beide Aktionen ausreichend vertreten sind.
        # Das Ganze soll die Positivitätsbedingung für kausales Lernen absichern: In den Trainingsdaten müssen sowohl call als auch wait beobachtet werden. Fehlt eine Aktion, kann das Modell deren Wirkung nicht lernen oder vergleichen.
        # call_incomplete_files
        # wait_incomplete_files
        # Und das Ganze wird wiederholt, bis beide Aktionen ausreichend vertreten sind.
        # ach maximal max_attempts – standardmäßig 20 – erscheint eine Warnung.
        attempt = 0
        seed_offset = 0
        while attempt < max_attempts:
            current_seed = random_seed + seed_offset
            data = set_delta(data=data_bank, data_RCT=data_rct, delta=delta, seed=current_seed)
            if all_actions_present(data, dataset_params["intervention_info"]["actions"], dataset_params["intervention_info"]["column"]):
                break
            else:
                attempt += 1
                seed_offset += 1
                print(f"Attempt {attempt}: Some actions missing, retrying with new seed {current_seed}")

        if attempt == max_attempts:
            print("Warning: Could not ensure all actions appear at least twice after max attempts.")

    else:
        # Old approach: inline per-decision-point randomness with probability (1-delta)
        data = run_simulation(random_seed=random_seed, n_cases=train_size, n_stages=n_stages, delta=delta)

        # drop unimportant columns
        columns_to_drop = ['avg_duration', 'length', 'prev_treatment', 'prev_treatment_effect', 'elapsed_time', 'treatment_effect', 'total_nr_treatments']
        data = data.drop(columns=columns_to_drop)

    # Dieser Block zerlegt die gemeinsame Konfiguration in eine Konfiguration pro Entscheidungsstufe.
    # Bei n_stages=6 entsteht eine Liste mit sechs Dictionaries:
    # Beispielsweise wird aus "action_width": [2, 2, 2, 2, 2, 2]
    # "action_width": 2 für Stufe 3 etc. 
    dataset_params_list = []
    for intervention in range(len(dataset_params["intervention_info"]["action_width"])):
        params = deepcopy(dataset_params)
        for key, value in params["intervention_info"].items():
            if isinstance(value, list):
                params["intervention_info"][key] = value[intervention]
        dataset_params_list.append(params)

    # ({"scale_cols":["duration","monthlycost","outcome"],"case_cols":["loangoal"],"event_cols":["activity","duration","monthlycost"],"intervention_info":{"name":["call_or_not"]*6,"action_width":[2,2,2,2,2,2],"actions":[["call_incomplete_files","wait_incomplete_files"]]*6,"RCT":False}}, [{"stage":0,"action_width":2},{"stage":1,"action_width":2},{"stage":2,"action_width":2},{"stage":3,"action_width":2},{"stage":4,"action_width":2},{"stage":5,"action_width":2}], DataFrame([{"case_nr":101,"activity":"validate_application","duration":1.2,"monthlycost":850.0,"loangoal":"car","outcome":4200.0}]))
    return dataset_params, dataset_params_list, data

def generate_eval_bpic17(args, dataset_params):
    eval_dfs = {}
    random_seed = 1984
    # Use conf_suffix (set by main.py) to differentiate paths for each confounding type
    # Sie erzeugt einen Zusatz für den Ordnernamen
    # Alles nur bzgl Location der Daten
    conf_suffix = "_case" if args.confounding_type == "case" else ""
    dataset_folder = args.dataset + conf_suffix
    eval_dir = os.path.join(os.getcwd(), "data", dataset_folder, str(args.n_stages), "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # us "action_width": is for example [2, 3], so you have 6 combinations of actions, and for each of them you need to evaluate the policy
    action_combos = list(product(*[range(width) for width in dataset_params["intervention_info"]["action_width"]]))
    for action_combo in action_combos:
        # Der Block lädt bereits gespeicherte Evaluationsergebnisse für eine feste Aktionsfolge, statt sie erneut zu simulieren.
        if (args.already_eval_generated or ('fixed' in args.already_eval_generated_list)) and os.path.exists(os.path.join(eval_dir, "fixed_" + str(action_combo) + "_performance.pkl")):
            # just load the data
            # performance: Gesamtleistung dieser Aktionsfolge
            performance = load_data(os.path.join(eval_dir, "fixed_" + str(action_combo) + "_performance"))
            # outcome_df: Ergebnis pro Fall
            outcome_df = load_data(os.path.join(eval_dir, "fixed_" + str(action_combo) + "_outcome_df"))
            # test_df: vollständiges simuliertes Test-Event-Log
            test_df = load_data(os.path.join(eval_dir, "fixed_" + str(action_combo) + "_test_df"))
            print('Fixed performance, combo: ', performance, action_combo)
        else:
            print("Evaluating action combo: ", action_combo)
            # Die Funktion bewertet genau eine Policy auf neu simulierten Testfällen.
            performance, outcome_df, test_df = generate_one_eval_bpic17(policy="fixed", args=args, action_combo=action_combo, random_seed=random_seed)
            eval_dfs[str(action_combo)] = {
                "performance": performance,
                "outcome_df": outcome_df,
                "test_df": test_df
            }

            save_data(performance, os.path.join(eval_dir, "fixed_" + str(action_combo) + "_performance"))
            save_data(outcome_df, os.path.join(eval_dir, "fixed_" + str(action_combo) + "_outcome_df"))
            save_data(test_df, os.path.join(eval_dir, "fixed_" + str(action_combo) + "_test_df"))

        eval_dfs[str(action_combo)] = {
            "performance": performance,
            "outcome_df": outcome_df,
            "test_df": test_df
        }

    # Dieser Block lädt oder erzeugt die Evaluation der historischen Bank-Policy.
    #Der Zweck ist, die Bankstrategie als Vergleichsbasis – eine Baseline – für SCOPE, feste und zufällige Policies zu bewerten.
    if (args.already_eval_generated or ('bank' in args.already_eval_generated_list)) and os.path.exists(os.path.join(eval_dir, "bank_performance.pkl")):
        # Load bank policy evaluation
        bank_performance = load_data(os.path.join(eval_dir, "bank_performance"))
        bank_outcome_df = load_data(os.path.join(eval_dir, "bank_outcome_df"))
        bank_test_df = load_data(os.path.join(eval_dir, "bank_test_df"))
        print('Bank performance: ', bank_performance)
    else:
        bank_performance, bank_outcome_df, bank_test_df = generate_one_eval_bpic17(policy="bank", args=args, random_seed=random_seed)
        eval_dfs["bank"] = {
            "performance": bank_performance,
            "outcome_df": bank_outcome_df,
            "test_df": bank_test_df
        }

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
            random_object_for_random_policy = random.Random(random_seed + 5*iter)
            random_performance, random_outcome_df, random_test_df = generate_one_eval_bpic17(policy="random", args=args, random_seed=random_seed, random_object_random_policy=random_object_for_random_policy)

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

        eval_dfs["optimal"] = {
            "performance": optimal_performance,
            "outcome_df": optimal_outcome_df,
            "test_df": optimal_test_df
        }
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
def generate_one_eval_bpic17(policy, args, action_combo=None, random_seed=1984, random_object_random_policy=None):
    print("Calculate True Performance for policy ", policy)

    delta = 0
    if policy == "bank":
        delta = 1
    elif policy == "fixed":
        action_combo = list(action_combo)

    # test_df: vollständiges Event-Log der Simulation
    test_df = run_simulation(random_seed=random_seed, n_cases=args.test_size, n_stages=args.n_stages, delta=delta, policy=action_combo, random_object_random_policy=random_object_random_policy)
    # calculate performance
    # performance: Gesamtwert über alle Fälle
    performance = test_df.groupby('case_nr')['outcome'].max().sum()
    # make outcome_df
    # outcome_df: ein Outcome pro Fall
    outcome_df = test_df.groupby('case_nr', as_index=False)['outcome'].max()

    return performance, outcome_df, test_df
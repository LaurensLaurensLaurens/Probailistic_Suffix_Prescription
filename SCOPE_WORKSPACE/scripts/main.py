import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from config.config import path
sys.path.append(path)
from SimBank.generate_sequential_simbank import generate_training_and_tuning, generate_eval
# Anpassen!
from SimBPIC17.simbpic17_run import generate_training_and_tuning_bpic17, generate_eval_bpic17
from SimRepair.simRepair_run import generate_training_and_tuning_simrepair, generate_eval_simrepair
from src.methods.method_main import Method
from src.utils.mini_tools import load_data, parse, save_data, make_dirs
from src.utils.prep_tools.main_prep import ProcessPreprocessor
import wandb

# Parsing
# Entspricht dem ausgefuehrten Befehl:
# python -u scripts/main.py --dataset SimRepair --train_size 1000 --test_size 20
#   --delta 0.5 --n_stages 1 --methods separate-S-reg-none --encodings agg
#   --model_categories ml ml ml ml --model_specifics xgb xgb xgb
#   --num_iterations 1
# args.dataset = "SimRepair"                 # SimRepair-Simulation verwenden
# args.train_size = 1000                     # Anzahl Trainingsfaelle
# args.test_size = 20                        # Anzahl Evaluationsfaelle
# args.delta = 0.5                            # Anteil/Richtung der Confounding-Mischung
# args.n_stages = 1                          # eine Entscheidungsstufe
# args.methods = ["separate-S-reg-none"]     # Separate S-Learner, Regression
# args.encodings = ["agg"]                   # aggregierte Prozessmerkmale
# args.model_categories = ["ml", "ml", "ml", "ml"]  # klassische ML-Modelle
# args.model_specifics = ["xgb", "xgb", "xgb"]        # XGBoost
# args.num_iterations = 1                    # ein Trainings-/Evaluationslauf

# args.cross_fitting = False
parser, args = parse()
print("Specified Arguments: ", args, "\n")
args.max_num_tuning_evals = 75
if not args.big_data:
    args.train_size = 500
    args.max_num_tuning_evals = 3

folder_to_add = ""
if args.dataset == "bpic17":
    conf_suffix = "_case" if args.confounding_type == "case" else ""
    folder_to_add = os.path.join("bpic17" + conf_suffix, str(args.n_stages))
elif args.dataset == "SimRepair":
    folder_to_add = "SimRepair"

PATH_BEGIN = str(args.train_size) + "_" + str(int(100*args.delta)) + "_"
DATA_FOLDER = os.path.join("data", folder_to_add, str(args.train_size), str(int(100 * args.delta)))
RESULTS_FOLDER = os.path.join("res", folder_to_add, str(args.train_size), str(int(100 * args.delta)))
make_dirs(args=args, DATA_FOLDER=DATA_FOLDER, RESULTS_FOLDER=RESULTS_FOLDER)

# log_cols=['case_nr', 'activity', 'timestamp', 'elapsed_time', 'duration',
# 					'bike_value', 'repair_severity', 'quality_uncertainty',
# 					'process_type', 'employee_competence', 'material_quality',
# 					'customer_patience', 'customer_friendliness', 'outcome']
# case_cols=['bike_value', 'repair_severity']
# event_cols=['activity', 'elapsed_time', 'quality_uncertainty',
# 						'process_type', 'employee_competence', 'material_quality']
# cat_cols=['activity', 'process_type']
# scale_cols=['elapsed_time', 'bike_value', 'repair_severity',
# 						'quality_uncertainty', 'employee_competence',
# 						'material_quality', 'outcome']
# last_state_cols=['elapsed_time']

# intervention_info des Stage-Dictionarys:
# 	name = 'choose_procedure'
# 	data_impact = 'direct'
# 	actions = ['start_standard', 'start_priority']
# 	action_width = 2
# 	action_depth = 1
# 	activities = ['start_standard', 'start_priority']
# 	column = 'activity'
# 	start_control_activity = ['initiate_case']
# 	end_control_activity = ['initiate_case']
# 	retain_method = 'precise'
# 	action_combinations = ('start_standard',)
# 	action_width_combinations = 2
# 	action_depth_combinations = 1
# 	len = 1
# 	RCT = False
# 	flat_activities = 'start_standard'

# filename = "repair_log_['choose_procedure']"
# policies_info = {
# 	'general': 'real',
# 	'take_employee': {'max_employee_waits': 3},
# 	'conduct_qc': {'max_qc': 3},
# 	'max_rerepairs': 3
# } )

# Generate Data
#Dieser Block prüft, ob Trainings- und Tuningdaten bereits früher erzeugt wurden.
# load_data() lädt die Daten, falls sie schon existieren, andernfalls werden sie neu generiert und gespeichert.
# Geht davon aus, dass die Dateien in pkl format vorliegen
if args.already_train_tune_generated:
    dataset_params = load_data(os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + "dataset_params"))
    dataset_params_list = load_data(os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + "dataset_params_list"))
    data = load_data(os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + "data"))
    
else:
    # Bei dataset == "bpic17" werden neue BPIC17-Daten simuliert. Dabei werden Trainingsgröße, delta, Anzahl der Entscheidungsstufen und Confounding-Art übergeben.
    if args.dataset == "bpic17":
        # DAS IST ERSTMAL NUR DAS GANZE ERSTELLEN UND RUNNEN DER BPIC2017 SIMULATION
        dataset_params, dataset_params_list, data = generate_training_and_tuning_bpic17(train_size=args.train_size, delta=args.delta, n_stages=args.n_stages, confounding_type=args.confounding_type)
    elif args.dataset == "SimRepair":
        print("Generating evaluation data for SimRepair")
        dataset_params, dataset_params_list, data = generate_training_and_tuning_simrepair(train_size=args.train_size, delta=args.delta, n_stages=args.n_stages, test_size=args.test_size)
    else:
        # Andernfalls werden SimBank-Daten erzeugt.
        dataset_params, dataset_params_list, data = generate_training_and_tuning(size=args.train_size, delta=args.delta)
    save_data(dataset_params, os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + "dataset_params"))
    save_data(dataset_params_list, os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + "dataset_params_list"))
    save_data(data, os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + "data"))

if args.dataset == "bpic17":
    eval_dfs = generate_eval_bpic17(args=args, dataset_params=dataset_params)
elif args.dataset == "SimRepair":
    eval_dfs = generate_eval_simrepair(args=args, dataset_params=dataset_params)
else:
    eval_dfs = generate_eval(args=args, dataset_params=dataset_params)

# Preprocessing
# Modelle trainieren und beim Tuning überwachen
prepped_data_dict = {"train": {}, "infer": {}, "utils": {}}
if args.already_train_tune_preprocessed:
    for encoding in args.encodings:
        data_train_list = load_data(os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + encoding + "preprocessed_data_train_list"))
        data_infer_list = load_data(os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + encoding + "preprocessed_data_infer_list"))
        prep_utils_list = load_data(os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + encoding + "preprocessed_utils_list"))
        prepped_data_dict["train"][encoding] = data_train_list
        prepped_data_dict["infer"][encoding] = data_infer_list
        prepped_data_dict["utils"][encoding] = prep_utils_list
else:

    # In Main preprocessor werden die Daten erstmal vorbereitet
    print("Preprocessing training and tuning data")
    preprocessor = ProcessPreprocessor(args=args, raw_data=data, DATASET_PARAMS_LIST=dataset_params_list)

    for encoding in args.encodings:
        data_train_list, data_infer_list, prep_utils_list = preprocessor.preprocess(encoding=encoding)
        #print(data_train_list)
        #print(data_infer_list)
        #print(prep_utils_list)
        # Save preprocessed data
        save_data(data_train_list, os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + encoding + "preprocessed_data_train_list"))
        save_data(data_infer_list, os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + encoding + "preprocessed_data_infer_list"))
        save_data(prep_utils_list, os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + encoding + "preprocessed_utils_list"))
        prepped_data_dict["train"][encoding] = data_train_list
        prepped_data_dict["infer"][encoding] = data_infer_list
        prepped_data_dict["utils"][encoding] = prep_utils_list

# NOTE: these are also preprocessed per datasize and delta, since the scaling will be different
# Vorberetung der Evaluationsdaten, die für die Inferenz verwendet werden. Diese Daten werden in eval_preps gespeichert.
eval_preps = {}
# Aus generate_eval_simrepair
# eval_dfs.keys ist immer Aktion 0, also start_standard, mmer Aktion 1, also start_priority
for action_combo in eval_dfs.keys():
    print(eval_dfs.keys())
    if action_combo == 'bank' or action_combo == 'optimal' or ('random') in action_combo: continue
    eval_preps[action_combo] = {}
    if args.already_eval_preprocessed:
        for encoding in args.encodings:
            data_eval_list = load_data(os.path.join(os.getcwd(), DATA_FOLDER, "eval", PATH_BEGIN + encoding + "_" + action_combo + "preprocessed_data_eval_list"))
            prep_utils_eval_list = load_data(os.path.join(os.getcwd(), DATA_FOLDER, "eval", PATH_BEGIN + encoding + "_" + action_combo + "preprocessed_utils_eval_list"))
            eval_preps[action_combo][encoding] = data_eval_list
    else:
        print(f"Preprocessing evaluation data for action combo: {action_combo}")
        preprocessor = ProcessPreprocessor(args=args, raw_data=eval_dfs[action_combo]["test_df"], DATASET_PARAMS_LIST=dataset_params_list)
        for encoding in args.encodings:
            _, data_eval_list, prep_utils_eval_list = preprocessor.preprocess(encoding=encoding, prep_utils_list=prepped_data_dict["utils"][encoding], eval=True, action_combo=action_combo)
            # Save preprocessed data
            save_data(data_eval_list, os.path.join(os.getcwd(), DATA_FOLDER, "eval", PATH_BEGIN + encoding + "_" + action_combo + "preprocessed_data_eval_list"))
            save_data(prep_utils_eval_list, os.path.join(os.getcwd(), DATA_FOLDER, "eval", PATH_BEGIN + encoding + "_" + action_combo + "preprocessed_utils_eval_list"))
            eval_preps[action_combo][encoding] = data_eval_list
print('\n')
        
# Tuning
# Method ist für das Tuning der modelle verantwortlich. Dabei werden die besten Parameter und Modelle für jede Methode gesammelt.
# Methods wird oben bei Aufruf von main.py übergeben
# "methods": [
#   "dtr-S-reg-R",
#   "dtr-T-reg-R",
#   "separate-S-reg-none",
#   "kmeans_q"
# ]
best_params_collection = {}
best_models_collection = {}
for method in args.methods:
    best_params_collection[method] = []
    best_models_collection[method] = []
    print(f"Tuning method: {method}")

    #-------Eigentlich wichtiger CodeBlock
    method_tune = Method(args=args, method=method, prepped_data_dict=prepped_data_dict)
    method_tune.run(tuning=True)

    best_params_list_of_dicts = method_tune.model_params_list_of_dicts
    best_models_list_of_dicts = method_tune.models_list_of_dicts
    # Collect the best parameters and models for each method
    best_params_collection[method] = best_params_list_of_dicts
    best_models_collection[method] = best_models_list_of_dicts

# # Training
# for iter in range(args.num_iterations):
#     print(f"Training iteration: {iter}")
#     if iter in args.iterations_to_skip: continue
#     for method in args.methods:
#         method_train = Method(args=args, method=method, prepped_data_dict=prepped_data_dict, best_model_params_list_of_dicts=best_params_collection[method], iter=iter)
#         method_train.run(tuning=False)

#         params_list_of_dicts = method_train.model_params_list_of_dicts
#         models_list_of_dicts = method_train.models_list_of_dicts

# # Evaluation
# avg_uplift = 0
# for iter in range(args.num_iterations):
#     print(f"Evaluating iteration: {iter}")
#     if iter in args.iterations_to_skip: continue
#     for method in args.methods:
#         print(f"    Evaluating method: {method}")
#         method_eval = Method(args=args, method=method, prepped_data_dict=prepped_data_dict, best_model_params_list_of_dicts=best_params_collection[method], iter=iter)
#         uplift, profit, df = method_eval.eval(preps_maps=eval_preps, dfs_map=eval_dfs)
#         avg_uplift += uplift

# # if args.wandb:
# #     wandb.log({
# #             "uplift": avg_uplift / args.num_iterations
# #         })
    
# #     wandb.finish()
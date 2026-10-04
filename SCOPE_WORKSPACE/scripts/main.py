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
from SimBPIC17.simbpic17_run import generate_training_and_tuning_bpic17, generate_eval_bpic17
from SimRepair.generate_sequential_simbank import generate_training_and_tuning as generate_training_and_tuning_simrepair
from SimRepair.generate_sequential_simbank import generate_eval as generate_eval_simrepair
from src.methods.method_main import Method
from src.utils.mini_tools import load_data, parse, save_data, make_dirs
from src.utils.prep_tools.main_prep import ProcessPreprocessor
import wandb

# Parsing
# z.b.
# args.dataset = "SimBank"
# args.n_stages = 2

# args.model_categories = [
#     "ml",
#     "ml",
#     "ml"
# ]

# args.model_specifics = [
#     "xgb",
#     "xgb",
#     "xgb"
# ]

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
else:
    conf_suffix = ""
args.conf_suffix = conf_suffix

PATH_BEGIN = str(args.train_size) + "_" + str(int(100*args.delta)) + "_"
DATA_FOLDER = os.path.join("data", folder_to_add, str(args.train_size), str(int(100 * args.delta)))
RESULTS_FOLDER = os.path.join("res", folder_to_add, str(args.train_size), str(int(100 * args.delta)))
make_dirs(args=args, DATA_FOLDER=DATA_FOLDER, RESULTS_FOLDER=RESULTS_FOLDER)

# if args.wandb:
#     wandb.login(key="")

#     wandb.init(
#         project="CLIPPS",      # e.g. "treatment-effect-nn"
#         config=args,         # logs your hyperparameters
#         name=str(args.kmeans_config) + args.dataset,           # optional, for clarity
#     )

# Generate Data
#Dieser Block prüft, ob Trainings- und Tuningdaten bereits früher erzeugt wurden.
if args.already_train_tune_generated:
    dataset_params = load_data(os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + "dataset_params"))
    dataset_params_list = load_data(os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + "dataset_params_list"))
    data = load_data(os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + "data"))
    
else:
    # Bei dataset == "bpic17" werden neue BPIC17-Daten simuliert. Dabei werden Trainingsgröße, delta, Anzahl der Entscheidungsstufen und Confounding-Art übergeben.
    if args.dataset == "bpic17":
        # DAS IST ERSTMAL NUR DAS GANZE ERSTELLEN UND RUNNEN DER BPIC2017 SIMULATION
        dataset_params, dataset_params_list, data = generate_training_and_tuning_bpic17(train_size=args.train_size, delta=args.delta, n_stages=args.n_stages, confounding_type=args.confounding_type)
    elif args.dataset == "Sim        dataset_params, dataset_params_list, data = generate_training_and_tuning_simrepair(size=args.train_size, delta=args.delta, test_size=args.test_size)
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

    preprocessor = ProcessPreprocessor(args=args, raw_data=data, DATASET_PARAMS_LIST=dataset_params_list)

    for encoding in args.encodings:
        data_train_list, data_infer_list, prep_utils_list = preprocessor.preprocess(encoding=encoding)
        # Save preprocessed data
        save_data(data_train_list, os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + encoding + "preprocessed_data_train_list"))
        save_data(data_infer_list, os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + encoding + "preprocessed_data_infer_list"))
        save_data(prep_utils_list, os.path.join(os.getcwd(), DATA_FOLDER, "training_and_tuning", PATH_BEGIN + encoding + "preprocessed_utils_list"))
        prepped_data_dict["train"][encoding] = data_train_list
        prepped_data_dict["infer"][encoding] = data_infer_list
        prepped_data_dict["utils"][encoding] = prep_utils_list

# NOTE: these are also preprocessed per datasize and delta, since the scaling will be different
eval_preps = {}
for action_combo in eval_dfs.keys():
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
best_params_collection = {}
best_models_collection = {}
for method in args.methods:
    best_params_collection[method] = []
    best_models_collection[method] = []
    print(f"Tuning method: {method}")
    method_tune = Method(args=args, method=method, prepped_data_dict=prepped_data_dict)
    method_tune.run(tuning=True)

    best_params_list_of_dicts = method_tune.model_params_list_of_dicts
    best_models_list_of_dicts = method_tune.models_list_of_dicts
    # Collect the best parameters and models for each method
    best_params_collection[method] = best_params_list_of_dicts
    best_models_collection[method] = best_models_list_of_dicts

# Training
for iter in range(args.num_iterations):
    print(f"Training iteration: {iter}")
    if iter in args.iterations_to_skip: continue
    for method in args.methods:
        method_train = Method(args=args, method=method, prepped_data_dict=prepped_data_dict, best_model_params_list_of_dicts=best_params_collection[method], iter=iter)
        method_train.run(tuning=False)

        params_list_of_dicts = method_train.model_params_list_of_dicts
        models_list_of_dicts = method_train.models_list_of_dicts

# Evaluation
avg_uplift = 0
for iter in range(args.num_iterations):
    print(f"Evaluating iteration: {iter}")
    if iter in args.iterations_to_skip: continue
    for method in args.methods:
        print(f"    Evaluating method: {method}")
        method_eval = Method(args=args, method=method, prepped_data_dict=prepped_data_dict, best_model_params_list_of_dicts=best_params_collection[method], iter=iter)
        uplift, profit, df = method_eval.eval(preps_maps=eval_preps, dfs_map=eval_dfs)
        avg_uplift += uplift

# if args.wandb:
#     wandb.log({
#             "uplift": avg_uplift / args.num_iterations
#         })
    
#     wandb.finish()
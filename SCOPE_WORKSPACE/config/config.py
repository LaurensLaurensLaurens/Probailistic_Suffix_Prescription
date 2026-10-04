import numpy as np
from hyperopt import hp
import os
import itertools

current_dir = os.path.dirname(os.path.abspath(__file__))
# Go one level up, then into folder_b/data
checkpoint_path = os.path.join(current_dir, "..", "res", "checkpoints", "checkpoint_")
# Normalize to an absolute path
checkpoint_path = os.path.abspath(checkpoint_path)

path = "c:\\Users\\u0166838\\OneDrive - KU Leuven\\Documents\\Doc\\Code\\SCOPE"

nn_space = {
    'lr': hp.loguniform('lr', np.log(0.001), np.log(0.01)),
    'batch_size': hp.choice('batch_size', [128, 256]),
    # Model dimensionality (keeping layer counts fixed)
    'dim_lstm': hp.choice('dim_lstm', [16, 32, 64]),
    'dim_dense': hp.choice('dim_dense', [16, 32, 64]),
    'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-3)),
    # Regularization and stability
    'dropout': hp.uniform('dropout', 0.0, 0.2),
    'grad_norm': hp.uniform('grad_norm', 0.5, 2.0),
    "num_epochs": hp.choice('nr_epochs', [50, 100, 200, 500])
}

logreg_space = {
    "C": hp.loguniform("logreg_C", -5, 2),  # ~exp(uniform(-5, 2)) → range ~ [0.0067, 7.4]
    "penalty": hp.choice("logreg_penalty", ["l1", "l2"]),
    "solver": hp.choice("logreg_solver", ["liblinear", "saga"])
}

xgb_space = {
    "n_estimators": hp.quniform("xgb_n_estimators", 50, 300, 10),     # Integers in [50, 300]
    "max_depth": hp.quniform("xgb_max_depth", 3, 10, 1),              # Depths 3–10
    "learning_rate": hp.loguniform("xgb_learning_rate", -4, 0)        # ~[0.018, 1.0]
}

rf_space = {
    "n_estimators": hp.quniform("rf_n_estimators", 50, 300, 10),
    "max_depth": hp.quniform("rf_max_depth", 3, 10, 1),
    "min_samples_leaf": hp.quniform("rf_min_samples_leaf", 1, 10, 1)
}

dt_space = {
    "max_depth": hp.choice("dt_max_depth", [None, 5, 10, 20]),
    "min_samples_split": hp.quniform("dt_min_samples_split", 2, 20, 1),
    "min_samples_leaf": hp.quniform("dt_min_samples_leaf", 1, 10, 1)
}

kmeans_space = {
    "n_clusters": hp.qloguniform("kmeans_n_clusters", np.log(2), np.log(500), 1),
}

q_learner_space = {
    "scale_factor_type": hp.choice("scale_factor_type", ["none", "linear", "smooth", "step"]),
    "scale_factor_step_smooth": hp.quniform("scale_factor_step_smooth", 1, 100, 1),  # Step size for the smooth scale factor
    "alpha_max": hp.uniform("alpha_max", 0.5, 0.95),  # Exploration rate for epsilon-greedy policy
    "alpha_min": hp.uniform("alpha_min", 0.01, 0.5),  # Exploration rate for epsilon-greedy policy
    "epsilon": hp.uniform("epsilon", 0.25, 0.95),  # Exploration rate for epsilon-greedy policy
}

def make_kmeans_q_space(feature_names, args):
    """
    Dynamically builds the hyperopt space given the feature list.
    """

    # drop 'case_nr', 'a', 'activity', 'outcome'
    feature_names = [f for f in feature_names if f not in ['case_nr', 'a', 'activity', 'outcome']]

    # max_factor_step = 100 if args.dataset == 'SimBank' else 10000
    if args.kmeans_config[0] == 'big':
        max_factor_step = 10000
    elif args.kmeans_config[0] == 'mid':
        max_factor_step = 1000
    else:
        max_factor_step = 100

    space = {
        "n_clusters": hp.qloguniform("n_clusters", np.log(2), np.log(500), 1),
        "scale_factor_type": hp.choice("scale_factor_type", ["none", "linear", "smooth", "step"]),
        "scale_factor_step_smooth": hp.quniform("scale_factor_step_smooth", 1, max_factor_step, 1),
        "alpha_max": hp.uniform("alpha_max", 0.000001, 0.99),
        "alpha_min": hp.uniform("alpha_min", 0.000001, 0.99),
        "epsilon": hp.uniform("epsilon", 0.000001, 0.99),
        # ⬇️ Only valid feature subsets (at least 2 features)
        "feature_names": {f: hp.choice(f"_keep_{f}", [0, 1]) for f in feature_names},
        "gamma": hp.uniform("gamma", 0.25, 1.0),
    }

    return space

space_dict = {
    'lstm': nn_space,
    'vanilla_nn': nn_space,
    'logreg': logreg_space,
    'xgb': xgb_space,
    'rf': rf_space,
    'dt': dt_space,
    'kmeans': kmeans_space,
    'q_learner': q_learner_space,
    'kmeans_q': None,  # to be set later dynamically
}

model_dict = {
    'nn': {
        'type': 'nn',
        'params': nn_space
    },
    'logreg': {
        'type': 'logreg',
        'params': logreg_space
    },
    'xgb': {
        'type': 'xgb',
        'params': xgb_space
    },
    'dt': {
        'type': 'dt',
        'params': dt_space
    }
}

model_params = {
    # Meta-parameters
    "method": "dtr-S-reg-R",
    "dataset": "SimBank",
    "pos_rewards": False,
    "encoding": "agg",
    "model_category": "dl",
    "model_specific": "lstm",
    "cross_fitting": False,  # Whether to use cross-fitting in the model training
    "target": "ps",
    "stage": 0,
    "ps_model_params": {"model_specific": "logreg", "encoding": "agg", "model_category": "ml"},  # For calibration, this is the model used to estimate the propensity score
    "prev_ps_model_params": {"model_specific": "logreg", "encoding": "agg", "model_category": "ml"},  # For outcome, this is the model used to estimate the propensity score
    "prev_outcome_model_params": {"model_specific": "lstm", "encoding": "agg", "model_category": "dl"},  # For outcome, this is the model used to estimate the previous outcome
    "cluster_model_specific": "kmeans",

    # Save paths
    "model_savepath_checkpoint": checkpoint_path,

    # Dataset parameters
    "dim_x_case": 0,  # To be set later
    "dim_x_event": 0,  # To be set later
    "dim_t": 0,  # To be set later
    "dim_output": 0,  # To be set later

    # DL parameters
    "n_lstm_layers": 2,
    "n_dense_layers_in_lstm": 1,
    "n_dense_layers": 3,
    "dim_dense": 64,
    "dim_lstm": 64,
    "masked": True,  # Whether to use masking in the LSTM
    "dropout": 0.0,  # Dropout rate
    "batch_size": 64,  # Batch size for training
    "lr": 0.0001,  # Learning rate for the optimizer
    "weight_decay": 1e-4,  # Weight decay for regularization
    "num_epochs": 100,  # Number of epochs for training
    "grad_norm": 1.0,  # Gradient clipping norm
    "print_every_iters": 100000,  # Print training progress every n iterations
    "eval_every": 2,  # Evaluate the model every n epochs
    "early_stop": False,  # Whether to use early stopping

    # Logistic Regression parameters
    "C": 1.0,  # Inverse of regularization strength
    "penalty": "l2",  # Regularization type
    "solver": "liblinear",  # Solver to use for optimization

    # XGBoost parameters & Random Forest parameters
    "n_estimators": 100,  # Number of trees in the ensemble
    "max_depth": 6,  # Maximum depth of a tree
    "learning_rate": 0.1,  # Step size shrinkage used in update to prevent overfitting
    "subsample": 0.8,  # Subsample ratio of the training instances
    "colsample_bytree": 0.9,  # Subsample ratio of
    "min_samples_leaf": 5,  # Minimum samples required to be at a leaf node
    
    # KMeans parameters
    "n_clusters": 10,  # Number of clusters for KMeans
    "init": "k-means++",  # Method for initialization
    "n_init": 10,  # Number of times the KMeans algorithm will be run with different centroid seeds
    "feature_names": {},  # To be set later for KMeansQ

    # Q-learning parameters
    "scale_factor_type": "none", # Either 'none', 'linear', 'smooth', or 'step'
    "scale_factor_step_smooth": 50,  # Step size for the smooth scale factor
    "alpha_max": 1,  # Maximum learning rate
    "alpha_min": 0.01,
    "gamma": 1.0,  # Discount factor for future rewards
    "epsilon": 0.1,  # Exploration rate for epsilon-greedy policy
    "normalize_reward": True,  # Whether to normalize the reward 
    "change_zero_reward": True, # THIS MAKES A LOT OF DIFFERENCE IN PERFORMANCE (True --> way better)
    "norm_mdp": True, # THIS MAKES A LOT OF DIFFERENCE IN PERFORMANCE (True --> way better)
    "loop_penalty": 0,  # Penalty for loops in the trace
    "exceeding_traces_length_penalty": -1000,  # Penalty for traces exceeding
    "max_trace_length": 100,  # Maximum length of a trace

    # seed
    "seed": 42,  # Random seed for reproducibility
}

# DATASET CONFIGURATION
dataset_configs = {
    "SimBank": {
        "case_id_col": "case_nr",
        "activity_col": "activity",
        "timestamp_col": "timestamp",
        "outcome_col": "outcome",
        "actions": [["start_priority", "start_standard"], 
                    [0.07, 0.08, 0.09]], # actions per stage
        "prev_activities": [["initiate_application"],
                            ["start_priority", "validate_application"]] # previous activities per stage
    },
    "bpic17": {
        "case_id_col": "case_nr",
        "activity_col": "activity",
        "timestamp_col": "timestamp",
        "outcome_col": "outcome",
        "actions": [["wait_incomplete_files", "call_incomplete_files"],
                    ["wait_incomplete_files", "call_incomplete_files"]], # actions NOTE: not per stage, just repeated two times, but the number of stages can be bigger
        "prev_activities": [["validate_application"],
                            ["wait_incomplete_files", "call_incomplete_files"]] # previous activities per stage
    },
}
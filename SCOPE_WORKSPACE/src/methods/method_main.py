import os
import time
from copy import deepcopy
from functools import reduce
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from hyperopt import STATUS_OK, Trials, fmin, tpe, space_eval
from functools import partial

from config.config import space_dict, make_kmeans_q_space
from src.methods.scope.scope_functions import SCOPEFunctions
from src.methods.separate.separate_functions import SeparateFunctions
from src.methods.kmeans_q.kmeans_q_functions import KMeansQFunctions
from src.utils.model_tools.model_training import ModelTrainer
from src.utils.model_tools.model_eval import ModelEval
from src.utils.mini_tools import (
    create_splits,
    get_model_params_list_of_dicts,
    save_data, load_data
)
# Das kommt ins Spiel nach dem Preprocessing Step
class Method():
    # args ist beispielsweise: Namespace(dataset="SimBank", methods=["dtr-S-reg-R"], n_stages=2, train_size=3, test_size=3, delta=0.95, encodings=["agg"], num_iterations=1)
    # "dtr-S-reg-R" ist die methode, also hier z.b. Dynamic Treatment Regime mit Single Stage, Regression, Ridge als Modell
    # DAS UNTERSCHEIDET SICH ABER JE NACH AGGREGATION UND kmeans
        # prepped_data_dict = { # Das kommt aus der main_prep.py Datei
        #     "train": { "train": Trainingsdaten auswählen.
        #         "tensor": [
        #             # Entscheidungsstufe 0
        #             {
        #                 "Y": tensor([ Zielwert, beispielsweise Profit oder Outcome als Tensor
        #                     [1200.0], # Zielwert für Fall 101
        #                     [-300.0] # Zielwert für Fall 104
        #                 ]),

        #                 "case_nr": tensor([ # ursprüngliche Fallnummer als Tensor
        #                     101.0,
        #                     104.0
        #                 ]),

        #                 "T": tensor([ # tatsächlich ausgeführte Behandlung/Aktion
        #                     [1.0], # Aktion 1 für Fall 101
        #                     [0.0] # Aktion 0 für Fall 104
        #                 ]),

        #                 "X_case": tensor([ statische Fallmerkmal
        #                     [ 1.0],
        #                     [-1.0]
        #                 ]),

        #                 "X_event": tensor([ bisheriger Prozessverlauf

        #                     # Fall 101: [Eventmerkmal, Präfixposition]
        #                     [
        #                         [ 1.0,  0.0, 0.0, 0.0],  # activity_start_standard
        #                         [ 0.0,  1.0, 0.0, 0.0],  # activity_start_priority
        #                         [-1.0,  0.2, 0.0, 0.0]   # elapsed_time
        #                     ],

        #                     # Fall 104
        #                     [
        #                         [ 1.0,  0.0, 0.0, 0.0],
        #                         [ 0.0,  0.0, 0.0, 0.0],
        #                         [-0.6,  0.0, 0.0, 0.0]
        #                     ]
        #                 ]),

        #                 "prefix_len": tensor([ Anzahl der echten Eventpositione
        #                     2.0,
        #                     1.0
        #                 ])
        #             },

        #             # Entscheidungsstufe 1
        #             {
        #                 "Y": tensor([
        #                     [1200.0]
        #                 ]),

        #                 "case_nr": tensor([
        #                     101.0
        #                 ]),

        #                 # Drei mögliche Zinsaktionen:
        #                 # Index 0 = 0.07
        #                 # Index 1 = 0.08
        #                 # Index 2 = 0.09
        #                 "T": tensor([
        #                     [0.0, 0.0, 1.0]
        #                 ]),

        #                 "X_case": tensor([
        #                     [1.0]
        #                 ]),

        #                 "X_event": tensor([
        #                     [
        #                         [1.0, 0.0, 0.0, 0.0],
        #                         [0.0, 1.0, 1.0, 0.0],
        #                         [-1.0, 0.2, 0.6, 0.0]
        #                     ]
        #                 ]),

        #                 "prefix_len": tensor([
        #                     3.0
        #                 ])
        #             }
        #         ]
        #     },

        #     "infer": {
        #         "tensor": [
        #             # Inferenzdaten für Stufe 0
        #             {
        #                 "Y": tensor([
        #                     [800.0]
        #                 ]),
        #                 "case_nr": tensor([
        #                     109.0
        #                 ]),
        #                 "T": tensor([
        #                     [1.0]
        #                 ]),
        #                 "X_case": tensor([
        #                     [0.35]
        #                 ]),
        #                 "X_event": tensor([
        #                     [
        #                         [1.0, 0.0, 0.0, 0.0],
        #                         [0.0, 1.0, 0.0, 0.0],
        #                         [0.1, 0.4, 0.0, 0.0]
        #                     ]
        #                 ]),
        #                 "prefix_len": tensor([
        #                     2.0
        #                 ])
        #             },

        #             # Inferenzdaten für Stufe 1
        #             {
        #                 "Y": tensor([
        #                     [800.0]
        #                 ]),
        #                 "case_nr": tensor([
        #                     109.0
        #                 ]),
        #                 "T": tensor([
        #                     [0.0, 1.0, 0.0]
        #                 ]),
        #                 "X_case": tensor([
        #                     [0.35]
        #                 ]),
        #                 "X_event": tensor([
        #                     [
        #                         [1.0, 0.0, 0.0, 0.0],
        #                         [0.0, 1.0, 1.0, 0.0],
        #                         [0.1, 0.4, 0.7, 0.0]
        #                     ]
        #                 ]),
        #                 "prefix_len": tensor([
        #                     3.0
        #                 ])
        #             }
        #         ]
        #     },

        #     "utils": {
        #         "tensor": [
        #             # Metadaten für Stufe 0
        #             {
        #                 "scaler_dict_train": {
        #                     "amount": StandardScaler(
        #                         mean=37500.0,
        #                         scale=12500.0
        #                     ),
        #                     "elapsed_time": StandardScaler(
        #                         mean=2.5,
        #                         scale=1.5
        #                     )
        #                 },

        #                 "oh_encoder_dict_train": {
        #                     "activity": OneHotEncoder(
        #                         categories=[
        #                             "start_standard",
        #                             "start_priority"
        #                         ]
        #                     )
        #                 },

        #                 "max_process_len": 4,

        #                 "case_cols_encoded": [
        #                     "amount"
        #                 ],

        #                 "event_cols_encoded": [
        #                     "activity_start_standard",
        #                     "activity_start_priority",
        #                     "elapsed_time"
        #                 ],

        #                 "dim_x_case": 1,
        #                 "dim_x_event": 3,
        #                 "dim_t": 1,
        #                 "dim_output": 1
        #             },

        #             # Metadaten für Stufe 1
        #             {
        #                 "scaler_dict_train": {
        #                     "amount": StandardScaler(
        #                         mean=37500.0,
        #                         scale=12500.0
        #                     ),
        #                     "elapsed_time": StandardScaler(
        #                         mean=3.0,
        #                         scale=1.8
        #                     )
        #                 },

        #                 "oh_encoder_dict_train": {
        #                     "activity": OneHotEncoder(
        #                         categories=[
        #                             "calculate_offer",
        #                             "receive_acceptance"
        #                         ]
        #                     )
        #                 },

        #                 "max_process_len": 4,

        #                 "case_cols_encoded": [
        #                     "amount"
        #                 ],

        #                 "event_cols_encoded": [
        #                     "activity_calculate_offer",
        #                     "activity_receive_acceptance",
        #                     "elapsed_time"
        #                 ],

        #                 "dim_x_case": 1,
        #                 "dim_x_event": 3,
        #                 "dim_t": 3,
        #                 "dim_output": 1
        #             }
        #         ]
        #     }
        # }
    # best_model_params_list_of_dicts = [{"method":"dtr-S-reg-R", "model_category":"ml", "model_specific":"xgb", "target":"outcome", "stage":0, "encoding":"agg", "seed":42}, {"method":"dtr-S-reg-R", "model_category":"ml", "model_specific":"xgb", "target":"effect", "stage":1, "encoding":"agg", "seed":42}]
    def __init__(self, args, method, prepped_data_dict, best_model_params_list_of_dicts=None, iter=0):
        to_add = ''
        if args.dataset == 'bpic17':
            conf_suffix = "_case" if args.confounding_type == "case" else ""
            to_add = os.path.join("bpic17" + conf_suffix, str(args.n_stages))
        elif args.dataset == "SimRepair":
            to_add = "SimRepair"
        self.RESULTS_FOLDER = os.path.join("res", to_add, str(args.train_size), str(int(100 * args.delta)))
        self.PATH_BEGIN = str(args.train_size) + "_" + str(int(100*args.delta)) + "_"

        self.iter = iter
        self.method = method
        self.args = args
        self.prepped_data_dict = prepped_data_dict

        if best_model_params_list_of_dicts is None:
            self.model_params_list_of_dicts = get_model_params_list_of_dicts(method=self.method, args=args, prep_utils=prepped_data_dict["utils"])
        else:
            self.model_params_list_of_dicts = deepcopy(best_model_params_list_of_dicts)

        self.models_list_of_dicts = [{} for _ in range(self.args.n_stages)]

        # SCOPEFunctions bei Methoden mit "dtr"
        # Für Dynamic Treatment Regimes. Die Stufen werden rückwärts verarbeitet. Ergebnisse späterer Stufen beeinflussen die Zielwerte früherer Stufen. Außerdem können Outcome-, Effekt- und Propensity-Score-Modelle kombiniert werden.
        if "dtr" in self.method:
            self.method_functions = SCOPEFunctions(model_params_list_of_dicts=self.model_params_list_of_dicts)
        elif "separate" in self.method:
            self.method_functions = SeparateFunctions(model_params_list_of_dicts=self.model_params_list_of_dicts)
        elif self.method == "kmeans_q":
            self.method_functions = KMeansQFunctions(model_params_list_of_dicts=self.model_params_list_of_dicts)
        
    # Wird von eval in main.py aufgerufen für training und evaluation
    # Für jede Entscheidungsstufe und jedes benötigte Modell die passenden Daten vorzubereiten, das Modell trainieren oder laden und das Ergebnis speichern.
    def run(self, tuning=False):
        # NOTE: here we start to go over the stages in reverse order for the backward induction
        # Runtime of the full method execution (all stages/targets in this run call)
        runtime_start = time.perf_counter()
        # range(Start, Ende, Schrittweite)
#         model_params_list_of_dicts = [
#     # Index 0 = Entscheidungsstufe 0
#     {
#         "ps": "nope",

#         "outcome": {
#             "method": "dtr-S-reg-R",
#             "dataset": "SimBank",
#             "stage": 0,

#             "learner_method": "S",
#             "action_recomm_method": "reg",
#             "value_function_method": "R",

#             "target": "outcome",
#             "encoding": "agg",
#             "model_category": "ml",
#             "model_specific": "xgb",

#             # Diese Werte kommen aus prep_utils["agg"][0].
#             # Zahlen hier nur beispielhaft:
#             "dim_x_case": 8,
#             "dim_x_event": 12,
#             "dim_t": 2,
#             "dim_output": 1,

#             # XGBoost-Einstellungen
#             "n_estimators": 100,
#             "max_depth": 6,
#             "learning_rate": 0.1,
#             "subsample": 0.8,
#             "colsample_bytree": 0.9,

#             "cross_fitting": False,

#             # DTR benötigt Informationen über die folgende Stufe:
#             "prev_ps_model_params": "nope",
#             "prev_outcome_model_params": {
#                 # Tatsächlich steht hier eine vollständige Kopie
#                 # der outcome-Konfiguration von Stufe 1.
#                 "stage": 1,
#                 "target": "outcome",
#                 "model_category": "ml",
#                 "model_specific": "xgb"
#             }
#         },

#         "effect": "nope"
#     },

#     # Index 1 = Entscheidungsstufe 1
#     {
#         "ps": "nope",

#         "outcome": {
#             "method": "dtr-S-reg-R",
#             "dataset": "SimBank",
#             "stage": 1,

#             "learner_method": "S",
#             "action_recomm_method": "reg",
#             "value_function_method": "R",

#             "target": "outcome",
#             "encoding": "agg",
#             "model_category": "ml",
#             "model_specific": "xgb",

#             # Aus prep_utils["agg"][1].
#             # Zahlen wieder nur beispielhaft:
#             "dim_x_case": 10,
#             "dim_x_event": 14,
#             "dim_t": 3,
#             "dim_output": 1,

#             "n_estimators": 100,
#             "max_depth": 6,
#             "learning_rate": 0.1,
#             "subsample": 0.8,
#             "colsample_bytree": 0.9,

#             "cross_fitting": False
#         },

#         "effect": "nope"
#     }
# ]
        for stage in range(len(self.model_params_list_of_dicts) - 1, -1, -1):
            self.stage = stage
            model_params_dict = self.model_params_list_of_dicts[stage]

            # Also beispielsweise zuerst outcome, dann effect oder ps
            for target, model_params in model_params_dict.items():
                print(f"    Stage: {stage}, Target: {target}")

                # Init variables
                if model_params == "nope": continue
                to_add_path = "tuning" if tuning else (str(self.iter) + "_training")
                to_add_folder = "tuning" if tuning else "training"
                to_add_cross_fitting = "cross_fitted_" if "dtr" in self.method and self.args.cross_fitting else ""
                model_params["savepath_ps_model"] = os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), to_add_folder, self.PATH_BEGIN + self.method + "_ps_" + model_params["ps_model_params"]["model_specific"] + "_" + str(stage) + "_" + to_add_cross_fitting + to_add_path  + "_model")
                model_params["seed"] = 42 + 5*self.iter
                self.model_params = model_params

                if (target in self.args.already_tuned_list and tuning) or (target in self.args.already_trained_list and not tuning) or (self.args.already_tuned and tuning) or (self.args.already_trained and not tuning):
                    # load the best model and params
                    self.best_model = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), to_add_folder, self.PATH_BEGIN + self.method + "_" + target + "_" + model_params["model_specific"] + "_" + str(stage) + "_" + to_add_cross_fitting + to_add_path + "_model"), is_state_dict=(self.model_params["model_category"] == "dl" and "S" in self.method and target == "outcome"))
                    best_params = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), to_add_folder, self.PATH_BEGIN + self.method + "_" + target + "_" + model_params["model_specific"] + "_" + str(stage) + "_" + to_add_cross_fitting + to_add_path + "_params"))
                    model_params.update(best_params)
                else:
                    data_train_list_ps, data_infer_list_ps = None, None
                    data_train_list_prev_ps, data_infer_list_prev_ps = None, None
                    data_train_list_prev_outcome, data_infer_list_prev_outcome = None, None
                    if "dtr" in self.method:
                        if (target == "effect"):
                            # still pass the class model_params to get the correct splits
                            # data_train_list_ps → Daten zum Trainieren des Propensity-Score-Modells
                            # data_infer_list_ps → Daten für unabhängige PS-Vorhersagen, eventuell None

                            # self.prepped_data_dict["train"][encoding]: Trainingsdaten in der vom PS-Modell benötigten Kodierung
                            # self.prepped_data_dict["infer"][encoding]: entsprechende Inferenzdaten
                            # model_params: bestimmt, wie die Daten aufgeteilt oder zusammengeführt werden

                            # create_splits() entscheidet anschließend modellabhängig:
                            # Trainings- und Inferenzdaten getrennt lassen,
                            # beide zusammenführen oder
                            # die Inferenzdaten nochmals aufteilen.

                            # data_train_list_ps = [
                            #     {   # Stage 0
                            # X_case: Fallmerkmale
                            #         "X_case": Tensor(...),
                            # X_event: Ereignisverlauf
                            #         "X_event": Tensor(...),
                            #         "T": Tensor(...),
                            #         "Y": Tensor(...),
                            #         "prefix_len": Tensor(...),
                            #         "case_nr": Tensor(...)
                            #     },
                            #     {   # Stage 1
                            #         "X_case": Tensor(...),
                            #         "X_event": Tensor(...),
                            #         "T": Tensor(...),
                            #         "Y": Tensor(...),
                            #         "prefix_len": Tensor(...),
                            #         "case_nr": Tensor(...)
                            #     }
                            # ]
                            data_train_list_ps, data_infer_list_ps = create_splits(data_train_list=self.prepped_data_dict["train"][model_params["ps_model_params"]["encoding"]], data_infer_list=self.prepped_data_dict["infer"][model_params["ps_model_params"]["encoding"]], model_params=model_params)
                        # Diese Zeile oben bereitet eigene Trainings- und Inferenzdaten für das Propensity-Score-Modell vor:
                        # self.prepped_data_dict["train"][encoding]: Trainingsdaten in der vom PS-Modell benötigten Kodierung
                        # self.prepped_data_dict["infer"][encoding]: entsprechende Inferenzdaten
                        # model_params: bestimmt, wie die Daten aufgeteilt oder zusammengeführt werden

                        # len(model_params_list_of_dicts) sind die Stages
                        if (target == "outcome" or target == "effect") and stage < len(self.model_params_list_of_dicts) - 1:
                            if self.model_params_list_of_dicts[stage + 1]["ps"] != "nope":
                                data_train_list_prev_ps, data_infer_list_prev_ps = create_splits(data_train_list=self.prepped_data_dict["train"][model_params["prev_ps_model_params"]["encoding"]], data_infer_list=self.prepped_data_dict["infer"][model_params["prev_ps_model_params"]["encoding"]], model_params=model_params)
                            data_train_list_prev_outcome, data_infer_list_prev_outcome = create_splits(data_train_list=self.prepped_data_dict["train"][model_params["prev_outcome_model_params"]["encoding"]], data_infer_list=self.prepped_data_dict["infer"][model_params["prev_outcome_model_params"]["encoding"]], model_params=model_params)
                            # BACKWARD INDUCTION:
                            # Die Zielwerte Y der nächsten Stage (stage + 1) werden durch die zuvor berechneten Outcome-Zielwerte ersetzt. Da die Stages rückwärts durchlaufen werden, wurden diese Werte bereits berechnet.
                            # prev_outcomes_train wird später erst erzeugt, das geht, da die Stage chronologisch später läuft
                            
                            # Bei Stage 0 bedeutet das:
                            # data_train_list_prev_outcome[1]["Y"] = (
                            #     prev_outcomes_train
                            # )
                            data_train_list_prev_outcome[stage + 1]["Y"] = prev_outcomes_train
                            #Dasselbe geschieht optional mit den Inferenzdaten:
                            if data_infer_list_prev_outcome is not None and prev_outcomes_infer is not None:
                                data_infer_list_prev_outcome[stage + 1]["Y"] = prev_outcomes_infer

                    # data_for_other_models
                    data_lists_for_other_models = {"ps": {"train": data_train_list_ps, "infer": data_infer_list_ps},
                                                "prev_ps": {"train": data_train_list_prev_ps, "infer": data_infer_list_prev_ps},
                                                "prev_outcome": {"train": data_train_list_prev_outcome, "infer": data_infer_list_prev_outcome}}

                    # Split data correctly
                    data_train_list, data_infer_list = create_splits(data_train_list=self.prepped_data_dict["train"][model_params["encoding"]], data_infer_list=self.prepped_data_dict["infer"][model_params["encoding"]], model_params=model_params)

                    # Prepare data if needed (e.g., to calculate the targets of outcome in stage 0)
                    # vorbereitete Inferenz-/Validierungsdaten für das aktuell zu trainierende Modell.
                    #   data_infer_list = [
                    #     # Stage 0
                    #     {
                    #         "case_nr": tensor([101., 102., 103.]),

                    #         "X_case": tensor([
                    #             [ 0.25, -0.60, 1.00, 0.30],
                    #             [-0.40,  0.20, 0.00, 0.75],
                    #             [ 0.80,  0.10, 1.00, 0.20]
                    #         ]),

                    #         "X_event": None,
                    #         "prefix_len": None,

                    #         "T": tensor([
                    #             [0.],
                    #             [1.],
                    #             [0.]
                    #         ]),

                    #         "Y": tensor([
                    #             [1200.],
                    #             [-300.],
                    #             [800.]
                    #         ])
                    #     },

                    #     # Stage 1
                    #     {
                    #         "case_nr": tensor([101., 103.]),

                    #         "X_case": tensor([
                    #             [0.35, -0.50, 1.00, 0.40, 2.00],
                    #             [0.90,  0.15, 1.00, 0.25, 3.00]
                    #         ]),

                    #         "X_event": None,
                    #         "prefix_len": None,

                    #         "T": tensor([
                    #             [0., 1., 0.],   # beispielsweise Aktion 0.08
                    #             [0., 0., 1.]    # beispielsweise Aktion 0.09
                    #         ]),

                    #         "Y": tensor([
                    #             [1200.],
                    #             [800.]
                    #         ])
                    #     }
                    # ]
                    #
                    # DAS PASSIERT FÜR JEDE STAGE
                    # Das heitß konkret z.n. : 
                    # Hier passiert die Ganze backward induction etc.
                    # Q_stage_1(0.07) = 900
                    # Q_stage_1(0.08) = 1250
                    # Q_stage_1(0.09) = 1050
                    self.data_train, self.data_infer, self.weights_train, self.weights_infer, self.data_train_ps, self.data_infer_ps = self.method_functions.prepare(data_train_list=data_train_list, data_infer_list=data_infer_list, stage=self.stage, model_params=model_params, data_lists_for_other_models=data_lists_for_other_models)

                    # DTR bedeutet Dynamische Entscheidungs- beziehungsweise Interventionsstrategie.
                    if "dtr" in self.method and (target == "outcome") and stage > 0:
                        # BACKWARD INDUCTION:
                        prev_outcomes_train = deepcopy(self.data_train["Y"])
                        if self.data_infer is not None:
                            prev_outcomes_infer = deepcopy(self.data_infer["Y"])

                    # Train or Tune
                    # Ja. self.best_model ist das Modell mit dem kleinsten Fehler (loss_infer) unter allen getesteten Hyperparameter-Kombinationen.
                    if tuning:
                        self.best_loss_infer = float('inf')
                        # Trials() speichert alle getesteten Hyperparameter und ihre Ergebnisse.
                        self.trials = Trials()

                        # SET NUM OF TRIALS FOR TUNING
                        # kleines Tuning: 3 Versuche
                        if not self.args.big_tuning:
                            num_tuning_evals = 3
                        elif model_params["model_category"] == "rl":
                            # KMeans and Q-learner get tuned simultaneously, so do twice the number of evals as usual
                            num_tuning_evals = 2*self.args.max_num_tuning_evals if (model_params["model_category"] != "dl" and self.args.big_data) else 2*42
                        else:
                            num_tuning_evals = self.args.max_num_tuning_evals if (model_params["model_category"] != "dl" and self.args.big_data) else 42
                        
                        algo = partial(tpe.suggest, n_startup_jobs=5)

                        # welchee Werte darf Hyperopt für das jeweilige Modell ausprobieren?
                        # model_params["model_specific"] = "xgb"
                        # model_space = deepcopy(
                        #     space_dict["xgb"]
                        # )
                        # z.B. 
                        # xgb_space = {
                        #     "n_estimators": hp.quniform(
                        #         "xgb_n_estimators",
                        #         50,
                        #         300,
                        #         10
                        #     ),

                        #     "max_depth": hp.quniform(
                        #         "xgb_max_depth",
                        #         3,
                        #         10,
                        #         1
                        #     ),

                        #     "learning_rate": hp.loguniform(
                        #         "xgb_learning_rate",
                        #         -4,
                        #         0
                        #     )
                        # }
                        model_space = deepcopy(space_dict[model_params["model_specific"]]) if model_params["method"] != "kmeans_q" else make_kmeans_q_space(feature_names=self.data_train.columns.tolist(), args=self.args)
                        # Hyperopt wählt daraus mehrere konkrete Kombinationen aus dem Model Space
                        best_params = fmin(fn=self._objective,
                                        space=model_space,
                                        algo=algo,
                                        max_evals=num_tuning_evals,
                                        trials=self.trials,
                                        rstate=np.random.default_rng(model_params["seed"]),
                                        show_progressbar=False)
                        # Update according to the model_space
                        best_params = space_eval(model_space, best_params)
                        model_params.update(best_params)
                    else:
                        model_trainer = ModelTrainer(args=self.args, data_train=self.data_train, data_infer=self.data_infer, weights_train=self.weights_train, weights_infer=self.weights_infer, model_params=model_params, data_train_ps=self.data_train_ps, data_infer_ps=self.data_infer_ps)
                        model_trainer.train()
                        self.best_model = model_trainer.get_model()

                # Get the best model
                self.models_list_of_dicts[self.stage][target] = self.best_model

                # IMPORTANT NOTE: Update
                self.method_functions.models_list_of_dicts = deepcopy(self.models_list_of_dicts)
                self.method_functions.model_params_list_of_dicts = deepcopy(self.model_params_list_of_dicts)

                # Save
                save_data(self.models_list_of_dicts[self.stage][target], os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), to_add_folder, self.PATH_BEGIN + self.method + "_" + target + "_" + self.model_params["model_specific"] + "_" + str(stage) + "_" + to_add_cross_fitting + to_add_path  + "_model"))
                save_data(self.model_params_list_of_dicts[self.stage][target], os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), to_add_folder, self.PATH_BEGIN + self.method + "_" + target + "_" + self.model_params["model_specific"] + "_" + str(stage) + "_" + to_add_cross_fitting + to_add_path  + "_params"))
                print('\n')

        runtime_seconds = time.perf_counter() - runtime_start
        to_add_folder = "tuning" if tuning else "training"
        to_add_path = "tuning" if tuning else (str(self.iter) + "_training")
        to_add_cross_fitting = "cross_fitted_" if "dtr" in self.method and self.args.cross_fitting else ""
        runtime_path = os.path.join(
            os.getcwd(),
            self.RESULTS_FOLDER,
            str(self.method),
            to_add_folder,
            self.PATH_BEGIN + self.method + "_" + to_add_cross_fitting + to_add_path + "_runtime",
        )
        # only save if no already_trained_list, no already_tuned_list, not already_trained, not already_tuned
        if not ((self.args.already_trained_list and not tuning) or (self.args.already_tuned_list and tuning) or (self.args.already_trained and not tuning) or (self.args.already_tuned and tuning)):
            save_data(runtime_seconds, runtime_path)
            print(f"    Runtime ({self.method}, {to_add_path}): {runtime_seconds:.2f}s")

    def _objective(self, params):
        # Ensure alpha_max > alpha_min
        if self.model_params["method"] == "kmeans_q":
            params["alpha_max"] = params["alpha_min"] + 0.0001 if params["alpha_max"] < params["alpha_min"] + 0.0001 else params["alpha_max"]
        model_params = deepcopy(self.model_params)
        model_params.update(params)
        self.method_functions.model_params.update(model_params)

        model_trainer = ModelTrainer(args=self.args, data_train=self.data_train, data_infer=self.data_infer, weights_train=self.weights_train, weights_infer=self.weights_infer, model_params=model_params, data_train_ps=self.data_train_ps, data_infer_ps=self.data_infer_ps)
        model_trainer.train(tuning=True)
        loss_infer = model_trainer.best_loss_infer
        if loss_infer < self.best_loss_infer:
            self.best_loss_infer = loss_infer
            if model_params["model_category"] == "ml":
                # Train the model once more on full set with the best parameters because we used cross-validation
                model_trainer.train(tuning=False)
            self.best_model = model_trainer.get_model()
            print('Best params:', model_params)

        return {'loss': loss_infer, 'status': STATUS_OK}
    
    # eval() bewertet die trainierte Entscheidungsstrategie auf den Testfällen.
    # Vergleichswerte laden/berechnen:
    # tatsächliche Bankstrategie
    # zufällige Strategie
    # optimale Strategie
    #
    # Das finale Modell auswählen:
    # meistens das effect-Modell
    # andernfalls das outcome-Modell
    # bei KMeans-Q das Q-Learning-Modell
    def eval(self, preps_maps, dfs_map):
        bank_profit, random_profit, random_uplift, optimal_profit, optimal_uplift = self.get_bank_optimal_random_results(dfs_map=dfs_map)
        
        # Getting the final target and model_params, so the model that actually recommends actions
        for target, model_params in reversed(list(self.model_params_list_of_dicts[0].items())):
            if model_params == "nope": continue
            final_target, final_model_category, final_model_specific = target, model_params["model_category"], model_params["model_specific"]
            to_add_cross_fitting = "cross_fitted_" if "dtr" in self.method and self.args.cross_fitting else ""
            break # model has been found

        to_add_model_specific = final_model_specific + "_" if (final_model_specific != "kmeans_q" and final_model_specific != "xgb" and final_model_specific != "lstm" and final_model_specific != "vanilla_nn") else ""

        if self.method in self.args.already_evaluated_list or self.args.already_evaluated:
            # jus load the results
            profit = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "profit_eval"))
            final_df = None
            # final_df = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "df_eval"))
            uplift = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "uplift_eval"))
        else:
            decisions = {}  # Store decisions for each stage
            case_nrs = list(range(self.args.test_size))
            # IMPORTANT NOTE: for q-learning, we train using 1 stage (0), but for evaluation we need to go through all stages, since the action in stage 0 decides the action in stage 1
            # So whenever we select something of a model, we select it by index_to_select, which is not the same as the stage when the method is "kmeans_q" (which only has one stage for training, so we always select 0)
            action_df = None
            total_eval_runtime_seconds = 0
            for stage in range(self.args.n_stages):
                # Stage 0 → Aktion auswählen
                # Stage 1 → Daten abhängig von Aktion 0 auswählen
                # Stage 2 → Daten abhängig von Aktionen 0 und 1 auswählen

                # Evaluation Stage 0 → Modell 0
                # Evaluation Stage 1 → Modell 1
                # Evaluation Stage 2 → Modell 2
                index_to_select = stage if self.method != "kmeans_q" else 0  # KMeansQ only has one stage for training, so we always have 0 to select the model
            #     {  # Stage 0
            #         "ps": {...},
            #         "outcome": {...},
            #         "effect": {...}
            #     },
            #     {  # Stage 1
            #         "ps": {...},
            #         "outcome": {...},
            #         "effect": {
            #             "method": "dtr-S-reg-R",
            #             "stage": 1,
            #             "target": "effect",
            #             "encoding": "agg",
            #             "model_category": "ml",
            #             "model_specific": "xgb",
            #             "dim_x_case": 20,
            #             "dim_t": 2,
            #             "n_estimators": 100,
            #             "max_depth": 6,
            #             "learning_rate": 0.1,
            #             "seed": 42
            #         }
            #     }
            # ]
                model_params = self.model_params_list_of_dicts[index_to_select][final_target].copy() 
                # Diese Zeile stellt die passenden Testdaten für die aktuelle Stage zusammen:
                # Entscheidend sind dabei die bereits empfohlenen Aktionen in decisions. Beispiel mit drei Stages: Für einen Fall wurden bisher diese Aktionen empfohlen:
                # Stage 0: Aktion 1
                # Stage 1: Aktion 0
                # Bei der Evaluation von Stage 2 sucht die Funktion deshalb die Daten für den Aktionsverlauf. Die letzte 0 ist nur ein Platzhalter für die noch nicht getroffene Entscheidung in Stage 2.
                #  key = "(1, 0, 0)"
                # preps_maps["(1, 0, 0)"][encoding][stage]
                # colated_prep = {
                #     "X_case": Tensor(...),
                #     "X_event": Tensor(...),
                #     "T": Tensor(...),
                #     "Y": Tensor(...),
                #     "prefix_len": Tensor(...),
                #     "case_nr": Tensor(...)
                # }
                # collated_prep enthält für jeden Fall genau die Daten, die zu den bisher vom Modell empfohlenen Aktionen passen.
                collated_prep = self.get_data_given_prev_actions(stage=stage, preps_maps=preps_maps, model_params=model_params, target=final_target, index_to_select=index_to_select, prev_decisions=decisions)
                # get_actions_recommended_current_stage() bewertet das aktuelle Modell auf den Testdaten.
                action_df, action_eval_seconds = self.get_actions_recommended_current_stage(stage=stage, target=final_target, index_to_select=index_to_select, model_params=model_params, collated_prep=collated_prep, case_nrs=case_nrs, to_add_cross_fitting=to_add_cross_fitting)
                total_eval_runtime_seconds += action_eval_seconds
                # Die Entscheidungen speichern:
                decisions[stage] = action_df

            # SAVE RUNTIME
            save_data(
                total_eval_runtime_seconds,
                os.path.join(
                    os.getcwd(),
                    self.RESULTS_FOLDER,
                    str(self.method),
                    "eval",
                    self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "runtime_eval",
                ),
            )
            print(f"                Eval runtime ({self.method}, iter {self.iter}): {total_eval_runtime_seconds:.2f}s")

            # Den Profit dieser Entscheidungen berechnen.
            profit, final_df = self.calculate_proft_of_decisions(decisions=decisions, dfs_map=dfs_map)

            # calculate the extra percentage profit compared to the bank
            # Den Uplift gegenüber der Bankstrategie bestimmen:
            uplift = (profit - bank_profit) / abs(bank_profit) * 100 if bank_profit != 0 else 0
            print(f"                Random uplift: {random_uplift:.2f}%; Optimal uplift {optimal_uplift:.2f}%; Uplift: {uplift:.2f}%")


            # Save the results
            # Rückgabe
            save_data(decisions, os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "decisions_eval"))
            save_data(profit, os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "profit_eval"))
            # save_data(final_df, os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "df_eval"))
            save_data(uplift, os.path.join(os.getcwd(), self.RESULTS_FOLDER, str(self.method), "eval", self.PATH_BEGIN + self.method + "_" + final_target + "_" + final_model_category + "_" + to_add_model_specific + str(self.iter) + "_" + to_add_cross_fitting + "uplift_eval"))

        return uplift, profit, final_df

    def get_bank_optimal_random_results(self, dfs_map):
        # BANK POLICY
        # calculate and save if the file does not exist
        if not os.path.exists(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "bank", self.PATH_BEGIN + "bank_profit")):
            # calculate the bank profit
            bank_profit = dfs_map["bank"]["test_df"].groupby('case_nr')['outcome'].first().sum()
            save_data(bank_profit, os.path.join(os.getcwd(), self.RESULTS_FOLDER, "bank", self.PATH_BEGIN + "bank_profit"))
        else:
            bank_profit = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "bank", self.PATH_BEGIN + "bank_profit"))

        # RANDOM POLICY
        # calculate and save if the file does not exist
        if not os.path.exists(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "random", self.PATH_BEGIN + str(self.iter) + "_random_profit")):
            # calculate the random profit
            random_profit = dfs_map["random_" + str(self.iter)]["test_df"].groupby('case_nr')['outcome'].first().sum()
            random_uplift = (random_profit - bank_profit) / abs(bank_profit) * 100 if bank_profit != 0 else 0
            save_data(random_profit, os.path.join(os.getcwd(), self.RESULTS_FOLDER, "random", self.PATH_BEGIN + str(self.iter) + "_random_profit"))
            save_data(random_uplift, os.path.join(os.getcwd(), self.RESULTS_FOLDER, "random", self.PATH_BEGIN + str(self.iter) + "_random_uplift"))
        else:
            random_profit = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "random", self.PATH_BEGIN + str(self.iter) + "_random_profit"))
            random_uplift = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "random", self.PATH_BEGIN + str(self.iter) + "_random_uplift"))
        
        # OPTIMAL POLICY
        if not os.path.exists(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "optimal", self.PATH_BEGIN + "optimal_profit")):
            # calculate the optimal profit
            optimal_profit = dfs_map["optimal"]["test_df"].groupby('case_nr')['outcome'].first().sum()
            optimal_uplift = (optimal_profit - bank_profit) / abs(bank_profit) * 100 if bank_profit != 0 else 0
            save_data(optimal_profit, os.path.join(os.getcwd(), self.RESULTS_FOLDER, "optimal", self.PATH_BEGIN + "optimal_profit"))
            save_data(optimal_uplift, os.path.join(os.getcwd(), self.RESULTS_FOLDER, "optimal", self.PATH_BEGIN + "optimal_uplift"))
        else:
            optimal_profit = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "optimal", self.PATH_BEGIN + "optimal_profit"))
            optimal_uplift = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, "optimal", self.PATH_BEGIN + "optimal_uplift"))

        return bank_profit, random_profit, random_uplift, optimal_profit, optimal_uplift

    def get_actions_recommended_current_stage(self, stage, target, index_to_select, model_params, collated_prep, case_nrs, to_add_cross_fitting):
        # Diese Zeile lädt das Modell für die aktuelle Stage und target:
        print(f"        Stage: {stage}, Target: {target}")

        model_to_load = load_data(os.path.join(os.getcwd(), self.RESULTS_FOLDER, self.method, "training", self.PATH_BEGIN + self.method + "_" + target + "_" + model_params["model_specific"] + "_" + str(index_to_select) + "_" + to_add_cross_fitting + str(self.iter) + "_training_model"), is_state_dict=(model_params["model_category"] == "dl" and "-S-" in model_params["method"]))

        # model_evaluator = ModelEval(
        #     data=collated_prep,
        #     model_params=model_params,
        #     model_to_load=model_to_load,
        #     stage=stage
        # )
        model_evaluator = ModelEval(data=collated_prep, model_params=model_params, model_to_load=model_to_load, stage=stage)

        # model_evaluator = ModelEval(
        #     data=collated_prep,
        #     model_params=model_params,
        #     model_to_load=model_to_load,
        #     stage=stage
        # )
        action_eval_start = time.perf_counter()

        # Aktionen bestimmen und Laufzeit messe
        # action df
        # case_nr   action
        # 0         1
        # 1         0
        # 2         1
        # {0: 1, 1: 0, 2: 1}
        action_df = model_evaluator.eval()
        action_eval_seconds = time.perf_counter() - action_eval_start

        # === Ensure all case_nrs have a decision ===
        action_map = dict(zip(action_df["case_nr"], action_df["action"]))

        # Fill missing case_nrs with action 0
        # Fehlende Fälle ergänzen
        full_action_list = [action_map.get(case_nr, 0) for case_nr in case_nrs]
        action_df = pd.DataFrame({"case_nr": case_nrs, "action": full_action_list})

        return action_df, action_eval_seconds
    
    def row_to_tuple(self, row, action_cols):
        return tuple(row[c] for c in action_cols)
    
    def get_data_given_prev_actions(self, stage, preps_maps, model_params, target, index_to_select, prev_decisions):
        # Diese Zeile stellt die passenden Testdaten für die aktuelle Stage zusammen:
        # Entscheidend sind dabei die bereits empfohlenen Aktionen in decisions. Beispiel mit drei Stages: Für einen Fall wurden bisher diese Aktionen empfohlen:
        # Stage 0: Aktion 1
        # Stage 1: Aktion 0
        # Stage 2: Aktion 0
        #  key = "(1, 0, 0)"
        # preps_maps["(1, 0, 0)"][encoding][stage]
        if stage == 0:
            key = str(tuple(0 for _ in range(self.args.n_stages)))
            collated_prep = preps_maps[key][model_params["encoding"]][0]
        else:
            encoding = self.model_params_list_of_dicts[index_to_select][target]["encoding"]
            selected_data = []

            dfs = []
            for s in range(stage):
                df = prev_decisions[s].copy()
                df = df[["case_nr", "action"]].rename(columns={"action": f"action_s{s}"})
                dfs.append(df)
            merged = reduce(lambda left, right: pd.merge(left, right, on="case_nr", how="inner"), dfs)
            action_cols = [f"action_s{s}" for s in range(stage)]
            merged["action_combo"] = merged.apply(lambda row: self.row_to_tuple(row=row, action_cols=action_cols), axis=1)
            combo_to_case_nrs: Dict[Tuple, List] = (merged.groupby("action_combo")["case_nr"].apply(list).to_dict())

            # NEW
            for combo, case_nrs in combo_to_case_nrs.items():
                str_items = ", ".join(repr(x) for x in combo + (0,)*(self.args.n_stages - len(combo)))
                key = f"({str_items})"

                try:
                    data_dict = preps_maps[key][encoding][index_to_select]
                    if model_params.get("method") == "kmeans_q":
                        # Fast vectorized filtering using pandas
                        filtered_rows = data_dict[data_dict["case_nr"].isin(case_nrs)]
                        selected_data.extend(filtered_rows.to_dict(orient="records"))
                    else:
                        # Torch tensor logic
                        case_nr_tensor = data_dict["case_nr"]
                        if not isinstance(case_nr_tensor, torch.Tensor):
                            case_nr_tensor = torch.tensor(case_nr_tensor.values if hasattr(case_nr_tensor, "values") else case_nr_tensor)

                        for case_nr in case_nrs:
                            try:
                                matching_indices = (case_nr_tensor == case_nr).nonzero(as_tuple=True)[0]
                                if len(matching_indices) > 0:
                                    i = matching_indices[0].item()
                                    selected_entry = {
                                        k: (v[i] if v is not None else None)
                                        for k, v in data_dict.items()
                                    }
                                    selected_data.append(selected_entry)
                            except (IndexError, RuntimeError, KeyError):
                                pass
                except KeyError:
                    pass

            if self.method == "kmeans_q":
                # just make a df again
                collated_prep = pd.DataFrame(selected_data)
                print('')
            else:
                # Collate into dict of tensors
                all_keys = selected_data[0].keys()  # Get all keys from one selected entry
                collated_prep = {}
                for key in all_keys:
                    values = [d[key] for d in selected_data]
                    if any(v is not None for v in values):
                        collated_prep[key] = torch.stack([v for v in values if v is not None])
                    else:
                        collated_prep[key] = None
        
        return collated_prep

    def calculate_proft_of_decisions(self, decisions, dfs_map):
        # Step 1: Merge all decision DataFrames on 'case_nr'
        all_decisions_df = None
        for decision_point, df in decisions.items():
            df = df[['case_nr', 'action']].rename(columns={'action': f'action_{decision_point}'})
            if all_decisions_df is None:
                all_decisions_df = df
            else:
                all_decisions_df = pd.merge(all_decisions_df, df, on='case_nr')

        # Step 2: Create action sequence tuple for each case
        decision_keys = sorted(decisions.keys())  # Ensure consistent order
        all_decisions_df['action_sequence'] = all_decisions_df.apply(
            lambda row: tuple(row[f'action_{k}'] for k in decision_keys), axis=1
        )

        # Step 3: Look up and collect matching rows from dfs_map
        final_dfs = []
        for action_sequence, group in all_decisions_df.groupby('action_sequence'):
            key = str(action_sequence)  # Convert tuple to string key for dfs_map
            if key in dfs_map:
                matching_cases = group['case_nr']
                filtered_df = dfs_map[key]["test_df"][dfs_map[key]["test_df"]['case_nr'].isin(matching_cases)]
                final_dfs.append(filtered_df)

        # Step 4: Concatenate all results
        final_df = pd.concat(final_dfs, ignore_index=True)

        # Sum up the whole df to get one float (so get one value of column 'outcome' for each case_nr in the final_df)
        profit = final_df.groupby('case_nr')['outcome'].first().sum()

        return profit, final_df
import torch
import numpy as np
from copy import deepcopy
from src.utils.mini_tools import get_model_functions
from sklearn.metrics import log_loss, accuracy_score, f1_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class SCOPEFunctions():
    # model_params_list_of_dicts sind schlicht die Bauanleitungen für die Modelle.
    # Diese params werden gebildet in get_model_params_list_of_dicts in mini_tools.py
    # Die Daten kommen aus args, config.py, prepped_data_dict['utils']
#     model_params_list_of_dicts = [
#     # =====================================================
#     # Entscheidungsstufe 0
#     # =====================================================
#     {
#         "ps": "nope", # model_params_list_of_dicts ist der vollständige Trainingsplan der ausgewählten Methode.

#         "outcome": {
#             "method": "dtr-S-reg-R",
#             "dataset": "SimBank",
#             "stage": 0,

#             "target": "outcome",

#             "encoding": "agg",
#             "model_category": "ml",
#             "model_specific": "xgb",

#             "learner_method": "S",
#             "action_recomm_method": "reg",
#             "value_function_method": "R",

#             "cross_fitting": False,

#             # Aus prepped_data_dict["utils"]["agg"][0]
#             "dim_x_case": 12,
#             "dim_x_event": 0,
#             "dim_t": 1,
#             "dim_output": 1,

#             "ps_model_params": {
#                 "model_specific": "logreg",
#                 "encoding": "agg",
#                 "model_category": "ml"
#             },

#             # Verweis auf Modelle der nächsten Stufe
#             "prev_ps_model_params": "nope",

#             "prev_outcome_model_params": {
#                 "method": "dtr-S-reg-R",
#                 "dataset": "SimBank",
#                 "stage": 1,
#                 "target": "outcome",
#                 "encoding": "agg",
#                 "model_category": "ml",
#                 "model_specific": "xgb",
#                 "dim_x_case": 10,
#                 "dim_x_event": 0,
#                 "dim_t": 3,
#                 "dim_output": 1
#             },

#             # Allgemeine Modellparameter aus config.py
#             "n_lstm_layers": 2,
#             "n_dense_layers": 3,
#             "dim_dense": 64,
#             "dim_lstm": 64,
#             "masked": True
#         },

#         "effect": "nope"
#     },

#     # =====================================================
#     # Entscheidungsstufe 1
#     # =====================================================
#     {
#         "ps": "nope",

#         "outcome": {
#             "method": "dtr-S-reg-R",
#             "dataset": "SimBank",
#             "stage": 1,

#             "target": "outcome",

#             "encoding": "agg",
#             "model_category": "ml",
#             "model_specific": "xgb",

#             "learner_method": "S",
#             "action_recomm_method": "reg", # Die Aktionsempfehlung wird über numerische Regressionswerte erzeugt.
#             "value_function_method": "R",

#             "cross_fitting": False,

#             # Aus prepped_data_dict["utils"]["agg"][1]
#             "dim_x_case": 10,
#             "dim_x_event": 0,
#             "dim_t": 3,
#             "dim_output": 1,

#             "ps_model_params": {
#                 "model_specific": "logreg",
#                 "encoding": "agg",
#                 "model_category": "ml"
#             },

#             "n_lstm_layers": 2,
#             "n_dense_layers": 3,
#             "dim_dense": 64,
#             "dim_lstm": 64,
#             "masked": True
#         },

#         "effect": "nope"
#     }
# ]
    def __init__(self, model_params_list_of_dicts):
        """
        Initialize the SCOPE calculations.
        """
        # Initialization only
        self.n_stages = len(model_params_list_of_dicts)
        self.models_list_of_dicts = [{} for _ in range(self.n_stages)]  # List of dictionaries for each stage
        self.model_params_list_of_dicts = model_params_list_of_dicts
    
    # Die prepare()-Methode bereitet die bereits vorverarbeiteten Daten für genau das Modell vor, das gerade trainiert werden soll.
    # Welches Y soll das aktuelle Modell als Trainingsziel bekommen?
    # Alles was passiert, ist das Y angepasst wird auf das Ergebnis der Backward induction: Y wird für die Backward Induction angepasst.
    def prepare(self, data_train_list, data_infer_list, stage, model_params, data_lists_for_other_models=None):
        self.stage = stage
        #     data_train_list = [
        # # data_train_list[0]: Stage 0
        # {
        #     "case_nr": tensor([
        #         1.,
        #         2.,
        #         3.
        #     ]),

        #     "X_case": tensor([
        #         # amount, quality, estimated_quality, ...
        #         [ 0.50,  0.80,  0.60, 1.00],
        #         [-0.20,  0.30,  0.40, 0.00],
        #         [ 0.90, -0.10,  0.75, 1.00]
        #     ]),

        #     "X_event": None,
        #     "prefix_len": None,

        #     # Beobachtete Aktion in Stage 0
        #     # 0 = start_standard
        #     # 1 = start_priority
        #     "T": tensor([
        #         [0.],
        #         [1.],
        #         [0.]
        #     ]),

        #     # Ergebnis des gesamten Falls
        #     "Y": tensor([
        #         [1200.],
        #         [-300.],
        #         [800.]
        #     ])
        # },

        # # data_train_list[1]: Stage 1
        # {
        #     # Fall 2 hat Stage 1 nicht erreicht.
        #     "case_nr": tensor([
        #         1.,
        #         3.
        #     ]),
        # Beide Listen in deisem Format
        self.data_train_list = data_train_list
        self.data_infer_list = data_infer_list

        # data for other models
        self.data_lists_for_other_models = data_lists_for_other_models

        self.model_params = model_params
        # Y = Outcome, also resultierendes Ergebnis
        self.learner_method = self.model_params_list_of_dicts[self.stage]["outcome"]["learner_method"]
        self.action_recomm_method = self.model_params_list_of_dicts[self.stage]["outcome"]["action_recomm_method"]
        self.value_function_method = self.model_params_list_of_dicts[self.stage]["outcome"]["value_function_method"]
       
        weights_train = None
        target_train = self.data_train_list[self.stage]["Y"]

        # If it is outcome model in stage <n_stages - 1>, we need to set the target outcomes using inference of previous model
        if model_params["target"] == "effect":
            target_train, weights_train = self.calc_target_effect()
        elif (model_params["target"] == "outcome" and stage < self.n_stages -1):
            target_train = self.calc_target_outcomes()
        
        # Replace the Y with targets
        data_train_adj = deepcopy(self.data_train_list[self.stage])
        # Sind jetzt die Daten aus der Backward Induction
        data_train_adj["Y"] = target_train

        # Wenn das Training ein angepasstes Pseudo-Outcome verwendet, darf die Validierung nicht weiterhin das ursprüngliche Outcome verwenden.
        # „Inferenzdaten“ bedeutet hier Validierungsdaten
        # NOCH NICHT GANZ VERSTANDEN
        if data_infer_list is not None:
            # If there is inference data, we need to adjust it as well
            weights_infer = None
            data_infer_adj = deepcopy(self.data_infer_list[self.stage])
            if model_params["target"] == "effect":
                target_infer, weights_infer = self.calc_target_effect(infer=True)
                data_infer_adj["Y"] = target_infer
            elif (model_params["target"] == "outcome" and stage < self.n_stages -1):
                data_infer_adj["Y"] = self.calc_target_outcomes(infer=True)
                weights_infer = None
            else:
                data_infer_adj["Y"] = data_infer_adj["Y"]
        else:
            data_infer_adj = None
            weights_infer = None

        self.data_train_ps = self.data_lists_for_other_models["ps"]["train"][self.stage] if self.data_lists_for_other_models["ps"]["train"] is not None else None
        self.data_infer_ps = self.data_lists_for_other_models["ps"]["infer"][self.stage] if self.data_lists_for_other_models["ps"]["infer"] is not None else None

        # Einfach statt Y [0.75, 0.10, 0.20] jetzt Y [1.00, 0.10, 0.35] durch Backward Induction angepasst.
        return data_train_adj, data_infer_adj, weights_train, weights_infer, self.data_train_ps, self.data_infer_ps

    def calc_target_effect(self, infer=False):
        data = self.data_infer_list[self.stage] if infer else self.data_train_list[self.stage]
        dataset_ps = (self.data_lists_for_other_models["ps"]["infer"][self.stage] if self.data_lists_for_other_models["ps"]["infer"] is not None else None) if infer else (self.data_lists_for_other_models["ps"]["train"][self.stage] if self.data_lists_for_other_models["ps"]["train"] is not None else None)

        outcome_model_functions = get_model_functions(model_params=self.model_params_list_of_dicts[self.stage]["outcome"], model_to_load=self.models_list_of_dicts[self.stage]["outcome"])
        
        target_outcomes = self.calc_target_outcomes(infer=infer)

        # Holen der Q Werte. Dort wird forward() aufgerufen, um die Q Werte zu berechnen.
        q_values_all_actions = self.get_q_values(model_functions=outcome_model_functions, data=data, target_outcomes=target_outcomes)

        ps_train = None
        if self.model_params_list_of_dicts[self.stage]["ps"] != "nope":
            ps_model_functions = get_model_functions(model_params=self.model_params_list_of_dicts[self.stage]["ps"], model_to_load=self.models_list_of_dicts[self.stage]["ps"])
            ps_train = self.get_propensity_scores(ps_model_functions=ps_model_functions, data=data, dataset_ps=dataset_ps)

        opt_actions, opt_estimates, contrast_all_actions, causal_estimates_tensor = self.calc_opt_actions_and_constrast(q_values_all_actions=q_values_all_actions, propensity_scores=ps_train, data=data, target_outcomes=target_outcomes)

        weights = self.get_weights_effect(contrast_all_actions=contrast_all_actions)
        targets = self.get_targets_effect(opt_actions=opt_actions, causal_estimates_tensor=causal_estimates_tensor)

        return targets, weights
    
    def get_targets_effect(self, opt_actions, causal_estimates_tensor):
        if "reg" in self.model_params["method"]:
            return causal_estimates_tensor
        else:
            return opt_actions

    def get_weights_effect(self, contrast_all_actions):
        if self.model_params["model_category"] == "dl" and 'class' in self.model_params["method"]:
            # just return all, since we use a custom loss function, where the loss depends on the 'predicted' optimal action

            # reshape so it is (n_samples, n_classes)
            weights = contrast_all_actions.permute(1, 0)  # shape: (n_samples, n_classes)

            # add 1 to all weights
            weights = weights + 1
        
        elif "reg" in self.model_params["method"]:
            # just return weights 1
            weights = torch.ones_like(contrast_all_actions[0])  # shape: (n_samples,)

        elif self.model_params["model_category"] == "ml":
            # Use the weights like CC-learning: just repeat the instances with artificial labels (every treatment level) and match with appropriate weights
            weights = contrast_all_actions.permute(1, 0)

        return weights

    # WICHTIG: BERECHNEN DER VALUE FUNKTION
    # calc_target_outcomes() transportiert den zukünftigen Wert von der nächsten Stage zurück zur aktuellen Stage
    # Das ist die Berechnnugng der 
    def calc_target_outcomes(self, infer=False):
        """
        Set the target variable for the outcome model in SCOPE.
        At stage <n_stages - 1>, this is the real Y, at stage <n_stages - 2>..., this setup using the M- (max) or R-method (regret).
        NOTE: Before training, using data_train.
        """
        # data = {
        #     "case_nr": tensor([
        #         101.,
        #         102.,
        #         103.
        #     ]),

        #     "X_case": tensor([
        #         # amount, elapsed_time, cum_cost, est_quality, unc_quality, call_count
        #         [0.25, 0.30, 0.12, 0.65, 0.20, 1.00],  # Fall 101
        #         [0.80, 0.55, 0.25, 0.75, 0.10, 2.00],  # Fall 102
        #         [0.45, 0.20, 0.08, 0.40, 0.50, 0.00]   # Fall 103
        #     ]),

        #     "X_event": None,

        #     "prefix_len": None,

        #     "T": tensor([
        #         [1., 0., 0.],   # Fall 101 → Aktion 0.07
        #         [0., 1., 0.],   # Fall 102 → Aktion 0.08
        #         [0., 0., 1.]    # Fall 103 → Aktion 0.09
        #     ]),

        #     "Y": tensor([
        #         [0.75],   # Outcome von Fall 101
        #         [0.30],   # Outcome von Fall 102
        #         [0.20]    # Outcome von Fall 103
        #     ])
        # }
        data = self.data_infer_list[self.stage] if infer else self.data_train_list[self.stage]

        if self.stage == self.n_stages - 1:
            # Stage 1 has the real Y
            target_outcomes = data["Y"]
        # STAGES SIND HIER ENTSHCIeDUNGSPUNKTE, DAS HEIßt ES KANN SEIN, DASS EIN FALL EINEN NÄCHSTEN ENTSCHeIDUNGSPUNKT GAR NICHT ERREICHT DURCH CANCEL oder einem anderen AST
        elif self.stage < self.n_stages - 1:
            # Get the right encoding for the previous outcome model
            prev_data = (self.data_lists_for_other_models["prev_outcome"]["infer"][self.stage+1] if self.data_lists_for_other_models["prev_outcome"]["infer"] is not None else self.data_infer_list[self.stage+1]) if infer else (self.data_lists_for_other_models["prev_outcome"]["train"][self.stage+1] if self.data_lists_for_other_models["prev_outcome"]["train"] is not None else self.data_train_list[self.stage+1])
            prev_data_ps = (self.data_lists_for_other_models["prev_ps"]["infer"][self.stage+1] if self.data_lists_for_other_models["prev_ps"]["infer"] is not None else None) if infer else (self.data_lists_for_other_models["prev_ps"]["train"][self.stage+1] if self.data_lists_for_other_models["prev_ps"]["train"] is not None else None)

            # Diese Zeile holt das bereits trainierte Outcome-Modell der nächsten Stage und macht es über eine einheitliche forward()-Schnittstelle benutzbar
            prev_outcome_model_functions = get_model_functions(model_params=self.model_params_list_of_dicts[self.stage+1]["outcome"], model_to_load=self.models_list_of_dicts[self.stage+1]["outcome"])
            prev_q_values_all_actions = self.get_q_values(model_functions=prev_outcome_model_functions, data=prev_data, target_outcomes=prev_data["Y"])
            
            # Stage 0 uses the M- or R-method to set up the target variable
            if self.model_params["value_function_method"] == "M":
                # Take the max over the Q-values for all actions
                target_outcomes = np.max(prev_q_values_all_actions, axis=0) if isinstance(prev_q_values_all_actions[0],np.ndarray) else torch.max(torch.stack(prev_q_values_all_actions), dim=0).values
            else:
                # Hier wird aus pre
                prev_obs_actions = torch.argmax(prev_data["T"], dim=1) if prev_data["T"].shape[1] > 2 else prev_data["T"].squeeze(1).long()  # Convert to class labels if one-hot encoded
                prev_q_values_obs_actions = self.get_correct_values(values_all_actions=prev_q_values_all_actions, actions=prev_obs_actions)
                q_obs = prev_q_values_obs_actions

                if self.model_params["value_function_method"] == "G":
                    # Used for example in g-computation
                    target_outcomes = q_obs

                elif self.model_params["value_function_method"] == "R":
                    # Grab any ps modelling if necessary
                    prev_ps = None
                    # Propensity Scores sind hier die vom Modell geschätzten Wahrscheinlichkeiten, dass die Bank bei einem gegebenen Fallzustand genau eine bestimmte Aktion auswählt. Sie bewerten also nicht den Erfolg der Aktion, sondern die historische Auswahlwahrscheinlichkeit der Aktion.
                    # Warum werden Propensity Scores benötigt?
                    # Angenommen, die bisherige Bank vergibt bei guten Kunden fast immer 0.08:
                    # gute Kunden    → meistens 0.08
                    # schlechte Kunden → meistens 0.09
                    # Dann könnte man in den Daten beobachten:
                    # 0.08 → hoher durchschnittlicher Gewinn
                    # 0.09 → niedriger durchschnittlicher Gewinn
                    # Das bedeutet aber nicht automatisch, dass 0.08 die bessere Aktion ist. Vielleicht waren die Kunden, die 0.08 erhielten, von Anfang an besser.
                    # Das ist Confounding
                    # Kundenzustand
                    # ├─ beeinflusst die gewählte Aktion
                    # └─ beeinflusst gleichzeitig das Outcome
                    # Dadurch kann AIPWE berücksichtigen, ob eine beobachtete Aktion:
                    # für diesen Fall sehr wahrscheinlich war oder
                    # eher selten beziehungsweise überraschend war. => ANSEHEN VON AIPWE
                    if self.model_params_list_of_dicts[self.stage+1]["ps"] != "nope":
                        prev_ps_model_functions = get_model_functions(model_params=self.model_params_list_of_dicts[self.stage+1]["ps"], model_to_load=self.models_list_of_dicts[self.stage+1]["ps"])

                        prev_ps = self.get_propensity_scores(ps_model_functions=prev_ps_model_functions, data=prev_data, dataset_ps=prev_data_ps)

                    # Grab the optimal action given by previous effect model if causal learners were used with pseudo-outcomes
                    # Das sind Learner mit seperatem effekt modell
                    # Effect-Modell
                    # → entscheidet, welche Aktion optimal ist

                    # Outcome-Modell
                    #     → liefert das erwartete Outcome dieser Aktion
                    if "RA" in self.learner_method or "AIPW" in self.learner_method:
                        # prev_effect_model_params = {
                        #     "method": "dtr-AIPWE-reg-R",
                        #     "dataset": "SimBank",
                        #     "stage": 1,

                        #     "target": "effect",

                        #     "learner_method": "AIPWE",
                        #     "action_recomm_method": "reg",
                        #     "value_function_method": "R",

                        #     "encoding": "agg",
                        #     "model_category": "ml",
                        #     "model_specific": "xgb",

                        #     "dim_x_case": 12,
                        #     "dim_x_event": 0,
                        #     "dim_t": 3,
                        #     "dim_output": 1,

                        #     "n_estimators": 150,
                        #     "max_depth": 6,
                        #     "learning_rate": 0.1
                        # }
                        # DAS FOLGENDE PASSIERT PRO STAGE! Wir sind hier in einem Loop
                        prev_effect_model_params = self.model_params_list_of_dicts[self.stage+1]["effect"]
                        prev_effect_model_functions = get_model_functions(
                            model_params=prev_effect_model_params,
                            # Trainiertes Modell holen
                            # self.models_list_of_dicts[1]["effect"] = [
                            #     xgb_effect_model_008,
                            #     xgb_effect_model_009
                            # ]
                            model_to_load=self.models_list_of_dicts[self.stage+1]["effect"],
                        )

                        # Drinenn wird effect_values_all_actions ausgeführt mit forward()
                        #prev_effect_values_all_actions = np.array([
                        #     # Case 101  Case 102  Case 103
                        #     [ 0.00,      0.00,      0.00],   # 0.07: Baseline
                        #     [ 0.35,     -0.10,     -0.05],   # Effekt von 0.08
                        #     [ 0.10,      0.25,     -0.20]    # Effekt von 0.09
                        # ])
                        prev_effect_values_all_actions = self.get_effect_values(model_functions=prev_effect_model_functions, data=prev_data, target_outcomes=prev_data["Y"])

                        #optimale Aktion?
                        # prev_opt_actions = np.array([
                        #     1,   # Fall 101 → 0.08
                        #     2,   # Fall 102 → 0.09
                        #     0    # Fall 103 → 0.07
                        # ]) optimale Aktion?
                        prev_opt_actions = np.argmax(prev_effect_values_all_actions, axis=0)

                    else:
                        # Just use the q-values to get optimal actions from the previous stage, since we are not using causal learners with pseudo-outcomes
                        prev_opt_actions, prev_opt_estimates, prev_contrast, _ = self.calc_opt_actions_and_constrast(q_values_all_actions=prev_q_values_all_actions, propensity_scores=prev_ps, data=prev_data, target_outcomes=prev_data["Y"])

                    # Dann werden aus der Outcome-Matrix folgende Werte genommen:
                    # Wir sind hier wieder in Stage X!
                    # Das Outcome-Modell schätzt für Fall 101 ein Outcome von 0.85, wenn in der späteren Stage Aktion 0.08 gewählt wird.
                    # prev_q_values_opt_actions = tensor([
                    #     [0.85],   # Fall 101, Aktion 0.08
                    #     [0.45],   # Fall 102, Aktion 0.09
                    #     [0.40]    # Fall 103, Aktion 0.07
                    # ])
                    prev_q_values_opt_actions = self.get_correct_values(values_all_actions=prev_q_values_all_actions, actions=prev_opt_actions)

                    # Use the regret function
                    # prev_data = {
                    #     "case_nr": tensor([
                    #         101.,
                    #         102.,
                    #         103.
                    #     ]),

                    #     "X_case": tensor([
                    #         [0.25, 0.60, 0.30],
                    #         [0.80, 0.40, 0.50],
                    #         [0.45, 0.75, 0.20]
                    #     ]),

                    #     "X_event": None,
                    #     "prefix_len": None,

                    #     # Tatsächlich ausgeführte Aktionen
                    #     "T": tensor([
                    #         [1., 0., 0.],   # Fall 101 → 0.07
                    #         [0., 1., 0.],   # Fall 102 → 0.08
                    #         [0., 0., 1.]    # Fall 103 → 0.09
                    #     ]),

                    #     # Beobachtete beziehungsweise vorbereitete Outcomes
                    #     "Y": tensor([
                    #         [0.75],   # Fall 101
                    #         [0.30],   # Fall 102
                    #         [0.20]    # Fall 103
                    #     ])
                    # }
                    v = prev_data["Y"]
                    q_opt = prev_q_values_opt_actions
                    # Welches Outcome hätte der Fall erzielt, wenn in der nächsten Stage die optimale statt der tatsächlich beobachteten Aktion gewählt worden wäre?
                    # q_opt - q_obs ist der vom Modell geschätzte Verbesserungsbetrag. Das heißt v + q_opt - q_obs ist das Outcome, das der Fall erzielen würde, wenn in der nächsten Stage die optimale Aktion gewählt worden wäre.
                    # target_outcomes = tensor([
                    #     [0.35],   # Fall 103
                    #     [1.00]    # Fall 101
                    # ])
                    target_outcomes = v + q_opt - q_obs

            # Das ist noch im Stage loop. Ein Fall erreicht die nächste Stage nur, wenn sein Prozess tatsächlich bis zum nächsten Interventionspunkt läuft. Wird er vorher storniert oder beendet, gibt es dort keine spätere Aktion, die optimiert werden könnte. 
            # Do this both for R and M methods
            # Fallnummern aus der nächsten Stage
            # prev_case_nrs = tensor([
            #     103.,
            #     101.
            # ])
            prev_case_nrs = prev_data['case_nr']  # shape (n_prev,)
            # Fallnummern aus der aktuellen Stage. 
            # data_case_nrs = tensor([
            #     101.,
            #     102.,
            #     103.
            # ])
            data_case_nrs = data['case_nr']       # shape (n_data,)
            # Ursprüngliche Outcomes merken. Diese Werte werden als Ersatz verwendet, wenn ein Fall die nächste Stage nicht erreicht hat.
            data_Y = data['Y']                    # shape (n_data, 1)
            # Create a mapping from prev_case_nr to its index in target_outcomes
            # Position der Fälle in den Daten von Stage 1
            # case_nr_to_index = {
            #     103: 0,
            #     101: 1
            # }
            case_nr_to_index = {int(c.item()): i for i, c in enumerate(prev_case_nrs)}
            # Build the aligned outcome
            aligned_outcomes = []
            # Fall 101 hat Stage 1 erreicht. Position des Falls in target_outcomes suchen, Das angepasste Outcome von Fall 101 steht also an Index 1.
            # Fall 102 hat Stage 1 nicht erreicht. Daher existiert für ihn kein optimiertes Outcome aus Stage 1. Das ursprüngliche Outcome bleibt also erhalten. data[Y]
            for i, c in enumerate(data_case_nrs):
                c_int = int(c.item())
                if c_int in case_nr_to_index:
                    # Position des Falls in target_outcomes suchen. Das angepasste Outcome von Fall 101 steht also an Index 1.
                    idx = case_nr_to_index[c_int]
                    # make sure you add a tensor (so convert target_outcomes[idx] to tensor if it is not already)
                    val = target_outcomes[idx]
                    # Tensor auf einheitliche Form bringen
                    tensor_element = (
                        val.detach().view(1) if isinstance(val, torch.Tensor)
                        else torch.tensor([float(val)], dtype=torch.float32)
                    )
                    aligned_outcomes.append(tensor_element)
                else:
                    aligned_outcomes.append(data_Y[i])

            # Stack into a single tensor
            # Liste in einen Tensor umwandeln
            target_outcomes = torch.stack(aligned_outcomes)  # shape (n_data, 1)
            print('')
        
        # change into a tensor if it is a numpy array
        target_outcomes = torch.tensor(target_outcomes, dtype=torch.float32) if isinstance(target_outcomes, np.ndarray) else target_outcomes
        # Ensure tensor has shape (n, 1)
        target_outcomes = target_outcomes.unsqueeze(1) if len(target_outcomes.shape) == 1 else target_outcomes

        # target_outcomes = tensor([
        #     [1.00], # Case 1: für optimale Aktion in Stage 1 angepasst
        #     [0.10], # Case 2: ursprüngliches Outcome, da Stage 1 nicht erreicht
        #     [0.35] # Case 3: für optimale Aktion in Stage 1 angepasst
        # ])
        return target_outcomes
    
    def calc_opt_actions_and_constrast(self, data, q_values_all_actions, propensity_scores, target_outcomes):
        """
        Estimate the optimal actions for SCOPE.
        NOTE: After training, using data_infer.
        """
        causal_estimates = []
        for action in range(len(q_values_all_actions)):
            estimate = self.get_causal_estimate(action=action, data=data, q_values_all_actions=q_values_all_actions, propensity_scores=propensity_scores, target_outcomes=target_outcomes)
            # causal_estimates.append(estimate)
            causal_estimates.append(estimate.squeeze(1))  # Squeeze to remove unnecessary dimensions

        causal_estimates_tensor = torch.stack(causal_estimates)  # shape: (num_actions, n)

        # Find the index of the max estimate (optimal action) for each data point (along actions dimension)
        opt_actions = torch.argmax(causal_estimates_tensor, dim=0)  # shape: (n,)

        # Gather the optimal estimates for each data point
        opt_estimates = causal_estimates_tensor[opt_actions, torch.arange(causal_estimates_tensor.shape[1])]  # shape: (n,)

        # Compute the contrast function: difference between optimal estimate and each estimate
        # We want result shape: (num_actions, n)
        contrast_function_values = opt_estimates.unsqueeze(0) - causal_estimates_tensor

        return opt_actions, opt_estimates, contrast_function_values, causal_estimates_tensor

    # Helper methods for SCOPE
    def get_q_values(self, model_functions, data, target_outcomes=None, target="outcome"):
        """
        Get the Q predictions for all possible actions in SCOPE.
        """

        q_values_all_actions = model_functions.forward(x_case=data["X_case"],
                                    x_event=data["X_event"],
                                    t=data["T"],
                                    prefix_len=data["prefix_len"],
                                    y=data["Y"],
                                    ret_counterfactuals=True)
        
        # quickly calculate mae, by taking the right q_value for the observed T, and by comparing it to the target outcomes
        obs_actions = torch.argmax(data["T"], dim=1) if data["T"].shape[1] > 2 else data["T"].squeeze(1).long()
        mae = torch.mean(torch.abs(self.get_correct_values(q_values_all_actions, obs_actions) - data["Y"]))
        print(f"MAE: {mae.item()}")

        # Für drei Fälle könnte die Rückgabe so aussehen
        # q_values_all_actions = np.array([
        #     # Case 1   Case 2   Case 3
        #     [  900.0,   500.0,  -200.0],   # Aktion 0.07
        #     [ 1250.0,   650.0,   100.0],   # Aktion 0.08
        #     [ 1050.0,   700.0,   -50.0]    # Aktion 0.09
        # ])
        return q_values_all_actions
    
    def get_effect_values(self, model_functions, data, target_outcomes=None):
        """
        Get the effect predictions for all possible actions in SCOPE.
        """
        effect_values_all_actions = model_functions.forward(x_case=data["X_case"],
                                    x_event=data["X_event"],
                                    t=data["T"],
                                    prefix_len=data["prefix_len"],
                                    y=data["Y"],
                                    ret_counterfactuals=True)
        
        return effect_values_all_actions
        
    def get_propensity_scores(self, ps_model_functions, data, dataset_ps=None):
        """
        Get the propensity scores for SCOPE (if used).
        """
        # NOTE: we use the calibration model to get the propensity scores if it exists, otherwise we use the propensity score model (which would be LogReg).
        propensity_scores = ps_model_functions.forward(x_case=data["X_case"],
                                    x_event=data["X_event"],
                                    t=data["T"],
                                    prefix_len=data["prefix_len"],
                                    y=data["Y"],
                                    dataset_ps=dataset_ps)
        
        # quickly calculate accuracy and log loss
        preds = np.argmax(propensity_scores, axis=1) if isinstance(propensity_scores, np.ndarray) else torch.argmax(propensity_scores, dim=1)
        if data["T"].shape[1] > 2 :
            t_true = np.argmax(data["T"], axis=1) if isinstance(data["T"], np.ndarray) else torch.argmax(data["T"], dim=1)
        else:
            t_true = data["T"]
        acc = accuracy_score(t_true, preds)
        ll = log_loss(t_true, propensity_scores)
        f1 = f1_score(t_true, preds, average='weighted')
        print(f"Accuracy: {acc}, Log Loss: {ll}, F1 Score: {f1}")
        return propensity_scores
        
    def get_correct_values(self, values_all_actions, actions):
        # Check if actions is a tensor or numpy array and edit accordingly
        actions = torch.tensor(actions, dtype=torch.int64) if isinstance(actions, (list, np.ndarray)) else actions
        stacked = torch.stack([
            (torch.tensor(v, dtype=torch.float32)) if isinstance(v, (list, np.ndarray)) else v for v in values_all_actions
        ]) if isinstance(values_all_actions, (list, np.ndarray)) else values_all_actions

        values_correct = stacked[actions.long(), torch.arange(stacked.shape[1])].unsqueeze(1)

        return values_correct
    
    def get_causal_estimate(self, action, data, q_values_all_actions, propensity_scores, target_outcomes):
        cutoff = 0.01

        if self.learner_method == "AIPWE":
            # Inidicator: I(A_k == s)
            indicator = data["T"][torch.arange(data["T"].shape[0]), action].reshape(-1, 1) if data["T"].shape[1] > 2 else (data["T"] == action).float()
            
            if isinstance(propensity_scores, torch.Tensor):
                propensity_scores = torch.clamp(propensity_scores, min=cutoff, max=1-cutoff)
                # renormalize after clipping
                propensity_scores = propensity_scores / propensity_scores.sum(dim=1, keepdim=True)
                π_hat_s = propensity_scores[torch.arange(propensity_scores.shape[0]), action].unsqueeze(1)
            else:
                propensity_scores = np.clip(propensity_scores, cutoff, 1-cutoff)
                # renormalize after clipping
                propensity_scores = propensity_scores / propensity_scores.sum(axis=1, keepdims=True)
                # If propensity scores is a numpy array, we need to index it correctly
                π_hat_s = propensity_scores[np.arange(propensity_scores.shape[0]), action][:, np.newaxis]

            V_hat_next = target_outcomes

            action_list = torch.tensor([action] * data["T"].shape[0], dtype=torch.int64, device=data["T"].device)

            Q_hat_s = self.get_correct_values(q_values_all_actions, action_list)

            # Term 1
            term1 = (indicator / π_hat_s) * V_hat_next
            # Term 2
            term2 = (1 - (indicator / π_hat_s)) * Q_hat_s
            # target outcomes M --> shape (n), also for R?
            estimate = term1 + term2
            return estimate
                
        elif "RA" in self.learner_method:
            n_classes = len(q_values_all_actions)

            def prep(X, T, Y):
                if isinstance(T, torch.Tensor):
                    if n_classes > 2:
                        T = torch.argmax(T, dim=1).to(torch.int)
                    else:
                        T = T.to(torch.int)
                    T = T.cpu().numpy() if self.model_params["model_category"] == "ml" else T
                else:  # Assume it's a NumPy array
                    if n_classes > 2:
                        T = np.argmax(T, axis=1).astype(int)
                    else:
                        T = T.astype(int)

                if len(np.unique(T)) == n_classes:
                    transformer = OneHotEncoder(sparse_output=False, drop='first')
                    T = transformer.fit_transform(T.reshape(-1, 1))
                    if Y.ndim == 2 and Y.shape[1] == 1:
                        Y = Y.flatten()
                
                if self.model_params["model_category"] == "dl":
                    Y = torch.tensor(Y, dtype=torch.float32)

                return X, T, Y
            
            def subcalc(t, T, Y, mu_hats):
                # NEW: baseline action is first one, so pseudo-outcome is standard 0 for all
                if t == 0:
                    return torch.zeros_like(Y) if isinstance(Y, torch.Tensor) else np.zeros_like(Y)
                mu_hat_0 = mu_hats[0]
                mu_hat_t = mu_hats[t]
                ind_t = get_ind(T, t)
                term1 = ind_t * (Y - mu_hat_0)
                for l in range(n_classes):
                    if l != t:
                        mu_hat_l = mu_hats[l]
                        ind_l = get_ind(T, l)
                        term2 = ind_l * (mu_hat_t - Y)
                        term3 = ind_l * (mu_hat_l - mu_hat_0)
                        term1 += term2 + term3
                return term1
            
            def get_ind(T, t):
                if n_classes == 2:
                    to_return = np.where(T == t, 1, 0).astype(float).flatten()
                else:
                    to_return = (T.sum(axis=1) == 0).astype(float) if t == 0 else T[:, t-1].astype(float)
                if self.model_params["model_category"] == "dl":
                    to_return = torch.tensor(to_return, dtype=torch.float32)
                return to_return
                
            _, T, Y = prep(data["X_case"], data["T"], data["Y"].cpu().numpy().reshape(-1))
            pseudo_outcomes = subcalc(t=action, T=T, Y=Y, mu_hats=q_values_all_actions)

            # return as tensor
            pseudo_outcomes = torch.tensor(pseudo_outcomes, dtype=torch.float32, device=data["Y"].device).unsqueeze(1)
            return pseudo_outcomes
            
        else:
            action_list = torch.tensor([action] * data["T"].shape[0], dtype=torch.int64, device=data["T"].device)
            estimate = self.get_correct_values(q_values_all_actions, action_list)
            return estimate
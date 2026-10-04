import torch
import src.utils.prep_tools.mini_prep_tools as mini_prep_tools
import pandas as pd
import numpy as np

class TensorPreprocessor():
    def __init__(self, data_train, data_infer, PREP_PARAMS, DATASET_PARAMS, add_properties, prep_utils=None):
        self.data_train = data_train
        self.data_infer = data_infer
        self.PREP_PARAMS = PREP_PARAMS
        self.DATASET_PARAMS = DATASET_PARAMS
        self.max_process_len = add_properties["max_process_len"]
        self.nr_treatment_columns = add_properties["nr_treatment_columns"]
        self.n_stages = len(DATASET_PARAMS["intervention_info"]["action_combinations"])
        self.missing_value = add_properties["missing_value"]
        self.prep_utils = prep_utils
        
    # Hier passiert das normale Preprocessing, also Skalieren, ordinal, one hot etc, 
    def preprocess(self):
        # In dataset columns sind die columns definiert kategrosich und numerisch
        if self.PREP_PARAMS["train_prop"] > 0:
            #TRAIN
            self.oh_encoder_dict_train, self.data_encoded_train, self.case_cols_encoded, self.event_cols_encoded = mini_prep_tools.one_hot_encode_columns(data = self.data_train, cat_cols = self.DATASET_PARAMS["cat_cols"], case_cols = self.DATASET_PARAMS["case_cols"], event_cols = self.DATASET_PARAMS["event_cols"])
            self.scaler_dict_train, self.data_scaled_train = mini_prep_tools.scale_columns(data = self.data_encoded_train, scale_cols = self.DATASET_PARAMS["scale_cols"], case_cols= self.DATASET_PARAMS["case_cols"])
            self.data_fill_train = self.handle_missing_values(data = self.data_scaled_train)
            self.data_treat_train = self.add_treatment_column_sequential(data = self.data_fill_train, scaler_dict_train=self.scaler_dict_train)

            if "call_or_not" in self.DATASET_PARAMS["intervention_info"]["name"]:
                self.data_train_prep = self.create_prefix_tensors_bpic17(data = self.data_treat_train, max_process_len = self.max_process_len, data_type="normal")
            else:
                self.data_train_prep = self.create_prefix_tensors(data = self.data_treat_train, max_process_len = self.max_process_len, data_type="normal")
            self.case_cols_encoded, self.event_cols_encoded = self.data_train_prep["case_cols_encoded"], self.data_train_prep["event_cols_encoded"]
        
            self.prep_utils = {"scaler_dict_train": self.scaler_dict_train, 
                                "oh_encoder_dict_train": self.oh_encoder_dict_train, 
                                "max_process_len": self.max_process_len,
                                "case_cols_encoded": self.case_cols_encoded,
                                "event_cols_encoded": self.event_cols_encoded,
                                "dim_x_case": self.data_train_prep["X_case"].shape[1],
                                "dim_x_event": self.data_train_prep["X_event"].shape[1],
                                "dim_t": self.data_train_prep["T"].shape[1] if self.data_train_prep["T"].shape[1] > 2 else 1,  # If binary treatment, we use one-hot encoding, so dim_t = 1
                                "dim_output": self.data_train_prep["Y"].shape[1],
            }

            # drop the case_cols_encoded and event_cols_encoded from the data
            self.data_train_prep.pop("case_cols_encoded")
            self.data_train_prep.pop("event_cols_encoded")
        else:
            self.data_train_prep = None

        #INFER
        self.oh_encoder_dict_infer, self.data_encoded_infer, _, _ = mini_prep_tools.one_hot_encode_columns(data = self.data_infer, oh_encoder_dict = self.prep_utils["oh_encoder_dict_train"], cat_cols = self.DATASET_PARAMS["cat_cols"], case_cols = self.DATASET_PARAMS["case_cols"], event_cols = self.DATASET_PARAMS["event_cols"])
        self.scaler_dict_infer, self.data_scaled_infer = mini_prep_tools.scale_columns(data = self.data_encoded_infer, scaler_dict = self.prep_utils["scaler_dict_train"], scale_cols = self.DATASET_PARAMS["scale_cols"], case_cols= self.DATASET_PARAMS["case_cols"])
        self.data_fill_infer = self.handle_missing_values(data = self.data_scaled_infer)
        self.data_treat_infer = self.add_treatment_column_sequential(data = self.data_fill_infer, scaler_dict_train=self.prep_utils["scaler_dict_train"])
        if "call_or_not" in self.DATASET_PARAMS["intervention_info"]["name"]:
            self.data_infer_prep = self.create_prefix_tensors_bpic17(data = self.data_treat_infer, max_process_len = self.prep_utils["max_process_len"], data_type="normal", case_cols_encoded=self.prep_utils["case_cols_encoded"], event_cols_encoded=self.prep_utils["event_cols_encoded"])
        else:
            self.data_infer_prep = self.create_prefix_tensors(data = self.data_treat_infer, max_process_len = self.prep_utils["max_process_len"], data_type="normal", case_cols_encoded=self.prep_utils["case_cols_encoded"], event_cols_encoded=self.prep_utils["event_cols_encoded"])

        # drop the case_cols_encoded and event_cols_encoded from the data
        self.data_infer_prep.pop("case_cols_encoded")
        self.data_infer_prep.pop("event_cols_encoded")

        return self.data_train_prep, self.data_infer_prep, self.prep_utils
    
    def handle_missing_values(self, data):
        #Only floats are missing normally
        data.fillna(self.missing_value, inplace=True)
        return data

    # Die Funktion erzeugt die Spalte treatment, welche die ausgeführte Intervention pro Ereignis kodiert.
    # Sie sorgt dafür, dass der Event-Log-DataFrame eine Spalte treatment enthält, die angibt, welche Intervention an einem Entscheidungspunkt gewählt wurde. Wie diese Spalte erzeugt wird, hängt davon ab, ob die Intervention über eine Activity oder über einen numerischen Wert wie interest_rate definiert ist.
    #                   data
    #                │
    #                ▼
    #     Gibt es "treatment" schon?
    #          /             \
    #        JA               NEIN
    #        │                  │
    #        ▼                  ▼
    #   return data       Welche Intervention?
    #                      /             \
    #               activity          interest_rate
    #                  │                  │
    #                  ▼                  ▼
    #           nächste Activity     mögliche Rates
    #               prüfen              skalieren
    #                  │                  │
    #                  ▼                  ▼
    #             binary T          One-Hot T
    #             0 oder 1          [1,0,0], ...
    #                  │                  │
    #                  └────────┬─────────┘
    #                           ▼
    #                      return data
    def add_treatment_column_sequential(self, data, print_debug=False, treatment_index=None, scaler_dict_train=None):
        # NOTE: we only preprocess per stage, so per intervention point

        if 'treatment' in data.columns:
            return data

        # Intervention ist eine Aktivität:
        # Es wird geprüft, ob das nächste Ereignis die relevante Interventionsaktivität ist.
        # Aktuelles Ereignis    Nächstes Ereignis    treatment
        # validate_application  wait                  1
        # validate_application  call                  0
        # shift(-1) verschiebt den Indikator des nächsten Ereignisses in die aktuelle Zeile. Fehlende Werte am Ende werden mit 0 ersetzt.
        # Wichtig: Verwendet wird actions[-1], also die letzte konfigurierte Aktion. Bei BPIC17 wäre das aktuell wait_incomplete_files.
        if self.DATASET_PARAMS["intervention_info"]["column"] == "activity":
            intervention_activity = "activity_" + self.DATASET_PARAMS["intervention_info"]["actions"][-1]
            data["treatment"] = data[intervention_activity].shift(-1).fillna(0).astype(int)
        else:
            if self.DATASET_PARAMS["intervention_info"]["column"] == "interest_rate":
                scaled_intervention_actions = pd.DataFrame(self.DATASET_PARAMS["intervention_info"]["actions"], columns=["interest_rate"])
                scaled_intervention_actions = mini_prep_tools.scale_column(col = "interest_rate", data = scaled_intervention_actions, case_cols=self.DATASET_PARAMS["case_cols"], scaler=scaler_dict_train["interest_rate"])[1]
                zeros_list = [0] * len(scaled_intervention_actions)
                data['treatment'] = [zeros_list for _ in range(len(data))]

                if treatment_index is not None:
                    new_zero_list = zeros_list.copy()
                    new_zero_list[treatment_index] = 1
                    activity_column = "activity_" + "calculate_offer"
                    case_nr_value_last_calc_offer = -1
                    for row_nr, row in data[data['interest_rate'] == scaled_intervention_actions["interest_rate"][treatment_index]].iterrows():
                        if row[activity_column] == 1.0:
                            if row["case_nr"] != case_nr_value_last_calc_offer:
                                case_nr_value_last_calc_offer = row["case_nr"]
                                data.at[row_nr, 'treatment'] = new_zero_list
                else:
                    activity_column = "activity_" + "calculate_offer"
                    case_nr_value_last_calc_offer = -1
                    for row_nr, row in data.iterrows():
                        if row[activity_column] == 1.0:
                            if row["case_nr"] != case_nr_value_last_calc_offer:
                                for i, option in enumerate(scaled_intervention_actions["interest_rate"]):
                                    if row["interest_rate"] == option:
                                        new_zero_list = zeros_list.copy()
                                        new_zero_list[i] = 1
                                        case_nr_value_last_calc_offer = row["case_nr"]
                                        data.at[row_nr - 1, 'treatment'] = new_zero_list

        if print_debug:
            print('data_treatment below')
        return data
    
    #bereitet die BPIC17-Prozessdaten für das neuronale Netz/LSTM auf. Genauer: Sie soll aus einem Event Log Tensoren erzeugen, die den Prozesszustand an den einzelnen Entscheidungspunkten repräsentieren.
    #    ┌─────────────────────────────────────────────┐
    # │ create_prefix_tensors_bpic17(...)           │
    # │                                             │
    # │ Input:                                      │
    # │ - data                                      │
    # │ - max_process_len                           │
    # │ - case_cols_encoded                         │
    # │ - event_cols_encoded                        │
    # └──────────────────────┬──────────────────────┘
    #                        │
    #                        ▼
    # ┌─────────────────────────────────────────────┐
    # │ Feature-Spalten bestimmen                   │
    # │                                             │
    # │ X_cols =                                    │
    # │   case features                             │
    # │ + event features                            │
    # │                                             │
    # │ treatment + prefix_len kommen zusätzlich    │
    # │ in den Tensor                               │
    # └──────────────────────┬──────────────────────┘
    #                        │
    #                        ▼
    # ┌─────────────────────────────────────────────┐
    # │ Leeren Tensor X erzeugen                    │
    # │                                             │
    # │ X.shape ≈                                   │
    # │ [len(data), features, max_process_len]      │
    # └──────────────────────┬──────────────────────┘
    #                        │
    #                        ▼
    #               ┌─────────────────┐
    #               │ Für jede Zeile  │
    #               │ in data         │
    #               └────────┬────────┘
    #                        │
    #                        ▼
    #               ┌───────────────────┐
    #               │ Neuer Case?       │
    #               └───────┬─────┬─────┘
    #                       │Ja   │Nein
    #                       ▼     ▼
    #             ┌────────────┐ ┌──────────────┐
    #             │ event_nr=0 │ │ event_nr +=1 │
    #             │ stage=0    │ └───────┬──────┘
    #             └──────┬─────┘         │
    #                    └────────┬──────┘
    #                             ▼
    # ┌─────────────────────────────────────────────┐
    # │ Aktuelle Activity aus One-Hot-Spalten       │
    # │ rekonstruieren                              │
    # │                                             │
    # │ activity_validate_application = 1           │
    # │              ↓                              │
    # │ activity = "validate_application"           │
    # └──────────────────────┬──────────────────────┘
    #                        │
    #                        ▼
    #           ┌─────────────────────────────┐
    #           │ Ist das ein Decision Point? │
    #           │                             │
    #           │ erstes validate_application │
    #           │ ODER                        │
    #           │ call_incomplete_files       │
    #           │ ODER                        │
    #           │ wait_incomplete_files       │
    #           │                             │
    #           │ und stage < n_stages        │
    #           └────────────┬───────────┬────┘
    #                        │ Ja        │ Nein
    #                        ▼           ▼
    #              ┌────────────────┐  ┌──────────┐
    #              │ stage += 1     │  │ continue │
    #              └───────┬────────┘  │ Zeile    │
    #                      │           │ ignorieren│
    #                      ▼           └──────────┘
    # ┌─────────────────────────────────────────────┐
    # │ Informationen in X schreiben               │
    # │                                             │
    # │ • case_nr                                   │
    # │ • treatment                                 │
    # │ • prefix_len                                │
    # │ • case features                             │
    # │ • event features                            │
    # └──────────────────────┬──────────────────────┘
    #                        │
    #                        ▼
    #              ┌────────────────────┐
    #              │ Weitere Datenzeile?│
    #              └────────┬───────┬───┘
    #                       │ Ja    │ Nein
    #                       │       ▼
    #                       │
    #                       └────── zurück zur
    #                               Schleife

    #                               │
    #                               ▼
    # ┌─────────────────────────────────────────────┐
    # │ Leere X-Samples entfernen                   │
    # │                                             │
    # │ mask = Sample enthält irgendeinen Wert      │
    # │ X = X[mask]                                 │
    # └──────────────────────┬──────────────────────┘
    #                        │
    #                        ▼
    # ┌─────────────────────────────────────────────┐
    # │ Tensor in Bestandteile zerlegen             │
    # │                                             │
    # │ case_nr                                     │
    # │ treatment                                   │
    # │ prefix_len                                  │
    # │ X_case                                      │
    # │ X_event / X_process                         │
    # └──────────────────────┬──────────────────────┘
    #                        │
    #                        ▼
    # ┌─────────────────────────────────────────────┐
    # │ Treatment T erzeugen                        │
    # │                                             │
    # │ Treatment über Zeitachse zusammenfassen     │
    # │                                             │
    # │ z.B. [0,0,1,0] → T = 1                     │
    # │      [0,0,0,0] → T = 0                     │
    # └──────────────────────┬──────────────────────┘
    #                        │
    #                        ▼
    # ┌─────────────────────────────────────────────┐
    # │ Unbrauchbare Event-Features entfernen       │
    # │                                             │
    # │ • nur 0 / Missing                           │
    # │ • konstante Features                        │
    # └──────────────────────┬──────────────────────┘
    #                        │
    #                        ▼
    # ┌─────────────────────────────────────────────┐
    # │ Outcome Y erzeugen                          │
    # │                                             │
    # │ Y = data["outcome"]                         │
    # └──────────────────────┬──────────────────────┘
    #                        │
    #                        ▼
    # ┌─────────────────────────────────────────────┐
    # │ RETURN                                      │
    # │                                             │
    # │ {                                           │
    # │   "Y": Y,                                   │
    # │   "case_nr": case_nr,                       │
    # │   "T": T,                                   │
    # │   "prefix_len": prefix_len,                 │
    # │   "X_case": X_case,                         │
    # │   "X_event": X_process,                     │
    # │   "case_cols_encoded": ...,                 │
    # │   "event_cols_encoded": ...                 │
    # │ }                                           │
    # └─────────────────────────────────────────────┘
    def create_prefix_tensors_bpic17(self, data, max_process_len, case_cols_encoded=None, event_cols_encoded=None, data_type="normal"):
        you_have_to_filter_cols_manually = False
        if case_cols_encoded is None:
            case_cols_encoded = self.case_cols_encoded
        if event_cols_encoded is None:
            event_cols_encoded = self.event_cols_encoded
            you_have_to_filter_cols_manually = True

        previous_case = -1
        inference_dataset_indices = []
        nr_decision_points_tracker = 0
        X_cols = ["case_nr", "prefix_len"] + case_cols_encoded + event_cols_encoded
        X = torch.zeros(size=(len(data), len(X_cols) + self.nr_treatment_columns, max_process_len))
        for row_nr, row in data.iterrows():
            current_case = row["case_nr"]
            if current_case != previous_case:
                if data_type == "inference_dataset" and row_nr > 0:
                    inference_dataset_indices.append(row_nr - 1 - 1) #NOTE, additional -1 to retain without intervention
                event_nr = 0
                nr_decision_points_tracker = 0
                previous_case = current_case
                prefix_condition = False
            else:
                event_nr += 1
            
            activity = [col[len("activity_"):] for col in row.index if col.startswith("activity_") and row[col] == 1][0]
            prefix_condition = (activity == "validate_application" and nr_decision_points_tracker == 0) or (activity == "call_incomplete_files" and nr_decision_points_tracker < self.n_stages) or (activity == "wait_incomplete_files" and nr_decision_points_tracker < self.n_stages)
            
            if prefix_condition:
                nr_decision_points_tracker += 1
            else:
                continue

            # add an event
            X[row_nr, 0, event_nr] = current_case

            # Process variable-length treatment list
            treatment_list = row["treatment"]
            X[row_nr, 1:1 + self.nr_treatment_columns, event_nr] = torch.tensor(treatment_list, dtype=torch.float32)
            last_index = 1+self.nr_treatment_columns

            X[row_nr, last_index, 0:event_nr + 1] = event_nr + 1
            last_index += 1

            X[row_nr, last_index:last_index + len(case_cols_encoded), 0] = torch.tensor(row[case_cols_encoded].values.astype(float))
            last_index += len(case_cols_encoded)
            X[row_nr, last_index: last_index + len(event_cols_encoded), event_nr] = \
                torch.tensor(row[event_cols_encoded].values.astype(float))
            
        # delete 
        mask = (X != 0).any(dim=2).any(dim=1)  # True if the sample has any non-zero value
        X = X[mask]
        prefix_len = X[:, 1 + self.nr_treatment_columns, 0]
        treatment = X[:, 1:1 + self.nr_treatment_columns, :]
        Y = torch.Tensor(data["outcome"].values).unsqueeze(1)  # Make sure Y is of shape [n, 1] instead of [n]
        case_nr = X[:, 0 ,0]
        # Make T so that if there is a True in T, than it is just True, otherwise False
        T = torch.any(treatment[:, :, :], dim=2)
        # make T not boolean, but float
        T = T.float()
        last_index = 1 + self.nr_treatment_columns
        last_index += 1
        X_case = X[:, last_index:last_index + len(case_cols_encoded), 0] #, :]
        last_index += len(case_cols_encoded)
        X_process = X[:, last_index: last_index + len(event_cols_encoded), :] #, :]

        # in X_process, if there are any 'cols' with all zeros, remove them, goes from 17 --> 8 for time_contact HQ, 17 --> 10 for calculate_offer
        if self.PREP_PARAMS["filter_useless_cols"] and you_have_to_filter_cols_manually:
            filter_mask = ((X_process == 0) | (X_process == self.missing_value)).all(dim=2).all(dim=0)
            event_cols_encoded = [col for i, col in enumerate(event_cols_encoded) if not filter_mask[i]]
            X_process = X_process[:, ~filter_mask, :]

            # also remove columns which have the same value for all rows
            constant_mask = (X_process == X_process[0:1]).all(dim=0).all(dim=1)  # shape: [num_features]
            event_cols_encoded = [col for i, col in enumerate(event_cols_encoded) if not constant_mask[i]]
            X_process = X_process[:, ~constant_mask, :]

        return {"Y": Y, "case_nr": case_nr, "T": T, "prefix_len": prefix_len, "X_case": X_case, "X_event": X_process, "case_cols_encoded": case_cols_encoded, "event_cols_encoded": event_cols_encoded}
    
    # bereitet die BPIC17-Prozessdaten für das neuronale Netz/LSTM auf. Genauer: Sie soll aus einem Event Log Tensoren erzeugen, die den Prozesszustand an den einzelnen Entscheidungspunkten repräsentieren.
    def create_prefix_tensors(self, data, max_process_len, case_cols_encoded=None, event_cols_encoded=None, data_type="normal"):
        you_have_to_filter_cols_manually = False
        if case_cols_encoded is None:
            case_cols_encoded = self.case_cols_encoded
        if event_cols_encoded is None:
            event_cols_encoded = self.event_cols_encoded
            you_have_to_filter_cols_manually = True

        # save the indices
        treated_indices = []
        control_indices = []
        
        previous_case = -1
        case_treated_condition = False
        inference_dataset_indices = []
        X_cols = ["case_nr", "prefix_len"] + case_cols_encoded + event_cols_encoded
        X = torch.zeros(size=(len(data), len(X_cols) + self.nr_treatment_columns, max_process_len))
        for row_nr, row in data.iterrows():
            current_case = row["case_nr"]
            if current_case != previous_case:
                if data_type == "inference_dataset" and row_nr > 0:
                    inference_dataset_indices.append(row_nr - 1 - 1) #NOTE, additional -1 to retain without intervention

                event_nr = 0
                previous_case = current_case
                case_treated_condition = False
            else:
                # if treated and the case is still the same
                if case_treated_condition:
                    # go to the next row (no need to go through the rest of this case)
                    continue
                # copy all previous prefixes
                event_nr += 1

                if event_nr >= max_process_len:
                    continue

                X[row_nr, :, 0:event_nr] = X[row_nr-1, :, 0:event_nr]
            # add an event
            X[row_nr, 0, event_nr] = current_case

            # Process variable-length treatment list
            treatment_list = row["treatment"]
            X[row_nr, 1:1 + self.nr_treatment_columns, event_nr] = torch.tensor(treatment_list, dtype=torch.float32)
            last_index = 1+self.nr_treatment_columns

            X[row_nr, last_index, 0:event_nr + 1] = event_nr + 1
            last_index += 1

            X[row_nr, last_index:last_index + len(case_cols_encoded), event_nr] = torch.tensor(row[case_cols_encoded].values.astype(float))
            last_index += len(case_cols_encoded)
            X[row_nr, last_index: last_index + len(event_cols_encoded), event_nr] = \
                torch.tensor(row[event_cols_encoded].values.astype(float))
            
            if data_type == "normal":
                case_treated_condition = False
                control_condition = False
                
                # TREATMENT RETAIN
                if X[row_nr, 1:1 + self.nr_treatment_columns, event_nr].sum() > 0:
                    case_treated_condition = True
                else:
                # CONTROL RETAIN
                    all_zero_condition = torch.all(X[row_nr, 1:1 + self.nr_treatment_columns, :] == 0)
                    if len(self.DATASET_PARAMS["intervention_info"]["start_control_activity"]) > 0:
                        start_control_condition = False
                        end_control_condition = False
                        for start_control in self.DATASET_PARAMS["intervention_info"]["start_control_activity"]:
                            if row["activity_" + start_control] == 1: start_control_condition = True
                            if start_control_condition:
                                break
                        for end_control in self.DATASET_PARAMS["intervention_info"]["end_control_activity"]:
                            if row["activity_" + end_control] == 1: end_control_condition = True
                            if end_control_condition:
                                break
                        control_condition = all_zero_condition and start_control_condition and end_control_condition
                
                if case_treated_condition:
                    treated_indices.append(row_nr)
                elif control_condition:
                    # save the indices of the prefixes that could be a control case, then if we go to the next case and we see there was never a treatment, we can add these indices to the control indices
                    control_indices.append(row_nr)

        prefix_len = X[:, 1 + self.nr_treatment_columns, 0]
        treatment = X[:, 1:1 + self.nr_treatment_columns, :]
        
        # Retaining
        treated_indices = torch.tensor(treated_indices)
        control_indices = torch.tensor(control_indices)
        if data_type == "inference_sample":
            last_index = prefix_len.size(0) - 1 - 1 #NOTE, additional -1 to retain without intervention
            retain_idx = torch.isin(torch.arange(treatment.size(0)), (last_index))
        elif data_type == "inference_dataset":
            # retain for all cases just the last possible prefix, also don't forget to add the last prefix of the last case
            inference_dataset_indices.append(prefix_len.size(0) - 1 - 1)
            retain_idx = torch.isin(torch.arange(treatment.size(0)), torch.tensor(inference_dataset_indices))
            lol = 1
        else:
            retain_idx = torch.isin(torch.arange(treatment.size(0)), torch.cat((control_indices, treated_indices)))

        Y = torch.Tensor(data["outcome"].values)[retain_idx].unsqueeze(1)  # Make sure Y is of shape [n, 1] instead of [n]
        case_nr = X[retain_idx, 0 ,0]
        # Make T so that if there is a True in T, than it is just True, otherwise False
        T = torch.any(treatment[retain_idx, :, :], dim=2)
        # make T not boolean, but float
        T = T.float()
        last_index = 1 + self.nr_treatment_columns
        prefix_len = prefix_len[retain_idx]
        last_index += 1
        X_case = X[retain_idx, last_index:last_index + len(case_cols_encoded), 0] #, :]
        last_index += len(case_cols_encoded)
        X_process = X[retain_idx, last_index: last_index + len(event_cols_encoded), :] #, :]

        # in X_process, if there are any 'cols' with all zeros, remove them, goes from 17 --> 8 for time_contact HQ, 17 --> 10 for calculate_offer
        if self.PREP_PARAMS["filter_useless_cols"] and you_have_to_filter_cols_manually:
            filter_mask = ((X_process == 0) | (X_process == self.missing_value)).all(dim=2).all(dim=0)
            event_cols_encoded = [col for i, col in enumerate(event_cols_encoded) if not filter_mask[i]]
            X_process = X_process[:, ~filter_mask, :]

            # also remove columns which have the same value for all rows
            constant_mask = (X_process == X_process[0:1]).all(dim=0).all(dim=1)  # shape: [num_features]
            event_cols_encoded = [col for i, col in enumerate(event_cols_encoded) if not constant_mask[i]]
            X_process = X_process[:, ~constant_mask, :]

        return {"Y": Y, "case_nr": case_nr, "T": T, "prefix_len": prefix_len, "X_case": X_case, "X_event": X_process, "case_cols_encoded": case_cols_encoded, "event_cols_encoded": event_cols_encoded}
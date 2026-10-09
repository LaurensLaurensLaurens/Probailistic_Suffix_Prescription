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
            # one hot encoding
            self.oh_encoder_dict_train, self.data_encoded_train, self.case_cols_encoded, self.event_cols_encoded = mini_prep_tools.one_hot_encode_columns(data = self.data_train, cat_cols = self.DATASET_PARAMS["cat_cols"], case_cols = self.DATASET_PARAMS["case_cols"], event_cols = self.DATASET_PARAMS["event_cols"])
            # Scaling
            self.scaler_dict_train, self.data_scaled_train = mini_prep_tools.scale_columns(data = self.data_encoded_train, scale_cols = self.DATASET_PARAMS["scale_cols"], case_cols= self.DATASET_PARAMS["case_cols"])
            # Missing Values
            self.data_fill_train = self.handle_missing_values(data = self.data_scaled_train)
            # Treatment Column wird eingefügt
            self.data_treat_train = self.add_treatment_column_sequential(data = self.data_fill_train, scaler_dict_train=self.scaler_dict_train)

            if "call_or_not" in self.DATASET_PARAMS["intervention_info"]["name"]:
                self.data_train_prep = self.create_prefix_tensors_bpic17(data = self.data_treat_train, max_process_len = self.max_process_len, data_type="normal")
            else:
                # WICHTIG: PREFIXE SCHAFFEN FÜR TRAINING
                self.data_train_prep = self.create_prefix_tensors(data = self.data_treat_train, max_process_len = self.max_process_len, data_type="normal")
            self.case_cols_encoded, self.event_cols_encoded = self.data_train_prep["case_cols_encoded"], self.data_train_prep["event_cols_encoded"]
        
            self.prep_utils = {"scaler_dict_train": self.scaler_dict_train, 
                                "oh_encoder_dict_train": self.oh_encoder_dict_train, 
                                "max_process_len": self.max_process_len,
                                "case_cols_encoded": self.case_cols_encoded,
                                "event_cols_encoded": self.event_cols_encoded,
                                # Suffix (nur create_prefix_tensors, nicht BPIC17): alle Event-Spalten + activity_EOS + outcome
                                "suffix_cols_encoded": self.data_train_prep.get("suffix_cols_encoded"),
                                "dim_x_suffix": self.data_train_prep["X_suffix"].shape[1] if "X_suffix" in self.data_train_prep else 0,
                                "dim_x_case": self.data_train_prep["X_case"].shape[1],
                                "dim_x_event": self.data_train_prep["X_event"].shape[1],
                                "dim_t": self.data_train_prep["T"].shape[1] if self.data_train_prep["T"].shape[1] > 2 else 1,  # If binary treatment, we use one-hot encoding, so dim_t = 1
                                "dim_output": self.data_train_prep["Y"].shape[1],
            }

            # drop the case_cols_encoded and event_cols_encoded from the data
            self.data_train_prep.pop("case_cols_encoded")
            self.data_train_prep.pop("event_cols_encoded")
            self.data_train_prep.pop("suffix_cols_encoded", None)
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
            # WICHTIG: PREFIXE SCHAFFEN FÜR TRAINING
            self.data_infer_prep = self.create_prefix_tensors(data = self.data_treat_infer, max_process_len = self.prep_utils["max_process_len"], data_type="normal", case_cols_encoded=self.prep_utils["case_cols_encoded"], event_cols_encoded=self.prep_utils["event_cols_encoded"], suffix_cols_encoded=self.prep_utils["suffix_cols_encoded"])

        # drop the case_cols_encoded and event_cols_encoded from the data
        self.data_infer_prep.pop("case_cols_encoded")
        self.data_infer_prep.pop("event_cols_encoded")
        self.data_infer_prep.pop("suffix_cols_encoded", None)

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

        intervention_info = self.DATASET_PARAMS["intervention_info"]
        #print("-----intervention info-----")
        #print(intervention_info)
        intervention_column = intervention_info["column"]

        if intervention_column == "activity":
            intervention_activity = "activity_" + intervention_info["actions"][-1]
            next_action = data.groupby("case_nr", sort=False)[intervention_activity].shift(-1)
            data["treatment"] = next_action.fillna(0).astype(int)
        #Sonderbehandlung für die ursprüngliche interest rate. Warum das genau sein muss, ist mir noch nicht ganz klar
        elif intervention_column == "interest_rate":
            scaled_intervention_actions = pd.DataFrame(
                intervention_info["actions"], columns=["interest_rate"]
            )
            scaled_intervention_actions = mini_prep_tools.scale_column(
                col="interest_rate",
                data=scaled_intervention_actions,
                case_cols=self.DATASET_PARAMS["case_cols"],
                scaler=scaler_dict_train["interest_rate"],
            )[1]
            zeros_list = [0] * len(scaled_intervention_actions)
            data["treatment"] = [zeros_list.copy() for _ in range(len(data))]

            if treatment_index is not None:
                new_zero_list = zeros_list.copy()
                new_zero_list[treatment_index] = 1
                activity_column = "activity_calculate_offer"
                case_nr_value_last_calc_offer = -1
                for row_nr, row in data[data["interest_rate"] == scaled_intervention_actions["interest_rate"][treatment_index]].iterrows():
                    if row[activity_column] == 1.0 and row["case_nr"] != case_nr_value_last_calc_offer:
                        case_nr_value_last_calc_offer = row["case_nr"]
                        data.at[row_nr, "treatment"] = new_zero_list
            else:
                activity_column = "activity_calculate_offer"
                case_nr_value_last_calc_offer = -1
                for row_nr, row in data.iterrows():
                    if row[activity_column] == 1.0 and row["case_nr"] != case_nr_value_last_calc_offer:
                        for action_index, option in enumerate(scaled_intervention_actions["interest_rate"]):
                            if row["interest_rate"] == option:
                                new_zero_list = zeros_list.copy()
                                new_zero_list[action_index] = 1
                                case_nr_value_last_calc_offer = row["case_nr"]
                                data.at[row_nr - 1, "treatment"] = new_zero_list
                                break
        
        # Das isr für numerische Interventionen, wie z.B. interest_rate. Hier wird die Spalte treatment als One-Hot-Vektor kodiert, der angibt, welche der möglichen Interventionen gewählt wurde. Die Skalierung der möglichen Werte erfolgt mit dem zuvor trainierten Scaler.
        elif intervention_column in data.columns:
            intervention_activity = intervention_info["activities"][-1]
            activity_column = "activity_" + intervention_activity
            actions = np.asarray(intervention_info["actions"], dtype=float)
            scaler = (scaler_dict_train or {}).get(intervention_column)
            scaled_actions = (
                scaler.transform(actions.reshape(-1, 1)).reshape(-1)
                if scaler is not None
                else actions
            )

            grouped_data = data.groupby("case_nr", sort=False)
            next_activity = grouped_data[activity_column].shift(-1).fillna(0).to_numpy() == 1
            next_action_value = grouped_data[intervention_column].shift(-1).to_numpy(dtype=float)
            treatment = np.zeros((len(data), len(actions)), dtype=np.int64)
            for action_index, action_value in enumerate(scaled_actions):
                treatment[next_activity & np.isclose(next_action_value, action_value), action_index] = 1
            data["treatment"] = treatment.tolist()
        else:
            raise ValueError(f"Unsupported intervention column: {intervention_column}")

        if print_debug:
            print('data_treatment below')
        # Hier ist die treatment column dann dabei. Sieht zum beispiel so aus für Stage 1
        #[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #print(data)
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
    def create_prefix_tensors(self, data, max_process_len, case_cols_encoded=None, event_cols_encoded=None, data_type="normal", suffix_cols_encoded=None):
        print("start of prefix tensors")
        with open(
            "/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/SCOPE_WORKSPACE/src/readable.txt",
            "w",
            encoding="utf-8",
        ) as readable_file:
            readable_file.write(
                data.rename_axis("row_nr").to_string(
                    index=True,
                    max_rows=None,
                    max_cols=None,
                    line_width=1_000_000,
                    float_format=lambda value: f"{value:.6g}",
                )
            )
            readable_file.write("\n")

        

        you_have_to_filter_cols_manually = False
        if case_cols_encoded is None:
            case_cols_encoded = self.case_cols_encoded
        if event_cols_encoded is None:
            event_cols_encoded = self.event_cols_encoded
            you_have_to_filter_cols_manually = True
        # Das Suffix bekommt ALLE Event-Spalten (ungefiltert), weil dort Aktivitäten vorkommen, die im Prefix nie auftreten.
        # Dazu kommen zwei Spalten: activity_EOS markiert das Ende des Falls, outcome steht nur am EOS-Event.
        # Das ist notwendig für das Training des PSP.
        if suffix_cols_encoded is None:
            suffix_cols_encoded = list(self.event_cols_encoded) + ["activity_EOS", "outcome"]

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
            # Hier sollen die Zeilen vor dem Entscheidungspunkt markiert und gesammelt werden 
            #       row_nr  case_nr  activity         bike_value  elapsed_time  treatment  outcome
            # 0       10       initiate_case    0.1         0.00          0          0.7
            # 1       10       start_standard   0.1         0.02          0          0.7
            # 2       10       choose_employee  0.1         0.05          0          0.7
            # 3       11       initiate_case    0.4         0.00          1          0.3   <- danach folgt start_priority
            # 4       11       start_priority   0.4         0.02          0          0.3
            # 5       12       initiate_case    0.8         0.00          0          0.9
            # 6       12       start_standard   0.8         0.02          0          0.9
            # 7       12       choose_employee  0.8         0.06          0          0.9

            if current_case != previous_case:
                if data_type == "inference_dataset" and row_nr > 0:
                    inference_dataset_indices.append(row_nr - 1 - 1) #NOTE, additional -1 to retain without intervention

                # neuer Fall
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
                
                # Baut den prefix stück für stück bis zur Treatment entscheidung
                # Kopiert aus den positionen davor
                X[row_nr, :, 0:event_nr] = X[row_nr-1, :, 0:event_nr]

            # add an event
            # # Die erste Tensors des cases, den wir gerade betrachten, bekommt die Fallnummer des Cases
            # row_nr zählt die Zeilen durch
            # 0 ist feature Feld für die Fallnummer
            # event_nr = die Position des aktuellen Events im Case
            # z.B. X[5, 0, 2] = 12 "Die sechste Zeile im Dataframe mit der eventnummer 2 gehört zum 12. Fall"
            X[row_nr, 0, event_nr] = current_case 

            # Process variable-length treatment list
            # Das ganze list zu nennen ist meines erachtesn falsch. Es sollte nur ein Wert sien.
            # row["treatment"] holt den Treatment-Wert der aktuellen Datenzeile. Die nächsten beiden Zeilen schreiben diesen Wert in den Tensor an die Position des aktuellen Events:
            treatment_list = row["treatment"]
            
            #self.nr_treatment_columns ist die Anzahl der Feature-Felder im Tensor, die für die Treatments reserviert sind. 
            # Ab dem ersten index, weil der 0te index für die Fallnummer reserviert ist
            # Nach dem ersten index kommen die treatments als feature-Felder, also um zu sehen, welche Interventionen an diesem Entscheidungspunkt gewählt wurden.
            # treatment_list ist keine liste, sondern ein wert und zeigt an, ob ein Treatment gewählt wurde oder nicht
            #                 X[0]          X[1]          X[2]          X[3]          X[4] Das sind Rows!! Nicht cases!
            # case_nr       [10, 0, 0]    [10,10, 0]    [10,10,10]    [11, 0, 0]    [0, 0, 0]
            # treatment     [ 0, 0, 0]    [ 0, 0, 0]    [ 0, 0, 0]    [ 1, 0, 0]    [0, 0, 0]
            # prefix_len    [ 1, 0, 0]    [ 2, 2, 0]    [ 3, 3, 3]    [ 1, 0, 0]    [0, 0, 0]
            # bike_value    [.1, 0, 0]    [.1,.1, 0]    [.1,.1,.1]    [.4, 0, 0]    [0, 0, 0]
            # elapsed_time  [ 0, 0, 0]    [ 0,.02,0]    [ 0,.02,.05]  [ 0, 0, 0]    [0, 0, 0]

            X[row_nr, 1:  1+self.nr_treatment_columns, event_nr] = torch.tensor(treatment_list, dtype=torch.float32)
            #print(X[row_nr])
            last_index = 1+self.nr_treatment_columns

            X[row_nr, last_index, 0:event_nr + 1] = event_nr + 1
            last_index += 1

            # Fall 12, Treatment 2 wird bei event_index 9 gewaehlt; der Prefix endet dort.
            # Zustand von X[row_nr] NACH den folgenden Zeilen (case- und event-Spalten werden hier
            # nur an Position event_nr geschrieben; 0..8 stammen aus der Kopie der Vorgaengerzeile).

            # Das isr eun Einzelner prefix X[row_nr]
            # Jede Row wird zu einem einzelnen prefix
            # Alle Positionen nach event_nr bleiben 0 (Padding bis max_process_len).
            X[row_nr, last_index:last_index + len(case_cols_encoded), event_nr] = torch.tensor(row[case_cols_encoded].values.astype(float))
            last_index += len(case_cols_encoded)
            X[row_nr, last_index: last_index + len(event_cols_encoded), event_nr] = \
                torch.tensor(row[event_cols_encoded].values.astype(float))
            
            # Welche Präfixe werden am Ende behalten?
            # Hier: Präfixe an Entscheidungspunkten
            # Kontroll- und Treated-Präfixe zusammen. Das ist der einzige Zweig, der aktuell läuft. Siehe andere mod unten
            # Normal ist für das normale training. Nicht zur Inferenzzeit
            if data_type == "normal":
                case_treated_condition = False
                control_condition = False
                
                # TREATMENT RETAIN
                # Guckt für jedes event, ob ein Treatment iregdnwo gewwählt wurde. Wenn ja, dann speicheren. 
                if X[row_nr, 1:1 + self.nr_treatment_columns, event_nr].sum() > 0:
                    case_treated_condition = True
                else:
                # CONTROL RETAIN. Jedes Prefix braucht ein outcome und ein Treatment
                    # Guckt ob, es ein Entscheidungspunkt ist, an dem hätte behandeln können, es aber nicht getan hat.
                    # Also zum vergeleich zwischen treated und control cases

                    # Hier steht für im gesamten prefix kein treatment 
                    all_zero_condition = torch.all(X[row_nr, 1:1 + self.nr_treatment_columns, :] == 0)
                    # Guckt ob die Start Control Aktivity und End Control activity in der Liste vorkommen.
                    # Die Kontroll-Aktivität ist das Event, an dem die Entscheidung fällt
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
                
                # ACHTUNG: Control hei0t hier "Stand an einem Entscheidungspunkt, blieb aber untreated"
                # Wenn weder control noch treated, dann wird verworfen
                # control_indices = [0, 5]
                # X[0]: Fall 10, treatment [0,0,0], Aktivität initiate_case   -> Control
                # X[5]: Fall 12, treatment [0,0,0], Aktivität initiate_case   -> Control (nicht in der Tabelle)

                # treated_indices = [3]
                # X[3]: Fall 11, treatment [1,0,0]                             -> Treated
                if case_treated_condition:
                    treated_indices.append(row_nr)
                elif control_condition:
                    # save the indices of the prefixes that could be a control case, then if we go to the next case and we see there was never a treatment, we can add these indices to the control indices
                    control_indices.append(row_nr)

        #Jede Zeile der prefix der ersten 3 events von Fall 0. 
        # Jede Zeile ist ein Feature
        # Jede Spalte ein Event
        # Ab Spalte 3 ist alle Padding, weil wir ja erst bei Event 3 sind
        #print(X[2])
        #return

        # liest für jede Zeile die Präfixlänge aus. Länge steht an allen positonen des prefixe
        # Nimmt alle Rows, geh jeweils in jeden der prefixe rein. Guck nach der Zeile mit der länge des prefixes, nimm das erste element daraus
        prefix_len = X[:, 1 + self.nr_treatment_columns, 0]
        #  nimmt die Treatment-Features über alle Event-Positionen. 
        # # Nimmt alle Rows, geh jeweils in jeden der prefixe rein. Nimm alle Elemente/Zeilen (Features) ab der ersten (statt 0ten) bis zur letzten Treatement spalte und nimm hier alle elemente als Liste
        #  Ist also für jeden Prefix, ob und wann welches treatment durchgeführt worden sit
        treatment = X[:, 1:1 + self.nr_treatment_columns, :]
        #print(treatment)
        
        # Retaining
        # Das ist ein Filter, welche präfixe behalten werden
        
        treated_indices = torch.tensor(treated_indices)
        control_indices = torch.tensor(control_indices)

        # nur der vorletzte Präfix überhaupt.
        # Zum Zeiptun der Inferenz läuft der Fall noch und man will wissen, ob man behandeln soll oder nicht.
        # Es gibt hier kein Control
        if data_type == "inference_sample":
            last_index = prefix_len.size(0) - 1 - 1 #NOTE, additional -1 to retain without intervention
            retain_idx = torch.isin(torch.arange(treatment.size(0)), (last_index))
        #der vorletzte Präfix jedes Falls.
        elif data_type == "inference_dataset":
            # retain for all cases just the last possible prefix, also don't forget to add the last prefix of the last case
            inference_dataset_indices.append(prefix_len.size(0) - 1 - 1)
            retain_idx = torch.isin(torch.arange(treatment.size(0)), torch.tensor(inference_dataset_indices))
            lol = 1
        else:
            # Das baut hier die Maske, ist ein prefix werden Control noch treated, so wird er verworfen, weil es sich nicht
            # um einen Entscheidungspunkt handelt. 

            # treatment.size(0) ist die Anzahl der Zeilen (Features), also das selbe wie len(data)
            # torch.cat((control_indices, treated_indices)) hängt die beiden Listen aneinander. Das sind die Zeilennummern, die in der Schleife als Entscheidungspunkt markiert wurden:
            # torch.isin prüft für jedes Element von a, ob es in b vorkommt:
            retain_idx = torch.isin(torch.arange(treatment.size(0)), torch.cat((control_indices, treated_indices)))

        # y nur die retainten
        # Wirft durch die Maske jene raus, die weder Control not Treatment sind, also keine Entscheidungspunkt
        Y = torch.Tensor(data["outcome"].values)[retain_idx].unsqueeze(1)  # Make sure Y is of shape [n, 1] instead of [n]
        case_nr = X[retain_idx, 0 ,0]
        # Make T so that if there is a True in T, than it is just True, otherwise False
        # Treatment alle rausschmeißen ohne
        T = torch.any(treatment[retain_idx, :, :], dim=2)
        # make T not boolean, but float
        T = T.float()
        last_index = 1 + self.nr_treatment_columns
        prefix_len = prefix_len[retain_idx]
        last_index += 1

        # Wir trennen hier case und Event-Varablen, weil wir keine unntöigen Wiederholungen haben wollen
        # Zudem verarbeiten die Modelle die Werte unterschiedlch. Bei LSTMs wird die event sequejz druchgereicht
        # und erst ganz am Ende werden die Case variablen dem Ergebnis angehängt
        # Bei psp auch, hier werden die case Variablen als Static nput verwendet


        # Nur case Variablen. Position 0 reicht, weil Case-Attribute im ganzen Fall gleich sind.
        X_case = X[retain_idx, last_index:last_index + len(case_cols_encoded), 0] #, :]
        last_index += len(case_cols_encoded)
        # Dann nur alle Event Variablen
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

        # --------------------WICHTIG------------------------
        # Suffixe für das PSP: zu jedem behaltenen Prefix die restlichen Events des Falls
        # Das brauchen wir für das Training des PSPs
        X_suffix, suffix_len = self.create_suffix_tensors(data=data, retain_idx=retain_idx, max_process_len=max_process_len, suffix_cols_encoded=suffix_cols_encoded)
        #print("suffix_tensors")
        #print(X_suffix)
        #print(suffix_len)

        return {"Y": Y, "case_nr": case_nr, "T": T, "prefix_len": prefix_len, "X_case": X_case, "X_event": X_process, "X_suffix": X_suffix, "suffix_len": suffix_len, "case_cols_encoded": case_cols_encoded, "event_cols_encoded": event_cols_encoded, "suffix_cols_encoded": suffix_cols_encoded}

    # Baut zu jedem behaltenen Prefix das Suffix, also alle Events des Falls NACH dem Entscheidungspunkt.
    # Das Suffix kommt direkt aus dem DataFrame und nicht aus X, weil X bei behandelten Fällen nach dem Treatment leer bleibt.
    #
    # Beispiel: Fall 10 mit 3 Events, behalten wird der Prefix aus Zeile 0 (initiate_case), max_process_len = 3
    #    suffix_position                 0               1                2     3
    #   elapsed_time                  0.02            0.05                0     0
    #   activity_start_standard          1               0                0     0
    #   activity_choose_employee         0               1                0     0
    #   ...
    #   activity_EOS                     0               0                1     0   <- steht an Position suffix_len
    #   outcome                          0               0              0.7     0   <- nur am EOS-Event
    #   suffix_len = 2 (Anzahl echter Events, ohne EOS)
    def create_suffix_tensors(self, data, retain_idx, max_process_len, suffix_cols_encoded):
        # Die letzten beiden Spalten (activity_EOS, outcome) gibt es im DataFrame nicht als Event-Spalten
        event_cols = suffix_cols_encoded[:-2]
        # alle Events als Matrix [len(data), n_event_cols], bereits skaliert und one-hot-kodiert
        event_matrix = torch.tensor(data[event_cols].values.astype(float), dtype=torch.float32)
        outcome = torch.tensor(data["outcome"].values.astype(float), dtype=torch.float32)
        # Wie viele Events folgen in diesem Fall noch nach dieser Zeile? z.B. Fall mit 3 Events: [2, 1, 0]
        remaining = data.groupby("case_nr", sort=False).cumcount(ascending=False).values

        # Zeilennummern der behaltenen Prefixe, gleiche Reihenfolge wie Y, T, X_case und X_event
        retained_rows = torch.nonzero(retain_idx).squeeze(1).tolist()
        # + 1, damit nach dem längsten Suffix noch Platz für das EOS-Event ist
        X_suffix = torch.zeros(size=(len(retained_rows), len(suffix_cols_encoded), max_process_len + 1))
        suffix_len = torch.zeros(len(retained_rows))
        for i, row_nr in enumerate(retained_rows):
            n_rest = min(int(remaining[row_nr]), max_process_len)
            # Zeilen row_nr+1 ... row_nr+n_rest sind die restlichen Events desselben Falls
            X_suffix[i, :len(event_cols), :n_rest] = event_matrix[row_nr + 1: row_nr + 1 + n_rest].T
            X_suffix[i, -2, n_rest] = 1
            X_suffix[i, -1, n_rest] = outcome[row_nr]
            suffix_len[i] = n_rest

        return X_suffix, suffix_len
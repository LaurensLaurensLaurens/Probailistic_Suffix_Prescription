import torch
import src.utils.prep_tools.mini_prep_tools as mini_prep_tools
import pandas as pd
import numpy as np
from copy import deepcopy

class AggPreprocessor():
    def __init__(self, data_train, data_infer, PREP_PARAMS, DATASET_PARAMS, add_properties, prep_utils=None):
        self.data_train = data_train
        self.data_infer = data_infer
        self.PREP_PARAMS = PREP_PARAMS
        self.DATASET_PARAMS = DATASET_PARAMS
        self.nr_treatment_columns = add_properties["nr_treatment_columns"]
        self.missing_value = add_properties["missing_value"]
        self.stage = add_properties["stage"]
        self.prep_utils = prep_utils
        self.n_stages = len(DATASET_PARAMS["intervention_info"]["action_combinations"])

    """Aggregated preprocessing"""
    def preprocess(self, eval=False):
        if self.PREP_PARAMS["train_prop"] > 0:
        # TRAIN
            self.data_treat_train = self.add_treatment_column_aggregated(self.data_train)
            if "call_or_not" in self.DATASET_PARAMS["intervention_info"]["name"]:
                self.data_pref_train = self.create_prefix_aggregations_bpic17(self.data_treat_train)
            else:
                self.data_pref_train = self.create_prefix_aggregations(self.data_treat_train)
            self.scale_cols_currently = self.DATASET_PARAMS["scale_cols"] + [col for col in self.data_pref_train.columns if "_count" in col] + ["prefix_len"]
            self.scaler_dict_train, self.data_scaled_train = mini_prep_tools.scale_columns(data = self.data_pref_train, scale_cols = self.scale_cols_currently, case_cols=self.DATASET_PARAMS["case_cols"])
            T_array_train = np.array(self.data_scaled_train["treatment"].tolist())
            if len(T_array_train.shape) == 1:
                T_array_train = T_array_train.reshape(-1, 1)
            self.data_train_prep = {
                "Y": torch.tensor(self.data_scaled_train["outcome"].values.reshape(-1, 1), dtype=torch.float32),
                "case_nr": torch.tensor(self.data_scaled_train["case_nr"].values, dtype=torch.float32),
                "T": torch.tensor(T_array_train, dtype=torch.float32),
                "X_case": torch.tensor(self.data_scaled_train.drop(columns=["outcome", "treatment", "case_nr"]).values, dtype=torch.float32),
                # Note: X_event is not used in aggregated preprocessing, but we keep it for consistency with the tensor preprocessing
                "X_event": None,
                "prefix_len": None,
            }

            self.prep_utils = {"scaler_dict_train": self.scaler_dict_train, "column_names": self.data_pref_train.columns, "scale_cols": self.scale_cols_currently,
                               "dim_x_case": self.data_train_prep["X_case"].shape[1], "dim_x_event": self.data_train_prep["X_event"].shape[1] if self.data_train_prep["X_event"] is not None else 0,
                               "dim_t": self.data_train_prep["T"].shape[1] if self.data_train_prep["T"].shape[1] > 2 else 1,  # If binary treatment, we use one-hot encoding, so dim_t = 1}
                               "dim_output": self.data_train_prep["Y"].shape[1],
            }
        else:
            self.data_train_prep = None

        # INFER
        self.data_treat_infer = self.add_treatment_column_aggregated(self.data_infer)
        if "call_or_not" in self.DATASET_PARAMS["intervention_info"]["name"]:
            self.data_pref_infer = self.create_prefix_aggregations_bpic17(self.data_treat_infer, cols=self.prep_utils["column_names"])
        else:
            self.data_pref_infer = self.create_prefix_aggregations(self.data_treat_infer, cols=self.prep_utils["column_names"])
        _, self.data_scaled_infer = mini_prep_tools.scale_columns(data = self.data_pref_infer, scaler_dict = self.prep_utils["scaler_dict_train"], scale_cols = self.prep_utils["scale_cols"], case_cols=self.DATASET_PARAMS["case_cols"])
        T_array_infer = np.array(self.data_scaled_infer["treatment"].tolist())
        if len(T_array_infer.shape) == 1:
            T_array_infer = T_array_infer.reshape(-1, 1)
        self.data_infer_prep = {
            "Y": torch.tensor(self.data_scaled_infer["outcome"].values.reshape(-1, 1), dtype=torch.float32),
            "case_nr": torch.tensor(self.data_scaled_infer["case_nr"].values, dtype=torch.float32),
            "T": torch.tensor(T_array_infer, dtype=torch.float32),
            "X_case": torch.tensor(self.data_scaled_infer.drop(columns=["outcome", "treatment", "case_nr"]).values, dtype=torch.float32),
            # Note: X_event is not used in aggregated preprocessing, but we keep it for consistency with the tensor preprocessing
            "X_event": None,
            "prefix_len": None,
        }

        return self.data_train_prep, self.data_infer_prep, self.prep_utils
    
    def create_prefix_aggregations_bpic17(self, data, data_type="normal", cols=None):
        to_one_hot_encode = [col for col in self.DATASET_PARAMS["cat_cols"] if col != "activity"]
        _, data, case_cols_encoded, event_cols_encoded = mini_prep_tools.one_hot_encode_columns(data = data, cat_cols = to_one_hot_encode, case_cols = self.DATASET_PARAMS["case_cols"], event_cols = self.DATASET_PARAMS["event_cols"])
        encoded = case_cols_encoded + event_cols_encoded
        last_state_cols_list = deepcopy(self.DATASET_PARAMS["last_state_cols"])
        for column in self.DATASET_PARAMS["last_state_cols"]:
            og = len(last_state_cols_list)
            last_state_cols_list += [col for col in encoded if (col.startswith(column) and col != column)]
            af = len(last_state_cols_list)
            if og != af:
                last_state_cols_list.remove(column)

        you_have_to_filter_cols_manually = True
        if cols is not None:
            activity_count_cols = [col for col in cols if "_count" in col]
            you_have_to_filter_cols_manually = False
        grouped_train = data.groupby("case_nr")
        self.scale_cols_aggregate = [col for col in self.DATASET_PARAMS["scale_cols"] if col not in last_state_cols_list]
        unique_activities = data["activity"].unique()

        train_prep = pd.DataFrame()

        for case_nr, group in grouped_train:
            # initiate a mean for every scale aggregate column
            sum_dict = {col: 0 for col in self.scale_cols_aggregate}
            scale_col_count = {col: 0 for col in self.scale_cols_aggregate}
            # initiate a count for every activity
            activity_count_dict = {str(activity) + "_count": 0 for activity in unique_activities}
            # initiate a last state for every last state column
            last_state_dict = {col: 0 for col in last_state_cols_list}

            nr_decision_points_tracker = 0

            for current_pos, (index, row) in enumerate(group.iterrows(), start=1):
                decision_condition = (row["activity"] == "validate_application") or (row["activity"] == "call_incomplete_files") or (row["activity"] == "wait_incomplete_files")
                prefix_condition = (row["activity"] == "validate_application" or row["activity"] == "call_incomplete_files" or row["activity"] == "wait_incomplete_files") and nr_decision_points_tracker == self.stage

                for col in self.scale_cols_aggregate:
                    if not pd.isna(row[col]):
                        sum_dict[col] += row[col]
                        scale_col_count[col] += 1
                activity_count_dict[row["activity"] + "_count"] += 1

                if decision_condition:
                    nr_decision_points_tracker += 1

                if prefix_condition:
                    for col in last_state_cols_list:
                        last_state_dict[col] = row[col]
                    
                    mean_dict = {col: ((sum_dict[col] / scale_col_count[col]) if scale_col_count[col] > 0 else 0) for col in self.scale_cols_aggregate}

                    prefix = pd.DataFrame({**mean_dict, **activity_count_dict, **last_state_dict}, index=[0])
                    prefix["treatment"] = [row["treatment"]]
                    prefix["prefix_len"] = current_pos
                    prefix["case_nr"] = [case_nr]
                    if cols is not None:
                        # if there is a col in activity_count_cols that is not in the prefix, add it with value 0
                        for col in activity_count_cols:
                            if col not in prefix.columns:
                                prefix[col] = 0
                    train_prep = pd.concat([train_prep, prefix], axis=0, ignore_index=True)
                else:
                    continue

        # if there are columns which have 0 as value in all rows, drop them --> goes from 21 until 12 columns
        if self.PREP_PARAMS["filter_useless_cols"]:
            if not you_have_to_filter_cols_manually: 
                # only get the columns in cols
                train_prep = train_prep[cols]
            else:
                train_prep = train_prep.loc[:, (train_prep != 0).any(axis=0)]

                # Also drop if it is only 1 value for the whole column (but not for outcome, treatment and case_nr)
                for col in train_prep.columns:
                    if col not in ["outcome", "treatment", "case_nr"]:
                        if len(train_prep[col].unique()) == 1:
                            train_prep = train_prep.drop(columns=[col])

        return train_prep
    
    def create_prefix_aggregations(self, data, data_type="normal", cols=None):
        you_have_to_filter_cols_manually = True
        if cols is not None:
            activity_count_cols = [col for col in cols if "_count" in col]
            you_have_to_filter_cols_manually = False
        grouped_train = data.groupby("case_nr")
        self.scale_cols_aggregate = [col for col in self.DATASET_PARAMS["scale_cols"] if col not in self.DATASET_PARAMS["last_state_cols"]]
        unique_activities = data["activity"].unique()

        train_prep = pd.DataFrame()

        for case_nr, group in grouped_train:
            case_treated = False

            # initiate a mean for every scale aggregate column
            sum_dict = {col: 0 for col in self.scale_cols_aggregate}
            scale_col_count = {col: 0 for col in self.scale_cols_aggregate}
            # initiate a count for every activity
            activity_count_dict = {str(activity) + "_count": 0 for activity in unique_activities}
            # initiate a last state for every last state column
            last_state_dict = {col: 0 for col in self.DATASET_PARAMS["last_state_cols"]}

            # if there is a row in the group with treatment == 1, only retain the the prefix ending at that row (not the previous ones)
            # if there is never a treatment == 1, retain all prefixes (for loop)
            for current_pos, (index, row) in enumerate(group.iterrows(), start=1):
                if case_treated:
                    break
                for col in self.scale_cols_aggregate:
                    if not pd.isna(row[col]):
                        sum_dict[col] += row[col]
                        scale_col_count[col] += 1
                activity_count_dict[row["activity"] + "_count"] += 1

                # check if row["treatment"] contains a 1 instead of row["treatment"] == 1
                if 1 in row["treatment"] if isinstance(row["treatment"], list) else row["treatment"] == 1:
                    for col in self.DATASET_PARAMS["last_state_cols"]:
                        last_state_dict[col] = row[col]
                    
                    mean_dict = {col: ((sum_dict[col] / scale_col_count[col]) if scale_col_count[col] > 0 else 0) for col in self.scale_cols_aggregate}

                    prefix = pd.DataFrame({**mean_dict, **activity_count_dict, **last_state_dict}, index=[0])
                    prefix["treatment"] = [row["treatment"]]
                    prefix["prefix_len"] = current_pos
                    prefix["case_nr"] = [case_nr]
                    if cols is not None:
                        # if there is a col in activity_count_cols that is not in the prefix, add it with value 0
                        for col in activity_count_cols:
                            if col not in prefix.columns:
                                prefix[col] = 0
                    train_prep = pd.concat([train_prep, prefix], axis=0, ignore_index=True)
                    case_treated = True
                
                # if datatyp is inference, we just want to retain if the current row is the second last row of the case
                else:
                    inference_condition = data_type == "infer_prefix" and current_pos == len(group) - 1

                    end_control_condition = False
                    if len(self.DATASET_PARAMS["intervention_info"]["end_control_activity"]) > 0:
                        for end_control in self.DATASET_PARAMS["intervention_info"]["end_control_activity"]:
                            end_control_condition = row["activity"] == end_control
                            if end_control_condition:
                                break
                    
                    if end_control_condition:
                        if inference_condition or data_type != "infer_prefix":
                            for col in self.DATASET_PARAMS["last_state_cols"]:
                                last_state_dict[col] = row[col]
                            
                            mean_dict = {col: ((sum_dict[col] / scale_col_count[col]) if scale_col_count[col] > 0 else 0) for col in self.scale_cols_aggregate}

                            prefix = pd.DataFrame({**mean_dict, **activity_count_dict, **last_state_dict}, index=[0])
                            prefix["treatment"] = [row["treatment"]]
                            prefix["prefix_len"] = current_pos
                            prefix["case_nr"] = [case_nr]
                            if cols is not None:
                                # if there is a col in activity_count_cols that is not in the prefix, add it with value 0
                                for col in activity_count_cols:
                                    if col not in prefix.columns:
                                        prefix[col] = 0
                            train_prep = pd.concat([train_prep, prefix], axis=0, ignore_index=True)
        
        # if there are columns which have 0 as value in all rows, drop them --> goes from 21 until 12 columns
        if self.PREP_PARAMS["filter_useless_cols"]:
            if not you_have_to_filter_cols_manually: 
                # only get the columns in cols
                train_prep = train_prep[cols]
            else:
                train_prep = train_prep.loc[:, (train_prep != 0).any(axis=0)]

                # Also drop if it is only 1 value for the whole column (but not for outcome, treatment and case_nr)
                for col in train_prep.columns:
                    if col not in ["outcome", "treatment", "case_nr"]:
                        if len(train_prep[col].unique()) == 1:
                            train_prep = train_prep.drop(columns=[col])

        return train_prep
    
    def add_treatment_column_aggregated(self, data, print_debug=False, treatment_index=None):
        # reset to be sure
        data = data.reset_index()

        # if there is already a treatment column, skip this function
        if 'treatment' in data.columns:
            return data

        if self.DATASET_PARAMS["intervention_info"]["column"] == "activity":
            intervention_activity = self.DATASET_PARAMS["intervention_info"]["actions"][-1]
            data['treatment'] = np.where(data['activity'].shift(-1) == intervention_activity, 1, 0)
        else:
            if self.DATASET_PARAMS["intervention_info"]["column"] == "interest_rate":
                intervention_actions = pd.DataFrame(self.DATASET_PARAMS["intervention_info"]["actions"], columns=["interest_rate"])
                zeros_list = [0] * len(intervention_actions)
                data['treatment'] = [zeros_list for _ in range(len(data))]
                # make sure the treatment column has as type list
                data['treatment'] = data['treatment'].apply(lambda x: [int(i) for i in x])

                if treatment_index is not None:
                    new_zero_list = zeros_list.copy()
                    new_zero_list[treatment_index] = 1
                    activity_column = "activity_" + "calculate_offer"
                    case_nr_value_last_calc_offer = -1
                    for row_nr, row in data[data['interest_rate'] == intervention_actions["interest_rate"][treatment_index]].iterrows():
                        if row[activity_column] == 1.0:
                            if row["case_nr"] != case_nr_value_last_calc_offer:
                                case_nr_value_last_calc_offer = row["case_nr"]
                                data.at[row_nr, 'treatment'] = new_zero_list
                else:
                    # activity_column = "activity_" + "calculate_offer"
                    case_nr_value_last_calc_offer = -1
                    for row_nr, row in data.iterrows():
                        if row["activity"] == "calculate_offer":
                            if row["case_nr"] != case_nr_value_last_calc_offer:
                                for i, option in enumerate(intervention_actions["interest_rate"]):
                                    if row["interest_rate"] == option:
                                        new_zero_list = zeros_list.copy()
                                        new_zero_list[i] = 1
                                        case_nr_value_last_calc_offer = row["case_nr"]
                                        data.at[row_nr - 1, 'treatment'] = new_zero_list

        if print_debug:
            print('data_treatment below')
        return data
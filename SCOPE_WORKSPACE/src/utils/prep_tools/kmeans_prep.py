import pandas as pd
from src.utils.prep_tools.mini_prep_tools import scale_columns, one_hot_encode_columns
from copy import deepcopy

class KMeansPreprocessor():
    def __init__(self, data_train, data_infer, PREP_PARAMS, DATASET_PARAMS, prep_utils=None, args=None):
        self.data = data_train
        self.data_infer = data_infer
        self.PREP_PARAMS = PREP_PARAMS
        self.DATASET_PARAMS = DATASET_PARAMS
        self.prep_utils = prep_utils
        self.n_stages = len(DATASET_PARAMS["intervention_info"]["action_combinations"])
        if "call_or_not" in self.DATASET_PARAMS["intervention_info"]["name"]:
            self.dataset = 'bpic17'
        else:
            self.dataset = 'SimBank'
        self.args = args

    def get_case_end_condition(self, row):
        condition = True
        if self.dataset == 'SimBank':
            condition = row["activity"] == "cancel_application" or row["activity"] == "receive_acceptance"
        elif self.dataset == 'bpic17':
            condition = row["activity"] == "end"
        return condition
    
    def get_intervention_condition(self, row, current_pos=0, group=None, nr_decision_points=0):
        condition = True
        action = None
        if self.dataset == 'SimBank':
            # add prefix to the dataframe, only if initiate_application is the 'current' activity, or if 'start_prior' is the activity, or if the current activity is 'validate_application'/'contact_headquarters' and the position is 10
            intervention_1_criterium = (row["activity"] == "initiate_application")
            
            intervention_2_criterium = (row["activity"] == "start_priority"
                                    or
                                        # check if the current activity is 'validate_application' and the position is 10 and contact_headquarters has occured in the case
                                        (row["activity"] == "validate_application" and current_pos == 10 and "contact_headquarters" in group["activity"].values) 
                                    or
                                        (row["activity"] == "contact_headquarters" and current_pos == 10)
                                    )
            condition = (intervention_1_criterium or intervention_2_criterium)

            if condition:
                # add the action: if intervention_1_criterium, the action is just the next activity
                if intervention_1_criterium:
                    # check if there is a next activity, if not, set it to None
                    action = group.iloc[current_pos]["activity"] if current_pos < len(group) else None
                # add the action: if intervention_2_criterium, the action is the interest rate in the next row
                elif intervention_2_criterium:
                    action = group.iloc[current_pos]["interest_rate"] if current_pos < len(group) else None

        elif self.dataset == 'bpic17':
            condition = (row["activity"] == "validate_application" and nr_decision_points == 0) or ( (row["activity"] == "call_incomplete_files" or row["activity"] == "wait_incomplete_files") and nr_decision_points < self.n_stages)
            action = group.iloc[current_pos]["activity"] if current_pos < len(group) else None

        return condition, action

    def preprocess(self):
        if self.PREP_PARAMS["train_prop"] > 0:
            self.prep_train, self.prep_utils = self.transform(self.data)
        else:
            self.prep_train = None
        
        self.prep_infer, self.prep_utils = self.transform(self.data_infer, prep_utils=self.prep_utils)

        return self.prep_train, self.prep_infer, self.prep_utils
    
    def transform(self, data, prep_utils=None):
        you_have_to_filter_cols_manually = True
        if prep_utils is not None:
            you_have_to_filter_cols_manually = False

        prep_enriched = pd.DataFrame()
        # create an activity dictionary with each unique activity in the event log as a key
        unique_activities = data["activity"].unique()
        activity_count_dict_max = {str(activity) + "_count": 0 for activity in unique_activities}
        activity_pos_dict_max = {str(activity) + "_pos": 0 for activity in unique_activities}

        grouped = data.groupby("case_nr")

        for case_nr, group in grouped:
            activity_count_dict = {str(activity) + "_count": 0 for activity in unique_activities}
            activity_pos_dict = {str(activity) + "_pos": 0 for activity in unique_activities}
            nr_decision_points = 0
            avg_dict = {column: 0 for column in self.DATASET_PARAMS["scale_cols"] if column not in self.DATASET_PARAMS["last_state_cols"]}
            for current_pos, (index, row) in enumerate(group.iterrows(), start=1):
                activity_count_dict[row["activity"] + "_count"] += 1
                # grab the position of the activity in the case length (starting from 1), so check which position the current row has in the case
                activity_pos_dict[row["activity"] + "_pos"] = current_pos

                # put both dictionaries in a dataframe
                prefix = pd.DataFrame({**activity_count_dict, **activity_pos_dict}, index=[0])

                columns_to_loop_through = list(dict.fromkeys(
                    (self.DATASET_PARAMS.get("scale_cols") or []) +
                    (self.DATASET_PARAMS.get("last_state_cols") or []) +
                    (self.DATASET_PARAMS.get("case_cols") or [])
                ))
                for col in columns_to_loop_through:
                    if col == 'outcome' and not self.get_case_end_condition(row=row):
                        prefix[col] = 0
                    else:
                        # check whether NaN or not
                        if pd.isna(row[col]):
                            prefix[col] = -1
                        else:
                            if col in avg_dict:
                                avg_dict[col] += row[col]
                                prefix[col] = avg_dict[col] / current_pos
                            else:
                                prefix[col] = row[col]
                
                intervention_condition, action = self.get_intervention_condition(row=row, current_pos=current_pos, group=group, nr_decision_points=nr_decision_points)
                if intervention_condition:
                    nr_decision_points += 1
                    # add the action to the prefix
                    prefix["a"] = action
                    # add the case_nr to the prefix
                    prefix["case_nr"] = case_nr
                    # add the outcome
                    prefix["outcome"] = row["outcome"]
                    # add the activity
                    prefix["activity"] = row["activity"]
                    # add the next activity
                    prep_enriched = pd.concat([prep_enriched, prefix], axis=0, ignore_index=True)

        # divide the count of each activity by the maximum count of that activity
        # divide the position of each activity by the maximum position of that activity
        if self.prep_utils is not None:
            activity_count_max = self.prep_utils["activity_count_max"]
            activity_pos_max = self.prep_utils["activity_pos_max"]
        else:
            for activity in unique_activities:
                count_col = activity + "_count"
                pos_col = activity + "_pos"

                if count_col in prep_enriched.columns:
                    activity_count_dict_max[activity + "_count"] = prep_enriched[activity + "_count"].max()
                if pos_col in prep_enriched.columns:
                    activity_pos_dict_max[activity + "_pos"] = prep_enriched[activity + "_pos"].max()

            # grab the all around maximum of the activity counts and positions
            # Filter the dicts to only include keys that still exist in prep_enriched
            valid_count_cols = [col for col in activity_count_dict_max.keys() if col in prep_enriched.columns]
            valid_pos_cols   = [col for col in activity_pos_dict_max.keys() if col in prep_enriched.columns]

            # Compute max and min only for valid (non-negative) values from existing columns
            activity_count_max = max(activity_count_dict_max[col] for col in valid_count_cols)
            activity_pos_max   = max(activity_pos_dict_max[col] for col in valid_pos_cols)

        for activity in unique_activities:
                count_col = activity + "_count"
                pos_col = activity + "_pos"

                if count_col in prep_enriched.columns:
                    prep_enriched[activity + "_count"] = (prep_enriched[activity + "_count"]) / (activity_count_max)
                if pos_col in prep_enriched.columns:
                    prep_enriched[activity + "_pos"] = (prep_enriched[activity + "_pos"]) / (activity_pos_max)

        scale_columns_list = prep_enriched.columns.tolist()
        # remove case_nr, activity, and 'a', next_activity
        scale_columns_list.remove("case_nr")
        scale_columns_list.remove("activity")
        scale_columns_list.remove("a")
        if self.args.kmeans_config[2] != "prep_outcome":
            scale_columns_list.remove("outcome") # we normalize when making MDP
        # also remove any column that has count or pos in its name
        scale_columns_list = [col for col in scale_columns_list if "count" not in col and "pos" not in col and col not in self.DATASET_PARAMS["cat_cols"]]
        scaler_dict, prep_scaled = scale_columns(prep_enriched, scale_columns_list, self.DATASET_PARAMS["case_cols"], scaler_dict=prep_utils["scaler_dict_train"] if prep_utils is not None else None)

        # one_hot_encode categorical cols
        cat_columns_list = deepcopy(self.DATASET_PARAMS["cat_cols"])
        cat_columns_list.remove("activity")
        if len(cat_columns_list) > 0:
            oh_encoder_dict, prep_scaled, case_cols_encoded, event_cols_encoded = one_hot_encode_columns(cat_cols=cat_columns_list,case_cols=self.DATASET_PARAMS["case_cols"], event_cols=self.DATASET_PARAMS["event_cols"], data=prep_scaled)
        
        # Remove all columns that have only one value for all rows, but only if prep_utils is None
        if you_have_to_filter_cols_manually:
            # Remove columns that have only one value for all rows
            prep = prep_scaled.loc[:, (prep_scaled != prep_scaled.iloc[0]).any()]
        else:
            # otherwise, use the columns from the prep_utils
            for col in prep_utils["columns_to_keep"]:
                if col not in prep_scaled.columns:
                    prep_scaled[col] = 0
            prep = prep_scaled[prep_utils["columns_to_keep"]]

        if prep_utils is None:
            prep_utils = {"scaler_dict_train": scaler_dict, 
                            "scale_cols": self.DATASET_PARAMS["scale_cols"],
                            "activity_count_max": activity_count_max,
                            "activity_pos_max": activity_pos_max,
                            "unique_activities": unique_activities,
                            "columns_to_keep": prep.columns.tolist(),}
            
        return prep, prep_utils
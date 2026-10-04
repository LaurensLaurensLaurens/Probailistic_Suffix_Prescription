import pandas as pd
from src.utils.model_tools.model_functions import KMeansClustering

class KMeansQFunctions():
    def __init__(self, model_params_list_of_dicts):
        """
        Initialize the KMeans_Q calculations.
        """
        # Initialization only
        self.n_stages = len(model_params_list_of_dicts)
        self.models_list_of_dicts = [{} for _ in range(self.n_stages)]  # List of dictionaries for each stage
        self.model_params_list_of_dicts = model_params_list_of_dicts

    def prepare(self, data_train_list, data_infer_list, stage, model_params, data_lists_for_other_models=None):
        self.stage = stage
        self.data_train_list = data_train_list
        self.data_infer_list = data_infer_list
        self.model_params = model_params

        data_train = self.data_train_list[self.stage]
        data_infer = self.data_infer_list[self.stage] if self.data_infer_list else None

        return data_train, data_infer, None, None, None, None  # No weights and no non-calibration data for KMeansQ
import random
import torch
import numpy as np
from src.utils.data_load_tools import DL_Dataset, ML_Dataset, RL_Dataset
from src.utils.mini_tools import get_model_functions
import pandas as pd
RESULTS_FOLDER = "res"

class ModelEval():
    def __init__(self, data, model_params, model_to_load, stage=0):
        self.data = data
        self.model_params = model_params
        self.stage = stage

        # Initialize the model functions based on the target and model specific parameters
        self.model_functions = get_model_functions(model_params=self.model_params, model_to_load=model_to_load)

        random.seed(self.model_params["seed"])
        torch.manual_seed(self.model_params["seed"])
        torch.cuda.manual_seed_all(self.model_params["seed"])
        np.random.seed(self.model_params["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    def eval(self):
        if self.model_params["model_category"] == "dl":
            dataset = DL_Dataset(x_case=self.data["X_case"],
                                 x_event=self.data["X_event"],
                                 prefix_len=self.data["prefix_len"],
                                 t=self.data["T"],
                                 y=self.data["Y"],
                                 weights=None)
        elif self.model_params["model_category"] == "ml":
            dataset = ML_Dataset(x_case=self.data["X_case"],
                                 x_event=self.data["X_event"],
                                 prefix_len=self.data["prefix_len"],
                                 t=self.data["T"],
                                 y=self.data["Y"],
                                 weights=None)
        elif self.model_params["model_category"] == "rl":
            if self.model_params["dataset"] == "SimBank":
                # filter: if the stage is 0, then use only data with 'activity' == 'initiate_application', else use only data with 'activity' != 'initiate_application'
                if self.stage == 0:
                    self.data = self.data[self.data["activity"] == "initiate_application"]
                else:
                    self.data = self.data[self.data["activity"] != "initiate_application"]
            elif self.model_params["dataset"] == "bpic17":
                if self.stage == 0:
                    self.data = self.data[self.data["activity"] == "validate_application"]
                else:
                    self.data = self.data[self.data["activity"] != "validate_application"]

            dataset = RL_Dataset(x_case=self.data,
                                 x_event=None,  # RL models do not use x_event
                                 prefix_len=None,  # RL models do not use prefix_len
                                 t=None, # RL models do not use t
                                 y=None,  # RL models do not use y
                                 weights=None) # weights are not used in RL models
        else:
            raise ValueError("Unknown model category: {}".format(self.model_params["model_category"]))
        
        with torch.no_grad():
            # action_values enthält danach für jeden Fall den geschätzten Wert jeder Aktion
            action_values = self.model_functions.forward(x_case=dataset.x_case,
                                        x_event=dataset.x_event,
                                        t=dataset.t,
                                        prefix_len=dataset.prefix_len,
                                        y=dataset.y,
                                        ret_counterfactuals=True)
        
        # Da es mehr Zeilen als Spalten gibt, wird transponiert:
        action_values = self.normalize_action_values(action_values)
        case_nrs = self.normalize_case_nrs(self.data["case_nr"])
        # Now get the 'highest' action for each case (so just argmax over the action values)
        actions = np.argmax(action_values, axis=0)
        # make a df with the case_nrs and actions
        action_df = {
            "case_nr": case_nrs,
            "action": actions
        }

        return action_df
    
    def normalize_action_values(self, action_values):
        if isinstance(action_values, (list, np.ndarray)):
            if isinstance(action_values[0], torch.Tensor):
                action_values = [a.detach().cpu().numpy() for a in action_values]
        # Convert to final 2D NumPy array
        action_values = np.array(action_values, dtype=np.float32)
        if action_values.ndim == 2 and action_values.shape[0] > action_values.shape[1]:
            action_values = action_values.T

        return action_values
    
    def normalize_case_nrs(self, case_nrs):
        case_nrs = case_nrs.detach().cpu().numpy() if not isinstance(case_nrs, (np.ndarray, list, pd.Series)) else case_nrs
        case_nrs = case_nrs.values if isinstance(case_nrs, pd.Series) else case_nrs

        return case_nrs
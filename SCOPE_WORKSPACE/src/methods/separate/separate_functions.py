from copy import deepcopy
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from src.utils.mini_tools import get_model_functions

class SeparateFunctions():
    def __init__(self, model_params_list_of_dicts):
        """
        Initialize the SEP calculations.
        """
        # Initialization only
        self.n_stages = len(model_params_list_of_dicts)
        self.models_list_of_dicts = [{} for _ in range(self.n_stages)]  # List of dictionaries for each stage
        self.model_params_list_of_dicts = model_params_list_of_dicts

    def prepare(self, data_train_list, data_infer_list, stage, model_params, data_lists_for_other_models=None):
        self.stage = stage
        self.data_train_list = data_train_list
        self.data_infer_list = data_infer_list

        # data for other models
        self.data_lists_for_other_models = data_lists_for_other_models

        self.model_params = model_params
        self.learner_method = self.model_params_list_of_dicts[self.stage]["outcome"]["learner_method"]
        self.action_recomm_method = self.model_params_list_of_dicts[self.stage]["outcome"]["action_recomm_method"]
        self.value_function_method = self.model_params_list_of_dicts[self.stage]["outcome"]["value_function_method"]
       
        weights_train = None
        target_train = self.data_train_list[self.stage]["Y"]
        
        if model_params["target"] == "effect":
            target_train, weights_train = self.calc_target_effect()
        
        # Replace the Y with targets
        data_train_adj = deepcopy(self.data_train_list[self.stage])
        data_train_adj["Y"] = target_train

        if data_infer_list is not None:
            # If there is inference data, we need to adjust it as well
            weights_infer = None
            data_infer_adj = deepcopy(self.data_infer_list[self.stage])
            if model_params["target"] == "effect":
                target_infer, weights_infer = self.calc_target_effect(infer=True)
                data_infer_adj["Y"] = target_infer
            else:
                data_infer_adj["Y"] = data_infer_adj["Y"]
        else:
            data_infer_adj = None
            weights_infer = None

        self.data_train_ps = self.data_lists_for_other_models["ps"]["train"][self.stage] if self.data_lists_for_other_models["ps"]["train"] is not None else None
        self.data_infer_ps = self.data_lists_for_other_models["ps"]["infer"][self.stage] if self.data_lists_for_other_models["ps"]["infer"] is not None else None

        return data_train_adj, data_infer_adj, weights_train, weights_infer, self.data_train_ps, self.data_infer_ps

    # SEP Calculations
    def calc_target_effect(self, infer=False):
        data = self.data_infer_list[self.stage] if infer else self.data_train_list[self.stage]

        outcome_model_functions = get_model_functions(model_params=self.model_params_list_of_dicts[self.stage]["outcome"], model_to_load=self.models_list_of_dicts[self.stage]["outcome"])
        
        target_outcomes = data["Y"]

        q_values_all_actions = self.get_q_values(outcome_model_functions=outcome_model_functions, data=data, target_outcomes=target_outcomes)

        ps_train = None
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
        if self.model_params["model_category"] == "dl":
            # just return all, since we use a custom loss function, where the loss depends on the 'predicted' optimal action

            # reshape so it is (n_samples, n_classes)
            weights = contrast_all_actions.permute(1, 0)  # shape: (n_samples, n_classes)

            # add 1 to all weights
            weights = weights + 1
        
        elif self.model_params["model_category"] == "ml" and "reg" in self.model_params["method"]:
            # just return weights 1
            weights = torch.ones_like(contrast_all_actions[0])  # shape: (n_samples,)

        elif self.model_params["model_category"] == "ml":
            # Use the weights like CC-learning: just repeat the instances with artificial labels (every treatment level) and match with appropriate weights
            weights = contrast_all_actions.permute(1, 0)

        return weights

    def calc_opt_actions_and_constrast(self, data, q_values_all_actions, propensity_scores, target_outcomes):
        """
        Estimate the optimal actions for the SEP method. 
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
        
    # Helper methods for SEP calculations
    def get_q_values(self, outcome_model_functions, data, target_outcomes=None):
        """
        Get the Q predictions for all possible actions in the SEP method.
        """

        q_values_all_actions = outcome_model_functions.forward(x_case=data["X_case"],
                                    x_event=data["X_event"],
                                    t=data["T"],
                                    prefix_len=data["prefix_len"],
                                    y=data["Y"],
                                    ret_counterfactuals=True)

        # quickly calculate mae, by taking the right q_value for the observed T, and by comparing it to the target outcomes
        obs_actions = torch.argmax(data["T"], dim=1) if data["T"].shape[1] > 2 else data["T"].squeeze(1).long()
        mae = torch.mean(torch.abs(self.get_correct_values(q_values_all_actions, obs_actions) - data["Y"]))
        print(f"MAE: {mae.item()}")

        return q_values_all_actions
        
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
                # NEW: baseline action is first one, so pseudo-outcome is standard 0
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
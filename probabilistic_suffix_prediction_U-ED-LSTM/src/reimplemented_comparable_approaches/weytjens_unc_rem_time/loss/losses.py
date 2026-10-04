"""
Use of same experimental setup as for our Probabilistic Suffix Prediction.

Reimplementation for comparison: (Use of exact the same code as provided for the loss function)
- Paper: Weytjens, Hans, and Jochen De Weerdt. "Learning uncertainty with artificial neural networks for predictive process monitoring." Applied Soft Computing 125 (2022): 109134.
- Github (code) from: https://github.com/hansweytjens/uncertainty-remaining_time/blob/main/LSTM.ipynb

"""

import torch
from typing import Optional

class Loss:
    def __init__(self):
        pass

    def regression_heteroscedastic_loss(self, true, mean, log_var, metric: Optional[str]="rmse"): 
        '''
        ARGUMENTS:
        true: true values. Tensor (batch_size x number of outputs)
        mean: predictions. Tensor (batch_size x number of outputs)
        log_var: Logaritms of uncertainty estimates. Tensor (batch_size x number of outputs)
        metric: "mae" or "rmse"

        OUTPUTS:
        loss. Tensor (0)
        '''
        
        precision = torch.exp(-log_var)
        if metric == "mae":
            return torch.mean(torch.sum((2 * precision) ** .5 * torch.abs(true - mean) + log_var / 2, 1), 0)
        elif metric == "rmse" or not metric:   # default is rmse
            return torch.mean(torch.sum(precision * (true - mean) ** 2 + log_var, 1), 0)
        else:
            print("Metric has to be 'rmse' or 'mae'")
            
    def regression_homoscedastic_loss(self, true, mean, metric: Optional[str]="rmse"):
        '''
        ARGUMENTS:
        true: true values. Tensor (batch_size x number of outputs)
        mean: predictions. Tensor (batch_size x number of outputs)
        metric: "mae" or "rmse"

        OUTPUTS:
        loss. Tensor (0)
        '''
        
        if metric == "mae":
            return torch.mean(torch.sum(torch.abs(true - mean), 1), 0)
        elif metric == "rmse" or not metric:   # default is rmse
            return torch.mean(torch.sum((true - mean) ** 2, 1), 0)
        else:
            print("Metric has to be 'rmse' or 'mae'")


 

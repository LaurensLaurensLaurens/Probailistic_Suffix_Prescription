"""
Use of same experimental setup as for our Probabilistic Suffix Prediction.

Reimplementation for comparison: 
- Paper: Weytjens, Hans, and Jochen De Weerdt. "Learning uncertainty with artificial neural networks for predictive process monitoring." Applied Soft Computing 125 (2022): 109134.
- Github (code) from: https://github.com/hansweytjens/uncertainty-remaining_time/blob/main/LSTM.ipynb

Delete terms:
- ConcreteDropout: We assume fixed non-learnable dropout rate.
- Bayes: We assume to be always bayesian.
- stop_dropout: We assume to no stop dropout during inference.

"""

import torch
from typing import Optional
from torch import nn

from stochasticLSTM.cell import StochasticLSTMCell

class StochasticLSTMWeytjens(nn.Module):
    def __init__(self, 
                 data_set_categories: list[tuple[str, dict[str, int]]],
                 model_feat: list,
                 hidden_size: int,
                 num_layers: int,
                 input_size: int,
                 weight_reg: float,
                 p_fix: Optional[float],
                 device: str = 'cpu'):
        """
        ARGS:
        data_set_categories: list of categorical and numerical feature metadata
        model_feat: list of features used in the model
        hidden_size: number of nodes in LSTM layers
        num_layers: number of LSTM layers
        input_size: number of numerical variables
        weight_reg: weight regularization parameter
        p_fix: dropout parameter
        device: device to run the model on ('cpu' or 'cuda')
        """
        super(StochasticLSTMWeytjens, self).__init__()
        
        self.device = torch.device(device)
        
        # Feature sizes
        self.data_set_categories = data_set_categories
        print("Data set categories: ", data_set_categories)
        self.model_feat = model_feat
        print("Model input features: ", model_feat)
        
        print("\n")
        
        # List containing two dicts: One for categorical, one for numerical
        cat_categories, num_categories = data_set_categories
        cat_feat_model, num_feat_model = model_feat
        
        # Use the first value in the tuple as the key and the second value as the value
        cat_dict = {cat[0]: cat[1] for cat in cat_categories}
        # Get input feature sizes to determine the model input size
        list_of_classes_per_cat = [cat_dict[cat_feat] for cat_feat in cat_feat_model if cat_feat in cat_dict]
        
        # Create embeddings for categorical features
        self.embeddings = nn.ModuleList([nn.Embedding(n_cat, min(600, round(1.6 * n_cat**0.56))) for n_cat in list_of_classes_per_cat])
        print("Embeddings: ", self.embeddings)
        # Compute total input size encoder
        embedding_size = sum([min(600, round(1.6 * n_cat**0.56)) for n_cat in list_of_classes_per_cat])
        print("Total embedding feature size: ", embedding_size)

        # Only add embedding to input size in case of training. When model is safed the input size expected is already correct:
        if input_size == 1:
            self.input_size = input_size + embedding_size
        else:
            self.input_size = input_size
            
        print("Input feature size: ", self.input_size) 
        self.hidden_size = hidden_size
        print("Cells hidden size: ", hidden_size)
        self.num_layers = num_layers
        print("Number of LSTM layer: ", num_layers)
        self.p_fix = p_fix
        print("Dropout rate: ", p_fix)
        
        print("\n")
        
        self.weight_reg = weight_reg

        # Create dictionaries to store feature names and their index positions
        num_feat_set = set(num_feat_model)
        self.output_feature = {num[0]: i for i, num in enumerate(num_categories) if num[0] in num_feat_set}         
        print("Output feature list of dicts (featue name, tensor index in dataset): ", self.output_feature)
        
        self.first_layer = StochasticLSTMCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            weight_reg=self.weight_reg,
            p_fix=self.p_fix,
            device=device
        )
        
        self.hidden_layers = nn.ModuleList([
            StochasticLSTMCell(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                weight_reg=self.weight_reg,
                p_fix=self.p_fix,
                device=device
            ) for _ in range(num_layers - 1)
        ])
        
        self.linear1 = nn.Linear(self.hidden_size, 5).to(self.device)
        self.relu = nn.ReLU()
        self.linear2_mu = nn.Linear(5, 1).to(self.device)
        self.linear2_logvar = nn.Linear(5, 1).to(self.device)

    def regularizer(self):
        total_weight_reg, total_bias_reg = self.first_layer.regularizer()
        
        for l in self.hidden_layers:
            weight, bias = l.regularizer()
            total_weight_reg += weight
            total_bias_reg += bias
            
        return total_weight_reg, total_bias_reg
    
    def __input_construction(self, data):
        """
        Construct input tensor with dim: seq len x batch size x input features.
        """
        cats, nums = data
        embedded_cats = [embedd(cats[i]) for i, embedd in enumerate(self.embeddings)]
        merged_cats = torch.cat(embedded_cats, dim=-1)
        
        if len(nums):
            merged_nums = torch.cat([num.unsqueeze(2) for num in nums], dim=-1)
        else:
            merged_nums = torch.tensor([], device=self.device)
        
        x = torch.cat((merged_cats, merged_nums), dim=-1).permute(1, 0, 2)
        return x
        
    def forward(self, input):
        """
        INPUTS:
        list of lists with tensors for (categorical, numerical event attributes)
        
        OUTPUTS:
        mean: point estimates, tensor (batch size x 1)
        log_var: log of uncertainty estimates, tensor (batch size x 1)
        """
        x = self.__input_construction(data=input)
        outputs, (h, c) = self.first_layer(x)
        
        for lstm_cell in self.hidden_layers:
            outputs, (h, c) = lstm_cell(input=outputs, hx=(h, c))
        
        final_output = outputs[-1]
        h_feat = self.relu(self.linear1(final_output))
        means = self.linear2_mu(h_feat)
        logvars = self.linear2_logvar(h_feat)
        
        return means, logvars
    
    def save(self, path: str):
        """
        Store the trained model at path.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'kwargs': {
                'data_set_categories': self.data_set_categories,
                'model_feat': self.model_feat,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'input_size': self.input_size,
                'weight_reg': self.weight_reg,
                'p_fix': self.p_fix,
                'device': str(self.device)
            }
        }
        return torch.save(checkpoint, path)

    @staticmethod
    def load(path: str, p_fix: Optional[float] = None, device: Optional[str] = None):
        """
        Load the stored model at path.
        """
        checkpoint = torch.load(path, weights_only=False, map_location=torch.device(device if device else 'cpu'))
        
        if p_fix is not None:
            checkpoint['kwargs']['p_fix'] = p_fix
        if device is not None:
            checkpoint['kwargs']['device'] = device
        
        model = StochasticLSTMWeytjens(**checkpoint['kwargs'])
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(checkpoint['kwargs']['device'])
        
        return model
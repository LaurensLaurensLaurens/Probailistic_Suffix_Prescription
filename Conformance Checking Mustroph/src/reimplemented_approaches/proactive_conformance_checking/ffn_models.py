"""
Reimplementaiton of FFN seperate, collective for deviation prediction:
Grohs, M., Pfeiffer, P., Rehse, J.: Proactive conformance checking: An approach for predicting deviations in business processes. Inf. Syst. 127, 102461 (2025)
"""
import os
# performance imports for torch: torch kernel uses one core only.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1" 

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

class FFNCollectiveIDP(nn.Module):
    def __init__(self,
                 input_size: int,
                 fc_hidden_1: int,
                 fc_hidden_2: int,
                 num_output_labels: int = None,
                 dropout: float = 0.1,
                 device: torch.device = torch.device("cuda")):
        
        super().__init__()
        if num_output_labels is None:
            raise ValueError("num_output_labels must be provided")
        self.device = torch.device(device)

        self.fc_hidden_1 = nn.Linear(input_size, fc_hidden_1)
        self.layer_norm_1 = nn.LayerNorm(fc_hidden_1)
        self.leaky_relu_1 = nn.LeakyReLU()

        self.fc_hidden_2 = nn.Linear(fc_hidden_1, fc_hidden_2)
        self.layer_norm_2 = nn.LayerNorm(fc_hidden_2)
        self.leaky_relu_2 = nn.LeakyReLU()

        self.dropout = nn.Dropout(dropout)

        self.fc_output = nn.Linear(fc_hidden_2, num_output_labels)
        
        # kwargs important to save the model
        self.init_kwargs = dict(input_size=input_size,
                                fc_hidden_1=fc_hidden_1,
                                fc_hidden_2=fc_hidden_2,
                                num_output_labels=num_output_labels,
                                dropout=dropout,
                                device=self.device.type)
        
        self.to(self.device)

    def forward(self,
                x: torch.Tensor,
                apply_sigmoid: bool = False) -> torch.Tensor:

        h = self.fc_hidden_1(x)
        h = self.layer_norm_1(h)
        h = self.leaky_relu_1(h)

        h2 = self.fc_hidden_2(h)
        h2 = self.layer_norm_2(h2)
        h2 = self.leaky_relu_2(h2)

        h2 = self.dropout(h2)

        logits = self.fc_output(h2)

        if apply_sigmoid:
            return torch.sigmoid(logits)
        return logits
    
    def save(self, path: str):
        """
        Store the trained model at path.
        """
        checkpoint = {"model_state_dict": self.state_dict(),
                      "kwargs": self.init_kwargs}
        torch.save(checkpoint, Path(path))

    @staticmethod
    def load(path: str,
             device: Optional[torch.device] = None) -> "FFNCollectiveIDP":
        """
        Load the stored model at path.
        """
        checkpoint = torch.load(Path(path), weights_only=False, map_location=device or torch.device("cpu"))
        kwargs = checkpoint["kwargs"]
        if device is not None:
            kwargs["device"] = device

        model = FFNCollectiveIDP(**kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(model.device)
        model.eval()
        return model
    
class _SingleLabelIDP(nn.Module):
    def __init__(self,
                 input_size: int,
                 fc_hidden_1: int,
                 fc_hidden_2: int,
                 dropout: float):
        
        super().__init__()
        
        self.fc_hidden_1 = nn.Linear(input_size, fc_hidden_1)
        self.layer_norm_1 = nn.LayerNorm(fc_hidden_1)
        self.leaky_relu_1 = nn.LeakyReLU()

        self.fc_hidden_2 = nn.Linear(fc_hidden_1, fc_hidden_2)
        self.layer_norm_2 = nn.LayerNorm(fc_hidden_2)
        self.leaky_relu_2 = nn.LeakyReLU()

        self.dropout = nn.Dropout(dropout)
        self.fc_output = nn.Linear(fc_hidden_2, 2)

    def forward(self,
                x_concat: torch.Tensor) -> torch.Tensor:

        x = self.fc_hidden_1(x_concat)
        x = self.layer_norm_1(x)
        x = self.leaky_relu_1(x)

        x = self.fc_hidden_2(x)
        x = self.layer_norm_2(x)
        x = self.leaky_relu_2(x)

        x = self.dropout(x)
        
        return self.fc_output(x)

class FFNSeparateIDP(nn.Module):
    def __init__(self,
                 input_size: int,
                 fc_hidden_1: int,
                 fc_hidden_2: int,
                 dropout: float = 0.1,
                 device: torch.device = torch.device("cuda")):
        
        super().__init__()

        self.device = torch.device(device)

        # Two scalar-logit heads stacked into a 2-class logit vector.
        self.head = _SingleLabelIDP(input_size=input_size,
                                    fc_hidden_1=fc_hidden_1,
                                    fc_hidden_2=fc_hidden_2,
                                    dropout=dropout)

        self.init_kwargs = dict(input_size=input_size,
                                fc_hidden_1=fc_hidden_1,
                                fc_hidden_2=fc_hidden_2,
                                dropout=dropout,
                                device=self.device.type)

        self.to(self.device)

    def forward(self,
                x: torch.Tensor,
                apply_softmax: bool = False) -> torch.Tensor:

        logits = self.head(x)

        if apply_softmax:
            return torch.softmax(logits, dim=1)
        return logits

    def save(self, path: str):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "kwargs": self.init_kwargs}
        
        torch.save(checkpoint, Path(path))

    @staticmethod
    def load(path: str,
             device: Optional[torch.device] = None) -> "FFNSeparateIDP":
        checkpoint = torch.load(Path(path), weights_only=False, map_location=device or torch.device("cpu"))
        kwargs = checkpoint["kwargs"]
        if device is not None:
            kwargs["device"] = device
        model = FFNSeparateIDP(**kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(model.device)
        model.eval()
        return model

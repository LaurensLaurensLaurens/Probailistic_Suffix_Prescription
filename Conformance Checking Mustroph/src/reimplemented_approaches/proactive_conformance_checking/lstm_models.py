"""
Reimplementaiton of LSTM seperate, collective for deviation prediction:
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

class LSTMCollectiveIDP(nn.Module):
    def __init__(self,
                 activity_vocab_size: int,
                 resource_vocab_size: int,
                 month_vocab_size: int,
                 num_trace_features: int,
                 embedding_dim: int = 16,
                 lstm_hidden: int = 128,
                 fc_hidden: int = 128,
                 num_output_labels: int = None,
                 dropout: float = 0.1,
                 device: torch.device = torch.device("cuda")):
        
        super().__init__()
        if num_output_labels is None:
            raise ValueError("num_output_labels must be provided")
        self.device = torch.device(device)

        # Embeddings
        self.embed_act = nn.Embedding(activity_vocab_size, embedding_dim)
        self.embed_res = nn.Embedding(resource_vocab_size, embedding_dim)
        self.embed_month = nn.Embedding(month_vocab_size, embedding_dim)

        # Separate LSTMs
        self.lstm_activity = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden, batch_first=True)
        self.lstm_resource = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden, batch_first=True)
        self.lstm_month = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden, batch_first=True)

        # FC for LSTM outputs (map hidden -> fc_hidden)
        self.fc_lstm = nn.Linear(lstm_hidden, fc_hidden)
        
        # FC for trace features
        self.fc_trace = nn.Linear(num_trace_features, fc_hidden)
        
        # LayerNorm over concatenated hidden vectors
        self.layer_norm = nn.LayerNorm(fc_hidden * 4)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout_rate = dropout
        
        # kwargs important to save the model
        self.init_kwargs = dict(activity_vocab_size=activity_vocab_size,
                                resource_vocab_size=resource_vocab_size,
                                month_vocab_size=month_vocab_size,
                                num_trace_features=num_trace_features,
                                embedding_dim=embedding_dim,
                                lstm_hidden=lstm_hidden,
                                fc_hidden=fc_hidden,
                                num_output_labels=num_output_labels,
                                device=self.device.type)

        self.dropout = nn.Dropout(dropout)
        
        # Final output layer
        self.fc_output = nn.Linear(fc_hidden * 4, num_output_labels)
        
        self.sigmoid = nn.Sigmoid()

        self.to(self.device)

    def forward(self,
                x_act: torch.Tensor,
                x_res: torch.Tensor,
                x_month: torch.Tensor,
                x_trace: torch.Tensor,
                apply_sigmoid: bool = False) -> torch.Tensor:
        
        x_act = x_act.to(self.device)
        x_res = x_res.to(self.device)
        x_month = x_month.to(self.device)
        x_trace = x_trace.to(self.device)

        emb_act = self.embed_act(x_act)
        emb_res = self.embed_res(x_res)
        emb_month = self.embed_month(x_month)

        _, (h_act, _) = self.lstm_activity(emb_act)
        _, (h_res, _) = self.lstm_resource(emb_res)
        _, (h_month, _) = self.lstm_month(emb_month)

        h_fc_act = self.fc_lstm(h_act[-1])
        h_fc_res = self.fc_lstm(h_res[-1])
        h_fc_month = self.fc_lstm(h_month[-1])
        h_fc_trace = self.fc_trace(x_trace.float())

        h_comb = torch.cat([h_fc_act, h_fc_res, h_fc_month, h_fc_trace], dim=-1)
        x = self.layer_norm(h_comb)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        logits = self.fc_output(x)

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
             device: Optional[torch.device] = None) -> "LSTMCollectiveIDP":
        """
        Load the stored model at path.
        """
        checkpoint = torch.load(Path(path), weights_only=False, map_location=device or torch.device("cpu"))
        kwargs = checkpoint["kwargs"]

        model = LSTMCollectiveIDP(**kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(model.device)
        model.eval()
        return model
    
class _SingleLabelIDP(nn.Module):
    def __init__(self,
                 activity_vocab_size: int,
                 resource_vocab_size: int,
                 month_vocab_size: int,
                 num_trace_features: int,
                 embedding_dim: int,
                 lstm_hidden: int,
                 fc_hidden: int,
                 dropout: float):
        
        super().__init__()
        
        self.embed_act = nn.Embedding(activity_vocab_size, embedding_dim)
        self.embed_res = nn.Embedding(resource_vocab_size, embedding_dim)
        self.embed_month = nn.Embedding(month_vocab_size, embedding_dim)

        self.lstm_activity = nn.LSTM(embedding_dim, lstm_hidden, batch_first=True)
        self.lstm_resource = nn.LSTM(embedding_dim, lstm_hidden, batch_first=True)
        self.lstm_month = nn.LSTM(embedding_dim, lstm_hidden, batch_first=True)

        self.fc_lstm = nn.Linear(lstm_hidden, fc_hidden)
        self.fc_trace = nn.Linear(num_trace_features, fc_hidden)

        self.layer_norm = nn.LayerNorm(fc_hidden * 4)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_output = nn.Linear(fc_hidden * 4, 2)

    def forward(self,
                x_act: torch.Tensor,
                x_res: torch.Tensor,
                x_month: torch.Tensor,
                x_trace: torch.Tensor) -> torch.Tensor:
        emb_act = self.embed_act(x_act)
        emb_res = self.embed_res(x_res)
        emb_month = self.embed_month(x_month)

        _, (h_act, _) = self.lstm_activity(emb_act)
        _, (h_res, _) = self.lstm_resource(emb_res)
        _, (h_month, _) = self.lstm_month(emb_month)

        h_fc_act = self.fc_lstm(h_act[-1])
        h_fc_res = self.fc_lstm(h_res[-1])
        h_fc_month = self.fc_lstm(h_month[-1])
        h_fc_trace = self.fc_trace(x_trace.float())

        h_comb = torch.cat([h_fc_act, h_fc_res, h_fc_month, h_fc_trace], dim=-1)
        x = self.layer_norm(h_comb)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # B, 2 logits
        return self.fc_output(x)

class LSTMSeparateIDP(nn.Module):
    def __init__(self,
                 activity_vocab_size: int,
                 resource_vocab_size: int,
                 month_vocab_size: int,
                 num_trace_features: int,
                 embedding_dim: int = 16,
                 lstm_hidden: int = 64,
                 fc_hidden: int = 128,
                 dropout: float = 0.1,
                 device: torch.device = torch.device("cuda")):
        super().__init__()

        self.device = torch.device(device)

        # self.label_heads = nn.ModuleList([_SingleLabelIDP(activity_vocab_size=activity_vocab_size,
        self.head = _SingleLabelIDP(activity_vocab_size=activity_vocab_size,
                                    resource_vocab_size=resource_vocab_size,
                                    month_vocab_size=month_vocab_size,
                                    num_trace_features=num_trace_features,
                                    embedding_dim=embedding_dim,
                                    lstm_hidden=lstm_hidden,
                                    fc_hidden=fc_hidden,
                                    dropout=dropout)

        self.init_kwargs = dict(activity_vocab_size=activity_vocab_size,
                                resource_vocab_size=resource_vocab_size,
                                month_vocab_size=month_vocab_size,
                                num_trace_features=num_trace_features,
                                embedding_dim=embedding_dim,
                                lstm_hidden=lstm_hidden,
                                fc_hidden=fc_hidden,
                                dropout=dropout,
                                device=self.device.type)

        self.to(self.device)

    def forward(self,
                x_act: torch.Tensor,
                x_res: torch.Tensor,
                x_month: torch.Tensor,
                x_trace: torch.Tensor,
                apply_softmax: bool = False) -> torch.Tensor:
        
        x_act = x_act.to(self.device)
        x_res = x_res.to(self.device)
        x_month = x_month.to(self.device)
        x_trace = x_trace.to(self.device)

        # logits = torch.stack([head(x_act, x_res, x_month, x_trace) for head in self.label_heads], dim=-1)
        logits = self.head(x_act, x_res, x_month, x_trace)
       
        if apply_softmax:
            logits = torch.softmax(logits, dim=-1) # class dim
            
        return logits

    def save(self, path: str):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "kwargs": self.init_kwargs}
        
        torch.save(checkpoint, Path(path))

    @staticmethod
    def load(path: str,
             device: Optional[torch.device] = None) -> "LSTMSeparateIDP":
        checkpoint = torch.load(Path(path), weights_only=False, map_location=device or torch.device("cpu"))
        kwargs = checkpoint["kwargs"]
        if device is not None:
            kwargs["device"] = device
        model = LSTMSeparateIDP(**kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(model.device)
        model.eval()
        return model

"""
Encoder consisting of two an two-layerd LSTM.

The ED-LSTM has LSTM cells using dropout as a Bayesian
approximation.
"""

# performance imports for torch: torch kernel uses one core only.
import os

from .dropout_uncertainty_LSTM_cell import DropoutUncertaintyLSTMCell

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn


class DropoutUncertaintyLSTMEncoder(nn.Module):
    """
    Encoder part of the Encoder-Decoder LSTM with MC-Dropout uncertainty estimation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        # dynamic attributes
        embeddings,
        data_indices_enc: list,
        input_size: int,
        # static attributes
        static_embeddings: Optional[nn.ModuleList] = None,
        static_data_indices: Optional[List[List[int]]] = None,
        static_input_size: Optional[int] = 0,
        # mc-dropout
        dropout: Optional[float] = None,
    ):
        """
        Encoder part of the Encoder-Decoder LSTM.

        Args:
            hidden_size (int): Size of the LSTM hidden state.
            num_layers (int): Number of stacked LSTM layers.
            embeddings (nn.ModuleList): Embedding modules for dynamic categorical
                encoder inputs.
            data_indices_enc (list): Indices selecting dynamic categorical and numerical
                tensors for the encoder.
            input_size (int): Number of dynamic input features per timestep.
            static_embeddings (Optional[nn.ModuleList]): Embedding modules for static
                categorical inputs.
            static_data_indices (Optional[List[List[int]]]): Indices selecting static
                categorical and numerical tensors.
            static_input_size (Optional[int]): Flattened size of all static features
                after embeddings.
            dropout (Optional[float]): Dropout probability used for MC dropout in the
                LSTM cells.
        """
        super(DropoutUncertaintyLSTMEncoder, self).__init__()

        # Embeddings for dynamic
        self.embeddings = embeddings
        # for static
        if static_embeddings is not None:
            self.static_embeddings = static_embeddings

        # List of two lists (categorical, numerical)
        # each containing the indices of tensors required for encoder
        self.data_indices_enc = data_indices_enc
        # for static
        if static_data_indices is not None:
            self.static_data_indices = static_data_indices
        # Static features are concatenated
        # to the dynamic per-timestep features and fed into the LSTM.
        self.static_input_size = static_input_size or 0

        # Linear projection before inserted into LSTM layers
        total_input_size = input_size + (
            self.static_input_size if self.static_input_size > 0 else 0
        )
        self.input_proj = nn.Linear(total_input_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.act = nn.ReLU()

        # Create a first cell:
        self.first_layer = DropoutUncertaintyLSTMCell(
            input_size=hidden_size, hidden_size=hidden_size, dropout=dropout
        )

        # Create multiple LSTM cells based on num_layer
        self.hidden_layers = nn.ModuleList(
            [
                DropoutUncertaintyLSTMCell(
                    input_size=hidden_size, hidden_size=hidden_size, dropout=dropout
                )
                for _ in range(num_layers - 1)
            ]
        )

    def regularizer(self) -> Tuple[float, float]:
        """
        L2 regularization of Encoder weights, biases and dropout.

        OUTPUTS:
        - total_weight_reg: L2 weight regularization term
        - total_bias_reg: L2 bias regularization term
        """
        total_weight_reg, total_bias_reg = self.first_layer.regularizer()

        for layer in self.hidden_layers:
            weight, bias = layer.regularizer()
            total_weight_reg += weight
            total_bias_reg += bias

        # Projection layer (weaker prior)
        proj_weight_reg = 0.1 * torch.sum(self.input_proj.weight**2)
        proj_bias_reg = 0.1 * torch.sum(self.input_proj.bias**2)
        total_weight_reg += proj_weight_reg
        total_bias_reg += proj_bias_reg

        return total_weight_reg, total_bias_reg

    def forward(
        self,
        input: List,
        static_inputs: Optional[Union[Tensor, List, Tuple, dict]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the encoder.

        Gets the final (hidden) state vector as input for the decoder.

        INPUTS:
        - input: Prefixes, Tensor: seq_len, batch_size, input_size
        - static_inputs: inputs that are static for the whole case
        - mask: zero padd mask

        OUTPUTS:
        - h,c: Last hidden and cell states of the last layer.
        """
        # Transform the input into a single tensor: [T, B, dyn_features]
        prefixes = self.__data_enc_model(
            data=input
        )  # dim: seq_len x batch_size x input_features

        # Optionally concatenate static features across the time axis T.
        if self.static_input_size > 0:
            static_tensor = self.__static_data_enc_model(
                static_inputs, device=prefixes.device, dtype=prefixes.dtype
            )
            if static_tensor is not None:
                # Expand to [T, B, static_features] and concat
                static_seq = static_tensor.unsqueeze(0).expand(
                    prefixes.shape[0], -1, -1
                )
                prefixes = torch.cat((prefixes, static_seq), dim=-1)

        # Project input features to hidden size
        prefixes = self.input_proj(prefixes)
        prefixes = self.layernorm(prefixes)
        prefixes = self.act(prefixes)

        # zero masking
        mask_seq = None
        if mask is not None:
            seq_len = prefixes.shape[0]
            if mask.shape[1] != seq_len:
                # Assuming left-aligned prefix (standard for [:-suffix]),  sk
                mask = mask[:, :seq_len]
            mask_seq = (
                mask.to(device=prefixes.device, dtype=prefixes.dtype)
                .transpose(0, 1)
                .contiguous()
            )

            # Apply mask to prefixes to ensure padded inputs are zero
            prefixes = prefixes * mask_seq.unsqueeze(-1)

        # Outputs: All hidden states of all cells in the layer, h,c: last hidden state
        # and cell state in the layer
        outputs, (h, c), _ = self.first_layer(
            input=prefixes, hx=None, z=None, mask=mask_seq
        )

        # Pass through the remaining LSTM cell: Layer gets for: input: h_n Tensor,
        # hx: (h, c)
        for _, layer in enumerate(self.hidden_layers):
            outputs, (h, c), _ = layer(input=outputs, hx=(h, c), z=None, mask=mask_seq)

        return (h, c)

    def __data_enc_model(self, data):
        """Dynamic attribute model encoder."""
        # cats dims: list (n categorical values):
        # Each with Tensor: batch_size x (window_size - suffix size)
        cats = [data[0][i] for i in self.data_indices_enc[0]]
        # nums dims: list (n numerical values):
        # Each with Tensor: batch_size x (window_size - suffix size)
        nums = [data[1][i] for i in self.data_indices_enc[1]]

        # Embedd categorical tensors
        embedded_cats = []
        for i, embedd in enumerate(self.embeddings):
            embedded_cats.append(embedd(cats[i]))

        # Merged categroical data
        merged_cats = torch.cat([cat for cat in embedded_cats], dim=-1)

        if len(nums):
            # Merged numerical inputs
            merged_nums = torch.cat([num.unsqueeze(2) for num in nums], dim=-1)
        else:
            merged_nums = torch.tensor([], device=merged_cats.device)
        prefixes = torch.cat((merged_cats, merged_nums), dim=-1).permute(
            1, 0, 2
        )  # dim: seq_len x batch_size x input_features
        return prefixes

    def __static_data_enc_model(
        self,
        static_inputs: Optional[Union[Tensor, List, Tuple, dict]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[Tensor]:
        """Static attribute model encoder."""
        if static_inputs is None or self.static_input_size == 0:
            return None

        # Allow passing either a (static_cats, static_nums) tuple or a dict.
        static_cats = None
        static_nums = None
        if isinstance(static_inputs, dict):
            static_cats = static_inputs.get(
                "static_cats", static_inputs.get("cats", None)
            )
            static_nums = static_inputs.get(
                "static_nums", static_inputs.get("nums", None)
            )
        elif isinstance(static_inputs, (list, tuple)):
            if len(static_inputs) != 2:
                raise TypeError(
                    "static_inputs tuple/list must be (static_cats, static_nums)"
                )
            static_cats, static_nums = static_inputs
        else:
            raise TypeError(
                "static_inputs must be a tuple/list(static_cats, static_nums) or a dict"
            )

        merged_static_cats = None
        if static_cats is not None and len(self.static_embeddings) > 0:
            # Support either a single tensor [B, n_static_cats]
            # or a list of tensors [B]
            if isinstance(static_cats, Tensor):
                static_cats = static_cats.long()
                # If 1D input [n_features] (inference single case),
                # add batch dim -> [1, n_features]
                if static_cats.dim() == 1:
                    static_cats = static_cats.unsqueeze(0)
            else:
                # List of tensors case
                static_cats = torch.stack([t.long() for t in static_cats], dim=1)

            embedded = []
            for i, emb in enumerate(self.static_embeddings):
                embedded.append(emb(static_cats[:, i]))
            merged_static_cats = torch.cat(embedded, dim=-1)

        merged_static_nums = None
        if static_nums is not None:
            if isinstance(static_nums, Tensor):
                # bring to size (features x B)
                if static_nums.dim() == 1:
                    static_nums = static_nums.unsqueeze(0)
                merged_static_nums = static_nums
            else:
                merged_static_nums = torch.cat(
                    [num.unsqueeze(1) for num in static_nums], dim=-1
                )

        if merged_static_cats is not None and device is not None:
            merged_static_cats = merged_static_cats.to(device=device, dtype=dtype)
        if merged_static_nums is not None and device is not None:
            merged_static_nums = merged_static_nums.to(device=device, dtype=dtype)

        if merged_static_cats is not None and merged_static_nums is not None:
            return torch.cat((merged_static_cats, merged_static_nums), dim=-1)
        elif merged_static_cats is not None:
            return merged_static_cats
        elif merged_static_nums is not None:
            return merged_static_nums
        else:
            return None
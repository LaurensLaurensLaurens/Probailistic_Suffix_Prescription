"""
LSTM cells using dropout as a Bayesian approximation.
"""

# performance imports for torch: torch kernel uses one core only.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class DropoutUncertaintyLSTMCell(nn.Module):
    """
    LSTM cell with MC Dropout for uncertainty estimation.
    """

    def __init__(
        self, input_size: int, hidden_size: int, dropout: Optional[float] = None
    ):
        """
        Initializes LSTM cell with MC Dropout.

        Args:
            input_size (int): Size of input features.
            hidden_size (int): Size of hidden layer.
            dropout (Optional[float]): Dropout probability, should be between 0 and 1.
        """
        super(DropoutUncertaintyLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize dropout
        if dropout is None:
            # Set p for dropout to random parameter
            self.p_logit = nn.Parameter(torch.empty(1).normal_())
        elif not 0 <= dropout < 1:
            # p dropout must be between 0 and 1
            raise Exception("Dropout rate should be between in [0, 1)")
        else:
            # Set p dropout to the fixed value
            self.p_logit = dropout

        # Input gate
        self.Wi = nn.Linear(self.input_size, self.hidden_size)
        self.Ui = nn.Linear(self.hidden_size, self.hidden_size)
        # Forget gate
        self.Wf = nn.Linear(self.input_size, self.hidden_size)
        self.Uf = nn.Linear(self.hidden_size, self.hidden_size)
        # Cell state gate
        self.Wc = nn.Linear(self.input_size, self.hidden_size)
        self.Uc = nn.Linear(self.hidden_size, self.hidden_size)
        # Output gate
        self.Wo = nn.Linear(self.input_size, self.hidden_size)
        self.Uo = nn.Linear(self.hidden_size, self.hidden_size)

        self.init_weights()

    def init_weights(self):
        """
        Initializes weight layers with initial values.
        """
        k = torch.tensor(self.hidden_size, dtype=torch.float32).reciprocal().sqrt()

        # Input gate weights:
        self.Wi.weight.data.uniform_(-k, k)
        self.Wi.bias.data.uniform_(-k, k)
        self.Ui.weight.data.uniform_(-k, k)
        self.Ui.bias.data.uniform_(-k, k)

        # Forget gate weights
        self.Wf.weight.data.uniform_(-k, k)
        self.Wf.bias.data.uniform_(-k, k)
        self.Uf.weight.data.uniform_(-k, k)
        self.Uf.bias.data.uniform_(-k, k)

        # Cell state gate weights
        self.Wc.weight.data.uniform_(-k, k)
        self.Wc.bias.data.uniform_(-k, k)
        self.Uc.weight.data.uniform_(-k, k)
        self.Uc.bias.data.uniform_(-k, k)

        # Output gate weights
        self.Wo.weight.data.uniform_(-k, k)
        self.Wo.bias.data.uniform_(-k, k)
        self.Uo.weight.data.uniform_(-k, k)
        self.Uo.bias.data.uniform_(-k, k)

    def _mc_dropout_sample_mask(
        self, B: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """
        Applies dropout to the LSTM Cell weight layers.

        INPUTS:
        B: Batch size

        OUTPUTS:
        zx: Dropout mask for weight layer before input
        zh: Dropout mask for weight layer before hidden

        Note: value p_logit at infinity can cause numerical instability.
        Dropout masks for 4 gates, scale input by 1 / (1 - p)
        """
        # Check dropout probability
        if isinstance(self.p_logit, float):
            p = self.p_logit
        else:
            p = torch.sigmoid(self.p_logit)

        # Four Weight matrix pairs: Perform dropout for each weight layer.
        GATES = 4

        eps = torch.tensor(1e-7, device=device, dtype=torch.float32)
        t = 1e-1

        # tensors with random values:
        ux = torch.rand(
            GATES, B, self.input_size, device=device, dtype=torch.float32
        )  # dim gates x batch_size x input_size
        uh = torch.rand(
            GATES, B, self.hidden_size, device=device, dtype=torch.float32
        )  # dim (gates=weight matrices per cell x batch_size x hidden_size)

        # Dropout masks: containing values near 1 for keeping weights,
        # and near 0 for dropping weights for each gate and batch
        if self.input_size == 1:
            zx = 1 - torch.sigmoid(
                (
                    torch.log(eps)
                    - torch.log(1 + eps)
                    + torch.log(ux + eps)
                    - torch.log(1 - ux + eps)
                )
                / t
            )
        else:
            # dim: gates x batch_size x input_features
            zx = (
                1
                - torch.sigmoid(
                    (
                        torch.log(p + eps)
                        - torch.log(1 - p + eps)
                        + torch.log(ux + eps)
                        - torch.log(1 - ux + eps)
                    )
                    / t
                )
            ) / (1 - p)
        # dim: gates x batch_size x input_features
        zh = (
            1
            - torch.sigmoid(
                (
                    torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(uh + eps)
                    - torch.log(1 - uh + eps)
                )
                / t
            )
        ) / (1 - p)

        return zx, zh

    def regularizer(self):
        """
        L2 regularization of weights and biases scaled for dropout.
        """
        # Compute dropout probability
        if isinstance(self.p_logit, float):
            p = self.p_logit
        else:
            p = torch.sigmoid(self.p_logit)

        # Weight L2 sum (keeps autograd).
        # For MC-dropout-as-variational-inference:
        # the KL/L2 term is typically scaled by (1-p) rather than 1/(1-p).
        keep_prob = 1.0 - p
        weight_sum = (
            sum(
                torch.sum(params**2)
                for name, params in self.named_parameters()
                if name.endswith("weight")
            )
            * keep_prob
        )

        # Bias L2 sum
        bias_sum = sum(
            torch.sum(params**2)
            for name, params in self.named_parameters()
            if name.endswith("bias")
        )

        return weight_sum, bias_sum

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
        z: Optional[Tuple[Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Performs forward pass of LSTM cell with MC Dropout.

        INPUTS:
        - input: Input tensor with shape (sequence, batch, input dimension)
        - hx: h_t: hidden state and c_t: cell state as tuple at time step (event t)
        - z: dropout masks for LSTM weights
        - mask:

        OUTPUTS:
        - hn: List of all hidden states: h_1, ... h_n
        - (h_t, c_t): Last hidden and cell state
        - (zx, zh): Applied MC dropout masks
        """
        device = input.device
        T, B, _ = input.shape

        # Initialize hidden and cell states
        if hx is None:
            h_t = torch.zeros(B, self.hidden_size, device=device, dtype=input.dtype)
            c_t = torch.zeros(B, self.hidden_size, device=device, dtype=input.dtype)
        else:
            h_t, c_t = hx
            h_t = h_t.to(device=device, dtype=input.dtype)
            c_t = c_t.to(device=device, dtype=input.dtype)

        # Prepare mask: [T, B, 1]
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            mask = mask.to(device=device, dtype=input.dtype)

        # MC dropout masks
        if z is None:
            zx, zh = self._mc_dropout_sample_mask(B, device=device)
            # Ensure same device as input
            zx = [m.to(device=device, dtype=input.dtype) for m in zx]
            zh = [m.to(device=device, dtype=input.dtype) for m in zh]
        else:
            zx, zh = z
            zx = [m.to(device=device, dtype=input.dtype) for m in zx]
            zh = [m.to(device=device, dtype=input.dtype) for m in zh]

        # Prepare output storage
        hn = torch.empty(T, B, self.hidden_size, device=device, dtype=input.dtype)

        for t in range(T):
            x = input[t]  # [B, input_size]

            # Apply MC dropout per gate explicitly
            x_i = x * zx[0]
            x_f = x * zx[1]
            x_c = x * zx[2]
            x_o = x * zx[3]

            h_i = h_t * zh[0]
            h_f = h_t * zh[1]
            h_c = h_t * zh[2]
            h_o = h_t * zh[3]

            # LSTM gates
            i = torch.sigmoid(self.Wi(x_i) + self.Ui(h_i))
            f = torch.sigmoid(self.Wf(x_f) + self.Uf(h_f))
            g = torch.tanh(self.Wc(x_c) + self.Uc(h_c))
            o = torch.sigmoid(self.Wo(x_o) + self.Uo(h_o))

            # Update cell and hidden
            c_new = f * c_t + i * g
            h_new = o * torch.tanh(c_new)

            # Apply prefix mask: keep old states where input is padding
            if mask is not None:
                step_mask = mask[t]  # [B, 1]
                c_t = step_mask * c_new + (1 - step_mask) * c_t
                h_t = step_mask * h_new + (1 - step_mask) * h_t
            else:
                c_t, h_t = c_new, h_new

            hn[t] = h_t

        return hn, (h_t, c_t), (zx, zh)
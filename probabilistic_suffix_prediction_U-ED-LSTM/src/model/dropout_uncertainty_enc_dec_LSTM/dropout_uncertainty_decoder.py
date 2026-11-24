"""
Decoder module of the Dropout Uncertainty Encoder-Decoder LSTM.

Decoder consisting of two an two-layerd LSTM with LSTM cells
using dropout as a Bayesian approximation.
"""

# performance imports for torch: torch kernel uses one core only.
import os

from .dropout_uncertainty_LSTM_cell import DropoutUncertaintyLSTMCell

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn


class DropoutUncertaintyLSTMDecoder(nn.Module):
    """
    Decoder part of the ED-LSTM with MC Dropout for uncertainty estimation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_sizes: dict,
        embeddings,
        data_indices_dec,
        num_layers: int,
        dropout: Optional[float] = None,
        categorical_conditioning: Optional[Tuple[str, str]] = None,
        count_size: Optional[int] = None,
        numeric_conditioning: Optional[Tuple[str, int]] = None,
    ):
        """
        Decoder part of the Encoder-Decoder LSTM.

        Args:
            input_size (int): Size of input event attributes
            hidden_size (int): Size of hidden layers
            output_sizes (dict): Tuple of dictionaries for categorical and numerical
                output feature sizes
            embeddings: Categorical event attributes embeddings
            data_indices_dec: Indices of event attributes
            num_layers (int): Number of hidden layers in the LSTM
            dropout (Optional[float]): Dropout probability
            categorical_conditioning: Optional (output, event) feature names. The
                output head additionally sees the current event, not the previous one.
            count_size: Optional number of event counters given to every step
                (forward argument counts: batch x count_size, already scaled).
            numeric_conditioning: Optional (event feature, number of event values).
                All numerical outputs additionally see the current event of that
                categorical output (same step), e.g. the gap of the drawn activity.
        """
        super(DropoutUncertaintyLSTMDecoder, self).__init__()

        self.embeddings = embeddings

        self.data_indices_dec = data_indices_dec

        self.input_proj = nn.Linear(input_size, hidden_size)
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
                for i in range(num_layers - 1)
            ]
        )

        self.output_sizes = output_sizes

        # Output_sizes is a list containing two dicts:
        # one for categorical features and one for numerical features
        cat_output_sizes, num_output_sizes = output_sizes

        # Create a ModuleDict to hold the layers
        self.output_layers = nn.ModuleDict()
        # Dynamically create mean and variance output linear layers
        # for categorical features and numerical features
        for key, size in cat_output_sizes.items():
            self.output_layers[f"{key}_mean"] = nn.Linear(hidden_size, size)
            self.output_layers[f"{key}_var"] = nn.Linear(hidden_size, size)

        for key, size in num_output_sizes.items():
            self.output_layers[f"{key}_mean"] = nn.Linear(hidden_size, size)
            self.output_layers[f"{key}_var"] = nn.Linear(hidden_size, size)

        self.categorical_conditioning = categorical_conditioning
        if categorical_conditioning is not None:
            conditioned, event = categorical_conditioning
            if conditioned == event or any(key not in cat_output_sizes for key in (conditioned, event)):
                raise ValueError("Categorical conditioning needs two different categorical outputs")
            size = cat_output_sizes[event]
            embedding_size = min(600, round(1.6 * size**0.56))
            self.condition_embedding = nn.Embedding(size, embedding_size)
            for suffix in ("mean", "var"):
                self.output_layers[f"{conditioned}_{suffix}"] = nn.Linear(
                    hidden_size + embedding_size, cat_output_sizes[conditioned]
                )

        # Optional event counters (zero-initialised map into the input projection).
        self.count_size = count_size
        if count_size:
            self.count_weight = nn.Parameter(torch.zeros(hidden_size, count_size))

        # Optional numerical outputs conditioned on the current event: the existing
        # heads on the hidden state plus a zero-initialised term of an event
        # embedding (mean and log-variance alike, so the attenuated loss is unchanged).
        self.numeric_conditioning = numeric_conditioning
        if numeric_conditioning is not None:
            event, size = numeric_conditioning
            if event not in cat_output_sizes:
                raise ValueError("Numeric conditioning needs a categorical output as event")
            embedding_size = min(600, round(1.6 * size**0.56))
            self.num_condition_embedding = nn.Embedding(size, embedding_size)
            self.num_condition_weights = nn.ParameterDict()
            for key, out_size in num_output_sizes.items():
                for suffix in ("mean", "var"):
                    self.num_condition_weights[f"{key}_{suffix}_w"] = nn.Parameter(torch.zeros(out_size, embedding_size))
                    self.num_condition_weights[f"{key}_{suffix}_b"] = nn.Parameter(torch.zeros(out_size))

    def add_event_time_parameters(self, slopes: bool = False):
        """
        Own parameters per event for every numerical output (2026-11-18, needs numeric_conditioning).
        The shared event embedding above feeds mean and log-variance alike; in trained models it
        shifts almost only the log-variance (e.g. the gap before A_Cancelled: mean offset ~0,
        log-variance offset +1.6). Added here, separately for each output (mean and log-variance
        of every numerical feature): a free offset per event and, with slopes, a slope vector per
        event on the hidden state (interaction event x hidden state). Zero-initialised: an
        existing model predicts exactly as before. Loss and drawing are unchanged.
        """
        numeric = getattr(self, "numeric_conditioning", None)
        if numeric is None:
            raise ValueError("Event time parameters need numeric_conditioning")
        size = numeric[1]
        self.num_event_offsets = nn.ParameterDict()
        self.num_event_slopes = nn.ParameterDict() if slopes else None
        for key, out_size in self.output_sizes[1].items():
            for suffix in ("mean", "var"):
                name = f"{key}_{suffix}"
                self.num_event_offsets[name] = nn.Parameter(torch.zeros(size, out_size))
                if slopes:
                    hidden_size = self.output_layers[name].in_features
                    self.num_event_slopes[name] = nn.Parameter(torch.zeros(size, out_size, hidden_size))

    def numeric_prediction(self, hidden: Tensor, event_ids: Tensor):
        """Numerical means/log-variances given the hidden state and the current event (no LSTM step)."""
        ids = event_ids.long().reshape(-1)
        embedded = self.num_condition_embedding(ids)
        offsets = getattr(self, "num_event_offsets", None)
        slopes = getattr(self, "num_event_slopes", None)
        means, variances = {}, {}
        for key in self.output_sizes[1]:
            for suffix, target in (("mean", means), ("var", variances)):
                name = f"{key}_{suffix}"
                weight = self.num_condition_weights[f"{name}_w"]
                bias = self.num_condition_weights[f"{name}_b"]
                value = self.output_layers[name](hidden) + embedded @ weight.t() + bias
                if offsets is not None:
                    value = value + offsets[name][ids]
                if slopes is not None:
                    value = value + torch.einsum("bh,boh->bo", hidden, slopes[name][ids])
                target[name] = value
        return means, variances

    def conditioned_prediction(self, hidden: Tensor, event_ids: Tensor):
        """Predict the conditional categorical logits/log-variance without another LSTM step."""
        conditioning = getattr(self, "categorical_conditioning", None)
        if conditioning is None:
            raise ValueError("This decoder has no event-conditioned output")
        features = torch.cat((hidden, self.condition_embedding(event_ids.long().reshape(-1))), dim=-1)
        key = conditioning[0]
        return self.output_layers[f"{key}_mean"](features), self.output_layers[f"{key}_var"](features)

    def regularizer(self):
        """
        L2 regularization of Encoder weights, biases and dropout.

        OUTPUTS:
        - total_weight_reg: L2 weight regularization term
        - total_bias_reg: L2 bias regularization term
        - total_dropout_reg: L2 dropout regularization term
        """
        total_weight_reg, total_bias_reg = self.first_layer.regularizer()

        for hidden_layer in self.hidden_layers:
            weight, bias = hidden_layer.regularizer()

            total_weight_reg += weight
            total_bias_reg += bias

        # Projection layer (weaker prior)
        proj_weight_reg = 0.1 * torch.sum(self.input_proj.weight**2)
        proj_bias_reg = 0.1 * torch.sum(self.input_proj.bias**2)
        total_weight_reg += proj_weight_reg
        total_bias_reg += proj_bias_reg

        return total_weight_reg, total_bias_reg

    def __data_enc_for_model(self, data, pred):
        """
        Transform the dataloader input.

        Transofrms prefix or suffix input into a tensor structure for the encoder.

        INPUTS:
            data: previous event data (either the last target or predicted).
            pred: Boolean: true if predicted.

        OUTPUTS:
            prefixes: Returns model input
                Tensor seq_len x batch_size x input features (also embedded)
        """
        if pred:
            cats, nums = data
        else:
            # cat dims: list (n categorical values):
            # Each with Tensor: batch_size x (window_size - suffix size)
            cats = [data[0][i] for i in self.data_indices_dec[0]]
            # num dims: list (n numerical values):
            # Each with Tensor: batch_size x (window_size - suffix size)
            nums = [data[1][i] for i in self.data_indices_dec[1]]

            assert len(cats) == len(self.data_indices_dec[0]) and len(nums) == len(
                self.data_indices_dec[1]
            ), "Decoder: Number of input tensor is unequal the number of indices"

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

        # Merged input
        next_event = torch.cat((merged_cats, merged_nums), dim=-1).permute(
            1, 0, 2
        )  # dim: seq_len x batch_size x input_features
        return next_event

    def forward(
        self,
        input: Tensor,
        hx: Tuple[Tensor, Tensor],
        z: Optional[Tuple[List, List]] = None,
        pred: Optional[bool] = True,
        condition_event: Optional[Tensor] = None,
        counts: Optional[Tensor] = None,
        num_condition_event: Optional[Tensor] = None,
    ) -> Tuple[list, Tuple[Tensor, Tensor]]:
        """
        Prediction of next event based on last hidden state and last event.

        INPUTS:
            input_event: Either last sequence event or next predicted, target event:
                Tensor: seq_len (1) x batch_size x input_features
            hx: Tuple containing last hidden state and cell state of the encoder:
                Tensor: batch_size x hidden_size

        OUTPUTS:
            predictions: List containing:
                activity mean,
                activity log variance,
                timestamp mean,
                timestamp log variance.
            h, c: Updated hidden and cell states.
        """
        prediction_means = [
            {},
            {},
        ]  # List of two dicts: [cat_pred_means, num_pred_means]
        prediction_vars = [{}, {}]  # List of two dicts: [cat_pred_vars, num_pred_vars]

        # Process the input event through the encoder
        event = self.__data_enc_for_model(
            data=input, pred=pred
        )  # dim: Tensor: seq_len x batch_size x input feature

        input_proj = self.input_proj(event)
        if getattr(self, "count_size", None):
            if counts is None:
                raise ValueError("This decoder expects event counters (counts)")
            input_proj = input_proj + (counts.to(input_proj.dtype) @ self.count_weight.t()).unsqueeze(0)
        input_proj = self.layernorm(input_proj)
        input_proj = self.act(input_proj)

        # first decoder call initialize sample mask
        if z is None:
            z_hidden_layers = []
            # Pass input_event through the first LSTM layer and all hidden layers
            outputs, (h, c), z_first_layer = self.first_layer(
                input=input_proj, hx=hx, z=None
            )

            for _, lstm_cell in enumerate(self.hidden_layers):
                outputs, (h, c), z_hidden_layer = lstm_cell(
                    input=outputs, hx=(h, c), z=None
                )
                z_hidden_layers.append(z_hidden_layer)

            z = (z_first_layer, z_hidden_layers)
        # Use same sample masks from previous iterations for decoder
        else:
            # Pass input_event through the first LSTM layer and all hidden layers
            outputs, (h, c), _ = self.first_layer(input=input_proj, hx=hx, z=z[0])

            for i, lstm_cell in enumerate(self.hidden_layers):
                outputs, (h, c), _ = lstm_cell(input=outputs, hx=(h, c), z=z[1][i])

        # Get the last output (outputs[-1]) for predictions
        final_output = outputs[-1]

        # Unpack output_sizes into categorical and numerical dicts
        cat_output_sizes, num_output_sizes = self.output_sizes

        # Predict means and variances for categorical features
        for key in cat_output_sizes:
            conditioning = getattr(self, "categorical_conditioning", None)
            if conditioning is not None and key == conditioning[0]:
                # Training/validation supply the current target event. During
                # generation this default uses argmax; the sampler recomputes the
                # head with the event it actually draws (or the forced option).
                event_ids = condition_event
                if event_ids is None:
                    event_ids = self.output_layers[f"{conditioning[1]}_mean"](final_output).argmax(dim=-1)
                mean, logvar = self.conditioned_prediction(final_output, event_ids)
                prediction_means[0][f"{key}_mean"] = mean
                prediction_vars[0][f"{key}_var"] = logvar
                continue
            pred_mean = self.output_layers[f"{key}_mean"](final_output)
            prediction_means[0][f"{key}_mean"] = (
                pred_mean  # Store in the first dict (for cat features)
            )

            pred_var = self.output_layers[f"{key}_var"](final_output)
            prediction_vars[0][f"{key}_var"] = (
                pred_var  # Store in the first dict (for cat features)
            )

        numeric = getattr(self, "numeric_conditioning", None)
        if numeric is not None:
            # Training/validation supply the current target event; otherwise the
            # argmax event (the sampler recomputes with the event it draws).
            event_ids = num_condition_event
            if event_ids is None:
                event_ids = prediction_means[0][f"{numeric[0]}_mean"].argmax(dim=-1)
            means, variances = self.numeric_prediction(final_output, event_ids)
            prediction_means[1].update(means)
            prediction_vars[1].update(variances)
            return [prediction_means, prediction_vars], (h, c), z

        # Predict means and variances for numerical features
        for key in num_output_sizes:
            pred_mean = self.output_layers[f"{key}_mean"](final_output)
            prediction_means[1][f"{key}_mean"] = pred_mean

            pred_var = self.output_layers[f"{key}_var"](final_output)
            prediction_vars[1][f"{key}_var"] = pred_var

            # Get raw log-variance prediction
            # raw_logvar = self.output_layers[f"{key}_var"](final_output)
            # LOGVAR_BOUND: 3.0 -> var = [exp(-3.0)=0.05, exp(3.0)=20]
            # LOGVAR_BOUND = 6.0
            # Bound the log-variance using tanh activation
            # bounded_logvar = LOGVAR_BOUND * torch.tanh(raw_logvar / LOGVAR_BOUND)
            # # Store the safe version
            # prediction_vars[1][f"{key}_var"] = bounded_logvar

        predictions = [prediction_means, prediction_vars]

        # Return the prediction dictionaries for means and variances
        # along with the hidden states
        return predictions, (h, c), z

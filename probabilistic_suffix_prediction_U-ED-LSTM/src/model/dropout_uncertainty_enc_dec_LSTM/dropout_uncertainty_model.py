"""
Enocder-Decoder LSTM modelling epistemic uncertainty via dropout.

- During training encoder and decoder use Variational Dropout.
- During testing encoder uses Variational Dropout, Decoder uses Naive Dropout.
"""

# performance imports for torch: torch kernel uses one core only.
import os

from .dropout_uncertainty_decoder import DropoutUncertaintyLSTMDecoder
from .dropout_uncertainty_encoder import DropoutUncertaintyLSTMEncoder, clock_days

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn


class DropoutUncertaintyEncoderDecoderLSTM(nn.Module):
    """
    Full Encoder-Decoder architecture with droput uncertainty LSTM.
    """

    def __init__(
        self,
        data_set_categories: list[tuple[str, dict[str, int]]],
        enc_feat: list,
        dec_feat: list,
        seq_len_pred: int,
        hidden_size: int,
        num_layers: int,
        dropout: Optional[float] = None,
        # optional static attributes (only for encoder)
        static_data_set_categories: Optional[list[tuple[str, dict[str, int]]]] = None,
        static_enc_feat: Optional[list] = None,
        categorical_conditioning: Optional[Tuple[str, str]] = None,
        count_feature: Optional[str] = None,
        numeric_conditioning: Optional[str] = None,
        count_scale: float = 10.0,
        clock_feature: Optional[str] = None,
        clock_stats: Optional[Tuple[float, float]] = None,
        clock_scale: float = 4.0,
    ):
        """
        Full Encoder-Decoder architecture with droput uncertainty LSTM.

        Args:
            data_set_categories: Event attributes, name and size
            enc_feat: Event attributes used by encoder as input
            dec_feat: Event attributes used by decoder as input and output
            seq_len_pred: Length of the predicted suffix sequence
            hidden_size: Hidden size for LSTM cells and fully connected layers
            num_layers: Number of hidden layers in both Encoder and Decoder
            dropout: Dropout probability
            static_data_set_categories: Event attribute categories for static encoder
                input
            static_enc_feat: Static event attributes used by encoder as input
            categorical_conditioning: Optional (output, event) categorical feature
                names, e.g. a gap class conditioned on the current event.
            count_feature: Optional categorical feature (e.g. the activity) whose
                values are counted: the encoder sees the cumulative counts at every
                prefix position, the decoder the counts up to its input event
                (deterministic bookkeeping, not predicted, no loss).
            numeric_conditioning: Optional categorical decoder output (e.g. the
                activity); the numerical outputs of a step then also see the event
                of the same step (true event in the loss, drawn event when sampling).
            count_scale: Counts are divided by this value.
            clock_feature: Optional numerical feature with the standardised gap to the previous event (seconds);
                needs count_feature. Encoder and decoder then also see, per counted value, the days since its last
                occurrence (log(1 + days) / clock_scale, 0 before the first one), updated from the fed gaps.
            clock_stats: (mean, std) of the standardisation of clock_feature (seconds).
        """
        super(DropoutUncertaintyEncoderDecoderLSTM, self).__init__()

        # Feature sizes encoder
        self.data_set_categories = data_set_categories
        print("Dynamic data set categories: ", data_set_categories)
        self.static_data_set_categories = static_data_set_categories
        print("Data set static categories: ", static_data_set_categories)

        self.enc_feat = enc_feat
        print("Encoder dynamic input features: ", enc_feat)

        self.static_enc_feat = static_enc_feat
        if self.static_enc_feat:
            print("Encoder static input features: ", self.static_enc_feat)

        self.dec_feat = dec_feat
        self.categorical_conditioning = categorical_conditioning
        print("Decoder input and output features: ", dec_feat)

        # Optional event counters and event-conditioned numerical outputs
        cat_entries = {entry[0]: (i, entry) for i, entry in enumerate(data_set_categories[0])}
        self.count_feature, self.count_scale = count_feature, float(count_scale)
        count_kwargs = {}
        if count_feature is not None:
            if count_feature not in cat_entries or count_feature not in dec_feat[0] or count_feature not in enc_feat[0]:
                raise ValueError("count_feature must be a categorical encoder and decoder feature")
            index, entry = cat_entries[count_feature]
            mapping = entry[2] if len(entry) > 2 else {}
            self.count_size = int(entry[1])
            self.count_index = index  # tensor index in the data set (prefixes and suffixes)
            self.count_ignore = tuple(sorted({0} | ({int(mapping["EOS"])} if "EOS" in mapping else set())))
            self.count_dec_position = [f for f in dec_feat[0] if f in cat_entries].index(count_feature)
            count_kwargs = {"count_size": self.count_size, "count_index": self.count_index,
                            "count_ignore": self.count_ignore, "count_scale": self.count_scale}
            print("Event counters: ", count_feature, self.count_size, "ignored ids", self.count_ignore)
        self.clock_feature = clock_feature
        self.clock_stats, self.clock_scale = clock_stats, float(clock_scale)
        if clock_feature is not None:
            num_entries = {entry[0]: i for i, entry in enumerate(data_set_categories[1])}
            if count_feature is None or clock_feature not in num_entries or clock_feature not in dec_feat[1] or clock_feature not in enc_feat[1]:
                raise ValueError("clock_feature must be a numerical encoder and decoder feature, and counters must be on")
            if clock_stats is None:
                raise ValueError("clock_stats (mean, std of the gap standardisation) is required")
            self.clock_index = num_entries[clock_feature]
            self.clock_dec_position = [f for f in dec_feat[1] if f in num_entries].index(clock_feature)
            count_kwargs.update({"clock_index": self.clock_index, "clock_mean": float(clock_stats[0]),
                                 "clock_std": float(clock_stats[1]), "clock_scale": self.clock_scale})
            print("Event clocks: ", clock_feature, "index", self.clock_index, "stats", clock_stats)
        self.numeric_conditioning = numeric_conditioning
        if numeric_conditioning is not None:
            if numeric_conditioning not in cat_entries or numeric_conditioning not in dec_feat[0]:
                raise ValueError("numeric_conditioning must be a categorical decoder output")
            print("Numerical outputs conditioned on the current event: ", numeric_conditioning)
        # Sequence lenght prediciton
        self.seq_len_pred = seq_len_pred
        print("Sequence length of decoder output: ", seq_len_pred)

        print("\n")

        # Parameters for encoder and decoder
        self.hidden_size = hidden_size
        print("LSTM cells and FC hidden size: ", hidden_size)
        self.num_layers = num_layers
        print("Number of LSTM layer: ", num_layers)
        self.dropout = dropout
        print("Dropout rate: ", dropout)

        print("\n")

        # Encoder (dynamic)
        print("Encoder dynamic:")
        # Get list of category label values for cat and num
        enc_label_cats, enc_label_nums = self.__get_list_labels_input(
            data_set_categories=data_set_categories, model_type_feats=enc_feat
        )
        self.data_labels_features_enc = [enc_label_cats, enc_label_nums]
        print(
            "Encoder number of labels for each input feature (categorical, numerical):",
            self.data_labels_features_enc,
        )
        data_cat_indices_enc, data_num_indices_enc = self.__get_list_tensor_indeces(
            data_set_categories=data_set_categories, model_type_feats=enc_feat
        )
        self.data_indices_enc = [data_cat_indices_enc, data_num_indices_enc]
        print(
            "Encoder indices of tensors in dataset used as input:",
            self.data_indices_enc,
        )

        # Create embeddings for categorical features
        self.embeddings_enc = nn.ModuleList(
            [
                nn.Embedding(n_cat, min(600, round(1.6 * n_cat**0.56)))
                for n_cat in enc_label_cats
            ]
        )
        print("Embeddings encoder: ", self.embeddings_enc)

        # Compute total input size encoder
        embedding_size_enc = sum(
            [min(600, round(1.6 * n_cat**0.56)) for n_cat in enc_label_cats]
        )
        print("Total embedding feature size encoder: ", embedding_size_enc)
        num_size_enc = sum(enc_label_nums)
        print("Total numerical feature size encoder: ", num_size_enc)
        self.input_size_enc = embedding_size_enc + num_size_enc
        print("Input feature size encoder: ", self.input_size_enc)

        print("\n")

        # Encoder (static)
        print("Encoder static:")
        if self.static_enc_feat:
            if self.static_data_set_categories is None:
                raise ValueError(
                    "Static encoder features provided but static categories aremissing."
                )
            static_enc_label_cats, static_enc_label_nums = self.__get_list_labels_input(
                data_set_categories=self.static_data_set_categories,
                model_type_feats=self.static_enc_feat,
            )
            self.data_labels_static_features_enc = [
                static_enc_label_cats,
                static_enc_label_nums,
            ]
            print(
                "Encoder number of labels for each input feature"
                " (categorical, numerical):",
                self.data_labels_static_features_enc,
            )
            static_data_cat_indices_enc, static_data_num_indices_enc = (
                self.__get_list_tensor_indeces(
                    data_set_categories=self.static_data_set_categories,
                    model_type_feats=self.static_enc_feat,
                )
            )
            self.static_data_indices_enc = [
                static_data_cat_indices_enc,
                static_data_num_indices_enc,
            ]
            print(
                "Encoder indices of tensors in dataset used as input (static):",
                self.static_data_indices_enc,
            )

            # Embedding
            if static_enc_label_cats:
                self.embeddings_static_enc = nn.ModuleList(
                    [
                        nn.Embedding(n_cat, min(600, round(1.6 * n_cat**0.56)))
                        for n_cat in static_enc_label_cats
                    ]
                )
                static_embedding_size = sum(
                    [
                        embedding.embedding_dim
                        for embedding in self.embeddings_static_enc
                    ]
                )
                print(
                    "Static encoder categorical embeddings:", self.embeddings_static_enc
                )
            else:
                self.embeddings_static_enc = None
                static_embedding_size = 0
            print(
                "Total embedding feature size encoder (static): ", static_embedding_size
            )

            if static_enc_label_nums:
                static_num_size = sum(static_enc_label_nums)
            else:
                static_num_size = 0
            print("Total numerical feature size encoder (static): ", static_num_size)

            # Total sizes
            self.static_input_size_enc = static_embedding_size + static_num_size
            print("Static encoder feature size: ", self.static_input_size_enc)

            # Define Encoder
            self.encoder = DropoutUncertaintyLSTMEncoder(
                hidden_size=hidden_size,
                # dynamics
                embeddings=self.embeddings_enc,
                data_indices_enc=self.data_indices_enc,
                input_size=self.input_size_enc,
                # layers
                num_layers=num_layers,
                # static feat
                static_embeddings=self.embeddings_static_enc,
                static_data_indices=self.static_data_indices_enc,
                static_input_size=self.static_input_size_enc,
                # dropout
                dropout=dropout,
                **count_kwargs,
            )

        else:
            print("No static encoder features configured.")

            # Define Encoder
            self.encoder = DropoutUncertaintyLSTMEncoder(
                hidden_size=hidden_size,
                # dynamics
                embeddings=self.embeddings_enc,
                data_indices_enc=self.data_indices_enc,
                input_size=self.input_size_enc,
                # layers
                num_layers=num_layers,
                # dropout
                dropout=dropout,
                **count_kwargs,
            )
        print("Encoder initialized! \n")

        # Decoder
        # Get list of category label values for cat and num
        dec_label_cats, dec_label_nums = self.__get_list_labels_input(
            data_set_categories=data_set_categories, model_type_feats=dec_feat
        )
        print(
            "Decoder label values size for each categorical input feature: ",
            dec_label_cats,
        )
        print(
            "Decoder label values size for each numerical input feature: ",
            dec_label_nums,
        )

        data_cat_indices_dec, data_num_indices_dec = self.__get_list_tensor_indeces(
            data_set_categories=data_set_categories, model_type_feats=dec_feat
        )
        self.data_indices_dec = [data_cat_indices_dec, data_num_indices_dec]
        print(
            "Decoder indices of tensors in dataset used as input: ",
            self.data_indices_dec,
        )

        # Create embeddings for categorical features
        self.embeddings_dec = nn.ModuleList(
            [
                nn.Embedding(n_cat, min(600, round(1.6 * n_cat**0.56)))
                for n_cat in dec_label_cats
            ]
        )
        print("Embeddings decoder: ", self.embeddings_dec)

        # Compute total input size decoder
        embedding_size_dec = sum(
            [min(600, round(1.6 * n_cat**0.56)) for n_cat in dec_label_cats]
        )
        print("Total embedding feature size decoder: ", embedding_size_dec)

        num_size_dec = sum(dec_label_nums)
        print("Total numerical feature size decoder: ", num_size_dec)

        self.input_size_dec = embedding_size_dec + num_size_dec
        print("Input feature size decoder: ", self.input_size_dec)

        # Dictionary of output features and output_sizes
        self.output_sizes = self.__get_list_dict_labels_output(
            data_set_categories=data_set_categories, model_type_feats=dec_feat
        )
        print(
            "Output feature list of dicts (featue name, feature output size)"
            " of decoder:",
            self.output_sizes,
        )

        # Define Decoder
        self.decoder = DropoutUncertaintyLSTMDecoder(
            input_size=self.input_size_dec,
            hidden_size=hidden_size,
            output_sizes=self.output_sizes,
            embeddings=self.embeddings_dec,
            data_indices_dec=self.data_indices_dec,
            num_layers=num_layers,
            dropout=dropout,
            categorical_conditioning=categorical_conditioning,
            count_size=(2 * count_kwargs["count_size"] if clock_feature is not None else count_kwargs.get("count_size")),
            numeric_conditioning=(numeric_conditioning, int(cat_entries[numeric_conditioning][1][1]))
            if numeric_conditioning is not None else None,
        )
        print("Decoder initialized! \n")

        # List containing two dicts: One for categorical, one for numerical
        self.output_feature_indeces = self.__get_list_dict_feature_index(
            data_set_categories=data_set_categories, model_type_feats=dec_feat
        )
        # print("Output feature list of dicts (featue name, tensor index in dataset)
        # of decoder: ", self.output_feature_indeces)

    def __get_list_labels_input(self, data_set_categories, model_type_feats):
        """
        Gets number of feature attributes used as input of model.

        Returns two lists (categorical, numerical) containing the number of feature
        attributes (e.g.: 28 activity labels for concept:name).
        """
        # Unpack categories
        cat_categories, num_categories = data_set_categories
        cat_feat_model, num_feat_model = model_type_feats

        # Use the first value in the tuple as the key and the second value as the value
        cat_dict = {cat[0]: cat[1] for cat in cat_categories}
        num_dict = {num[0]: num[1] for num in num_categories}

        # Get input feature sizes to determine the model input size
        label_cats = [
            cat_dict[cat_feat] for cat_feat in cat_feat_model if cat_feat in cat_dict
        ]
        label_nums = [
            num_dict[num_feat] for num_feat in num_feat_model if num_feat in num_dict
        ]

        return label_cats, label_nums

    def __get_list_tensor_indeces(self, data_set_categories, model_type_feats):
        """
        Gets indices of tensors in dataset used as input of model.

        Returns two lists (categorical, numerical) containing the indices of the tensors
        in the datset used as input for the encoder.
        """
        # Unpack categories
        cat_categories, num_categories = data_set_categories
        cat_feat_model, num_feat_model = model_type_feats

        # Convert cat_feat_model and num_feat_model to sets for O(1) membership checks
        cat_feat_set = set(cat_feat_model)
        num_feat_set = set(num_feat_model)

        # Get indices of tensors used as input of model
        cat_indices = [
            i for i, cat in enumerate(cat_categories) if cat[0] in cat_feat_set
        ]
        num_indices = [
            i for i, num in enumerate(num_categories) if num[0] in num_feat_set
        ]

        return cat_indices, num_indices

    def __get_list_dict_labels_output(self, data_set_categories, model_type_feats):
        """
        Return list of dictionary labels.

        Returns a list of two dicts (categorical, numerical)
        containing the key: feature name
        and the value: number of labels of feature.
        Decoder, Output only!
        """
        # Unpack categories
        cat_categories, num_categories = data_set_categories
        cat_feat_model, num_feat_model = model_type_feats

        # Use the first value in the tuple as the key and the second value as the value
        cat_dict = {cat[0]: cat[1] for cat in cat_categories}
        num_dict = {num[0]: num[1] for num in num_categories}

        # Create separate dictionaries for categorical and numerical features
        cat_labels_dict = {
            cat_feat: cat_dict[cat_feat]
            for cat_feat in cat_feat_model
            if cat_feat in cat_dict
        }
        num_labels_dict = {
            num_feat: num_dict[num_feat]
            for num_feat in num_feat_model
            if num_feat in num_dict
        }

        # Return a list containing two dicts: one for categorical and one for numerical
        # features
        return [cat_labels_dict, num_labels_dict]

    def __get_list_dict_feature_index(self, data_set_categories, model_type_feats):
        """
        Gets lisft of dicts of feature names and their tensor indices in dataset.

        Returns a list of two dicts (categorical, numerical)
        containing the key: feature name and
        the value: indices of the tensors in the datset
        used as input for the encoder.
        Decoder, Output only!
        """
        # Unpack categories
        cat_categories, num_categories = data_set_categories
        cat_feat_model, num_feat_model = model_type_feats

        # Convert cat_feat_model and num_feat_model to sets for O(1) membership checks
        cat_feat_set = set(cat_feat_model)
        num_feat_set = set(num_feat_model)

        # Create dictionaries to store feature names and their index positions
        cat_index_dict = {
            cat[0]: i for i, cat in enumerate(cat_categories) if cat[0] in cat_feat_set
        }
        num_index_dict = {
            num[0]: i for i, num in enumerate(num_categories) if num[0] in num_feat_set
        }

        # Return a list of two dicts: one for categorical and one for numerical features
        return [cat_index_dict, num_index_dict]

    def _decoder_condition(self, suffixes, step):
        """Condition the loss on the current target event, independently of history teacher forcing."""
        result = {}
        conditioning = getattr(self, "categorical_conditioning", None)
        if conditioning is not None:
            index = self.output_feature_indeces[0][conditioning[1]]
            result["condition_event"] = suffixes[0][index][:, step : step + 1]
        numeric = getattr(self, "numeric_conditioning", None)
        if numeric is not None:
            index = self.output_feature_indeces[0][numeric]
            result["num_condition_event"] = suffixes[0][index][:, step : step + 1]
        return result

    def _counts_of(self, ids: Tensor) -> Tensor:
        """Scaled counts of the counted feature in ids (batch x k) -> batch x count_size."""
        onehot = nn.functional.one_hot(ids.long().reshape(ids.shape[0], -1), self.count_size).sum(dim=1).float()
        onehot[:, list(self.count_ignore)] = 0
        return onehot / self.count_scale

    def _counting(self) -> bool:
        return getattr(self, "count_feature", None) is not None

    def _clocking(self) -> bool:
        return getattr(self, "clock_feature", None) is not None

    def _state_of_prefix(self, cat_prefixes, num_prefixes):
        """Counter state after the prefix: counts (as before) or, with clocks, (counts, days since, seen) at its last event."""
        if not self._counting():
            return None
        counts = self._counts_of(cat_prefixes[self.count_index])
        if not self._clocking():
            return counts
        since, seen = clock_days(cat_prefixes[self.count_index], num_prefixes[self.clock_index], self.count_size,
                                 self.count_ignore, self.clock_stats[0], self.clock_stats[1])
        return (counts, since[:, -1], seen[:, -1])

    def _advance(self, state, ids: Tensor, gaps: Optional[Tensor] = None):
        """State after feeding one event per row (ids, gaps: batch x 1; gaps standardised seconds)."""
        if state is None:
            return None
        if not isinstance(state, tuple):
            return state + self._counts_of(ids)
        counts, since, seen = state
        ids = ids.reshape(ids.shape[0], -1)[:, -1]
        days = (gaps.reshape(gaps.shape[0], -1)[:, -1].float() * self.clock_stats[1] + self.clock_stats[0]).clamp(min=0.0) / 86400.0
        occurred = nn.functional.one_hot(ids.long(), self.count_size).bool()
        occurred[:, list(self.count_ignore)] = False
        since = (since + days.unsqueeze(1)).masked_fill(occurred, 0.0)
        return (counts + self._counts_of(ids.unsqueeze(1)), since, seen | occurred)

    def _count_kwargs(self, counts):
        if counts is None:
            return {}
        if isinstance(counts, tuple):  # counters and clocks in one vector for the decoder
            values, since, seen = counts
            clocks = (torch.log1p(since) / self.clock_scale * seen).to(values.dtype)
            return {"counts": torch.cat([values, clocks], dim=1)}
        return {"counts": counts}

    def forward(
        self,
        prefixes: List,
        static_inputs: Optional[Union[Tensor, List, Tuple, dict]] = None,
        suffixes: Optional[List] = None,
        teacher_forcing_ratio: Optional[float] = 0.0,
        prefix_mask: Optional[Tensor] = None,
    ):
        """
        Full forward pass through the Encoder-Decoder architecture.

        INPUTS:
            prefixes: Input prefix sequence:
                list(list(tensor(categorical), list(tensor(numerical)))
            static_inputs: Optional static attribute tensor(s) aligned with
                the batch dimension.
                Expected format: (static_cat_tensor, static_num_tensor) or a
                pre-projected tensor.
            suffixes: Suffix to predict:
                Tensor: list(list(tensor(categorical), list(tensor(numerical)))
            teacher_forcing_ratio: Value between 0 and 1 to select pred or target as
                last event.
            prefix_mask: masking of zero padding for prefix for encoder.

        OUTPUTS:
            predictions: Predicted outcome.
                [categorical dict (key: feature name, value tensor),
                numerical dict (key: feature name, value tensor)]
            (h,c): Predicted last hidden and cell state.
            self.seq_len_pred: Sequence length.
            self.output_feature_indeces: Target data indices:
                [categorical dict(key: feature name, value:
                    index of tensor in categorical list of dataset),
                    numerical dict(key: feature name, value: index of tensor in
                    numerical list of dataset)]
        """
        # Model is in training mode and suffixes are provided
        training = self.training and suffixes is not None
        # Model is in evaluation (validation) mode and suffixes are provided
        validation = not self.training and suffixes is not None

        # Call encoder: Differentiate between static and dynamic attributes
        # and apply zero padding mask:
        (h_enc, c_enc) = self.encoder(
            input=prefixes, static_inputs=static_inputs, mask=prefix_mask
        )

        # Get SOS event: Last prefx event:
        cat_prefixes, num_prefixes = prefixes
        cat_sos_events = [cat_tens[:, -1:] for cat_tens in cat_prefixes]
        num_sos_events = [num_tens[:, -1:] for num_tens in num_prefixes]
        sos_event = [cat_sos_events, num_sos_events]
        # Event counters of the whole prefix (the SOS event is its last event)
        counts = self._state_of_prefix(cat_prefixes, num_prefixes)

        # output_sizes is a list of two dicts: [cat_dict, num_dict]
        cat_output_features_labels, num_output_features_labels = self.output_sizes
        # Prediction dictionary for categorical features
        cat_predictions = {
            f"{key}_{suffix}": None
            for key in cat_output_features_labels
            for suffix in ["mean", "var"]
        }
        # Prediction dictionary for numerical features
        num_predictions = {
            f"{key}_{suffix}": None
            for key in num_output_features_labels
            for suffix in ["mean", "var"]
        }
        predictions = [cat_predictions, num_predictions]

        # Training
        if training:
            # Timestep iterations: 0, 1, ..., n-1
            for t in range(self.seq_len_pred):
                # SOS Event
                if t == 0:
                    # preds: list containing two dicts one for all means (cat, num),
                    # one for all vars (cat, num)
                    preds, (h, c), z = self.decoder(
                        input=sos_event, hx=(h_enc, c_enc), z=None, pred=False,
                        **self._decoder_condition(suffixes, t), **self._count_kwargs(counts)
                    )
                    pred_means, pred_vars = preds

                # Next Event
                # Decide per timestep whether to use teacher forcing
                else:
                    # Random value for teacher forcing for each timestep: If smaller use
                    # target else predicted. For high teacher forcing use target
                    if torch.rand(1).item() < teacher_forcing_ratio:
                        # Use ground-truth previous event
                        cat_t_suffix_event = [
                            cat_tens[:, t - 1 : t] for cat_tens in suffixes[0]
                        ]
                        num_t_suffix_event = [
                            num_tens[:, t - 1 : t] for num_tens in suffixes[1]
                        ]
                        t_suffix_event = [cat_t_suffix_event, num_t_suffix_event]
                        if counts is not None:
                            counts = self._advance(counts, suffixes[0][self.count_index][:, t - 1 : t],
                                                   suffixes[1][self.clock_index][:, t - 1 : t] if self._clocking() else None)

                        preds, (h, c), _ = self.decoder(
                            input=t_suffix_event, hx=(h, c), z=z, pred=False,
                            **self._decoder_condition(suffixes, t), **self._count_kwargs(counts)
                        )
                        pred_means, pred_vars = preds

                    else:
                        # Use model prediction
                        last_pred_event = self.__transform_pred_into_next_event(
                            pred_means=pred_means, pred_index=t, suffix=suffixes, decoder_hidden=h
                        )
                        # For prediction, we assume valid input (or we could carry over
                        # mask if we wanted to propagate padding)
                        # But usually we don't mask predictions during generation unless
                        # we track finished state.
                        if counts is not None:
                            counts = self._advance(counts, last_pred_event[0][self.count_dec_position],
                                                   last_pred_event[1][self.clock_dec_position] if self._clocking() else None)
                        preds, (h, c), _ = self.decoder(
                            input=last_pred_event, hx=(h, c), z=z, pred=True,
                            **self._decoder_condition(suffixes, t), **self._count_kwargs(counts)
                        )
                        pred_means, pred_vars = preds

                cat_pred_means, num_pred_means = pred_means
                cat_pred_vars, num_pred_vars = pred_vars

                # Add categorical tensors to output
                for key in cat_output_features_labels:
                    if t == 0:
                        predictions[0][f"{key}_mean"] = cat_pred_means[
                            f"{key}_mean"
                        ].unsqueeze(0)
                        predictions[0][f"{key}_var"] = cat_pred_vars[
                            f"{key}_var"
                        ].unsqueeze(0)
                    else:
                        predictions[0][f"{key}_mean"] = torch.cat(
                            (
                                predictions[0][f"{key}_mean"],
                                cat_pred_means[f"{key}_mean"].unsqueeze(0),
                            ),
                            dim=0,
                        )
                        predictions[0][f"{key}_var"] = torch.cat(
                            (
                                predictions[0][f"{key}_var"],
                                cat_pred_vars[f"{key}_var"].unsqueeze(0),
                            ),
                            dim=0,
                        )

                # Add numerical tensors to output
                for key in num_output_features_labels:
                    if t == 0:
                        predictions[1][f"{key}_mean"] = num_pred_means[
                            f"{key}_mean"
                        ].unsqueeze(0)
                        predictions[1][f"{key}_var"] = num_pred_vars[
                            f"{key}_var"
                        ].unsqueeze(0)
                    else:
                        predictions[1][f"{key}_mean"] = torch.cat(
                            (
                                predictions[1][f"{key}_mean"],
                                num_pred_means[f"{key}_mean"].unsqueeze(0),
                            ),
                            dim=0,
                        )
                        predictions[1][f"{key}_var"] = torch.cat(
                            (
                                predictions[1][f"{key}_var"],
                                num_pred_vars[f"{key}_var"].unsqueeze(0),
                            ),
                            dim=0,
                        )

        # Validation:
        if validation:
            for k in range(self.seq_len_pred):
                if k == 0:
                    preds, (h, c), z = self.decoder(
                        input=sos_event, hx=(h_enc, c_enc), z=None, pred=False,
                        **self._decoder_condition(suffixes, k), **self._count_kwargs(counts)
                    )
                    pred_means, pred_vars = preds
                else:
                    last_pred_event = self.__transform_pred_into_next_event(
                        pred_means=pred_means, decoder_hidden=h
                    )
                    if counts is not None:
                        counts = self._advance(counts, last_pred_event[0][self.count_dec_position],
                                               last_pred_event[1][self.clock_dec_position] if self._clocking() else None)
                    preds, (h, c), z = self.decoder(
                        input=last_pred_event, hx=(h, c), z=z, pred=True,
                        **self._decoder_condition(suffixes, k), **self._count_kwargs(counts)
                    )
                    pred_means, pred_vars = preds

                cat_pred_means, num_pred_means = pred_means
                cat_pred_vars, num_pred_vars = pred_vars

                # Add categorical tensors to output
                for key in cat_output_features_labels:
                    if k == 0:
                        predictions[0][f"{key}_mean"] = cat_pred_means[
                            f"{key}_mean"
                        ].unsqueeze(0)
                        predictions[0][f"{key}_var"] = cat_pred_vars[
                            f"{key}_var"
                        ].unsqueeze(0)
                    else:
                        predictions[0][f"{key}_mean"] = torch.cat(
                            (
                                predictions[0][f"{key}_mean"],
                                cat_pred_means[f"{key}_mean"].unsqueeze(0),
                            ),
                            dim=0,
                        )
                        predictions[0][f"{key}_var"] = torch.cat(
                            (
                                predictions[0][f"{key}_var"],
                                cat_pred_vars[f"{key}_var"].unsqueeze(0),
                            ),
                            dim=0,
                        )

                # Add numerical tensors to output
                for key in num_output_features_labels:
                    if k == 0:
                        predictions[1][f"{key}_mean"] = num_pred_means[
                            f"{key}_mean"
                        ].unsqueeze(0)
                        predictions[1][f"{key}_var"] = num_pred_vars[
                            f"{key}_var"
                        ].unsqueeze(0)
                    else:
                        predictions[1][f"{key}_mean"] = torch.cat(
                            (
                                predictions[1][f"{key}_mean"],
                                num_pred_means[f"{key}_mean"].unsqueeze(0),
                            ),
                            dim=0,
                        )
                        predictions[1][f"{key}_var"] = torch.cat(
                            (
                                predictions[1][f"{key}_var"],
                                num_pred_vars[f"{key}_var"].unsqueeze(0),
                            ),
                            dim=0,
                        )

        # Return training or validation output
        return predictions, (h, c), self.seq_len_pred, self.output_feature_indeces

    def __transform_pred_into_next_event(
        self,
        pred_means,
        pred_index: Optional[int] = None,
        suffix: Optional[list] = None,
        decoder_hidden: Optional[Tensor] = None,
    ):
        """
        Transform predictions into next event for decoder input.

        Gets the predicted values (means) and transform it into input for decoder model
        -> input: list(list(categorical tensors(batch size x 1)),
            list(numerical tensors(batch size x 1))).

        INPUTS:
            pred_means: predicted values
            pred_index: index of event for next prediction
            suffix: Target data

        OUTPUTS:
            next_event: event in decoder input data format
        """
        cat_pred_means, num_pred_means = pred_means
        conditioning = getattr(self, "categorical_conditioning", None)
        if conditioning is not None:
            # The loss uses the true current event; autoregressive feedback must
            # instead use the class conditional on the event fed back into the net.
            event_ids = cat_pred_means[f"{conditioning[1]}_mean"].argmax(dim=-1)
            logits, _ = self.decoder.conditioned_prediction(decoder_hidden, event_ids)
            cat_pred_means = {**cat_pred_means, f"{conditioning[0]}_mean": logits}
        numeric = getattr(self, "numeric_conditioning", None)
        if numeric is not None:
            # Same for the numerical outputs: the loss saw the true current event,
            # the fed-back times belong to the event that is fed back (argmax).
            event_ids = cat_pred_means[f"{numeric}_mean"].argmax(dim=-1)
            num_pred_means, _ = self.decoder.numeric_prediction(decoder_hidden, event_ids)

        # Create index tensor based on predicted logits
        cat_preds = [
            torch.argmax(tensor, dim=1).unsqueeze(1)
            for _, tensor in enumerate(cat_pred_means.values())
        ]

        # Create time tesnors (if training:
        # next prediction time must always be >= last prediction)
        if pred_index is None or pred_index == 0:
            num_preds = [
                pred_means for _, pred_means in enumerate(num_pred_means.values())
            ]
        # Replace num preds with previous target event in case
        # the predicted one is smaller the previous target events one
        else:
            assert suffix is not None, "Suffix is None"
            # Get numerical targets
            _, num_suffix = suffix
            # Get the previous true event tesnors (which are the inputs for the decoder)
            num_suffix_dec = [
                num_suffix[i][:, pred_index - 1 : pred_index]
                for i in self.data_indices_dec[1]
            ]

            # Get the predicted event tensors
            num_preds_list = [
                pred_means for _, pred_means in enumerate(num_pred_means.values())
            ]

            assert len(num_preds_list) == len(num_suffix_dec), (
                "Not same lenght of elements in previous decoder and prediction numeric"
            )
            " values."

            num_preds = [
                torch.max(num_suffix_dec[i], num_preds_list[i])
                for i in range(len(num_suffix_dec))
            ]

        last_event = [cat_preds, num_preds]

        return last_event

    # During test time:
    def inference(
        self,
        # dynamic event attributes for encoder
        prefix: Optional[list] = None,
        # static event attributes for encoder
        static_inputs: Optional[Union[Tensor, List, Tuple, dict]] = None,
        mask: Optional[Tensor] = None,
        # last prefix event (decoder (dynamic) event attributes only)
        last_event: Optional[list] = None,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
        z: Optional[Tuple[List, List]] = None,
    ):
        """
        Inference method fo scenario analysis based on Monte Carlo sampling.

        INPUTS:
            prefix: Input sequence of the model to be analyzed by encoder.
                (Set param only for the first model call)
            static_inputs: Optional static attribute tensor(s) to merge with the latent
                space when encoding a prefix.
                Expected format: (static_cat_tensor, static_num_tensor)
                or a pre-projected tensor.
            mask: Zero padding mask for prefix.
            last_event: Last event which was the output of the decoder.
                (Set param only after the first model call)
            hx: Last hidden state which was the output of the decoder.
                (Set param only for the first model call)

        OUTPUTS:
            predictions: Predicted outcome.
                [categorical dict (key: feature name, value tensor),
                numerical dict (key: feature name, value tensor)]
                (h,c): Predicted last hidden and cell state
        """
        with torch.no_grad():
            # First Prediciton
            if prefix is not None:
                # Call encoder (static inputs are only used here)
                (h_enc, c_enc) = self.encoder(
                    input=prefix, static_inputs=static_inputs, mask=mask
                )

                # Get SOS event: Last prefx event:
                cat_prefixes, num_prefixes = prefix
                cat_sos_events = [cat_tens[:, -1:] for cat_tens in cat_prefixes]
                num_sos_events = [num_tens[:, -1:] for num_tens in num_prefixes]
                sos_event = [cat_sos_events, num_sos_events]

                # Event counters: kept on the model between the calls of one
                # sampling loop (the rows of a batch stay in the same order).
                counts = None
                if self._counting():
                    counts = self._state_of_prefix(cat_prefixes, num_prefixes)
                    self._inference_counts = counts
                preds, (h, c), z = self.decoder(
                    input=sos_event, hx=(h_enc, c_enc), z=None, pred=False, **self._count_kwargs(counts)
                )

                # Return the sample masks for consistent variational inference
                return preds, (h, c), z

            # Second-n_th prediction
            else:
                (h, c) = hx
                counts = None
                if self._counting():
                    counts = self._advance(self._inference_counts, last_event[0][self.count_dec_position],
                                           last_event[1][self.clock_dec_position] if self._clocking() else None)
                    self._inference_counts = counts
                preds, (h, c), _ = self.decoder(
                    input=last_event, hx=(h, c), z=z, pred=True, **self._count_kwargs(counts)
                )
                return preds, (h, c)

    # save and load the trained models
    #
    def save(self, path: str):
        """
        Store the trained model at path.
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "kwargs": {
                "data_set_categories": self.data_set_categories,
                "enc_feat": self.enc_feat,
                "dec_feat": self.dec_feat,
                "seq_len_pred": self.seq_len_pred,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "static_data_set_categories": self.static_data_set_categories,
                "static_enc_feat": self.static_enc_feat,
                "categorical_conditioning": getattr(self, "categorical_conditioning", None),
                "count_feature": getattr(self, "count_feature", None),
                "numeric_conditioning": getattr(self, "numeric_conditioning", None),
                "count_scale": getattr(self, "count_scale", 10.0),
                "clock_feature": getattr(self, "clock_feature", None),
                "clock_stats": getattr(self, "clock_stats", None),
                "clock_scale": getattr(self, "clock_scale", 4.0),
            },
        }
        return torch.save(checkpoint, path)

    @staticmethod
    def load(path: str, dropout: Optional[float] = None):
        """
        Load the stored model at path.
        """
        checkpoint = torch.load(
            path, weights_only=False, map_location=torch.device("cpu")
        )
        if dropout is not None:
            checkpoint["kwargs"]["dropout"] = dropout
        checkpoint["kwargs"].setdefault("static_data_set_categories", None)
        checkpoint["kwargs"].setdefault("static_enc_feat", None)
        checkpoint["kwargs"].pop("static_input_size_enc", None)
        model = DropoutUncertaintyEncoderDecoderLSTM(**checkpoint["kwargs"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

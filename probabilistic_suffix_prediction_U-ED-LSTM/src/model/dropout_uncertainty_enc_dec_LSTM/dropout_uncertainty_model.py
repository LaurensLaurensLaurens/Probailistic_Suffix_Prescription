"""
Enocder-Decoder LSTM modelling epistemic uncertainty via dropout.

- During training encoder and decoder use Variational Dropout.
- During testing encoder uses Variational Dropout, Decoder uses Naive Dropout.
"""

# performance imports for torch: torch kernel uses one core only.
import os

from .dropout_uncertainty_decoder import DropoutUncertaintyLSTMDecoder
from .dropout_uncertainty_encoder import DropoutUncertaintyLSTMEncoder

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

# Autoregressives Model:
# prefix
#   ↓
# encoder
#   ↓
# letztes Prefix-Event als SOS
#   ↓
# decoder
#   ↓
# preds₀
#   ↓
# __transform_pred_into_next_event(preds₀)
#   ↓
# last_event₀
#   ↓
# inference(last_event=last_event₀, hx=...)
#   ↓
# preds₁
#   ↓
# __transform_pred_into_next_event(preds₁)
#   ↓
# last_event₁
#   ↓
# ...
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
        """
        super(DropoutUncertaintyEncoderDecoderLSTM, self).__init__()

        # Feature sizes encoder
            #data_set_categories = [
            # kategoriale Features
            #[
            #    ("Activity", 5),
            #    ("Resource", 10),
            #    ("Department", 4),
            #],

            # numerische Features
            #[
            #    ("amount", 1),
            #    ("duration", 1),
            #]
            #]
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
        print("Decoder input and output features: ", dec_feat)
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

        # __get_list_labels_input: "wie groß ist jedes gewählte Feature?"
        # data_set_categories: alle Dataset-Spalten mit Größe
        #   z. B. Activity 5 Klassen, Resource 4, amount Breite 1
        # enc_feat: welche Namen der Encoder nutzt
        #   z. B. [["Activity"], ["amount"]] → [5], [1]
        #data_set_categories = [
        # categorical
        #[
        #    ("Activity", 5),
        #    ("Resource", 10),
        #    ("Department", 4),
        #],

        # numerical
        #[
        #    ("amount", 1),
        #    ("duration", 1),
        #]
        #]

        #enc_feat = [
        #    ["Activity"],   # categorical Features für Encoder
        #   ["amount"]      # numerical Features für Encoder
        #]
        #
        # Es ist die Größe der MODELL-Schichten:
        #   5  → nn.Embedding(5, d)   Tabelle mit 5 Zeilen (eine pro Id) z.b. Id 1 = Activity 1 id 2 = Activity 2 id 3 = Activity 3 id 4 = Activity 4 id 5 = EOS etc

        # LSTM kann mit reiner zahl wie 2 nichtsanafnegn. Braucht Embedding, also gelernter Vektor.
        #            Id     Zeile in der Tabelle          Vektor (Länge d, z. B. 4)
        #    0  →   [ 0.1, -0.3,  0.0,  0.2 ]     Padding
        #    1  →   [ 0.8,  0.1, -0.4,  0.0 ]     A
        #    2  →   [ 0.2,  0.9,  0.1, -0.2 ]     B
        #    3  →   …
        #    4  →   …                             EOS
        #   1  → amount bleibt 1 Zahl pro Zeitschritt. Wird twar skaliert, aber bleibt eine Zahl, nicht Vektor 
        #   danach: d + 1 = input_size_enc  (LSTM-Input pro Event) d ist padding
        # ENC sind die dynamischen Attribute, die der Encoder nutzt und auf denen er hidden state traininert (Er trainiert den hidden state also nicht auf allen Features in dataset)
        # Die werden dann nochmal aufgespalten in kategoriale und numerische Features
        enc_label_cats, enc_label_nums = self.__get_list_labels_input(
            data_set_categories=data_set_categories, model_type_feats=enc_feat
        )
        self.data_labels_features_enc = [enc_label_cats, enc_label_nums]
        print(
            "Encoder feature sizes (cat vocab sizes, num widths):",
            self.data_labels_features_enc,
        )

        # B — list slots (used every forward, not per event / not per case)
        # prefixes kommen sp#ter von extern 
        #   data_indices_enc = [[0], [0]]
        #   means: cats = prefixes[0][0]  (Activity of all prefixes in batch)
        #          nums = prefixes[1][0]  (amount  of all prefixes in batch)
        #   Resource at prefixes[0][1] is never read if it is not in enc_feat.
        data_cat_indices_enc, data_num_indices_enc = self.__get_list_tensor_indeces(
            data_set_categories=data_set_categories, model_type_feats=enc_feat
        )
        self.data_indices_enc = [data_cat_indices_enc, data_num_indices_enc]
        print(
            "Encoder dataset tensor indices (cat, num):",
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
        # Wir haben oben die dynamischen Features berechnet. Jetzt berechnen wir die statischen Features.
        if self.static_enc_feat:
            if self.static_data_set_categories is None:
                raise ValueError(
                    "Static encoder features provided but static categories are missing."
                )
            static_enc_label_cats, static_enc_label_nums = self.__get_list_labels_input(
                data_set_categories=self.static_data_set_categories,
                model_type_feats=self.static_enc_feat,
            )
            # Sieht z.b. so aus: [5], [1]

            # Ist gegeben
            self.data_labels_static_features_enc = [
                static_enc_label_cats,
                static_enc_label_nums,
            ]
            print(
                "Encoder number of labels for each input feature"
                " (categorical, numerical):",
                self.data_labels_static_features_enc,
            )
            # Wieder wie oben
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
        )
        print("Decoder initialized! \n")

        # List containing two dicts: One for categorical, one for numerical
        self.output_feature_indeces = self.__get_list_dict_feature_index(
            data_set_categories=data_set_categories, model_type_feats=dec_feat
        )
        # print("Output feature list of dicts (featue name, tensor index in dataset)
        # of decoder: ", self.output_feature_indeces)

    def __get_list_labels_input(self, data_set_categories, model_type_feats):
        # Wir brauchen das, um die Größe der Embeddingstabelle zu berechnen. Anonsten wüsste Pytorch nicht, wie groß die Tabelle sein soll.
        # Unpack categories
        cat_categories, num_categories = data_set_categories
        # cat_categories = [
        #    ("Activity", 5),
        #    ("Resource", 10),
        #    ("Department", 4),
        #]
        # num_categories = [
        #    ("amount", 1),
        #    ("duration", 1),
        #]
        cat_feat_model, num_feat_model = model_type_feats
        # cat_feat_model = ["Activity"]
        # num_feat_model = ["amount"]

        cat_dict = {cat[0]: cat[1] for cat in cat_categories}
        num_dict = {num[0]: num[1] for num in num_categories}
        # Einfach Liste in Dictionary umwandeln
        # Use the first value in the tuple as the key and the second value as the value.
        #cat_dict = {
        #    "Activity": 5,
        #    "Resource": 10,
        #    "Department": 4
        #}

        #num_dict = {
        #    "amount": 1,
            #"duration": 1
        #}

        # No,, jede spalte, die wir für den Encoder nutzen. Steht sie im Dict, hol die Größe und packe sie in eine Liste
        label_cats = [
            cat_dict[cat_feat] for cat_feat in cat_feat_model if cat_feat in cat_dict
        ]
        # label_cats = [5]
        label_nums = [
            num_dict[num_feat] for num_feat in num_feat_model if num_feat in num_dict
        ]
        # label_nums = [1]

        return label_cats, label_nums
        # return [5], [1]  → Embedding(5, d) for Activity, 1 float channel for amount

    def __get_list_tensor_indeces(self, data_set_categories, model_type_feats):
        """Return list positions of ``model_type_feats`` inside the dataset tensors.

        Step-by-step demo
        -----------------
        Input::

            data_set_categories = [
                [("Activity", 5), ("Resource", 10), ("Department", 4)],
                [("amount", 1), ("duration", 1)],
            ]
            model_type_feats = [["Activity"], ["amount"]]

        1) Unpack::

            cat_categories = [("Activity", 5), ("Resource", 10), ("Department", 4)]
            num_categories = [("amount", 1), ("duration", 1)]
            cat_feat_model = ["Activity"]
            num_feat_model = ["amount"]

        2) Sets (only for fast "is this name wanted?")::

            cat_feat_set = {"Activity"}
            num_feat_set = {"amount"}

        3) Walk dataset order, keep index if name is in the set::

            cat  i=0 Activity    in set? yes → keep 0
                 i=1 Resource    in set? no
                 i=2 Department  in set? no
            num  i=0 amount      in set? yes → keep 0
                 i=1 duration    in set? no

        4) Return::

            cat_indices = [0] Da Activity in cat_feat_set ist und i=0 ist
            num_indices = [0] Da amount in num_feat_set ist und i=0 ist

        Later encoder.forward::

            prefixes[0][0]  → Activity tensor  (batch, seq) batch ist die Anzahl der Beispiele in der Batch, seq ist die Länge der Sequenz
            prefixes[1][0]  → amount tensor
            Resource / Department / duration are never read.
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

        # z.B. [0], [0] 
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

    # EncoderDecoder.forward()
    #     │
    #     ▼
    # Encoder.forward()
    #     │
    #     ▼
    # LSTMCell.forward()
    #     │
    #     ▼
    #   h_enc,c_enc
    #     │
    #     ▼
    # Decoder.forward()
    #     │
    #     ▼
    # LSTMCell.forward()
    #     │
    #     ▼
    # Prediction t=0
    #     │
    #     ▼
    # Decoder.forward()
    #     │
    #     ▼
    # LSTMCell.forward()
    #     │
    #     ▼
    # Prediction t=1
    #     │
    #    ...
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
                könnte so aussehen
                    prefixes = [

                        # ============================================================
                        # CATEGORICAL FEATURES
                        # Jede kategoriale Variable hat einen eigenen Tensor [B, W]
                        # B = Batchgröße / Anzahl Prefix-Samples im Batch
                        # W = window_size
                        # ============================================================
                        [
                            # categorical feature 1: activity
                            tensor([
                                # Prefix-Sample 1
                                [0, 0, 1, 4, 7],
                                # 0,0 = Padding
                                # 1 = Create Application
                                # 4 = Delete Application
                                # 7 = Assign Developer

                                # Prefix-Sample 2
                                [0, 2, 3, 5, 8],

                                # Prefix-Sample 3
                                [0, 0, 0, 1, 6],
                            ]),

                            # categorical feature 2: resource
                            tensor([
                                # Resource-Werte für GENAU dieselben 3 Prefix-Samples
                                [0, 0, 12, 5, 9],

                                [0, 7,  3, 2, 4],

                                [0, 0,  0, 8, 6],
                            ]),

                            # categorical feature 3: department
                            tensor([
                                # Department-Werte für dieselben Prefix-Samples
                                [0, 0, 2, 2, 3],

                                [0, 1, 1, 4, 4],

                                [0, 0, 0, 2, 5],
                            ]),
                        ],


                        # ============================================================
                        # NUMERICAL FEATURES
                        # Auch jedes numerische Feature hat einen eigenen Tensor [B, W]
                        # ============================================================
                        [
                            # numerical feature 1: elapsed_time
                            tensor([
                                # Prefix-Sample 1
                                [0.0, 0.0, 0.10, 0.20, 0.55],

                                # Prefix-Sample 2
                                [0.0, 0.10, 0.30, 0.40, 0.80],

                                # Prefix-Sample 3
                                [0.0, 0.0, 0.0, 0.00, 0.35],
                            ]),

                            # numerical feature 2: time_since_last_event
                            tensor([
                                # Prefix-Sample 1
                                [0.0, 0.0, 0.05, 0.10, 0.35],

                                # Prefix-Sample 2
                                [0.0, 0.10, 0.20, 0.30, 0.40],

                                # Prefix-Sample 3
                                [0.0, 0.0, 0.0, 0.00, 0.35],
                            ]),
                        ]
                    ]
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
        #sos_event: das Event, mit dem der Decoder startet, typischerweise das letzte Event des Prefixes
        cat_prefixes, num_prefixes = prefixes
        cat_sos_events = [cat_tens[:, -1:] for cat_tens in cat_prefixes]
        num_sos_events = [num_tens[:, -1:] for num_tens in num_prefixes]
        sos_event = [cat_sos_events, num_sos_events]
        # Nimm für jeden Prefix im Batch nur die letzte Spalte, behalte aber die Dimension [B,1]
        #         prefix = [
        #     # categorical
        #     [
        #         # Activity [B,T]
        #         tensor([
        #             [0, 0, 3, 7, 12], Prefix 1 Activity None -> None -> Create Application -> Submit Application -> Assign Developer
        #             [0, 0, 0, 3,  9], Prefix 2 ...
        #         ]),

        #         # Lifecycle [B,T]
        #         tensor([
        #             [0, 0, 1, 1, 3],
        #             [0, 0, 0, 1, 2],
        #         ]),

        #         # Resource [B,T]
        #         tensor([
        #             [0, 0, 5, 5, 8],
        #             [0, 0, 0, 2, 7],
        #         ]),
        #     ],

        #     # numerical
        #     [
        #         # case_elapsed_time
        #         tensor([
        #             [0., 0., -1.2, -0.4, 0.3],
        #             [0., 0.,  0.0, -0.8, 0.1],
        #         ]),

        #         # event_elapsed_time
        #         tensor([
        #             [0., 0., -0.9, 0.2, 0.5],
        #             [0., 0.,  0.0, 0.1, 0.7],
        #         ]),

        #         # day_in_week
        #         tensor([
        #             [0., 0., -0.5, -0.5, -0.5],
        #             [0., 0.,  0.0,  0.8,  0.8],
        #         ]),

        #         # seconds_in_day
        #         tensor([
        #             [0., 0., 0.2, 0.3, 0.4],
        #             [0., 0., 0.0, 0.5, 0.6],
        #         ]),
        #     ],
        # ]
        # ZU 
        #sos_event = [
            #     # categorical
            #     [
            #         # Activity
            #         tensor([
            #             [12],
            #             [ 9],
            #         ]),

            #         # Lifecycle
            #         tensor([
            #             [3],
            #             [2],
            #         ]),

            #         # Resource
            #         tensor([
            #             [8],
            #             [7],
            #         ]),
            #     ],

            #     # numerical
            #     [
            #         # case_elapsed_time
            #         tensor([
            #             [0.3],
            #             [0.1],
            #         ]),

            #         # event_elapsed_time
            #         tensor([
            #             [0.5],
            #             [0.7],
            #         ]),

            #         # day_in_week
            #         tensor([
            #             [-0.5],
            #             [ 0.8],
            #         ]),

            #         # seconds_in_day
            #         tensor([
            #             [0.4],
            #             [0.6],
            #         ]),
            #     ],
            # ]


        # output_sizes is a list of two dicts: [cat_dict, num_dict]
        cat_output_features_labels, num_output_features_labels = self.output_sizes
        # Prediction dictionary for categorical features.
        # Wir sagen für jede kategorielle Variable, die wir vorhersagen wollen, ein Dictionary mit den Keys "mean" und "var"
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
            # Training auf S-1 nächsten Schritten
            for t in range(self.seq_len_pred):
                # SOS Event
                if t == 0:
                    # preds: list containing two dicts one for all means (cat, num),
                    # one for all vars (cat, num)
                    # c_enc ist cell state des encoders nach dem letzten prefix event. Langzeitgedächtnis 
                    # h_enc ist hidden state des encoders nach dem letzten prefix event
                    # preds ist ein Tupel preds = (pred_means, pred_vars)
                    preds, (h, c), z = self.decoder(
                        input=sos_event, hx=(h_enc, c_enc), z=None, pred=False
                    )
                    pred_means, pred_vars = preds
                    
                    #predictions = [
                    #     # categorical predictions
                    #     {
                    #         "activity_mean": tensor([
                    #             [
                    #                 [0.10, 0.70, 0.20],   # prefix 1 , timestep t=0
                    #                 [0.60, 0.10, 0.30],   # prefix 2 , timestep t=0
                    #             ]
                    #         ]),

                    #         "activity_var": tensor([
                    #             [
                    #                 [0.01, 0.03, 0.02],
                    #                 [0.02, 0.01, 0.04],
                    #             ]
                    #         ]),

                    #         "resource_mean": tensor([
                    #             [
                    #                 [0.20, 0.80],
                    #                 [0.75, 0.25],
                    #             ]
                    #         ]),

                    #         "resource_var": tensor([
                    #             [
                    #                 [0.02, 0.04],
                    #                 [0.01, 0.03],
                    #             ]
                    #         ]),
                    #     },

                    #     # numerical predictions
                    #     {
                    #         "elapsed_time_mean": tensor([
                    #             [
                    #                 [0.42],   # Case 1, timestep 0
                    #                 [0.73],   # Case 2, timestep 0
                    #             ]
                    #         ]),

                    #         "elapsed_time_var": tensor([
                    #             [
                    #                 [0.05],
                    #                 [0.08],
                    #             ]
                    #         ]),
                    #     }
                    # ]

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

                        # echtes vorheriges Event
                        # t_suffix_event ist ein Tupel t_suffix_event = (cat_t_suffix_event, num_t_suffix_event). Es ist das echte vorherige Event.
                        # h ist hidden state des decoders nach dem letzten prefix event
                        # c ist cell state des decoders nach dem letzten prefix event
                        # z ist das Zustandsvektor des decoders
                        # pred=False: wir verwenden das echte vorherige Event
                        preds, (h, c), _ = self.decoder(
                            input=t_suffix_event, hx=(h, c), z=z, pred=False
                        )
                        pred_means, pred_vars = preds

                    else:
                        # Use model prediction
                        last_pred_event = self.__transform_pred_into_next_event(
                            pred_means=pred_means, pred_index=t, suffix=suffixes
                        )
                        # For prediction, we assume valid input (or we could carry over
                        # mask if we wanted to propagate padding)
                        # But usually we don't mask predictions during generation unless
                        # we track finished state.
                        preds, (h, c), _ = self.decoder(
                            input=last_pred_event, hx=(h, c), z=z, pred=True
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
                        # Dann unsqueeze(0) mit Zeitdimension
                        # activity vorher:
                        #     [B, classes]
                        #     [2, 3]

                        #     nach unsqueeze(0):
                        #     [1, B, classes]
                        #     [1, 2, 3]
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
                # predictions[0]["activity_mean"] = tensor([
                #     # t = 0
                #     [   z.B. Acttivity X, Activity Y, Activity Z
                #         [0.10, 0.70, 0.20],   # Prefix 1 -------v
                #         [0.60, 0.10, 0.30],   # Prefix 2 -------> Anzahl prefix samples im batch
                #     ],

                #     # t = 1
                #     [
                #         [0.20, 0.30, 0.50],  
                #         [0.10, 0.80, 0.10],  
                #     ],
                # ])
                #   [2, 2, 3]
                #   ↑  ↑  ↑
                #   T  B  Klassen
                # T = Anzahl vorhergesagter Zeitschritte im Suffix
                # B = Anzahl Prefix-Samples gleichzeitig im Batch
                # C = Anzahl Klassen des kategorialen Features

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
                # predictions = [
                #     {
                #         "activity_mean": ...,   # [4, B, num_activity_classes]
                #         "activity_var":  ...,   # [4, B, num_activity_classes]

                #         "resource_mean": ...,   # [4, B, num_resource_classes]
                #         "resource_var":  ...,   # [4, B, num_resource_classes]
                #     },

                #     {
                #         "elapsed_time_mean": ...,  # [4, B, 1]
                #         "elapsed_time_var":  ...,  # [4, B, 1]
                #     }
                # ]

                #     predictions
                # ├── [0] categorical
                # │   ├── activity_mean   [T, B, C_activity]
                # │   ├── activity_var    [T, B, C_activity]
                # │   ├── resource_mean   [T, B, C_resource]
                # │   └── resource_var    [T, B, C_resource]
                # │
                # └── [1] numerical
                #     ├── elapsed_time_mean   [T, B, 1]
                #     └── elapsed_time_var    [T, B, 1]

        # Validation:
        if validation:
            for k in range(self.seq_len_pred):
                if k == 0:
                    preds, (h, c), z = self.decoder(
                        input=sos_event, hx=(h_enc, c_enc), z=None, pred=False
                    )
                    pred_means, pred_vars = preds
                else:
                    last_pred_event = self.__transform_pred_into_next_event(
                        pred_means=pred_means
                    )
                    preds, (h, c), z = self.decoder(
                        input=last_pred_event, hx=(h, c), z=z, pred=True
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

    # Transform predictions into next event for decoder input
    def __transform_pred_into_next_event(
        self,
        pred_means,
        pred_index: Optional[int] = None,
        suffix: Optional[list] = None,
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

        pred_means = [
            {
                "activity_mean": tensor([
                    [0.1, 0.7, 0.2], # case 1 activity X, activity Y, activity Z
                    [0.8, 0.1, 0.1], # case 2
                ])
            },
            {
                "time_mean": tensor([
                    [4.2],
                    [7.5],
                ])
            },
        ]

        OUTPUTS:
            next_event: event in decoder input data format

        last_pred_event = [next_event = [
            [tensor([[1], [0]])],      # kategorial: [B, 1]
            [tensor([[4.2], [7.5]])],  # numerisch:   [B, 1]
        ]
        next_event = [
            [cat_feature_1, cat_feature_2, ...],
            [num_feature_1, num_feature_2, ...],
        ]

        """
        cat_pred_means, num_pred_means = pred_means

        # Create index tensor based on predicted logits
        # cat_preds = [
        #     tensor([[1], [0]]),  # z. B. Aktivität
        #     tensor([[2], [1]]),  # z. B. Ressource
        # ]
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
        # last_event
        # ├── [0] kategoriale Features
        # │   ├── [0] Aktivität: [B, 1]
        # │   └── [1] Ressource: [B, 1]
        # └── [1] numerische Features
        #     ├── [0] Zeit: [B, 1]
        #     └── [1] weiteres Feature: [B, 1]

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

                preds, (h, c), z = self.decoder(
                    input=sos_event, hx=(h_enc, c_enc), z=None, pred=False
                )

                # Return the sample masks for consistent variational inference
                return preds, (h, c), z

            # Second-n_th prediction
            else:
                (h, c) = hx
                preds, (h, c), _ = self.decoder(
                    input=last_event, hx=(h, c), z=z, pred=True
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
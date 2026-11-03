"""The candidate next-event models: an n-gram baseline, an LSTM and a Transformer decoder."""
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

from .data_prep import BOS, END, PAD, Encoder, Trace


class NGramModel:
    """Next event from the counts of the last order-1 events; shorter contexts if a context was never seen."""

    def __init__(self, order: int = 5):
        self.order = order
        self.counts: Dict[tuple, Counter] = defaultdict(Counter)

    def fit(self, sequences: List[List[int]]) -> None:
        for sequence in sequences:
            padded = [BOS] * (self.order - 1) + sequence + [END]
            for position in range(self.order - 1, len(padded)):
                for length in range(self.order):
                    self.counts[tuple(padded[position - length:position])][padded[position]] += 1

    def sample(self, n_traces: int, max_length: int, rng: np.random.Generator) -> List[List[int]]:
        traces = []
        for _ in range(n_traces):
            history = [BOS] * (self.order - 1)
            trace: List[int] = []
            while len(trace) < max_length:
                for length in range(self.order - 1, -1, -1):
                    counter = self.counts.get(tuple(history[len(history) - length:]))
                    if counter:
                        break
                tokens = list(counter)
                weights = np.fromiter(counter.values(), dtype=float)
                token = tokens[rng.choice(len(tokens), p=weights / weights.sum())]
                if token == END:
                    break
                trace.append(token)
                history.append(token)
            traces.append(trace)
        return traces


def build_network(kind: str, vocabulary_size: int, category_sizes: List[int], n_numeric: int, max_length: int,
                  width: int = 128, depth: int = 2, heads: int = 4, dropout: float = 0.1):
    """
    Network that maps a sequence of events and the case attributes to the logits of the next event at
    every position. kind is "lstm" or "transformer". The case attributes are added to every position.
    """
    import keras
    from keras import layers, ops

    class AddPositions(layers.Layer):
        """Adds a learned embedding of the position in the sequence."""

        def __init__(self, max_length, width, **kwargs):
            super().__init__(**kwargs)
            self.embedding = layers.Embedding(max_length, width)

        def call(self, x):
            return x + self.embedding(ops.arange(ops.shape(x)[1]))

    tokens = keras.Input(shape=(None,), dtype="int32", name="tokens")
    categorical = keras.Input(shape=(len(category_sizes),), dtype="int32", name="categorical")
    numeric = keras.Input(shape=(n_numeric,), dtype="float32", name="numeric")

    attribute_parts = [layers.Embedding(size, 8)(categorical[:, i]) for i, size in enumerate(category_sizes)] + [numeric]
    attributes = layers.Dense(width)(layers.Concatenate()(attribute_parts))
    x = layers.Embedding(vocabulary_size, width)(tokens) + ops.expand_dims(attributes, 1)

    if kind == "lstm":
        for _ in range(depth):
            x = layers.LSTM(width, return_sequences=True, dropout=dropout)(x)
    elif kind == "transformer":
        x = layers.Dropout(dropout)(AddPositions(max_length, width)(x))
        for _ in range(depth):
            h = layers.LayerNormalization()(x)
            h = layers.MultiHeadAttention(num_heads=heads, key_dim=width // heads, dropout=dropout)(h, h, use_causal_mask=True)
            x = x + h
            h = layers.LayerNormalization()(x)
            h = layers.Dense(4 * width, activation="gelu")(h)
            h = layers.Dropout(dropout)(layers.Dense(width)(h))
            x = x + h
        x = layers.LayerNormalization()(x)
    else:
        raise ValueError(f"Unknown network kind: {kind}")
    return keras.Model([tokens, categorical, numeric], layers.Dense(vocabulary_size)(x))


class NeuralModel:
    """LSTM or Transformer over event tokens, conditioned on the case attributes."""

    def __init__(self, kind: str, encoder: Encoder, max_length: int, **network_options):
        self.kind = kind
        self.encoder = encoder
        self.max_length = max_length
        self.network_options = network_options
        self.network = build_network(
            kind, encoder.vocabulary_size, [len(values) for values in encoder.categories.values()], 1, max_length, **network_options
        )
        self._logits_fn = None

    def _arrays(self, traces: List[Trace]):
        sequences = [self.encoder.encode_events(trace.events)[: self.max_length - 1] for trace in traces]
        attributes = [self.encoder.encode_attributes(trace.attributes) for trace in traces]
        categorical = np.array([a[0] for a in attributes], dtype=np.int32)
        numeric = np.array([a[1] for a in attributes], dtype=np.float32)
        return sequences, categorical, numeric

    def _batches(self, traces: List[Trace], batch_size: int, shuffle: bool, seed: int = 0):
        """Batches of similar length (less padding): inputs BOS + events, targets events + END."""
        import keras

        sequences, categorical, numeric = self._arrays(traces)
        order = np.argsort([len(sequence) for sequence in sequences], kind="stable")
        chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        # Keras averages the loss over all positions of a batch. Batches hold cases of similar length,
        # so without correction an event of a long case would count far less than one of a short case
        # and the model would learn to generate cases that are too short. The weights below undo
        # that: every event counts the same, whatever batch it is in.
        events_per_batch = batch_size * (np.mean([len(sequence) for sequence in sequences]) + 1)

        class Batches(keras.utils.PyDataset):
            def __init__(self):
                super().__init__()
                self.rng = np.random.default_rng(seed)
                self.order = np.arange(len(chunks))

            def __len__(self):
                return len(chunks)

            def __getitem__(self, index):
                chunk = chunks[self.order[index]]
                length = max(len(sequences[i]) for i in chunk) + 1
                inputs = np.full((len(chunk), length), PAD, dtype=np.int32)
                targets = np.full((len(chunk), length), PAD, dtype=np.int32)
                for row, i in enumerate(chunk):
                    inputs[row, : len(sequences[i]) + 1] = [BOS] + sequences[i]
                    targets[row, : len(sequences[i]) + 1] = sequences[i] + [END]
                weights = (targets != PAD).astype(np.float32) * (targets.size / events_per_batch)
                return (inputs, categorical[chunk], numeric[chunk]), targets, weights

            def on_epoch_end(self):
                if shuffle:
                    self.rng.shuffle(self.order)

        return Batches()

    def fit(self, train: List[Trace], validation: List[Trace], epochs: int = 40, batch_size: int = 128,
            learning_rate: float = 1e-3, patience: int = 4, verbose: int = 2):
        import keras

        self.network.compile(
            optimizer=keras.optimizers.Adam(learning_rate, clipnorm=1.0),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            weighted_metrics=[],
        )
        return self.network.fit(
            self._batches(train, batch_size, shuffle=True),
            validation_data=self._batches(validation, batch_size, shuffle=False),
            epochs=epochs,
            verbose=verbose,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, patience // 3), min_lr=1e-5),
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            ],
        )

    def loss(self, traces: List[Trace], batch_size: int = 256) -> float:
        """
        Mean negative log-likelihood per event (including the end of the case). Unlike the loss Keras
        reports during training, padding is not counted, so values of different sets are comparable.
        """
        import tensorflow as tf

        total, count = 0.0, 0
        batches = self._batches(traces, batch_size, shuffle=False)
        for index in range(len(batches)):
            inputs, targets, weights = batches[index]
            logits = self.network(list(inputs), training=False)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits).numpy()
            total += float(losses[targets != PAD].sum())
            count += int((targets != PAD).sum())
        return total / count

    def _next_logits(self, tokens: np.ndarray, categorical: np.ndarray, numeric: np.ndarray) -> np.ndarray:
        import tensorflow as tf

        if self._logits_fn is None:
            network = self.network
            signature = [
                tf.TensorSpec([None, None], tf.int32),
                tf.TensorSpec([None, categorical.shape[1]], tf.int32),
                tf.TensorSpec([None, numeric.shape[1]], tf.float32),
            ]

            @tf.function(input_signature=signature)
            def logits_fn(tokens, categorical, numeric):
                return network([tokens, categorical, numeric], training=False)[:, -1, :]

            self._logits_fn = logits_fn
        return self._logits_fn(tf.constant(tokens), tf.constant(categorical), tf.constant(numeric)).numpy()

    def sample(self, traces: List[Trace], rng: np.random.Generator, batch_size: int = 2048,
               min_probability: float = 0.0) -> List[List[int]]:
        """
        One generated trace per given case (only its attributes are used).

        Events are drawn from the predicted distribution. Events the model gives less than
        min_probability are never drawn: a softmax gives every event a tiny probability, and over
        hundreds of thousands of draws that produces transitions that never occur in the log.
        """
        _, categorical, numeric = self._arrays(traces)
        results: List[Optional[List[int]]] = [None] * len(traces)
        for start in range(0, len(traces), batch_size):
            rows = np.arange(start, min(start + batch_size, len(traces)))
            tokens = np.full((len(rows), 1), BOS, dtype=np.int32)
            active = np.arange(len(rows))
            while len(active) and tokens.shape[1] < self.max_length:
                logits = self._next_logits(tokens[active], categorical[rows[active]], numeric[rows[active]]).astype(np.float64)
                logits[:, [PAD, BOS]] = -np.inf
                if min_probability > 0:
                    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                    logits[probabilities < min_probability] = -np.inf
                # Gumbel trick: argmax of logits plus Gumbel noise is a draw from the softmax
                drawn = np.argmax(logits + rng.gumbel(size=logits.shape), axis=1)
                tokens = np.concatenate([tokens, np.full((len(rows), 1), PAD, dtype=np.int32)], axis=1)
                tokens[active, -1] = drawn
                active = active[drawn != END]
            for row, sequence in zip(rows, tokens):
                results[row] = [int(t) for t in sequence[1:] if t not in (PAD, END)]
        return results

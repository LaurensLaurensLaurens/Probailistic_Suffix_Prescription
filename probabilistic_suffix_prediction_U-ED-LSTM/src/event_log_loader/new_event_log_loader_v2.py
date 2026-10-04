"""


"""
from functools import partial
from typing import Any, Dict, Iterable, Optional
import warnings

import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import torch
from pandas.errors import PerformanceWarning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset
from tqdm.notebook import tqdm


def _categorical_token(value: object) -> object:
    """Normalize categorical values.

    Goal: keep integer-coded categories stable (e.g. 1, 1.0 -> "1") so they
    don't end up as float-looking categories ("1.0").

    We keep tokens as `object` (typically strings) to avoid mixed-type sorting
    issues inside sklearn encoders.
    """
    if pd.isna(value):
        return value
    if isinstance(value, str):
        return value

    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        return str(value)

    return str(value)

def _normalize_categorical_series(series: pd.Series) -> pd.Series:
    return series.map(_categorical_token).astype(object)

def _ensure_group_key_column(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Ensure `key` exists as a DataFrame column.

    Pandas groupby/apply pipelines may move the grouping key into the index.
    Most of this codebase expects `key` to be a normal column.
    """
    if key in df.columns:
        return df

    if df.index.name == key:
        return df.reset_index()

    if isinstance(df.index, pd.MultiIndex) and key in (df.index.names or []):
        return df.reset_index(level=key)

    return df

def _groupby_apply_preserve_key(
    df: pd.DataFrame, key: str, func, *, sort: bool = False
) -> pd.DataFrame:
    """Run `df.groupby(key).apply(func)` and keep `key` as a column."""
    df = _ensure_group_key_column(df, key)
    applied = df.groupby(key, sort=sort).apply(func)

    if key not in applied.columns:
        if applied.index.name == key:
            applied = applied.reset_index()
        elif isinstance(applied.index, pd.MultiIndex) and key in (applied.index.names or []):
            applied = applied.reset_index(level=key)

    return applied


def _marking_signature(marking: object, *, sort_tokens: bool = True) -> tuple[str, ...]:
    """Turn a Petri-net marking into a hashable token tuple.

    Called by
        ``PrefixesDataFrameLoader.remove_prefixes_with_unique_markings`` —
        so train/val/test markings can be compared as sets (order of tokens
        must not matter, tensors/arrays must be reduced to strings).

    Example
    -------
    Input
        marking = ["p3", "p1"]   (or a tensor / ndarray of the same tokens)

    Processing
        1. Unwrap tensor/ndarray → Python list.
        2. ``str`` every token; optionally sort.

    Output
        ``("p1", "p3")``  — usable as a dict/set key.
        ``None`` / NaN → ``("<NA>",)``.
    """
    if marking is None or (isinstance(marking, float) and np.isnan(marking)):
        return ("<NA>",)

    if isinstance(marking, torch.Tensor):
        marking = marking.detach().cpu().tolist()
    elif isinstance(marking, np.ndarray):
        marking = marking.tolist()

    # If it's already an iterable of tokens (most common case)
    if isinstance(marking, (list, tuple, set)):
        tokens = [str(t) for t in marking]
        if sort_tokens:
            tokens.sort()
        return tuple(tokens)

    # Fallback: treat as a single token
    return (str(marking),)


class RawDataFrameLoader:
    """
    Base class for loading raw event log data into a DataFrame.
    """

    def __init__(
        self,
        event_log_dir: str,
        timestamp_name: str,
        case_name: str,
        categorical_columns: list[str],
        continuous_columns: list[str],
        continuous_positive_columns: list[str],
        static_categorical_columns: Optional[list[str]] = None,
        static_continuous_columns: Optional[list[str]] = None,
        time_since_case_start_column: str | None = None,
        time_since_last_event_column: str | None = None,
        day_in_week_column: str | None = None,
        seconds_in_day_column: str | None = None,
        date_format: str = "%Y-%m-%d %H:%M:%S.%f",
        min_suffix_size: int = 1,
        **kwargs,
    ):
        """
        Initialize the raw data frame loader.
        """
        self.df = pd.read_csv(event_log_dir)

        self.case_name = case_name

        self.timestamp_name = timestamp_name

        self.time_since_case_start_column = time_since_case_start_column
        self.time_since_last_event_column = time_since_last_event_column
        self.day_in_week_column = day_in_week_column
        self.seconds_in_day_column = seconds_in_day_column
        self.date_format = date_format
        self.min_suffix_size = min_suffix_size

        # dynamic attributes
        self.categorical_columns = list(categorical_columns or [])
        self.continuous_columns = list(continuous_columns or [])
        # dynamic (log-normal) continuous attributes
        self.continuous_positive_columns = list(continuous_positive_columns or [])
        # static attributes
        self.static_categorical_columns = list(static_categorical_columns or [])
        self.static_continuous_columns = list(static_continuous_columns or [])

        self.df[self.timestamp_name] = pd.to_datetime(
            self.df[self.timestamp_name], format=date_format, errors="coerce"
        )


    # Extrahieren der Case Values. Das heißt jener Values, die sich nicht ändern für einen case
    @staticmethod
    def __extract_static_value(series: pd.Series) -> object:
        """
        Extract a static value from a series, ignoring NaNs and "EOS" values.
        """
        cleaned = series.dropna()
        if cleaned.empty:
            return np.nan
        if cleaned.dtype == object or cleaned.dtype.name == "category":
            cleaned = cleaned[cleaned != "EOS"]
            if cleaned.empty:
                return np.nan
        return cleaned.iloc[0]

    def create_case_level_dataframe(self, event_level_df: pd.DataFrame) -> pd.DataFrame:
        """Collapse one-row-per-event into one-row-per-case.

        Dynamic columns become lists (the trace). Static columns stay a single
        scalar (first non-NaN / non-``"EOS"`` value).

        Called by
            ``CSV2EventLog.__init__`` — builds ``self.raw_df`` right after the
            CSV is loaded (before EOS rows are appended). Notebooks then call
            ``PrefixesDataFrameLoader.get_raw_dataframe()`` to inspect that
            case-level view.

        Example
        -------
        Input (event-level, one case)::

            case_id  activity  amount  resource
            C1       A         10.0    Alice
            C1       B         20.0    Alice
            C1       C         30.0    Alice

        Processing
            group by case_id
            activity / amount → ``.tolist()``
            resource (static) → first value ``"Alice"``

        Output (one row)::

            case_id  activity       amount              resource
            C1       [A, B, C]      [10.0, 20.0, 30.0]  Alice
        """
        event_level_df = _ensure_group_key_column(event_level_df, self.case_name)
        grouped = event_level_df.groupby(self.case_name, sort=False)
        records = []
        for case_id, group in grouped:
            row = {self.case_name: case_id}
            for col in (
                self.categorical_columns
                + self.continuous_columns
                + self.continuous_positive_columns
            ):
                if col in group.columns:
                    row[col] = group[col].tolist()
            for col in self.static_categorical_columns:
                if col in group.columns:
                    row[col] = self.__extract_static_value(group[col])
            for col in self.static_continuous_columns:
                if col in group.columns:
                    row[col] = self.__extract_static_value(group[col])
            records.append(row)
        return pd.DataFrame(records)


class CSV2EventLog(RawDataFrameLoader):
    """
    Base class for loading event logs from CSV files.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the event log loader with additional processing.
        """
        # load raw constructor
        super().__init__(*args, **kwargs)

        # Time values
        # create new time since case started column if desired
        if self.time_since_case_start_column:
            self.__create_time_since_case_start_column()

        # create new offset time to last event column if desired
        if self.time_since_last_event_column:
            self.__create_time_since_last_event_column()

        # create new day in week column if desired
        if self.day_in_week_column:
            self.__create_day_in_week_column()

        # create new seconds in day column if desired
        if self.seconds_in_day_column:
            self.__create_seconds_in_day_column()

        # raw dataframe before split containing categorical and continuous attributes:
        # Reduced to only the selected attributes
        self.raw_df = self.create_case_level_dataframe(self.df.copy())

        applied = _groupby_apply_preserve_key(
            self.df,
            self.case_name,
            lambda group: self.__add_last_rows(group),
            sort=False,
        )
        self.df = applied.reset_index(drop=True)

        # Normalize categorical columns (dynamic + static) so integer-coded
        # categories stay stable (avoid "1.0" artifacts).
        for categorical_col in (self.categorical_columns + self.static_categorical_columns):
            if categorical_col in self.df.columns:
                self.df[categorical_col] = _normalize_categorical_series(
                    self.df[categorical_col]
                )

        for continuous_col in self.continuous_columns:
            self.df[continuous_col] = self.df[continuous_col].astype("float32")
        for continuous_col in self.continuous_positive_columns:
            self.df[continuous_col] = self.df[continuous_col].astype("float32")

    def __create_time_since_case_start_column(self):
        """
        Create a new column representing the time since the case started.
        """
        case_start_times = self.df.groupby(self.case_name)[
            self.timestamp_name
        ].transform("min")
        time_offset = self.df[self.timestamp_name] - case_start_times
        time_offset_seconds = time_offset.dt.total_seconds()
        self.df[self.time_since_case_start_column] = time_offset_seconds
        self.max_case_length = self.df.groupby(self.case_name).size().max()

    @staticmethod
    def __min_timestamp_before_event(group, timestamp_name, new_column_name):
        """
        Find the minimum timestamp before each event in the group.
        """
        min_values = []
        for _i, row in group.iterrows():
            before_values = group[(group[timestamp_name] < row[timestamp_name])][
                timestamp_name
            ]
            if not before_values.empty:
                min_values.append(before_values.max())
            else:
                min_values.append(np.nan)
        group[new_column_name] = min_values
        return group

    def __create_time_since_last_event_column(self):
        """
        Create a new column representing the time since the last event.
        """
        # Vectorized and dtype-stable implementation.
        # The old implementation used `np.nan` for missing previous timestamps,
        # which can upcast the column to object dtype and break datetime
        # subtraction (pandas then raises the non-vectorized dtype mismatch
        # error you saw).
        df = _ensure_group_key_column(self.df, self.case_name)

        # Keep original row order, but compute within-case diffs in time order.
        work = df.copy()
        work["_orig_row"] = np.arange(len(work))
        work = work.sort_values(
            [self.case_name, self.timestamp_name], kind="mergesort"
        )

        # Ensure timestamps are datetime-like before shifting/subtracting.
        # RawDataFrameLoader already parses this, but groupby/apply pipelines or
        # mixed inputs may still result in object dtype.
        work[self.timestamp_name] = pd.to_datetime(
            work[self.timestamp_name], errors="coerce"
        )

        # Match old semantics: previous strictly earlier timestamp within the
        # same case (not just the previous row). This matters when a case has
        # multiple events with identical timestamps.
        grouped = work.groupby(self.case_name, sort=False)[self.timestamp_name]

        # Identify blocks of equal timestamps per case (after sorting).
        prev_row_ts = grouped.shift(1)
        is_new_timestamp = work[self.timestamp_name].ne(prev_row_ts)
        ts_block = is_new_timestamp.astype(int).groupby(work[self.case_name], sort=False).cumsum()

        # For each (case, timestamp-block), get the block timestamp and the
        # previous block timestamp (strictly earlier by construction).
        block_ts = work.groupby([work[self.case_name], ts_block], sort=False)[
            self.timestamp_name
        ].first()
        prev_block_ts = block_ts.groupby(level=0, sort=False).shift(1)

        # Broadcast previous block timestamp back to each row.
        row_index = pd.MultiIndex.from_arrays([work[self.case_name], ts_block])
        prev_strict_ts = prev_block_ts.reindex(row_index).to_numpy()

        delta = work[self.timestamp_name] - prev_strict_ts
        # Keep the first (minimum-timestamp) events per case as NaN.
        work[self.time_since_last_event_column] = delta.dt.total_seconds()

        work = work.sort_values("_orig_row", kind="mergesort").drop(
            columns=["_orig_row"]
        )
        self.df = work

    def __create_day_in_week_column(self):
        """
        Create a new column representing the day of the week.

        0 = Monday, 6 = Sunday
        """
        self.df[self.day_in_week_column] = self.df[self.timestamp_name].dt.weekday

    def __create_seconds_in_day_column(self):
        """
        Create a new column.

        The column representing the number of seconds elapsed since
        the start of the day.
        """
        self.df[self.seconds_in_day_column] = (
            self.df[self.timestamp_name].dt.hour * 3600
            + self.df[self.timestamp_name].dt.minute * 60
            + self.df[self.timestamp_name].dt.second
        )

    def __add_last_rows(self, group):
        """
        Adds EOS rows to each case in the event log dataframe.
        """
        new_row = {}
        for col in group.columns:
            if col == self.case_name:
                new_row[col] = group.name
            elif col in self.categorical_columns:
                new_row[col] = "EOS"
            elif col in self.static_categorical_columns:
                # Static categoricals should not receive EOS; keep them missing on
                # the appended rows so per-case extraction uses the original value.
                new_row[col] = np.nan
            elif group[col].dtype == "object" or group[col].dtype.name == "category":
                new_row[col] = "EOS"

        # Adds everywhere min_suffix_size
        max_case_len = self.min_suffix_size
        # Adds new rows with EOS:
        eos_rows = pd.DataFrame(max_case_len * [new_row])
        # concat standard dataframe
        concat_case = pd.concat([group.sort_values(by=self.timestamp_name), eos_rows])
        return concat_case


class EventLogSplitter:
    """
    Split event log into train, train_validation, and test_validation sets.
    """

    def __init__(
        self, train_validation_size: float, test_validation_size: float, **kwargs
    ):
        """
        Initialize the splitter with train/validation and test/validation sizes.
        """
        self.train_validation_size = train_validation_size
        self.test_validation_size = test_validation_size

    def split(self, event_log: CSV2EventLog):
        """
        Split the event log into train, train_validation, and test_validation sets.
        """
        # Ensure we shuffle a plain NumPy array (not a pandas StringArray),
        # otherwise NumPy warns that shuffle may behave unexpectedly.
        cases = np.asarray(event_log.df[event_log.case_name].unique(), dtype=object)
        np.random.shuffle(cases)

        train_validation_ix = int(self.train_validation_size * len(cases))
        test_validation_ix = train_validation_ix + int(
            self.test_validation_size * len(cases)
        )

        train_validation_cases = cases[:train_validation_ix]
        test_validation_cases = cases[train_validation_ix:test_validation_ix]
        train_cases = cases[test_validation_ix:]

        train_df = event_log.df[event_log.df[event_log.case_name].isin(train_cases)]
        train_validation_df = event_log.df[
            event_log.df[event_log.case_name].isin(train_validation_cases)
        ]
        test_validation_df = event_log.df[
            event_log.df[event_log.case_name].isin(test_validation_cases)
        ]

        return train_df, train_validation_df, test_validation_df


class PositiveStandardizer_normed(BaseEstimator, TransformerMixin):
    """
    Standard scaler for log normal attributes. Das ist meines verstädnisses nach für strikt positive werte, da
    wir sonst für zeitwerte einen starken skew hätten => Schlecht für 
    """

    def __init__(self):
        """
        Initialize the standardizer.
        """
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        """
        Fit the standardizer by computing mean and std of log-transformed data.
        """
        print("Positive Standardization")
        log_x = np.log1p(X)
        print("min,25%,50%,75%,max:", np.percentile(log_x, [0, 25, 50, 75, 100]))
        # Standardize values
        self.mean_ = np.mean(log_x, axis=0)
        print("Mean: ", self.mean_)
        self.std_ = np.std(log_x, axis=0)
        print("Std: ", self.std_)
        return self

    def transform(self, X):
        """
        Transform the data by applying log transformation and standardization.
        """
        # log the observations to assume normal PDF
        log_x = np.log1p(X)
        x_enc = (log_x - self.mean_) / self.std_
        return x_enc

    def inverse_transform(self, X_enc):
        """
        Inverse transform the encoded data back to the original scale.
        """
        # Destandardization
        log_x = X_enc * self.std_ + self.mean_
        # Exponentiation:
        x = np.expm1(log_x)
        return x


class TensorEncoderDecoder:
    """
    Responsible for tensor encoding of event log data.
    """

    def __init__(
        self,
        event_log: pd.DataFrame,
        case_name: str,
        concept_name: str,
        window_size: int,
        min_suffix_size: int,
        categorical_columns: Optional[list[str]] = None,
        continuous_columns: Optional[list[str]] = None,
        continuous_positive_columns: Optional[list[str]] = None,
        static_categorical_columns: Optional[list[str]] = None,
        static_continuous_columns: Optional[list[str]] = None,
        **kwargs,
    ):
        """Pick encoders from five feature lists (not one generic 'numeric' list).

        Called by
            ``EventLogLoader.__init__`` — unpacks the same keys from
            ``event_log_properties`` (set in the loader notebooks). The
            encoder/decoder LSTM then treats dynamic vs static and
            categorical vs numeric differently.

        Why five lists
            Two axes, plus a special numeric transform:

            **When does the value change?**
            - *dynamic* (``categorical_columns``, ``continuous_*``):
              one value **per event** → sequence of length ``window_size``, # also für werte, die pro case sich verändern innerhalb eines cases
              embeddings/LSTM timesteps, predicted in the suffix.
            - *static* (``static_*``): one value **per case** (resource type, # Wert der sich nicht ändern für einen case
              variant, …) → a single vector, merged into the encoder state,
              **not** a time series and not predicted as suffix tokens.

            **What is the value?**
            - *categorical*: discrete labels → ``OrdinalEncoder`` (+1 so
              ``0`` is padding) → embedding table.
            - *continuous*: real numbers → mean-impute + ``StandardScaler``.
            - *continuous_positive*: same, but ``log1p`` first
              (``PositiveStandardizer_normed``). Use this for durations /
              elapsed times: always ≥ 0, heavy right tail. Raw seconds
              would let a few long waits dominate the scaler.

        Example (Helpdesk-style)::

            categorical_columns          = [Activity, Resource]
                C1: [Register, Resolve]          ← changes along the trace
            continuous_columns           = [case_elapsed_time, day_in_week]
                C1: [0.0, 3600.0] / [0, 0]       ← per event, roughly symmetric
            continuous_positive_columns  = [event_elapsed_time]   # if log-scaled
                C1: [0.0, 3600.0]                ← ≥ 0, skewed → log1p
            static_categorical_columns   = [VariantIndex]
                C1: 3  on every event            ← one embedding for the case
            static_continuous_columns    = []    # e.g. case amount if you had one

        Empty lists are fine; they just skip that encoder path.
        """
        self.event_log = event_log
        self.case_name = case_name
        self.concept_name = concept_name
        self.min_suffix_size = min_suffix_size
        # Maximal betrachtete Länge der Sequenz
        if window_size == "auto":
            # get max. length of (100-1.5)% of the longest cases as prefix
            # and add the min. suffix_size
            event_log_df = _ensure_group_key_column(self.event_log, case_name)
            case_sizes = event_log_df.groupby(case_name).size()
            self.window_size = (
                round(case_sizes.quantile(1 - 0.015)) + self.min_suffix_size
            )
        else:
            self.window_size = window_size
        self.categorical_columns = list(categorical_columns or [])
        self.continuous_columns = list(continuous_columns or [])
        self.continuous_positive_columns = list(continuous_positive_columns or [])
        self.static_categorical_columns = list(static_categorical_columns or [])
        self.static_continuous_columns = list(static_continuous_columns or [])

        # --- unfitted sklearn transformers, one per column name ----------------
        # This loop only *constructs* objects. Fitting happens later in
        # ``train_imputers_encoders`` (called by ``EventLogLoader.__init__``).
        # encode_df then looks them up by column name — hence dicts, not lists.
        #
        # Dynamic + static of the same type share the same transformer class:
        # the difference is *when* the value is used (sequence vs case vector),
        # not how it is scaled.
        #
        # Example after this block (still unfitted):
        #   categorical_encoders["Activity"]  → OrdinalEncoder(unknown=-1)
        #   continuous_imputers["amount"]     → SimpleImputer(mean)
        #   continuous_encoders["amount"]     → StandardScaler
        #   continuous_encoders["elapsed"]    → PositiveStandardizer (log1p)
        #
        # Missing categoricals: no imputer. OrdinalEncoder maps NaN / unknown
        # to -1, then encode_* adds +1 so padding can stay 0.
        # ``categorical_imputers`` is unused leftover.

        self.categorical_imputers: dict[str, SimpleImputer] = {}
        self.categorical_encoders: dict[str, sklearn.preprocessing.OrdinalEncoder] = {}
        for categorical_column in (
            self.categorical_columns + self.static_categorical_columns
        ):
            if categorical_column not in self.categorical_encoders:
                # labels → 0..K-1  (later +1). unknown/NaN → -1
                self.categorical_encoders[categorical_column] = (
                    self.__get_categorical_encoder()
                )

        self.continuous_imputers = {}
        self.continuous_encoders: dict[str, sklearn.preprocessing.StandardScaler] = {}

        # unbounded / roughly symmetric numerics (e.g. day_in_week)
        for continuous_column in (
            self.continuous_columns + self.static_continuous_columns
        ):
            if continuous_column not in self.continuous_imputers:
                self.continuous_imputers[continuous_column] = (
                    self.__get_continuous_imputer()
                )
                self.continuous_encoders[continuous_column] = (
                    self.__get_continuous_encoder()
                )

        # durations ≥ 0, heavy tail: same imputer, but log1p then scale
        for continuous_positive_column in self.continuous_positive_columns:
            self.continuous_imputers[continuous_positive_column] = (
                self.__get_continuous_positive_imputer()
            )
            self.continuous_encoders[continuous_positive_column] = (
                self.__get_continuous_positive_encoder()
            )

    def train_imputers_encoders(self):
        """Fit the transformers built in ``__init__`` on the train event log.

        Called by
            ``EventLogLoader.__init__`` — immediately after constructing
            ``TensorEncoderDecoder(self.train_df, ...)``, so val/test are
            encoded with train statistics only (no leakage).

        Input
            unfitted dicts from ``__init__`` + ``self.event_log`` (train split)

        Processing
            categorical: fit ``OrdinalEncoder`` on the column (static cols
            drop ``"EOS"`` so EOS is not a static class).
            continuous: ``imputer.fit_transform`` then ``encoder.fit``
            (mean/std, or log1p-mean/std for positive cols).

        Output
            same dicts, now fitted. ``encode_df`` / ``decode_event`` call
            ``.transform`` / ``.inverse_transform`` on them.
        """
        # categorical encoders: fit on 2D numpy arrays with dtype=object
        for col, categorical_encoder in self.categorical_encoders.items():
            column_series = self.event_log[col].astype(object)

            if col in self.static_categorical_columns:
                column_series = column_series[column_series != "EOS"]

            column_data = column_series.to_numpy().reshape(-1, 1)
            categorical_encoder.fit(column_data)

        # continuous encoders / imputers: fit on 2D numpy arrays (n_samples, 1)
        for col, continuous_encoder in self.continuous_encoders.items():
            continuous_imputer = self.continuous_imputers[col]
            column_data = self.event_log[[col]].to_numpy()  # DataFrame -> ndarray (n,1)
            column_data = continuous_imputer.fit_transform(column_data)  # still (n,1)
            continuous_encoder.fit(
                column_data
            )  # StandardScaler or custom transformer expects 2D

    # Aufgerufen von encode_case, um eine einzelne kategorielle Variable zu encoden
    # PRO CASE
    def _single_encode_categorical_column(
        self, df_case: pd.DataFrame, col: str
    ) -> torch.Tensor:
        case_values = np.array(df_case[[col]], dtype=object)
        # case_values =
        # [
        #     ["Create Application"],
        #     ["Delete Application"],
        #     ["Assign Developer"]
        # ]

        #OrdinalEncoder encoded nach ...
        case_values_enc = (
            self.categorical_encoders[col].transform(case_values) + 1
        )  # shape (n,1)
        # case_values_enc =
        # [
        #     [1],
        #     [4],
        #     [7]
        # ]
        # Pad encodings - clearer prefix loop (prefix_len from min_suffix_size .. len)
        case_values_enc_pad = self.pad_to_window_size(case_values_enc)
        return (
            torch.tensor(np.array(case_values_enc_pad, dtype=int), dtype=torch.long)
            .squeeze(-1)
            .unsqueeze(0)
        )

        # Padden und unsqueeze:
        #  tensor([
        #     [0, 0, 1, 4, 7]
        # ])

    # Aufgerufen von encode_case, um eine einzelne kontinuierliche Variable zu encoden
    # PRO CASE
    def _single_encode_continuous_column(
        self, df_case: pd.DataFrame, col: str
    ) -> torch.Tensor:

        case_values = df_case[[col]].values  # shape (n,1)
        # z. B. col = "elapsed_time"
        # case_values =
        # [
        #     [10.0],
        #     [25.0],
        #     [40.0]
        # ]

        # Fehlende Werte werden mit dem bereits trainierten Imputer ersetzt
        case_values_imputed = self.continuous_imputers[col].transform(case_values)
        # z. B.
        # [
        #     [10.0],
        #     [25.0],
        #     [40.0]
        # ]

        # Numerische Werte werden mit dem bereits trainierten Encoder/Scaler skaliert
        case_values_enc = self.continuous_encoders[col].transform(
            case_values_imputed
        )
        # z. B.
        # [
        #     [0.20],
        #     [0.55],
        #     [0.80]
        # ]

        # Padding auf window_size, z. B. window_size = 5
        case_values_enc_pad = self.pad_to_window_size(case_values_enc)
        # [
        #     [0.0],
        #     [0.0],
        #     [0.20],
        #     [0.55],
        #     [0.80]
        # ]

        return (
            torch.tensor(
                np.array(case_values_enc_pad, dtype=float),
                dtype=torch.float,
            )
            .squeeze(-1)
            .unsqueeze(0)
        )

        # Nach squeeze(-1) und unsqueeze(0):
        #
        # tensor([
        #     [0.0, 0.0, 0.20, 0.55, 0.80]
        # ])
        #
        # Shape: [1, window_size]

    def encode_case(
        self, df_case: pd.DataFrame
    ) -> tuple[
        tuple[torch.Tensor],
        tuple[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        df_case = df_case.copy()

        for col in self.categorical_columns:
            if col not in df_case:
                df_case[col] = np.nan
        for col in self.continuous_columns + self.continuous_positive_columns:
            if col not in df_case:
                df_case[col] = np.nan

        categorical_tensors = []
        continuous_tensors = []

        for col in self.categorical_columns:
            cat_columns = self._single_encode_categorical_column(df_case, col)
            categorical_tensors.append(cat_columns)
        # Beispiel:
        # activity:
        # ["Create Application",
        #  "Delete Application",
        #  "Assign Developer"]
        # -> tensor([[0, 0, 1, 4, 7]]) Wird appended an die categorical_tensors tensoren
        # bei window_size = 5

        # Beispiel nach der Schleife:
        # categorical_tensors =
        # [
        #     tensor([[0, 0, 1, 4, 7]]),     # activity
        #     tensor([[0, 0, 12, 5, 9]]),    # resource
        #     tensor([[0, 0, 2, 2, 3]])      # department
        # ]
        # Jeder Tensor:
        # Shape [1, window_size]
        # Jede numerische Spalte desselben Cases
        # wird ebenfalls separat encodiert.
        # Beispiel elapsed_time:
        # [10.0, 25.0, 40.0]
        # -> tensor([[0.0, 0.0, 0.20, 0.55, 0.80]])

        for col in self.continuous_columns + self.continuous_positive_columns:
            cont_columns = self._single_encode_continuous_column(df_case, col)
            continuous_tensors.append(cont_columns)
        # Jede numerische Spalte desselben Cases
        # wird ebenfalls separat encodiert.
        # Beispiel elapsed_time:
        # [10.0, 25.0, 40.0]
        # -> tensor([[0.0, 0.0, 0.20, 0.55, 0.80]])

        # Beispiel:
        # continuous_tensors =
        # [
        #     tensor([[0.0, 0.0, 0.20, 0.55, 0.80]]),  # elapsed_time
        #     tensor([[0.0, 0.0, 0.10, 0.35, 0.25]])   # time_since_last_event
        # ]

        # static attributes
        case_id = df_case[self.case_name].iloc[0] if not df_case.empty else None
        static_cat_tensor, static_cont_tensor = self._encode_static_attributes(
            df_case, [case_id] if case_id is not None else []
        )

        # Beispiele:
        #   customer_type = "Premium"
        #   country       = "Germany"
        #   case_amount   = 5000

        # Beispiel:
        # static_cat_tensor =
        # tensor([[2, 5]])
        # static_cont_tensor =
        # tensor([[0.73]])

        seq_len = min(len(df_case), self.window_size)
        zero_mask = torch.zeros((1, self.window_size), dtype=torch.float32)
        if seq_len > 0:
            zero_mask[:, -seq_len:] = 1.0
        # Beispiel:
        # zero_mask = tensor([[0, 0, 0, 0, 0]])

        return (
            tuple(categorical_tensors),
            tuple(continuous_tensors),
            static_cat_tensor,
            static_cont_tensor,
            zero_mask,
        )
        # Bei seq_len = 3 und window_size = 5:
        # tensor([[0, 0, 1, 1, 1]])
        # Bedeutung:
        # [0, 0, 1, 1, 1]
        #  |  |  |  |  |
        #  |  |  +--+--+--> echte Events
        #  +--+------------> Padding

    def encode_df(
        self, df
    ) -> tuple[
        dict[str, object],
        tuple[list[tuple[str, int, dict[str, int]]]],
        tuple[list[tuple[str, int, dict[str, int]]]],
    ]:
        categorical_tensors = []
        all_categories = [[], []]
        static_categories = [[], []]
        eos_padding_tensor = None
        zero_padding_tensor = None
        case_ids = None

        for col in tqdm(self.categorical_columns, desc="categorical tensors"):
            if col == self.concept_name:
                (
                    case_ids,
                    enc_column,
                    eos_padding_tensor,
                    zero_padding_tensor,
                    categories,
                    max_classes,
                ) = self.encode_categorical_column(
                    df, col, return_case_ids_and_eos_paddings=True
                )
            else:
                enc_column, categories, max_classes = self.encode_categorical_column(
                    df, col
                )
            categorical_tensors.append(enc_column)
            all_categories[0].append((col, max_classes, categories))

        if (
            case_ids is None
            or eos_padding_tensor is None
            or zero_padding_tensor is None
        ):
            raise ValueError(
                "Concept column must be part of the categorical_columns to"
                "compute padding metadata."
            )

        continuous_tensors = []
        for col in tqdm(
            self.continuous_columns + self.continuous_positive_columns,
            desc="continouous tensors",
        ):
            continuous_tensors.append(self.encode_continuous_column(df, col))
            all_categories[1].append((col, 1, {}))

        for col in self.static_categorical_columns:
            filtered_categories = [
                category
                for category in self.categorical_encoders[col].categories_[0]
                if category != "EOS" and not pd.isna(category)
            ]
            categories = {
                category: idx + 1 for idx, category in enumerate(filtered_categories)
            }
            max_classes = len(filtered_categories) + 1
            static_categories[0].append((col, max_classes, categories))
        for col in self.static_continuous_columns:
            static_categories[1].append((col, 1, {}))

        static_cat_tensor, static_cont_tensor = self._encode_static_attributes(
            df, case_ids
        )

        tensor_bundle = {
            "categorical": categorical_tensors,
            "continuous": continuous_tensors,
            "eos_padding": eos_padding_tensor,
            "zero_padding": zero_padding_tensor,
            "case_ids": tuple(case_ids),
            "static_categorical": static_cat_tensor,
            "static_continuous": static_cont_tensor,
        }

        return tensor_bundle, tuple(all_categories), tuple(static_categories)

    def encode_categorical_column(
        self, df, col, return_case_ids_and_eos_paddings=False
    ):
        df = _ensure_group_key_column(df, self.case_name)
        grouped = df.groupby(self.case_name)
        windows = []
        eos_masks = []
        zero_masks = []
        categories = {
            category: idx + 1
            for idx, category in enumerate(
                self.categorical_encoders[col].categories_[0]
            )
        }
        # z.B.
        #
        # categories =
        # {
        #     "Create Application": 1,
        #     "Delete Application": 4,
        #     "Assign Developer": 7,
        #     "Approve Application": 8,
        #     "EOS": 9
        # }
        eos_token_id = categories.get("EOS", 0)
        # z.B.
        #
        # eos_token_id = 9

        case_ids = []
        for case_id, group in tqdm(grouped, desc=col, leave=False):
            case_values = np.array(group[[col]], dtype=object)
            case_values_enc = (
                self.categorical_encoders[col].transform(case_values) + 1
            )  # shape (n,1)
            # Beispiel:
            # case_values_enc =
            # [
            #     [1],
            #     [4],
            #     [7],
            #     [9]
            # ]
            padded_encodings = []

            for prefix_len in range(self.min_suffix_size + 1, len(case_values_enc) + 1):
                # bei min_suffix_size = 1
                # und Case-Länge = 4:
                # prefix_len = 2
                # prefix_len = 3
                # prefix_len = 4
                padded_slice = self.pad_to_window_size(case_values_enc[:prefix_len])
                # prefix_len = 2
                # case_values_enc[:2]
                # [
                #     [1],
                #     [4]
                # ]
                # nach Padding auf window_size = 5:
                # [
                #     [0],
                #     [0],
                #     [0],
                #     [1],
                #     [4]
                # ]
                #
                # => [0, 0, 0, 1, 4]

                padded_encodings.append(padded_slice)
                if return_case_ids_and_eos_paddings:
                    case_ids.append(case_id)
                    # Beispiel:
                    # case_ids =
                    # [
                    #     "Case_1",
                    #     "Case_1",
                    #     "Case_1"
                    # ]
                    flattened = np.array(padded_slice, dtype=int).squeeze(-1)
                    # z.B. flattened = [0, 1, 4, 7, 9]
                    eos_mask = np.ones_like(flattened, dtype=float)
                    # zunächst: [1, 1, 1, 1, 1]
                    if eos_token_id and eos_token_id > 0:
                        eos_positions = np.flatnonzero(flattened == eos_token_id)
                         # bei:
                         # [0, 1, 4, 7, 9]
                         # EOS liegt bei Index 4
                        if eos_positions.size > 0:
                            first_eos_idx = int(eos_positions[0])
                            eos_mask[first_eos_idx + 1 :] = 0.0
                    # Beispiel ohne Werte nach EOS:
                    # [0, 1, 4, 7, 9]
                    # eos_mask = [1, 1, 1, 1, 1]
                    # Falls z.B.
                    # [1, 4, 9, 9, 9]
                    # dann:
                    # eos_mask =
                    # [1, 1, 1, 0, 0]
                    zero_mask = np.zeros_like(flattened, dtype=float)
                    non_zero_positions = np.flatnonzero(flattened != 0)
                    if non_zero_positions.size > 0:
                        first_valid_idx = int(non_zero_positions[0])
                        zero_mask[first_valid_idx:] = 1.0
                         # Beispiel:
                    # Prefix
                    # [0, 0, 1, 4, 7]
                    # zero_mask:
                    # [0, 0, 1, 1, 1]
                    # 0 = Padding
                    # 1 = echtes Event
                    eos_masks.append(eos_mask.tolist())
                    zero_masks.append(zero_mask.tolist())
            windows.extend(padded_encodings)

        # Beispiel:
        # Case_1 erzeugt:
        # [0, 0, 0, 1, 4]
        # [0, 0, 1, 4, 7]
        # [0, 1, 4, 7, 9]
        #
        # Case_2 mit [1,8,9] erzeugt:
        # [0, 0, 0, 1, 8]
        # [0, 0, 1, 8, 9]
        #
        # ....
   

        if len(windows) == 0:
            # avoid creating empty numpy array with ambiguous dtype
            padded_array = np.zeros((0, self.window_size), dtype=int)
        else:
            padded_array = np.array(windows, dtype=int)
        t = torch.tensor(padded_array, dtype=torch.long)

        # t ungefähr:
        # tensor([
        #     [[0],[0],[0],[1],[4]],
        #     [[0],[0],[1],[4],[7]],
        #     [[0],[1],[4],[7],[9]],
        #     [[0],[0],[0],[1],[8]],
        #     [[0],[0],[1],[8],[9]]
        # ])

        # shape ungefähr:
        # [5, 5, 1]

        max_classes = len(self.categorical_encoders[col].categories_[0]) + 1
        if return_case_ids_and_eos_paddings:
            if len(eos_masks) == 0:
                eos_padded_array = np.zeros((0, self.window_size), dtype=float)
                zero_padded_array = np.zeros((0, self.window_size), dtype=float)
            else:
                eos_padded_array = np.array(eos_masks, dtype=float)
                zero_padded_array = np.array(zero_masks, dtype=float)

            eos_padded_tensor = torch.tensor(eos_padded_array, dtype=torch.float32)
            zero_padded_tensor = torch.tensor(zero_padded_array, dtype=torch.float32)
            return (
                case_ids,
                t.squeeze(-1),
                eos_padded_tensor,
                zero_padded_tensor,
                categories,
                max_classes,
            )
        else:
            return t.squeeze(-1), categories, max_classes
        
        # Prefix 1    [0, 0, 0, 1, 4]   ← Case_1
        # Prefix 2    [0, 0, 1, 4, 7]   ← Case_1
        # Prefix 3    [0, 1, 4, 7, 9]   ← Case_1

        # Prefix 4    [0, 0, 0, 1, 8]   ← Case_2
        # Prefix 5    [0, 0, 1, 8, 9]   ← Case_2

        #              ↑
        #              num_prefixes


    def encode_continuous_column(self, df, col):
        """Impute, scale, and prefix-window one continuous column.

        Same prefix loop as ``encode_categorical_column``:
        ``prefix_len = min_suffix_size + 1 … n``, then left-pad with ``0.0``.
        EOS / missing values are filled by the fitted imputer **before**
        scaling (so the last prefix usually has a real number, not NaN).

        Called by
            ``TensorEncoderDecoder.encode_df`` — once per column in
            ``continuous_columns + continuous_positive_columns``.
            ``EventLogLoader.get_dataset`` is the public caller of
            ``encode_df``. No EOS/zero masks here; those come from the
            concept column.

        Example
        -------
        Settings: ``window_size = 4``, ``min_suffix_size = 1``.
        Fitted scaler: mean=15, std=5. Imputer fills NaN with 15.

        Input (one case)::

            amount = [10.0, 20.0, NaN]   n = 3
            after impute + scale ≈ [-1.0, 1.0, 0.0]
            prefix_len ∈ {2, 3}

        Processing
            prefix 2: [-1.0, 1.0]      pad → [0.0, 0.0, -1.0, 1.0]
            prefix 3: [-1.0, 1.0, 0.0] pad → [0.0, -1.0, 1.0, 0.0]

        Output
            ``torch.float32`` tensor of shape ``(2, 4)`` (one row per prefix).
        """
        df = _ensure_group_key_column(df, self.case_name)
        grouped = df.groupby(self.case_name)
        windows = []
        for _case_id, group in tqdm(grouped, desc=col, leave=False):
            case_values = group[[col]].values  # shape (n,1)
            case_values_imputed = self.continuous_imputers[col].transform(case_values)
            case_values_enc = self.continuous_encoders[col].transform(
                case_values_imputed
            )
            padded_encodings = []

            # check
            for prefix_len in range(self.min_suffix_size + 1, len(case_values_enc) + 1):
                padded_encodings.append(
                    self.pad_to_window_size(case_values_enc[:prefix_len])
                )

            windows.extend(padded_encodings)

        if len(windows) == 0:
            padded_array = np.zeros((0, self.window_size), dtype=float)
        else:
            padded_array = np.array(windows, dtype=float)
        t = torch.tensor(padded_array, dtype=torch.float32)
        return t.squeeze(-1)
        #     tensor([
        # [0.0, 0.0, 0.0, 0.20, 0.55],   # Prefix Länge 2
        # [0.0, 0.0, 0.20, 0.55, 0.80],   # Prefix Länge 3
        # [0.0, 0.20, 0.55, 0.80, 1.10],  # Prefix Länge 4
        # ])

    def pad_to_window_size(self, previous_values):
        """
        Pad or truncate the previous values to match the window size.
        """
        prev_list = np.asarray(previous_values).tolist()
        if len(prev_list) > self.window_size:
            return prev_list[-self.window_size :]
        else:
            pad_count = self.window_size - len(prev_list)
            # use 0.0 for continuous;
            # for categorical it will be cast to int later when dtype=int
            return [[0.0]] * pad_count + prev_list

    def _encode_static_attributes(
        self, df: pd.DataFrame, case_ids: list[object]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ordered_case_ids = list(case_ids)
        num_samples = len(ordered_case_ids)
        if not num_samples:
            return (
                torch.zeros(
                    (0, len(self.static_categorical_columns)), dtype=torch.long
                ),
                torch.zeros(
                    (0, len(self.static_continuous_columns)), dtype=torch.float32
                ),
            )

        if not self.static_categorical_columns and not self.static_continuous_columns:
            return (
                torch.zeros((num_samples, 0), dtype=torch.long),
                torch.zeros((num_samples, 0), dtype=torch.float32),
            )

        case_static_values = self._collect_static_case_values(df)

        if self.static_categorical_columns:
            cat_rows = []
            for case_id in tqdm(
                ordered_case_ids, desc="static categorical", leave=False
            ):
                row = []
                case_record = case_static_values.get(case_id, {})
                for col in self.static_categorical_columns:
                    value = case_record.get(col, np.nan)
                    if value == "EOS" or pd.isna(value):
                        row.append(0)
                        continue
                    value_arr = np.array([[value]], dtype=object)
                    encoded_value = (
                        self.categorical_encoders[col].transform(value_arr) + 1
                    )
                    row.append(int(encoded_value.squeeze()))
                cat_rows.append(row)
            static_cat_tensor = torch.tensor(cat_rows, dtype=torch.long)
        else:
            static_cat_tensor = torch.zeros((num_samples, 0), dtype=torch.long)

        if self.static_continuous_columns:
            cont_rows = []
            for case_id in tqdm(
                ordered_case_ids, desc="static continuous", leave=False
            ):
                row = []
                case_record = case_static_values.get(case_id, {})
                for col in self.static_continuous_columns:
                    value = case_record.get(col, np.nan)
                    value_arr = np.array([[value]], dtype=float)
                    imputed = self.continuous_imputers[col].transform(value_arr)
                    encoded_value = self.continuous_encoders[col].transform(imputed)
                    row.append(float(encoded_value.squeeze()))
                cont_rows.append(row)
            static_cont_tensor = torch.tensor(cont_rows, dtype=torch.float32)
        else:
            static_cont_tensor = torch.zeros((num_samples, 0), dtype=torch.float32)

        return static_cat_tensor, static_cont_tensor
        # return (
        #     tensor([
        #         [2, 5],   # Case_1: customer_type=2, country=5
        #         [1, 3],   # Case_2
        #         [4, 2],   # Case_3
        #     ]),

        #     tensor([
        #         [0.73],   # Case_1: case_amount
        #         [0.41],   # Case_2
        #         [0.88],   # Case_3
        #     ])
        # )

    def _collect_static_case_values(
        self, df: pd.DataFrame
    ) -> dict[str, dict[str, object]]:
        df = _ensure_group_key_column(df, self.case_name)
        grouped = df.groupby(self.case_name, sort=False)
        case_values: dict[str, dict[str, object]] = {}
        for case_id, group in grouped:
            record: dict[str, object] = {}
            for col in self.static_categorical_columns:
                if col in group.columns:
                    record[col] = self.__extract_static_value(group[col])
                else:
                    record[col] = np.nan
            for col in self.static_continuous_columns:
                if col in group.columns:
                    record[col] = self.__extract_static_value(group[col])
                else:
                    record[col] = np.nan
            case_values[case_id] = record
        return case_values

    @staticmethod
    def __extract_static_value(series: pd.Series) -> object:
        cleaned = series.dropna()
        if cleaned.empty:
            return np.nan
        return cleaned.iloc[0]

    # einfach wieder von dem tensor zurück in die original werte
    # z.B.
    #
    # return pd.DataFrame({
    #     "activity": [
    #         np.nan,                 # Padding 0
    #         np.nan,                 # Padding 0
    #         "Create Application",
    #         "Delete Application",
    #         "Assign Developer",
    #     ],

    #     "resource": [
    #         np.nan,
    #         np.nan,
    #         "Alice",
    #         "Bob",
    #         "Charlie",
    #     ],

    #     "elapsed_time": [
    #         0.0,    # nach inverse_transform ggf. ursprünglicher Wert
    #         0.0,
    #         10.0,
    #         25.0,
    #         40.0,
    #     ],

    #     "case_id": [
    #         "Case_42",
    #         "Case_42",
    #         "Case_42",
    #         "Case_42",
    #         "Case_42",
    #     ],
    # })
    def decode_event(self, event_tuple: tuple):
        cat, cont, *_, case_id = event_tuple
        decoded_event = {}
        for i, col in enumerate(self.categorical_columns):
            enc_col = cat[i].unsqueeze(-1).numpy()
            if col in self.categorical_encoders:
                categories = self.categorical_encoders[col].categories_[0]
                dec_col = np.array(
                    [
                        (
                            categories[idx - 1]
                            if idx > 0 and idx <= len(categories)
                            else np.nan
                        )
                        for idx in enc_col.flatten()
                    ]
                )
            else:
                dec_col = enc_col
            decoded_event[col] = dec_col.tolist()
        for i, col in enumerate(
            self.continuous_columns + self.continuous_positive_columns
        ):
            enc_col = cont[i].unsqueeze(-1).numpy()
            if col in self.continuous_encoders:
                dec_col = self.continuous_encoders[col].inverse_transform(enc_col)
            else:
                dec_col = enc_col
            decoded_event[col] = dec_col.flatten().tolist()
        decoded_event[self.case_name] = [case_id] * len(
            decoded_event[self.categorical_columns[0]]
        )
        return pd.DataFrame(decoded_event)

    def __get_continuous_imputer(self):
        """
        Get the imputer for continuous variables.
        """
        return SimpleImputer(strategy="mean")

    def __get_categorical_encoder(self):
        """
        Get the encoder for categorical variables.
        """
        return sklearn.preprocessing.OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )

    def __get_continuous_encoder(self):
        """
        Get the encoder for continuous variables.
        """
        return sklearn.preprocessing.StandardScaler()

    def __get_continuous_positive_imputer(self):
        """
        Get the imputer for continuous positive variables.
        """
        return SimpleImputer(strategy="mean")

    def __get_continuous_positive_encoder(self):
        """
        Get the encoder for continuous positive variables.
        """
        standardizer = PositiveStandardizer_normed()
        return standardizer


class PrefixesDataFrameLoader:
    """
    Loader that creates dataframes of prefixes from event log data.

    Class that provides intermediate steps:
    create dataframe of prefixes for marking determination:
    """

    def __init__(self, event_log_location: str, event_log_properties: Dict[str, Any]):
        """
        Initialize the PrefixesDataFrameLoader.
        """
        if not event_log_properties:
            raise ValueError("event_log_properties are required")
        self.event_log_properties = event_log_properties

        self.case_name = event_log_properties["case_name"]
        self.min_suffix_size = event_log_properties.get("min_suffix_size", 1)
        self.window_size_setting = event_log_properties.get("window_size", "auto")

        self.categorical_columns = list(
            event_log_properties.get("categorical_columns") or []
        )
        self.continuous_columns = list(
            event_log_properties.get("continuous_columns") or []
        )
        self.continuous_positive_columns = list(
            event_log_properties.get("continuous_positive_columns") or []
        )

        self.static_categorical_columns = list(
            event_log_properties.get("static_categorical_columns") or []
        )
        self.static_continuous_columns = list(
            event_log_properties.get("static_continuous_columns") or []
        )

        # create processed event log with EOS rows and engineered columns
        self.csv2event_log = CSV2EventLog(event_log_location, **event_log_properties)
        # dataframe from CSV2EventLog
        self.event_log = self.csv2event_log.df.copy()

        # configure window size so we can trim sequences later on
        self.window_size = self._resolve_window_size()

        # splitting initialization
        train_size = event_log_properties.get("train_validation_size")
        test_size = event_log_properties.get("test_validation_size")
        splitter = EventLogSplitter(
            train_validation_size=train_size, test_validation_size=test_size
        )

        train_df, val_df, test_df = splitter.split(self.csv2event_log)

        self.processed_splits: Dict[str, pd.DataFrame] = {
            "train": train_df.reset_index(drop=True).copy(),
            "val": val_df.reset_index(drop=True).copy(),
            "test": test_df.reset_index(drop=True).copy(),
        }
        self.train_df = self.processed_splits["train"]
        self.val_df = self.processed_splits["val"]
        self.test_df = self.processed_splits["test"]

    def get_raw_dataframe(self) -> pd.DataFrame:
        """
        Return the raw full dataframe.

        The dataframe contains dynamic columns
        as lists and static columns as values.
        """
        return self.csv2event_log.raw_df.copy()

    def _resolve_window_size(self) -> int:
        """Pick the maximum prefix length kept per case.

        ``"auto"`` uses the 98.5th percentile of case lengths (incl. EOS rows)
        and then adds ``min_suffix_size``. The longest 1.5% of cases are treated
        as outliers so a few very long traces do not blow up the window.

        Called by
            ``PrefixesDataFrameLoader.__init__`` — once at construction, to set
            ``self.window_size``. That value is later used by ``transform`` →
            ``_limit_sequence`` so every prefix list has a hard length cap
            before the data are turned into tensors by ``EventLogLoader`` /
            ``TensorEncoderDecoder``.

        Example
        -------
        Input
            window_size_setting = "auto"
            min_suffix_size     = 1
            case lengths (rows per case_id, already including 1 EOS row)::

                case_id  length
                C1       5
                C2       6
                C3       6
                C4       7
                C5       8
                C6       10
                C7       40     ← outlier

        Processing
            1. ``case_sizes = [5, 6, 6, 7, 8, 10, 40]``
            2. ``quantile(0.985)`` sits near the upper end of the bulk
               (around 10), not at 40.
            3. ``auto_window = round(quantile) + min_suffix_size``
               e.g. ``round(10) + 1 = 11``
            4. Clamp: ``max(min_suffix_size, auto_window)``

        Output
            ``11``  — later ``_limit_sequence`` keeps at most the last 11
            events of every prefix. A fixed integer setting is returned as-is.
        """
        setting = (
            self.window_size_setting if self.window_size_setting is not None else "auto"
        )
        if isinstance(setting, str) and setting.lower() == "auto":
            event_log_df = _ensure_group_key_column(self.event_log, self.case_name)
            case_sizes = event_log_df.groupby(self.case_name).size()
            if case_sizes.empty:
                return self.min_suffix_size
            auto_window = round(case_sizes.quantile(1 - 0.015)) + self.min_suffix_size
            return max(self.min_suffix_size, int(auto_window))
        return int(setting)

    @staticmethod
    def _extract_static_value(series: pd.Series) -> object:
        cleaned = series.dropna()
        if cleaned.empty:
            return np.nan
        cleaned = cleaned[cleaned != "EOS"]
        if cleaned.empty:
            return np.nan
        return cleaned.iloc[0]

    def _limit_sequence(self, values):
        seq = list(values)
        if len(seq) <= self.window_size:
            return seq
        return seq[-self.window_size :]

    def transform(
        self, df: Optional[pd.DataFrame] = None, with_eos: Optional[bool] = False
    ) -> pd.DataFrame:
        """Turn event-level rows into one prefix row per growing prefix.

        Each original case with ``n`` events (including ``min_suffix_size`` EOS
        rows) yields ``n - min_suffix_size`` prefix rows. Dynamic columns become
        lists of the first ``prefix_len`` events (then trimmed to
        ``window_size``). Static columns stay a single scalar (first non-NaN,
        non-``"EOS"`` value of the case).

        Called by
            ``PrefixesDataFrameLoader.get_dataset("train"|"val"|"test")`` —
            the public entry that notebooks call (e.g. Repair / Helpdesk /
            Sepsis / BPIC loaders) to get prefix dataframes for Petri-net
            marking determination (proactive conformance checking).

            ``EventLogLoader`` does **not** call ``transform``. It only
            reuses this loader's already-split ``train_df`` / ``val_df`` /
            ``test_df`` so tensor encoding and prefix markings stay aligned
            on the same cases.

        Example
        -------
        Settings: ``min_suffix_size = 1``, ``window_size = 10``.

        Input (one case after EOS was appended by ``CSV2EventLog``)::

            case_id  activity  amount  resource
            C1       A         10.0    Alice
            C1       B         20.0    Alice
            C1       C         30.0    Alice
            C1       EOS       NaN     NaN      ← suffix reserved for prediction

        Processing (this case)
            total_len      = 4
            max_prefix_len = 4 - 1 = 3   → prefixes of length 1, 2, 3
            EOS is never copied into a prefix.

            prefix_len=1 → activity=[A],     amount=[10]
            prefix_len=2 → activity=[A, B],  amount=[10, 20]
            prefix_len=3 → activity=[A, B, C], amount=[10, 20, 30]
            resource is static → "Alice" on every prefix row
            (``_limit_sequence`` would keep only the last ``window_size``
            events if a prefix were longer than the window.)

        Output (this case contributes 3 rows)::

            case_id  prefix_length  activity       amount          resource
            C1       1              [A]            [10.0]          Alice
            C1       2              [A, B]         [10.0, 20.0]    Alice
            C1       3              [A, B, C]      [10.0, 20.0, 30.0] Alice
        """
        working_df = self.event_log if df is None else df
        working_df = _ensure_group_key_column(working_df, self.case_name)
        rows = []
        grouped = working_df.groupby(self.case_name, sort=False)
        for case_id, group in grouped:
            group = group.reset_index(drop=True)

            # Keep min_suffix_size events (the EOS rows) out of every prefix.
            total_len = len(group)
            max_prefix_len = total_len - self.min_suffix_size

            for prefix_len in range(1, max_prefix_len + 1):
                row = {self.case_name: case_id, "prefix_length": prefix_len}
                # categorical: growing list of activity tokens, then window-trim
                for col in self.categorical_columns:
                    values = (
                        group[col].iloc[:prefix_len].tolist()
                        if col in group.columns
                        else []
                    )
                    row[col] = self._limit_sequence(values)

                # continuous: growing list of numeric values, then window-trim
                for col in self.continuous_columns + self.continuous_positive_columns:
                    values = (
                        group[col].iloc[:prefix_len].tolist()
                        if col in group.columns
                        else []
                    )
                    row[col] = self._limit_sequence(values)

                # static: one scalar per case, copied onto every prefix row
                for col in self.static_categorical_columns:
                    row[col] = (
                        self._extract_static_value(group[col])
                        if col in group.columns
                        else np.nan
                    )

                for col in self.static_continuous_columns:
                    row[col] = (
                        self._extract_static_value(group[col])
                        if col in group.columns
                        else np.nan
                    )

                rows.append(row)
        return pd.DataFrame(rows)

    def get_dataset(self, type: str):
        """
        Return the transformed dataframe for the specified dataset type.
        """
        if type == "train":
            df = self.transform(df=self.train_df)
        elif type == "val":
            df = self.transform(df=self.val_df)
        elif type == "test":
            df = self.transform(df=self.test_df)
        return df

    def get_all_datasets(self):
        """
        Return the raw full dataframes for train, validation, and test datasets.
        """
        return self.train_df, self.val_df, self.test_df

    def extract_feature_info(self):
        """Collect vocabularies and numeric ranges from the processed event log.

        Uses ``self.event_log`` (event-level, including EOS rows). Categorical
        values are unique sorted strings; continuous columns get min/max/mean/std.

        Called by
            Loader notebooks (Helpdesk / Repair / Sepsis / BPIC) — usually
            commented in as ``categories = pref_adopt_dataframe.extract_feature_info()``
            to inspect which activity labels and value ranges exist before
            training. ``EventLogLoader`` / ``TensorEncoderDecoder`` do **not**
            call this; they fit encoders on the train split themselves.

        Example
        -------
        Input (``self.event_log``)::

            case_id  activity  amount
            C1       A         10.0
            C1       B         30.0
            C1       EOS       NaN

        Processing
            activity → unique non-null strings, sorted
            amount   → min/max/mean/std of non-null values

        Output::

            {
              "categorical": {"activity": ["A", "B", "EOS"]},
              "continuous":  {"amount": {"min": 10.0, "max": 30.0, "mean": 20.0, "std": …}},
            }
        """
        categories_info = {}
        ranges_info = {}

        # Get all unique categories for each categorical column
        categorical_columns = self.event_log_properties.get("categorical_columns", [])
        for col in categorical_columns:
            if col in self.event_log.columns:
                # Get all unique values (excluding NaN)
                unique_values = self.event_log[col].dropna().unique()
                # Convert to list and sort for consistency
                categories_info[col] = sorted(
                    [str(val) for val in unique_values if pd.notna(val)]
                )
            else:
                categories_info[col] = []

        # Get min/max ranges for each continuous column
        continuous_columns = self.event_log_properties.get(
            "continuous_columns", []
        ) + self.event_log_properties.get("continuous_positive_columns", [])
        for col in continuous_columns:
            if col in self.event_log.columns:
                # Get non-null values
                col_data = self.event_log[col].dropna()
                if len(col_data) > 0:
                    ranges_info[col] = {
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                    }
                else:
                    ranges_info[col] = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
            else:
                ranges_info[col] = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        return {"categorical": categories_info, "continuous": ranges_info}

    def remove_prefixes_with_unique_markings(
        self,
        train_pref_df: pd.DataFrame,
        val_pref_df: pd.DataFrame,
        test_pref_df: pd.DataFrame,
        markings_train: list,
        markings_val: list,
        markings_test: list,
        *,
        sort_tokens: bool = True,
    ):
        """Drop prefixes whose Petri-net marking appears in only one split.

        A marking is unique to a split if ``_marking_signature`` of it occurs
        there but in neither of the other two. Those rows are removed so
        train/val/test share the same marking vocabulary for evaluation.

        Called by
            Loader notebooks (``proactive_conf_check_v2`` / ``version_2``)
            after they have: (1) ``get_dataset`` prefix DFs, (2) replayed
            each prefix on the Petri net. They then keep only prefixes whose
            marking also exists in at least one other split.

        Example
        -------
        Input (3 prefix rows per split, markings already computed)::

            train prefixes / markings : P0→{p1}, P1→{p1}, P2→{p2}
            val   prefixes / markings : P0→{p1}, P1→{p3}
            test  prefixes / markings : P0→{p1}

        Processing
            signatures → sets
            unique_train = {p2}   (only in train) → drop train row P2
            unique_val   = {p3}   (only in val)   → drop val row P1
            unique_test  = {}     (p1 is shared)

        Output
            filtered prefix DFs + markings, plus kept positional indices
            ``(keep_train_ix, keep_val_ix, keep_test_ix)``
            e.g. train keeps [0, 1], val keeps [0], test keeps [0].
        """

        def _validate_lengths(df: pd.DataFrame, markings: list, name: str) -> None:
            if len(df) != len(markings):
                raise ValueError(
                    f"Length mismatch for {name}: prefix_df has {len(df)} rows, "
                    f"markings has {len(markings)} entries"
                )

        _validate_lengths(train_pref_df, markings_train, "train")
        _validate_lengths(val_pref_df, markings_val, "val")
        _validate_lengths(test_pref_df, markings_test, "test")

        train_sigs = [
            _marking_signature(m, sort_tokens=sort_tokens) for m in list(markings_train)
        ]
        val_sigs = [
            _marking_signature(m, sort_tokens=sort_tokens) for m in list(markings_val)
        ]
        test_sigs = [
            _marking_signature(m, sort_tokens=sort_tokens) for m in list(markings_test)
        ]

        train_set = set(train_sigs)
        val_set = set(val_sigs)
        test_set = set(test_sigs)

        unique_train = train_set - (val_set | test_set)
        unique_val = val_set - (train_set | test_set)
        unique_test = test_set - (train_set | val_set)

        keep_train_ix = [i for i, s in enumerate(train_sigs) if s not in unique_train]
        keep_val_ix = [i for i, s in enumerate(val_sigs) if s not in unique_val]
        keep_test_ix = [i for i, s in enumerate(test_sigs) if s not in unique_test]

        filtered_train_df = train_pref_df.iloc[keep_train_ix].reset_index(drop=True)
        filtered_val_df = val_pref_df.iloc[keep_val_ix].reset_index(drop=True)
        filtered_test_df = test_pref_df.iloc[keep_test_ix].reset_index(drop=True)

        filtered_markings_train = [list(markings_train)[i] for i in keep_train_ix]
        filtered_markings_val = [list(markings_val)[i] for i in keep_val_ix]
        filtered_markings_test = [list(markings_test)[i] for i in keep_test_ix]

        return (
            filtered_train_df,
            filtered_val_df,
            filtered_test_df,
            filtered_markings_train,
            filtered_markings_val,
            filtered_markings_test,
            (keep_train_ix, keep_val_ix, keep_test_ix),
        )


class EventLogDataset(Dataset):
    """
    Dataset class for event log data.
    """

    def __init__(
        self,
        tensor_bundle: dict[str, object],
        all_categories: tuple[list[tuple[str, int, dict[str, int]]]],
        all_static_categories: tuple[list[tuple[str, int, dict[str, int]]]],
        encoder_decoder: TensorEncoderDecoder,
    ):
        """
        Initialize the EventLogDataset with tensor data and encoding information.
        """
        self.tensor_bundle = tensor_bundle

        self.case_ids: list[object] = list(tensor_bundle["case_ids"])

        self.categorical_tensors: list[torch.Tensor] = tensor_bundle["categorical"]
        self.continuous_tensors: list[torch.Tensor] = tensor_bundle["continuous"]

        self.static_categorical_tensor: torch.Tensor = tensor_bundle[
            "static_categorical"
        ]
        self.static_continuous_tensor: torch.Tensor = tensor_bundle["static_continuous"]

        self.eos_padding: torch.Tensor = tensor_bundle["eos_padding"]
        self.zero_padding: torch.Tensor = tensor_bundle["zero_padding"]

        num_rows = self.eos_padding.shape[0]
        self.prefixes_petri_net_marking: list[Optional[object]] = [0] * num_rows

        self.all_categories: tuple[list[tuple[str, int, dict[str, int]]]] = (
            all_categories
        )
        self.all_static_categories: tuple[list[tuple[str, int, dict[str, int]]]] = (
            all_static_categories
        )
        self.encoder_decoder: TensorEncoderDecoder = encoder_decoder
        self.min_suffix_size: Optional[int] = getattr(
            encoder_decoder, "min_suffix_size", None
        )

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return self.eos_padding.shape[0]

    def __getitem__(self, idx):
        """
        Return the sample at the specified index.
        """
        case_id = self.case_ids[idx]

        categorical_items = [tensor[idx] for tensor in self.categorical_tensors]
        continuous_items = [tensor[idx] for tensor in self.continuous_tensors]

        eos_mask = self.eos_padding[idx]
        zero_mask = self.zero_padding[idx]

        static_cat = self.static_categorical_tensor[idx]
        static_cont = self.static_continuous_tensor[idx]

        prefixes_petri_net_marking = self.prefixes_petri_net_marking[idx]

        return (
            case_id,
            tuple(categorical_items),
            tuple(continuous_items),
            eos_mask,
            zero_mask,
            static_cat,
            static_cont,
            prefixes_petri_net_marking,
        )

    def subset(self, indices: Iterable[int]) -> "EventLogDataset":
        """Keep only selected prefix rows; return a **new** dataset.

        Slices every tensor in ``tensor_bundle`` along dim 0 and copies the
        matching Petri-net markings. Vocab / encoders
        (``all_categories``, ``encoder_decoder``) stay shared — you are not
        refitting, only dropping rows.

        Called by
            Loader notebooks after
            ``remove_prefixes_with_unique_markings``: that method returns
            ``keep_train_ix`` / ``keep_val_ix`` / ``keep_test_ix``. Then
            ``train_set.subset(keep_train_ix)`` so the tensor dataset has
            the same rows as the filtered prefix DataFrame before
            ``set_prefix_markings``.

        Example
        -------
        Input
            dataset length 4, ``indices = [0, 2]``
            ``case_ids = (C1, C1, C2, C2)``
            ``categorical[activity]`` shape ``(4, window_size)``

        Processing
            ``index_select(0, [0, 2])`` on every feature / mask / static
            tensor; ``case_ids`` and markings indexed the same way.
            ``indices = []`` → empty tensors, same number of feature
            columns (so DataLoader structure stays valid).

        Output
            new ``EventLogDataset`` of length 2:
            ``case_ids = (C1, C2)``, tensors shape ``(2, window_size)``.
            Original dataset is unchanged.
        """
        idx = list(indices)
        if len(idx) == 0:
            # Keep structure but return empty tensors
            idx_tensor = torch.tensor([], dtype=torch.long)
        else:
            idx_tensor = torch.tensor(idx, dtype=torch.long)

        new_tensor_bundle = {
            "categorical": [t.index_select(0, idx_tensor) for t in self.categorical_tensors],
            "continuous": [t.index_select(0, idx_tensor) for t in self.continuous_tensors],
            "eos_padding": self.eos_padding.index_select(0, idx_tensor),
            "zero_padding": self.zero_padding.index_select(0, idx_tensor),
            "case_ids": tuple(self.case_ids[i] for i in idx),
            "static_categorical": self.static_categorical_tensor.index_select(0, idx_tensor),
            "static_continuous": self.static_continuous_tensor.index_select(0, idx_tensor),
        }

        new_ds = EventLogDataset(
            new_tensor_bundle,
            self.all_categories,
            self.all_static_categories,
            self.encoder_decoder,
        )
        new_ds.prefixes_petri_net_marking = [self.prefixes_petri_net_marking[i] for i in idx]
        return new_ds

    def set_prefix_markings(
        self, markings, indices: Optional[Iterable[int]] = None
    ) -> None:
        """
        Set the prefix Petri net markings for the dataset.
        """
        markings_list = list(markings)

        if indices is None:
            if len(markings_list) != len(self.prefixes_petri_net_marking):
                raise ValueError(
                    "Number of markings must match dataset length when indices are"
                    " omitted"
                )
            target_indices = range(len(self.prefixes_petri_net_marking))
        else:
            target_indices = list(indices)
            if len(target_indices) != len(markings_list):
                raise ValueError(
                    "indices and markings must reference the same number of rows"
                )

        for idx, marking in zip(target_indices, markings_list, strict=False):
            if isinstance(marking, torch.Tensor):
                marking = marking.detach().cpu().tolist()
            elif isinstance(marking, np.ndarray):
                marking = marking.tolist()
            self.prefixes_petri_net_marking[idx] = marking


class EventLogLoader:
    """
    Loader that creates datasets from event log data.
    """

    def __init__(
        self,
        event_log_location,
        event_log_properties,
        prefix_df: PrefixesDataFrameLoader = None,
    ):
        """
        Initialize the EventLogLoader.

        Using event log location, properties,
        and optional prefix dataframe.
        """
        if prefix_df is not None:
            # event log
            self.event_log = prefix_df.csv2event_log
            #
            self.train_df = prefix_df.train_df.copy()
            self.val_df = prefix_df.val_df.copy()
            self.test_df = prefix_df.test_df.copy()
        else:
            self.event_log = CSV2EventLog(event_log_location, **event_log_properties)
            splitter = EventLogSplitter(**event_log_properties)
            self.train_df, self.val_df, self.test_df = splitter.split(self.event_log)

        self.encoder_decoder = TensorEncoderDecoder(
            self.train_df, **event_log_properties
        )
        # Data are transformed
        self.encoder_decoder.train_imputers_encoders()

    # get encoded dataframe
    def get_encoded_dataframe(self, type: str) -> pd.DataFrame:
        """
        Returns the raw dataframe for the specified dataset type.
        """
        if type == "train":
            df = self.train_df
        elif type == "val":
            df = self.val_df
        elif type == "test":
            df = self.test_df
        else:
            raise ValueError("type must be one of 'train', 'val', or 'test'")
        return df

    # get encoded data as tensors
    def get_dataset(self, type: str) -> EventLogDataset:
        """
        Return the dataset for the specified type as an EventLogDataset.
        """
        if type == "train":
            df = self.train_df
        elif type == "val":
            df = self.val_df
        elif type == "test":
            df = self.test_df
        else:
            raise ValueError("type must be one of 'train', 'val', or 'test'")
        encoded_data, all_categories, all_static_categories = (
            self.encoder_decoder.encode_df(df)
        )
        return EventLogDataset(
            encoded_data, all_categories, all_static_categories, self.encoder_decoder
        )
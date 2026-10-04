import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Any, Union
import pandas as pd
import numpy as np
from tqdm import trange
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import OneSidedSelection

# performance imports for torch: torch kernel uses one core only.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1" 

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.serialization import add_safe_globals

class TraceAttrScaler:
    """
    Fit scalers on train only, then apply to val/test. Use only if you have numeric trace attributes.
    """
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.scalers: Dict[str, StandardScaler] = {}

    def fit(self, df_train: pd.DataFrame):
        for col in self.cols:
            if col not in df_train.columns:
                continue
            # apply standard scaler to trace attributes
            sc = StandardScaler()
            sc.fit(df_train[[col]].astype(float))
            self.scalers[col] = sc
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        apply scaling
        """
        out = df.copy()
        for col, sc in self.scalers.items():
            if col not in out.columns:
                continue
            out[col] = sc.transform(out[[col]].astype(float)).astype(np.float32).ravel()
        return out

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """
        learn and apply scaling
        """
        return self.fit(df_train).transform(df_train)

# apply scaling to train, val and test
def scale_trace_attrs_after_split(train_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                                  val_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                                  test_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                                  trace_attr_cols: List[str],
                                  label_strategy: str = "collective"):
    """
    Returns scaled (train, val, test, scaler or dict[label]->scaler).
    """
    if label_strategy == "collective":
        scaler = TraceAttrScaler(trace_attr_cols)
        train_s = scaler.fit_transform(train_data)
        val_s = scaler.transform(val_data) if isinstance(val_data, pd.DataFrame) else val_data
        test_s = scaler.transform(test_data) if isinstance(test_data, pd.DataFrame) else test_data
        return train_s, val_s, test_s, scaler

    # separate: fit per label dataset to avoid leakage
    scalers: Dict[str, TraceAttrScaler] = {}
    train_out, val_out, test_out = {}, {}, {}
    for label in train_data.keys():
        sc = TraceAttrScaler(trace_attr_cols)
        train_out[label] = sc.fit_transform(train_data[label])
        val_out[label] = sc.transform(val_data[label])
        test_out[label] = sc.transform(test_data[label])
        scalers[label] = sc
    return train_out, val_out, test_out, scalers


class DeviationLabeling:
    def __init__(self,
                 log_name: str,
                 case_name: str,
                 activity_name: str,
                 resource_name: str,
                 time_name: str,
                 path_event_log: str,
                 path_process_model: str,
                 label_strategy: str = 'collective'):
        
        self.log_name = log_name
        self.path_event_log = path_event_log
        self.path_process_model = path_process_model
        
        self.case_name = case_name
        self.activity_name = activity_name
        self.time_name = time_name
        self.resource_name = resource_name

        if label_strategy not in {"collective", "separate"}:
            raise ValueError("label_strategy must be 'collective' or 'separate'")
        self.label_strategy = label_strategy

    def _load_log_csv(self):
        df = pd.read_csv(self.path_event_log)
        rename = {self.case_name: "case:concept:name",
                  self.activity_name: "concept:name",
                  self.time_name: "time:timestamp",
                  self.resource_name: "org:resource"}
        df = df.rename(columns=rename)

        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")

        ev_log = log_converter.apply(df, 
                                     variant=log_converter.Variants.TO_EVENT_LOG,
                                     parameters={log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"})
        return df, ev_log

    def _pre_process_process_model(self):
        pm = pm4py.read.read_bpmn(self.path_process_model)
        pn, im, fm = pm4py.convert.convert_to_petri_net(pm)
        return pn, im, fm

    def _extract_deviations_from_alignment(self, ev_log, alignment_results: List[dict]):
        dev_pos_by_case: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        dev_types: Set[str] = set()

        for trace, res in zip(ev_log, alignment_results):
            cid = trace.attributes.get("concept:name")

            aln_steps = res.get("alignment") or []
            # keep ">>" but drop None
            aln_steps = [(a, b) for a, b in aln_steps if a is not None and b is not None]
            if not aln_steps:
                continue

            log_idx = 0
            for a, b in aln_steps:
                if a == b:
                    log_idx += 1
                    continue

                if a != ">>" and b == ">>":
                    dt = str((a, b))
                    dev_pos_by_case[cid].append((log_idx, dt))
                    dev_types.add(dt)
                    # log_idx must increment
                    log_idx += 1
                    continue

                if a == ">>" and b != ">>":  # model move, no log consumption
                    dt = str((a, b))
                    dev_pos_by_case[cid].append((log_idx, dt))
                    dev_types.add(dt)
                    continue

        return sorted(dev_types), dict(dev_pos_by_case)

    def generate_individual_labels(self,
                                   trace_attr: List[str],
                                   max_prefix_cap: int = None,
                                   conf_runs: int = 100) -> Tuple[Any, Any]:
        
        if self.label_strategy not in {"collective", "separate"}:
            raise ValueError("label_strategy must be 'collective' or 'separate'")

        df_raw, ev_log = self._load_log_csv()
        net, im, fm = self._pre_process_process_model()

        best_D = ([], {})  # (dev_types, dev_pos_by_case)
        for _ in trange(conf_runs):
            conformance = pm4py.conformance.conformance_diagnostics_alignments(ev_log, net, im, fm, multi_processing=False)
            deviations, dev_pos_by_case = self._extract_deviations_from_alignment(ev_log, conformance)
            
            if len(best_D[0]) == 0 or len(best_D[0]) >= len(deviations):
                best_D = (deviations, dev_pos_by_case)

        dev_types, dev_pos_by_case = best_D

        df_sorted = df_raw.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
        
        # new: added weekdays
        case_start_ts = df_sorted.groupby("case:concept:name")["time:timestamp"].transform("first")
        case_end_ts   = df_sorted.groupby("case:concept:name")["time:timestamp"].transform("last")

        # old: ISO weekday number (Mon=1..Sun=7) -> commented out
        # df_sorted["weekday_start"] = case_start_ts.dt.isocalendar().day.astype("Int64").astype("string").fillna("NA")
        # df_sorted["weekday_end"]   = case_end_ts.dt.isocalendar().day.astype("Int64").astype("string").fillna("NA")

        # new: weekday names (Mon..Sun) using the same semantics as datetime.weekday() (Mon=0..Sun=6)
        weekDaysMapping = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
        _day_map = {i: weekDaysMapping[i] for i in range(7)}

        df_sorted["weekday_start"] = case_start_ts.dt.weekday.map(_day_map).fillna("NA").astype("string")
        df_sorted["weekday_end"]   = case_end_ts.dt.weekday.map(_day_map).fillna("NA").astype("string")

        # df_sorted["weekday_start"] = case_start_ts.dt.isocalendar().day.astype("Int64").astype("string").fillna("NA")
        # df_sorted["weekday_end"]   = case_end_ts.dt.isocalendar().day.astype("Int64").astype("string").fillna("NA")

        activities = df_sorted["concept:name"].fillna("NA").astype(str).unique().tolist()
        act2idx = {act: (i + 1) for i, act in enumerate(sorted(activities))}  # 0 reserved for PAD

        resource_source_col = "org:resource"
        resources = df_sorted[resource_source_col].fillna("NA").astype(str).unique().tolist()
        res2idx = {res: (i + 1) for i, res in enumerate(sorted(resources))}

        months = df_sorted["time:timestamp"].apply(lambda x: f"{x.month}_{x.year}" if pd.notna(x) else "NA")
        unique_months = sorted(months.unique())
        month2idx = {month: (i + 1) for i, month in enumerate(unique_months)}

        case_attr_list = list(dict.fromkeys((trace_attr or []) + ["weekday_start", "weekday_end"]))

        # initialize LabelEncoders per trace attr
        case_attr_encoders: Dict[str, LabelEncoder] = {}
        for ca in case_attr_list:
            le = LabelEncoder()
            if ca not in df_sorted.columns:
                df_sorted[ca] = "NA"
            col = df_sorted.groupby("case:concept:name")[ca].first().fillna("NA").astype(str).values
            le.fit(col)
            case_attr_encoders[ca] = le

        # max length
        case_lengths = df_sorted.groupby("case:concept:name").size().to_dict()
        L_max = max(case_lengths.values()) if len(case_lengths) > 0 else 0
        if max_prefix_cap is not None:
            L_max = min(L_max, max_prefix_cap)

        rows = []
        for cid, g in df_sorted.groupby("case:concept:name", sort=False):
            g = g.reset_index(drop=True)
            n = len(g)

            encoded_case_attrs = {}
            for ca in case_attr_list:
                raw = g[ca].iloc[0] if ca in g.columns else "NA"
                raw = "NA" if pd.isna(raw) else raw
                le = case_attr_encoders[ca]
                try:
                    enc = int(le.transform([str(raw)])[0])
                except Exception:
                    # keep your behavior but note: this creates an "UNK" id
                    enc = len(le.classes_)
                encoded_case_attrs[ca] = enc

            start_ts = g.loc[0, "time:timestamp"] if n > 0 else pd.NaT

            for i in range(n):
                act_row = [0] * L_max
                res_row = [0] * L_max
                month_row = [0] * L_max

                for p in range(1, L_max + 1):
                    if p - 1 <= i:
                        act = str(g.loc[p - 1, "concept:name"]) if "concept:name" in g.columns else "NA"
                        res = str(g.loc[p - 1, resource_source_col]) if resource_source_col in g.columns else "NA"
                        ts = g.loc[p - 1, "time:timestamp"]

                        act_row[p - 1] = act2idx.get(act, 0)
                        res_row[p - 1] = res2idx.get(res, 0)

                        if pd.isna(ts) or pd.isna(start_ts):
                            month_row[p - 1] = 0
                        else:
                            month_str = f"{ts.month}_{ts.year}"
                            month_row[p - 1] = month2idx.get(month_str, 0)

                # suffix deviations: strictly after prefix end i
                devs_in_suffix = []
                for pos, dt in dev_pos_by_case.get(cid, []):
                    if pos > i:
                        devs_in_suffix.append(dt)

                prefix_len = i + 1

                # keep case_id for splitting; keep prefix_id as (case, pref_len)
                base = {"case_id": str(cid), # used for splitting (no leakage)                   
                        "prefix_len": int(prefix_len),
                        "prefix_id": str((str(cid), prefix_len)),  
                        "activities": act_row,
                        "resources": res_row,
                        "months": month_row}

                for ca in case_attr_list:
                    base[f"trace_attr_{ca}"] = encoded_case_attrs[ca]

                for dt in dev_types:
                    base[f"y_{dt}"] = 1 if dt in devs_in_suffix else 0

                rows.append(base)

        df_flat = pd.DataFrame(rows)

        # remove scaling from here: scale after split with TraceAttrScaler if needed.
        # Store cardinalities for one-hot pipeline (incl. UNK bucket if used)
        trace_attr_cardinalities = {ca: (len(le.classes_) + 1)  # +1 for the UNK id you create via enc=len(classes_)
                                    for ca, le in case_attr_encoders.items()}

        base_encoders = {"activity_ids": act2idx,
                         "resource_ids": res2idx,
                         "month_ids": month2idx,
                         "trace_attr_encoders": case_attr_encoders,
                         "trace_attr_cardinalities": trace_attr_cardinalities,
                         "L_max": L_max}

        if self.label_strategy == "collective":
            encoders = {**base_encoders, "deviations": dev_types}
            return df_flat, encoders

        label_dfs: Dict[str, pd.DataFrame] = {}
        label_encoders: Dict[str, Dict[str, Any]] = {}
        dev_cols = [f"y_{dt}" for dt in dev_types]

        for dt in dev_types:
            y_col = f"y_{dt}"
            if y_col not in df_flat.columns:
                continue
            df_label = df_flat.drop(columns=[col for col in dev_cols if col != y_col]).copy()
            label_dfs[dt] = df_label
            label_encoders[dt] = {**base_encoders, "deviations": [dt]}

        return label_dfs, label_encoders


class TrainTestSplit:
    def __init__(self,
                 df_labled_deviations: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 label_strategy: str = "collective",
                 seed: int = 42):
        
        if label_strategy not in {"collective", "separate"}:
            raise ValueError("label_strategy must be 'collective' or 'separate'")
        
        self.df_labeled_deviations = df_labled_deviations
        self.label_strategy = label_strategy
        self.seed = seed

    def _split_dataframe_by_cases(self,
                                  df: pd.DataFrame,
                                  seed: int,
                                  train_frac: float,
                                  val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        if df is None or df.empty:
            empty = pd.DataFrame(columns=df.columns if df is not None else [])
            return empty.copy(), empty.copy(), empty.copy()

        # split by case_id (present in all rows)
        cases = df["case_id"].dropna().astype(str).unique()
        if len(cases) == 0:
            empty = df.iloc[0:0].copy()
            return empty, empty.copy(), empty.copy()

        cases = np.array(sorted(cases.tolist(), key=str), dtype=object)
        rng = np.random.default_rng(seed)
        cases = rng.permutation(cases)

        n_train = int(len(cases) * train_frac)
        n_train = max(1, min(n_train, len(cases)))
        train_cases = cases[:n_train]

        val_cases = np.array([], dtype=object)
        if val_frac > 0 and len(train_cases) > 1:
            n_val = max(1, int(len(train_cases) * val_frac))
            n_val = min(n_val, len(train_cases) - 1)
            val_cases = train_cases[:n_val]
            train_cases = train_cases[n_val:]

        train_mask = df["case_id"].isin(train_cases)
        val_mask = df["case_id"].isin(val_cases) if len(val_cases) > 0 else pd.Series(False, index=df.index)
        test_mask = ~(train_mask | val_mask)

        train_df = df[train_mask].reset_index(drop=True)
        val_df = df[val_mask].reset_index(drop=True) if len(val_cases) > 0 else df.iloc[0:0].copy()
        test_df = df[test_mask].reset_index(drop=True)
        
        return train_df, val_df, test_df

    def data_split(self, 
                   train_frac: float = 2/3,
                   val_frac: float = 0.0):
        
        data = self.df_labeled_deviations
        seed = self.seed

        if self.label_strategy == "collective":
            if not isinstance(data, pd.DataFrame):
                raise TypeError("For collective strategy, df_labled_deviations must be a DataFrame.")
            return self._split_dataframe_by_cases(data, seed, train_frac, val_frac)

        if not isinstance(data, dict):
            raise TypeError("For separate strategy, df_labled_deviations must be a dict[label -> DataFrame].")

        train_dict: Dict[str, pd.DataFrame] = {}
        val_dict: Dict[str, pd.DataFrame] = {}
        test_dict: Dict[str, pd.DataFrame] = {}

        for idx, (label, df_label) in enumerate(data.items()):
            split_seed = seed + idx
            train_df, val_df, test_df = self._split_dataframe_by_cases(df_label, split_seed, train_frac, val_frac)
            train_dict[label] = train_df
            val_dict[label] = val_df
            test_dict[label] = test_df

        if val_frac == 0:
            val_dict = {}
        return train_dict, val_dict, test_dict


class Undersampling:
    """
    Training data consists of many prefixes (many rows):
    The paper by Grohs says it undersamples traces (cases), meaning:
    - If a case is removed, all its prefixes are removed.
    - If a case is kept, all its prefixes are kept.
    
    Therefore, we take one representative row per case, usually the earliest prefix: OSS expects one instance per item being sampled
    """
    def __init__(self,
                 train_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 list_dynamic_cols: List[str],
                 label_strategy: str = "collective"):
        
        self.train_data = train_data
        self.list_dynamic_cols = list_dynamic_cols
        self.strategy = label_strategy

    @staticmethod
    def _flat_list_cols(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        df_flat = df.copy()
        mapping = {}
        for col in cols:
            if col not in df_flat.columns:
                continue
            expanded = pd.DataFrame(df_flat[col].tolist(), index=df_flat.index)
            expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
            df_flat = pd.concat([df_flat.drop(columns=[col]), expanded], axis=1)
            mapping[col] = expanded.columns.tolist()
        return df_flat, mapping

    @staticmethod
    def _get_case_representatives(df: pd.DataFrame) -> pd.DataFrame:
        """
        One row per case (trace-level). We take the smallest prefix_len row per case.
        This corresponds to prefix_len=1 in your generated dataset.
        """
        if "case_id" not in df.columns or "prefix_len" not in df.columns:
            raise ValueError("Trace-level undersampling requires 'case_id' and 'prefix_len' columns.")
        idx = df.groupby("case_id")["prefix_len"].idxmin()
        return df.loc[idx].reset_index(drop=True)

    @staticmethod
    def _oss_select_case_ids(df_case: pd.DataFrame,
                            list_dynamic_cols: List[str],
                            y: np.ndarray) -> Set[str]:
        """
        Runs OSS on the case-level dataframe and returns selected case_ids.
        """
        df_flat, _ = Undersampling._flat_list_cols(df_case, list_dynamic_cols)

        feature_cols = [
            c for c in df_flat.columns
            if not c.startswith("y_")
            and c not in {"case_id", "prefix_id", "prefix_len"}
        ]
        if not feature_cols:
            # no features => cannot do OSS meaningfully; keep all cases
            return set(df_case["case_id"].astype(str).tolist())

        if np.sum(y == 1) == 0:
            # no positives => keep all (or you could keep all negatives too)
            return set(df_case["case_id"].astype(str).tolist())

        X = df_flat[feature_cols].to_numpy()
        oss = OneSidedSelection(sampling_strategy="auto", random_state=42)
        oss.fit_resample(X, y)
        selected_idx = sorted(set(oss.sample_indices_))
        return set(df_case.iloc[selected_idx]["case_id"].astype(str).tolist())

    def _collective_oss(self, df: pd.DataFrame):
        """
        TRACE-LEVEL collective: build y_any at case-level (OR over all y_*),
        run OSS once on cases, then keep all rows for selected cases.
        """
        target_cols = [c for c in df.columns if c.startswith("y_")]
        if not target_cols:
            return df.copy(), []

        # Case-level representatives (one per case)
        df_case = self._get_case_representatives(df)

        # track labels with no positives at CASE-level (optional)
        no_true_class = [c for c in target_cols if np.sum(df_case[c].values == 1) == 0]

        # collective case label: any deviation in that trace
        y_any = (df_case[target_cols].to_numpy(dtype=np.int64).sum(axis=1) > 0).astype(np.int64)

        selected_case_ids = self._oss_select_case_ids(df_case, self.list_dynamic_cols, y_any)

        df_out = df[df["case_id"].astype(str).isin(selected_case_ids)].reset_index(drop=True)
        # optional: drop labels that have zero positives at case-level
        df_out = df_out.drop(columns=no_true_class, errors="ignore")
        return df_out, no_true_class

    def _undersample_single_label(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        TRACE-LEVEL separate: df must have exactly one y_* column.
        Run OSS on cases for that label, keep all rows for selected cases.
        """
        target_cols = [c for c in df.columns if c.startswith("y_")]
        if not target_cols:
            return df.copy(), []
        if len(target_cols) > 1:
            raise ValueError("Separate strategy expects exactly one target column per dataframe.")
        target_col = target_cols[0]

        df_case = self._get_case_representatives(df)
        y = df_case[target_col].to_numpy()

        if np.sum(y == 1) == 0:
            # IMPORTANT: do NOT drop the label column; just signal skip
            return df.copy(), [target_col]

        selected_case_ids = self._oss_select_case_ids(df_case, self.list_dynamic_cols, y)
        df_out = df[df["case_id"].astype(str).isin(selected_case_ids)].reset_index(drop=True)
        return df_out, []

    def _separate_oss(self, data: Dict[str, pd.DataFrame]):
        undersampled: Dict[str, pd.DataFrame] = {}
        no_true_class: List[str] = []

        for label, df_label in data.items():
            df_oss, missing = self._undersample_single_label(df_label)
            undersampled[label] = df_oss
            no_true_class.extend(missing)

        return undersampled, list(dict.fromkeys(no_true_class))

    def one_sided_selection_undersampling(self):
        if self.strategy == "collective":
            return self._collective_oss(self.train_data)
        return self._separate_oss(self.train_data)

# used for LSTMs
class PrefixDataset(Dataset):
    def __init__(self,
                 df_train: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 df_val: Union[None, pd.DataFrame, Dict[str, pd.DataFrame]],
                 df_test: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 activity_col: str,
                 resource_col: str,
                 month_col: str,
                 trace_cols,
                 y_cols: Union[List[str], Dict[str, List[str]]],
                 label_strategy: str = "collective"):
        
        if label_strategy not in {"collective", "separate"}:
            raise ValueError("label_strategy must be 'collective' or 'separate'")
        self.label_strategy = label_strategy

        if label_strategy == "collective":
            if not isinstance(df_train, pd.DataFrame) or not isinstance(df_test, pd.DataFrame):
                raise TypeError("Collective strategy expects single train/val/test DataFrames.")
            if df_val is not None and not isinstance(df_val, pd.DataFrame):
                raise TypeError("Collective strategy expects df_val as DataFrame (or None).")
            if not isinstance(y_cols, list):
                raise TypeError("Collective strategy expects y_cols as a list of column names.")
            self.df_train = df_train.reset_index(drop=True)
            self.df_val = (df_val.reset_index(drop=True) if isinstance(df_val, pd.DataFrame)
                           else self.df_train.iloc[0:0].copy())
            self.df_test = df_test.reset_index(drop=True)
            self.df_train_dict = None
            self.df_val_dict = None
            self.df_test_dict = None
        else:
            if not isinstance(df_train, dict) or not isinstance(df_test, dict):
                raise TypeError("Separate strategy expects dict[label -> DataFrame] inputs.")
            if df_val is not None and not isinstance(df_val, dict):
                raise TypeError("Separate strategy expects df_val as dict[label -> DataFrame] (or None).")
            if not isinstance(y_cols, dict):
                raise TypeError("Separate strategy expects y_cols as dict[label -> List[str]].")
            val_dict = df_val or {k: v.iloc[0:0].copy() for k, v in df_train.items()}
            missing_keys = (set(df_train.keys()) - set(df_test.keys())) | (set(df_train.keys()) - set(val_dict.keys()))
            if missing_keys:
                raise ValueError(f"Missing splits for labels: {sorted(missing_keys)}")

            self.df_train = None
            self.df_val = None
            self.df_test = None
            self.df_train_dict = {k: v.reset_index(drop=True) for k, v in df_train.items()}
            self.df_val_dict = {k: val_dict[k].reset_index(drop=True) for k in df_train.keys()}
            self.df_test_dict = {k: df_test[k].reset_index(drop=True) for k in df_train.keys()}

        self.activity_col = activity_col
        self.resource_col = resource_col
        self.month_col = month_col
        self.trace_cols = trace_cols or []
        self.y_cols = y_cols

    def __len__(self, split: str = "train"):
        if self.label_strategy != "collective":
            raise RuntimeError("__len__ is only supported in collective mode.")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        df_map = {"train": self.df_train, "val": self.df_val, "test": self.df_test}
        return len(df_map[split])

    def __getitem__(self, idx, split: str = "train"):
        if self.label_strategy != "collective":
            raise RuntimeError("__getitem__ is only supported in collective mode.")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        df = {"train": self.df_train, "val": self.df_val, "test": self.df_test}[split]

        row = df.iloc[idx]
        x_act = torch.tensor(row[self.activity_col], dtype=torch.long)
        x_res = torch.tensor(row[self.resource_col], dtype=torch.long)
        x_month = torch.tensor(row[self.month_col], dtype=torch.long)

        if self.trace_cols:
            trace_cols = self._resolve_trace_columns(df)
            trace_vals = np.asarray(row[trace_cols], dtype=np.float32)
            trace_feats = torch.tensor(trace_vals, dtype=torch.float32)
        else:
            trace_feats = torch.zeros(0, dtype=torch.float32)

        y_columns = self.y_cols
        if y_columns:
            y_vals = np.asarray(row[y_columns], dtype=np.int64)
            y = torch.tensor(y_vals, dtype=torch.long)
        else:
            y = torch.zeros(0, dtype=torch.long)

        return x_act, x_res, x_month, trace_feats, y

    def _to_tensor_dataset(self, df: pd.DataFrame, y_columns: List[str], device=None):
        device = torch.device(device) if device is not None else torch.device("cpu")

        act_arr = np.asarray(df[self.activity_col].tolist(), dtype=np.int64)
        res_arr = np.asarray(df[self.resource_col].tolist(), dtype=np.int64)
        month_arr = np.asarray(df[self.month_col].tolist(), dtype=np.int64)

        if self.trace_cols:
            trace_cols = self._resolve_trace_columns(df)
            trace_arr = df[trace_cols].to_numpy(dtype=np.float32, copy=True)
        else:
            trace_arr = np.zeros((len(df), 0), dtype=np.float32)

        if y_columns:
            y_arr = df[y_columns].to_numpy(dtype=np.int64, copy=True)
        else:
            y_arr = np.zeros((len(df), 0), dtype=np.int64)

        x_act = torch.tensor(act_arr, dtype=torch.long, device=device)
        x_res = torch.tensor(res_arr, dtype=torch.long, device=device)
        x_month = torch.tensor(month_arr, dtype=torch.long, device=device)
        trace_tensor = torch.tensor(trace_arr, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_arr, dtype=torch.long, device=device)

        return TensorDataset(x_act, x_res, x_month, trace_tensor, y_tensor)

    def tensor_datset_encoding(self, device=None):
        """
        Embedding pipeline output.
        - collective: (train, val, test) TensorDataset
        - separate: dict[label]-> TensorDataset
        """
        if self.label_strategy == "collective":
            train_dataset = self._to_tensor_dataset(self.df_train, self.y_cols, device)
            val_dataset = self._to_tensor_dataset(self.df_val, self.y_cols, device)
            test_dataset = self._to_tensor_dataset(self.df_test, self.y_cols, device)
            return train_dataset, val_dataset, test_dataset

        train_dict: Dict[str, TensorDataset] = {}
        val_dict: Dict[str, TensorDataset] = {}
        test_dict: Dict[str, TensorDataset] = {}
        for label in self.df_train_dict:
            y_columns = self.y_cols.get(label, [])
            train_dict[label] = self._to_tensor_dataset(self.df_train_dict[label], y_columns, device)
            val_dict[label] = self._to_tensor_dataset(self.df_val_dict[label], y_columns, device)
            test_dict[label] = self._to_tensor_dataset(self.df_test_dict[label], y_columns, device)
        return train_dict, val_dict, test_dict

    @staticmethod
    def save_datasets(train_dataset, test_dataset, val_dataset, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        train_path = os.path.join(save_path, "train_set.pkl")
        val_path = os.path.join(save_path, "val_set.pkl")
        test_path = os.path.join(save_path, "test_set.pkl")
        torch.save(train_dataset, train_path)
        torch.save(val_dataset, val_path)
        torch.save(test_dataset, test_path)
        return train_path, val_path, test_path

    @staticmethod
    def load_datasets(save_path: str, map_location=None):
        train_path = os.path.join(save_path, "train_set.pkl")
        val_path = os.path.join(save_path, "val_set.pkl")
        test_path = os.path.join(save_path, "test_set.pkl")

        if add_safe_globals is not None:
            add_safe_globals([TensorDataset])

        def _torch_load(path):
            load_kwargs = {}
            if map_location is not None:
                load_kwargs["map_location"] = map_location
            try:
                return torch.load(path, weights_only=False, **load_kwargs)
            except TypeError:
                return torch.load(path, **load_kwargs)

        train_dataset = _torch_load(train_path)
        val_dataset = _torch_load(val_path)
        test_dataset = _torch_load(test_path)
        return train_dataset, val_dataset, test_dataset

    def _resolve_trace_columns(self, df: pd.DataFrame) -> List[str]:
        resolved = []
        missing = []
        for name in self.trace_cols:
            prefixed = name if name.startswith("trace_attr_") else f"trace_attr_{name}"
            if prefixed in df.columns:
                resolved.append(prefixed)
            elif name in df.columns:
                resolved.append(name)
            else:
                missing.append(name)
        if missing:
            raise KeyError(f"Trace attributes not found in dataframe columns: {missing}")
        return resolved

# new for FFN
class PrefixDatasetTabularFFN:
    """
    FFN tabular encoding with SAME init params as PrefixDataset.

    X consists of:
    - one-hot over (activity/resource/month) sequences per position (flattened)
    - trace attributes taken same logic as PrefixDataset: df[trace_cols]
    """
    def __init__(self,
                 df_train: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 df_val: Union[None, pd.DataFrame, Dict[str, pd.DataFrame]],
                 df_test: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 activity_col: str,
                 resource_col: str,
                 month_col: str,
                 trace_cols,
                 y_cols: Union[List[str], Dict[str, List[str]]],
                 label_strategy: str = "collective",
                 drop_pad: bool = True):
        if label_strategy not in {"collective", "separate"}:
            raise ValueError("label_strategy must be 'collective' or 'separate'")
        self.label_strategy = label_strategy

        # same params/fields as PrefixDataset
        self.activity_col = activity_col
        self.resource_col = resource_col
        self.month_col = month_col
        self.trace_cols = trace_cols or []
        self.y_cols = y_cols

        # FFN-specific
        self.drop_pad = bool(drop_pad)

        # store splits (same logic as PrefixDataset)
        if label_strategy == "collective":
            if not isinstance(df_train, pd.DataFrame) or not isinstance(df_test, pd.DataFrame):
                raise TypeError("Collective strategy expects single train/val/test DataFrames.")
            if df_val is not None and not isinstance(df_val, pd.DataFrame):
                raise TypeError("Collective strategy expects df_val as DataFrame (or None).")
            if not isinstance(y_cols, list):
                raise TypeError("Collective strategy expects y_cols as a list of column names.")

            self.df_train = df_train.reset_index(drop=True)
            self.df_val = (df_val.reset_index(drop=True) if isinstance(df_val, pd.DataFrame)
                           else self.df_train.iloc[0:0].copy())
            self.df_test = df_test.reset_index(drop=True)

            self.df_train_dict = None
            self.df_val_dict = None
            self.df_test_dict = None
        else:
            if not isinstance(df_train, dict) or not isinstance(df_test, dict):
                raise TypeError("Separate strategy expects dict[label -> DataFrame] inputs.")
            if df_val is not None and not isinstance(df_val, dict):
                raise TypeError("Separate strategy expects df_val as dict[label -> DataFrame] (or None).")
            if not isinstance(y_cols, dict):
                raise TypeError("Separate strategy expects y_cols as dict[label -> List[str]].")

            val_dict = df_val or {k: v.iloc[0:0].copy() for k, v in df_train.items()}
            missing_keys = (set(df_train.keys()) - set(df_test.keys())) | (set(df_train.keys()) - set(val_dict.keys()))
            if missing_keys:
                raise ValueError(f"Missing splits for labels: {sorted(missing_keys)}")

            self.df_train = None
            self.df_val = None
            self.df_test = None
            self.df_train_dict = {k: v.reset_index(drop=True) for k, v in df_train.items()}
            self.df_val_dict = {k: val_dict[k].reset_index(drop=True) for k in df_train.keys()}
            self.df_test_dict = {k: df_test[k].reset_index(drop=True) for k in df_train.keys()}

        # infer vocab sizes from available training data (PAD=0)
        ref_df = self._get_reference_df()
        self.activity_vocab_size = self._infer_max_id(ref_df, self.activity_col)
        self.resource_vocab_size = self._infer_max_id(ref_df, self.resource_col)
        self.month_vocab_size = self._infer_max_id(ref_df, self.month_col)

    def _get_reference_df(self) -> pd.DataFrame:
        if self.label_strategy == "collective":
            return self.df_train
        # pick first non-empty df
        for _, df in self.df_train_dict.items():
            if len(df) > 0:
                return df
        # fallback: first key even if empty
        return self.df_train_dict[next(iter(self.df_train_dict.keys()))]

    @staticmethod
    def _infer_max_id(df: pd.DataFrame, col: str) -> int:
        if col not in df.columns or len(df) == 0:
            return 0
        arr = np.asarray(df[col].tolist(), dtype=np.int64)
        return int(arr.max()) if arr.size else 0

    def _one_hot_seq(self, x: torch.Tensor, num_classes_with_pad: int) -> torch.Tensor:
        oh = F.one_hot(x, num_classes=num_classes_with_pad)  # [N,L,C]
        if self.drop_pad:
            oh = oh[..., 1:]  # drop PAD channel => PAD becomes all zeros
        return oh.float()

    def _resolve_trace_columns(self, df: pd.DataFrame) -> List[str]:
        resolved = []
        missing = []
        for name in (self.trace_cols or []):
            prefixed = name if name.startswith("trace_attr_") else f"trace_attr_{name}"
            if prefixed in df.columns:
                resolved.append(prefixed)
            elif name in df.columns:
                resolved.append(name)
            else:
                missing.append(name)
        if missing:
            raise KeyError(f"Trace attributes not found in dataframe columns: {missing}")
        return resolved

    def _build_X(self, df: pd.DataFrame) -> torch.Tensor:
        # sequences -> one-hot -> flatten
        act = torch.tensor(np.asarray(df[self.activity_col].tolist(), dtype=np.int64), dtype=torch.long)
        res = torch.tensor(np.asarray(df[self.resource_col].tolist(), dtype=np.int64), dtype=torch.long)
        mon = torch.tensor(np.asarray(df[self.month_col].tolist(), dtype=np.int64), dtype=torch.long)

        act_oh = self._one_hot_seq(act, self.activity_vocab_size + 1).reshape(len(df), -1)
        res_oh = self._one_hot_seq(res, self.resource_vocab_size + 1).reshape(len(df), -1)
        mon_oh = self._one_hot_seq(mon, self.month_vocab_size + 1).reshape(len(df), -1)

        parts = [act_oh, res_oh, mon_oh]

        # trace attributes AS-IS (PrefixDataset logic): take df[resolved_trace_cols] and append
        if self.trace_cols:
            trace_cols = self._resolve_trace_columns(df)
            trace_arr = df[trace_cols].to_numpy(dtype=np.float32, copy=True)
            parts.append(torch.tensor(trace_arr, dtype=torch.float32))
        else:
            # keep shape consistent; no-op
            pass

        return torch.cat(parts, dim=1)

    def _to_tensor_dataset(self, df: pd.DataFrame, y_columns: List[str], device=None) -> TensorDataset:
        device = torch.device(device) if device is not None else torch.device("cpu")
        X = self._build_X(df).to(device)

        if y_columns:
            y = torch.tensor(df[y_columns].to_numpy(dtype=np.int64, copy=True),
                             dtype=torch.long, device=device)
        else:
            y = torch.zeros((len(df), 0), dtype=torch.long, device=device)

        return TensorDataset(X, y)

    def tensor_dataset_encoding(self, device=None):
        """
        Tabular one-hot pipeline output.
        - collective: (train, val, test) TensorDataset
        - separate: dict[label]->TensorDataset
        """
        if self.label_strategy == "collective":
            return (
                self._to_tensor_dataset(self.df_train, self.y_cols, device),
                self._to_tensor_dataset(self.df_val, self.y_cols, device),
                self._to_tensor_dataset(self.df_test, self.y_cols, device),
            )

        train_dict: Dict[str, TensorDataset] = {}
        val_dict: Dict[str, TensorDataset] = {}
        test_dict: Dict[str, TensorDataset] = {}
        for label in self.df_train_dict:
            y_columns = self.y_cols.get(label, [])
            train_dict[label] = self._to_tensor_dataset(self.df_train_dict[label], y_columns, device)
            val_dict[label] = self._to_tensor_dataset(self.df_val_dict[label], y_columns, device)
            test_dict[label] = self._to_tensor_dataset(self.df_test_dict[label], y_columns, device)

        return train_dict, val_dict, test_dict
    
    @staticmethod
    def save_datasets(train_dataset, test_dataset, val_dataset, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        train_path = os.path.join(save_path, "train_set.pkl")
        val_path = os.path.join(save_path, "val_set.pkl")
        test_path = os.path.join(save_path, "test_set.pkl")
        torch.save(train_dataset, train_path)
        torch.save(val_dataset, val_path)
        torch.save(test_dataset, test_path)
        return train_path, val_path, test_path

    @staticmethod
    def load_datasets(save_path: str, map_location=None):
        train_path = os.path.join(save_path, "train_set.pkl")
        val_path = os.path.join(save_path, "val_set.pkl")
        test_path = os.path.join(save_path, "test_set.pkl")

        if add_safe_globals is not None:
            add_safe_globals([TensorDataset])

        def _torch_load(path):
            load_kwargs = {}
            if map_location is not None:
                load_kwargs["map_location"] = map_location
            try:
                return torch.load(path, weights_only=False, **load_kwargs)
            except TypeError:
                return torch.load(path, **load_kwargs)

        train_dataset = _torch_load(train_path)
        val_dataset = _torch_load(val_path)
        test_dataset = _torch_load(test_path)
        return train_dataset, val_dataset, test_dataset
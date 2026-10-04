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
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.serialization import add_safe_globals

class DeviationLabeling:
    def __init__(self, log_name: str, path_event_log: str, path_process_model: str, label_strategy: str = 'collective'):
        self.log_name = log_name
        self.path_event_log = path_event_log
        self.path_process_model = path_process_model
        
        if label_strategy not in {"collective", "separate"}:
            raise ValueError("label_strategy must be 'collective' or 'separate'")
        self.label_strategy = label_strategy

    def _load_log_csv(self):
        df = pd.read_csv(self.path_event_log)
        if self.log_name == "Helpdesk":
            
            # Rename: for example important for helpdesk:
            rename = {"CaseID": "case:concept:name",
                      "Activity": "concept:name",
                      "CompleteTimestamp": "time:timestamp",
                      "Resource": "org:resource"}
            
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
        """
        From one alignment result per trace, produce:
          - dev_pos_by_case: case -> list of (position_index, dev_type)
          - dev_types: set of all deviation types encountered
        """
        # Position of deviation in suffix
        dev_pos_by_case: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        # All deviations
        dev_types: Set[str] = set()

        # iterate through the alignments in each trace:
        for trace, res in zip(ev_log, alignment_results):
            # Get the case id:
            cid = trace.attributes.get("concept:name")
            trace_event_labels = [t.get("concept:name") for t in trace]
            
            # Get the alignments
            aln_steps = res.get("alignment")
            aln_cleaned_steps = [(a,b) for a,b in aln_steps if a not in (None, "None") and b not in (None, "None")]
            
            if not aln_cleaned_steps:
                continue
            
            assert len(trace_event_labels) <= len(aln_cleaned_steps), \
                "Alignment length mismatch: trace has {} events, alignment has {}".format(trace_event_labels, aln_cleaned_steps)
            
            log_idx = 0
            for a, b in aln_cleaned_steps:
                # (a,b) and b == a: synchronuous move:
                if a == b:
                    log_idx += 1
                    continue 
                # (a, >>) log move on activity a:  activity a executed in the trace should not have been executed according to the model.
                if a != ">>" and b == ">>":
                    dt = str((a,b))
                    dev_pos_by_case[cid].append((log_idx, dt))
                    dev_types.add(dt) 
                # (>>, b) model move on b:activity b is prescribed by model but missing in the trace.
                elif a == ">>" and b != ">>":
                    dt = str((a,b))
                    dev_pos_by_case[cid].append((log_idx, dt))
                    dev_types.add(dt)
                else:
                    continue
                    
        return sorted(dev_types), dict(dev_pos_by_case), 

    def generate_individual_labels(self,
                                   trace_attr: List[str],
                                   max_prefix_cap: int = None,
                                   conf_runs: int = 100) -> Tuple[Any, Any]:
        """
        Run conformance and produce labeled prefixes.
        label_strategy:
        - "collective": one DataFrame containing all deviation targets + single encoder dict.
        - "separate":  dict[label] -> DataFrame (only that label's target), dict[label] -> encoder dict.
        """
        if self.label_strategy not in {"collective", "separate"}:
            raise ValueError("label_strategy must be 'collective' or 'separate'")
        # Get dataframe and event log objects
        df_raw, ev_log = self._load_log_csv()
        # Get petri net object out of the process model
        net, im, fm = self._pre_process_process_model()

        # Run 100 conformance alignment:
        best_D = ([], {})  # (dev_types, dev_pos_by_case)
        for _ in trange(conf_runs):
            conformance = pm4py.conformance.conformance_diagnostics_alignments(ev_log, net, im, fm, multi_processing=False)
            deviations, dev_pos_by_case = self._extract_deviations_from_alignment(ev_log, conformance)
            # Choose the smallest deviation set:
            if len(best_D[0]) == 0 or len(best_D[0]) >= len(deviations):
                best_D = (deviations, dev_pos_by_case)
        # D_LB with fewest deviations, positions in suffixes of deviations
        dev_types, dev_pos_by_case = best_D
        
        # Build prefixes and targets:
        df_sorted = df_raw.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
        
        # vocabularies for activities/resources
        activities = df_sorted["concept:name"].fillna("NA").astype(str).unique().tolist()
        act2idx = {act: (i+1) for i, act in enumerate(sorted(activities))}   # 0 reserved for PAD

        # Some logs store resources in `org:resource`, others in `org:group`.
        if "org:resource" in df_sorted.columns:
            resource_source_col = "org:resource"
        elif "org:group" in df_sorted.columns:
            resource_source_col = "org:group"
        else:
            # Ensure downstream logic always has a resource column.
            resource_source_col = "org:resource"
            df_sorted[resource_source_col] = "NA"

        resources = df_sorted[resource_source_col].fillna("NA").astype(str).unique().tolist()
        res2idx = {res: (i+1) for i, res in enumerate(sorted(resources))}
            
        
        months = df_sorted["time:timestamp"].apply(lambda x: f"{x.month}_{x.year}")
        unique_months = sorted(months.unique())
        month2idx = {month: (i+1) for i, month in enumerate(unique_months)}
        
        # case attribute encoders
        case_attr_list = trace_attr or []
        
        # initializes and fits LabelEncoder objects for each trace attribute
        case_attr_encoders: Dict[str, LabelEncoder] = {}
        for ca in case_attr_list:
            le = LabelEncoder()
            if ca not in df_sorted.columns:
                df_sorted[ca] = "NA"
            col = df_sorted.groupby("case:concept:name")[ca].first().fillna("NA").astype(str).values
            le.fit(col)
            case_attr_encoders[ca] = le
            
        # determine the maximum case length to pad the 
        case_lengths = df_sorted.groupby("case:concept:name").size().to_dict()
        L_max = max(case_lengths.values()) if len(case_lengths) > 0 else 0
        if max_prefix_cap is not None:
            L_max = min(L_max, max_prefix_cap)
            
        # build rows (store lists inside DataFrame cells)
        rows = []
        for cid, g in df_sorted.groupby("case:concept:name", sort=False):
            g = g.reset_index(drop=True)
            n = len(g)

            # encode additional trace attrs once per case
            encoded_case_attrs = {}
            for ca in case_attr_list:
                
                raw = g[ca].iloc[0] if ca in g.columns else "NA"
                
                raw = "NA" if pd.isna(raw) else raw
                le = case_attr_encoders[ca]
                try:
                    enc = int(le.transform([str(raw)])[0])
                except Exception:
                    enc = len(le.classes_)
                encoded_case_attrs[ca] = enc

            start_ts = g.loc[0, "time:timestamp"]

            for i in range(n):  # prefix ends at index i
                # prepare lists of length L_max
                act_row = [0] * L_max
                res_row = [0] * L_max
                month_row = [0] * L_max  # only month number

                for p in range(1, L_max + 1):
                    if p-1 <= i:
                        # values from the log
                        act = str(g.loc[p-1, "concept:name"]) if "concept:name" in g.columns else "NA"

                        res = str(g.loc[p-1, resource_source_col]) if resource_source_col in g.columns else "NA"
                        ts  = g.loc[p-1, "time:timestamp"]
                        # encoded values
                        act_row[p-1] = act2idx.get(act, 0)
                        res_row[p-1] = res2idx.get(res, 0)

                        if pd.isna(ts) or pd.isna(start_ts):
                            month_row[p-1] = 0
                        else:
                            month_str = f"{ts.month}_{ts.year}" if pd.notna(ts) else "NA"
                            month_row[p-1] = month2idx.get(month_str, 0)
                    else:
                        # PAD remains 0
                        pass

                # for prefixes build classes for suffix labels
                devs_in_suffix = []
                for pos, dt in dev_pos_by_case.get(cid, []):
                    if pos > i or (pos >= i and i == n - 1):
                        devs_in_suffix.append(dt)

                # base row fields
                base = {"case_id": str((cid, i+1)),
                        "activities": act_row,
                        "resources": res_row,
                        "months": month_row}
                
                # add case_attr_<name> columns
                for ca in case_attr_list:
                    base[f"trace_attr_{ca}"] = encoded_case_attrs[ca]

                # add y_<dev_label> target columns
                for dt in dev_types:
                    base[f"y_{dt}"] = 1 if dt in devs_in_suffix else 0

                rows.append(base)

        df_flat = pd.DataFrame(rows)
        
        trace_attr_cols = [col for col in df_flat.columns if col.startswith("trace_attr_")]
        trace_attr_scalers: Dict[str, StandardScaler] = {}
        for col in trace_attr_cols:
            scaler = StandardScaler()
            col_values = df_flat[col].astype(float).to_numpy().reshape(-1, 1)
            df_flat[col] = scaler.fit_transform(col_values).astype(np.float32).ravel()
            trace_attr_scalers[col] = scaler

        base_encoders = {"activity_ids": act2idx,
                         "resource_ids": res2idx,
                         "month_ids": month2idx,
                         "trace_attr_encoders": case_attr_encoders,
                         "trace_attr_scalers": trace_attr_scalers,
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

        # Sort case IDs before shuffling so the split is stable even if df row order changes.
        cases = df["case_id"].dropna().astype(str).unique()
        if len(cases) == 0:
            empty = df.iloc[0:0].copy()
            return empty, empty.copy(), empty.copy()

        cases = np.array(sorted(cases.tolist(), key=str), dtype=object)

        rng = np.random.default_rng(seed)
        cases = rng.permutation(cases)

        n_train = int(len(cases) * train_frac)
        if len(cases) > 0:
            n_train = max(1, min(n_train, len(cases)))
        train_cases = cases[:n_train]
        # remaining_cases = cases[n_train:]

        val_cases = np.array([], dtype=train_cases.dtype if len(train_cases) else object)
        if val_frac > 0 and len(train_cases) > 1:
            n_val = max(1, int(len(train_cases) * val_frac))
            n_val = min(n_val, len(train_cases) - 1)
            if n_val > 0:
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
                   seed: Union[int, None] = None,
                   train_frac: float = 2/3,
                   val_frac: float = 0.0):
        data = self.df_labeled_deviations
        seed = self.seed if seed is None else seed

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
            split_seed = (seed + idx) if seed is not None else None
            train_df, val_df, test_df = self._split_dataframe_by_cases(df_label, split_seed, train_frac, val_frac)
            train_dict[label] = train_df
            if val_frac > 0:
                val_dict[label] = val_df
            test_dict[label] = test_df

        if val_frac == 0:
            val_dict = {}

        return train_dict, val_dict, test_dict
        

class Undersampling:
    def __init__(self,
                 train_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 list_dynamic_cols: List[str],
                 label_strategy: str = "collective"):
        """
        strategy:
        -"collective": run OSS per label but keep the union of sampled rows so every deviation type contributes equally to one shared dataset.
        -"separate":  run OSS per label and return a dict[label] -> undersampled dataframe.
        """
        if label_strategy not in {"collective", "separate"}:
            raise ValueError("strategy must be 'collective' or 'separate'")
        if label_strategy == "collective" and not isinstance(train_data, pd.DataFrame):
            raise TypeError("Collective strategy expects a single DataFrame.")
        if label_strategy == "separate" and not isinstance(train_data, dict):
            raise TypeError("Separate strategy expects a dict[label -> DataFrame].")
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
    def _reflat_list_cols(df_flat: pd.DataFrame, mapping: Dict[str, List[str]], original_cols: List[str]) -> pd.DataFrame:
        df_restored = df_flat.copy()
        for original_col, expanded_cols in mapping.items():
            existing = [c for c in expanded_cols if c in df_restored.columns]
            if not existing:
                continue
            df_restored[original_col] = df_restored[existing].values.tolist()
            df_restored = df_restored.drop(columns=existing)
        df_restored = df_restored[[c for c in original_cols if c in df_restored.columns]]
        return df_restored

    def _undersample_single_label(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        original_cols = df.columns.tolist()
        df_flat, mapping = self._flat_list_cols(df, self.list_dynamic_cols)
        feature_cols = [col for col in df_flat.columns if not col.startswith('y_') and col != 'case_id']
        target_cols = [col for col in df_flat.columns if col.startswith('y_')]

        if not target_cols:
            return df.copy(), []
        if len(target_cols) > 1:
            raise ValueError("Separate strategy expects exactly one target column per dataframe.")

        target_col = target_cols[0]
        y = df_flat[target_col].values
        if np.sum(y == 1) == 0:
            return df.drop(columns=[target_col], errors="ignore").copy(), [target_col]

        if not feature_cols:
            return df.copy(), []

        X = df_flat[feature_cols].values
        oss = OneSidedSelection(sampling_strategy='auto', random_state=42)
        oss.fit_resample(X, y)
        selected = sorted(set(oss.sample_indices_))

        df_resampled_flat = df_flat.iloc[selected].reset_index(drop=True)
        df_restored = self._reflat_list_cols(df_resampled_flat, mapping, original_cols)
        return df_restored, []

    def _collective_oss(self, df: pd.DataFrame):
        original_cols = df.columns.tolist()
        df_flat, mapping = self._flat_list_cols(df, self.list_dynamic_cols)

        feature_cols = [col for col in df_flat.columns if not col.startswith('y_') and col != 'case_id']
        target_cols = [col for col in df_flat.columns if col.startswith('y_')]

        selected_by_label: Dict[str, Set[int]] = {}
        no_true_class: List[str] = []

        for target_col in target_cols:
            y = df_flat[target_col].values
            if np.sum(y == 1) == 0:
                no_true_class.append(target_col)
                continue
            if not feature_cols:
                continue

            X = df_flat[feature_cols].values
            oss = OneSidedSelection(sampling_strategy='auto', random_state=42)
            oss.fit_resample(X, y)
            selected_by_label[target_col] = set(oss.sample_indices_)

        if not selected_by_label:
            df_resampled = df.drop(columns=no_true_class, errors="ignore").copy()
            df_resampled = df_resampled[[c for c in original_cols if c in df_resampled.columns]]
            return df_resampled, no_true_class

        union_idx = sorted(set.union(*selected_by_label.values()))
        df_resampled_flat = df_flat.iloc[union_idx].reset_index(drop=True)
        df_resampled_flat = df_resampled_flat.drop(columns=no_true_class, errors="ignore")
        df_restored = self._reflat_list_cols(df_resampled_flat, mapping, original_cols)
        return df_restored, no_true_class

    def _separate_oss(self, data: Dict[str, pd.DataFrame]):
        undersampled: Dict[str, pd.DataFrame] = {}
        no_true_class: List[str] = []

        for label, df_label in data.items():
            df_oss, missing = self._undersample_single_label(df_label)
            undersampled[label] = df_oss
            if missing:
                no_true_class.extend(missing)

        return undersampled, list(dict.fromkeys(no_true_class))

    def one_sided_selection_undersampling(self):
        """
        Apply OSS according to the chosen strategy.
        Returns:
            - collective: (undersampled_df, no_true_class)
            - separate: (dict[label] -> undersampled_df, no_true_class)
        """
        if self.strategy == "collective":
            return self._collective_oss(self.train_data)

        return self._separate_oss(self.train_data)
        
        
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

    def _to_tensor_dataset(self,
                           df: pd.DataFrame,
                           y_columns: List[str],
                           device=None):
        
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
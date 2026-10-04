import random
import numpy as np
from typing import Any, Dict, List, Tuple, Union
from collections import Counter
from pathlib import Path
import pickle

class DeviationPrediction:
    def __init__(self, pred_conf_set):
        # list of dicts containing: target case, alignment, fitness, cost
        self.pred_conf_set = pred_conf_set

    def __get_case_meta(self, n: int):
        case_ids = self.pred_conf_set.get("case_id")
        labels = self.pred_conf_set.get("label")

        if case_ids is not None and len(case_ids) != n:
            raise ValueError("Mismatched length for pred_conf_set['case_id'].")
        if labels is not None and len(labels) != n:
            raise ValueError("Mismatched length for pred_conf_set['label'].")

        return case_ids, labels
        
    def __get_target_aligns_pref_suf(self) -> List[Any]:
        """
        Returns the list of target alignments.
        """
        prefs = [tgt['prefix'] for tgt in self.pred_conf_set['target_conformance']]
        tgt_aligns = [tgt['suffix_alignment'] for tgt in self.pred_conf_set['target_conformance']]
        tgt_sufs = [tgt['target_suffix'] for tgt in self.pred_conf_set['target_conformance']]

        return prefs, tgt_aligns, tgt_sufs

    def __get_aggregated_alignments(self, aggregation: str='median') -> List[List[Any]]:
        """
        Gets for each prefix the list of 100 sampled alignments. 
        Then takes the median fitness and filters the samples with this fitness scores, then takes a random sample of those filtered.
        """
        prefs = []        
        pred_aligns = []
        pred_sufs = []

        for smpls in self.pred_conf_set["samples_conformance"]:
            # Extract fitness values
            fitness_values = np.array([smpl["suffix_fitness"] for smpl in smpls])
            
            if aggregation == 'median':
                # Get aggregated fitness score
                fitness = np.median(fitness_values)
            else:
                fitness = np.mean(fitness_values)
            
            # Get alignments of suffix
            alignments = [smpl["suffix_alignment"] for smpl in smpls if fitness == smpl['fitness']]
            # Get prefix                                  
            prefixes = [smpl["prefix"] for smpl in smpls if fitness == smpl['fitness']]
            # Get sampled suffixes
            suffixes = [smpl["sampled_suffix"] for smpl in smpls if fitness == smpl['fitness']]
            
            # Randomly choose one alignment from median list
            if len(alignments) > 0:
                # Randomness makes results non-deterministic
                idx = random.randrange(len(alignments))
                # Get fixed suffix and alignments
                # idx = 0
                prefs.append(prefixes[idx])
                pred_aligns.append(alignments[idx])
                pred_sufs.append(suffixes[idx])
                
            else:
                # idx = random.randrange(len(smpls))
                idx = 0
                prefs.append([smpl["prefix"] for smpl in smpls][idx])
                pred_aligns.append([smpl["suffix_alignment"] for smpl in smpls][idx])
                pred_sufs.append([smpl["sampled_suffix"] for smpl in smpls][idx])
                
        # return pred_aligns, pred_prefs, pred_sufs, pred_aligns_prob
        return prefs, pred_aligns, pred_sufs
    
    def get_aggregated_deviations(self, aggregation: str = 'median', include_positions: bool = False, eval_purpose: bool = False):
        """
        Return per-case deviation info.
        - Removes ('>>', None), (None, '>>') filler from alignments.
        - Clears entries that belong to the prefix (prefix matches are removed).
        - Collects deviations only in the suffix region (indices >= len(prefix)).
        """
        tgt_prefs = None
        tgt_aligns = None
        tgt_sufs = None
        if eval_purpose:
            tgt_prefs, tgt_aligns, tgt_sufs = self.__get_target_aligns_pref_suf()

        prefs, pred_aligns, pred_sufs = self.__get_aggregated_alignments(aggregation=aggregation)

        # Basic sanity checks for length consistency
        n = len(prefs)
        
        if eval_purpose:
            if not (len(prefs) == n and len(tgt_aligns) == n and len(pred_aligns) == n and len(tgt_sufs) == n and len(pred_sufs) == n):
                raise ValueError("Mismatched lengths between target/predicted prefixes/aligns/suffixes.")

        case_ids, labels = self.__get_case_meta(n)
        
        # Remove filler form suffix alignments:
        cleaned_tgt_alignments = None
        tgt_deviations = None
        if eval_purpose:
            cleaned_tgt_alignments = [[a for a in align if a != ('>>', None) and a != (None, '>>')] for align in tgt_aligns]
            tgt_deviations = [[(a,b) for (a,b) in align if a != b] for align in cleaned_tgt_alignments]
        
        cleaned_pred_alignments = [[a for a in align if a != ('>>', None) and a != (None, '>>')] for align in pred_aligns]
        pred_deviations = [[(a,b) for (a,b) in align if a != b] for align in cleaned_pred_alignments]
        
        results = []
        for i in range(n):
            if eval_purpose:
                result = {"case_id": case_ids[i],
                          "label": int(labels[i]),
                          # prefix of case
                          "prefix": tgt_prefs[i] if tgt_prefs is not None else prefs[i],
                          # target and aggregated (median+random) 
                          "tgt_suffix": tgt_sufs[i],
                          "pred_suffix": pred_sufs[i],
                          # All suffix aligning (synchronous) and deviating moves 
                          "tgt_cleaned_aligns": cleaned_tgt_alignments[i],
                          "pred_cleaned_aligns": cleaned_pred_alignments[i],
                          # All suffix deviating only moves
                          "tgt_deviations": tgt_deviations[i],
                          "pred_deviations": pred_deviations[i]}
            else:
                result = {"case_id": case_ids[i],
                          "label": int(labels[i]),
                          # prefix of case
                          "prefix": prefs[i],
                          # aggregated (median+random) 
                          "pred_suffix": pred_sufs[i],
                          # All suffix aligning (synchronous) and deviating moves 
                          "pred_cleaned_aligns": cleaned_pred_alignments[i],
                          # All suffix deviating only moves
                          "pred_deviations": pred_deviations[i]}

            if include_positions:
                # Target positions: when evaluating, include for all cases.
                if eval_purpose:
                    tgt_align = result.get("tgt_cleaned_aligns") or []
                    tgt_model, tgt_log = _extract_positions_sequence_index(tgt_align)
                    result["tgt_model_moves"] = tgt_model
                    result["tgt_log_moves"] = tgt_log

                # Predicted positions: keep existing behavior (risk cases only).
                if int(labels[i]) == 0:
                    pred_align = result.get("pred_cleaned_aligns") or []
                    pred_model, pred_log = _extract_positions_sequence_index(pred_align)
                    result["pred_model_moves"] = pred_model
                    result["pred_log_moves"] = pred_log

            results.append(result)

        return results

    def get_aggregated_deviations_with_positions(self, eval_purpose: bool = False, aggregation: str = 'median'):
        """
        Backward-compatible wrapper.

        Position data is computed inside get_aggregated_deviations(..., include_positions=True).
        """
        return self.get_aggregated_deviations(aggregation=aggregation, include_positions=True, eval_purpose=eval_purpose)

    def __get_all_alignments(self) -> List[List[Any]]:
        """
        Collects every sampled alignment, prefix, and suffix for each case without aggregation.
        """
        prefs = []
        all_aligns = []
        all_sufs = []

        for smpls in self.pred_conf_set["samples_conformance"]:
            prefixes = [smpl["prefix"] for smpl in smpls]
            alignments = [smpl["suffix_alignment"] for smpl in smpls]
            suffixes = [smpl["sampled_suffix"] for smpl in smpls]

            prefs.append(prefixes)
            all_aligns.append(alignments)
            all_sufs.append(suffixes)

        return prefs, all_aligns,  all_sufs
    
    def get_probabilistic_deviations(self, deviation_thresholds: dict, include_positions: bool = True, eval_purpose: bool = False):
        """
        Like get_aggregated_deviations() but uses probabilistic predicted alignments and deviations.
        """
        tgt_prefs = None
        if eval_purpose:
            tgt_prefs, tgt_aligns, tgt_sufs = self.__get_target_aligns_pref_suf()
            # Remove filler from target alignments
            cleaned_tgt_alignments = [[a for a in align if a != ('>>', None) and a != (None, '>>')] for align in tgt_aligns]
            tgt_deviations = [[(a, b) for (a, b) in align if a != b] for align in cleaned_tgt_alignments]

        _prefs_samples, pred_aligns_all, pred_sufs_all = self.__get_all_alignments()
        
        n = len(pred_aligns_all)

        if eval_purpose:
            if not (len(tgt_aligns) == len(tgt_sufs) == len(pred_aligns_all) == len(pred_sufs_all) == n):
                raise ValueError("Mismatched lengths between target/predicted prefixes/alignments/suffixes.")

        case_ids, labels = self.__get_case_meta(n)

        # Aggregate probabilistic deivations
        results = []
        for i in range(n):            
            # Get all samples for this case (prefix)
            smpls = self.pred_conf_set["samples_conformance"][i]
            
            total_samples = len(smpls)

            # Collect cleaned alignments and deviations for each sample
            sampled_suffixes = []
            cleaned_aligns = []
            sample_devs = []
            for smpl in smpls:
                 suffix = smpl["sampled_suffix"]
                 sampled_suffixes.append(suffix)
                     
                 # Alignment 
                 align = smpl["suffix_alignment"]
                 cleaned_align = [a for a in align if a != ('>>', None) and a != (None, '>>')]
                 cleaned_aligns.append(cleaned_align)
                    
                 # Deviations
                 devs = [(a, b) for (a, b) in cleaned_align if a != b]
                 # List of deviations across all samples:
                 for dev in devs:
                     sample_devs.append(dev)

            # Count frequencies
            counter_devs = Counter(sample_devs) 
            all_devs_with_prob = [(k, v / total_samples) for k, v in counter_devs.items()]
            
            pred_deviations = []
            for (k,p) in all_devs_with_prob:
                # If deviation k has calibrated threshold:
                if k in deviation_thresholds.keys():
                    # If prob of k is >= the calibrated threshold add:
                    if p >= deviation_thresholds[k]:
                        pred_deviations.append(k)
                    else:
                        continue
                # If k is not in the deviation label list -> unseen:
                else:
                    # Add the deviation if prob >= as the default prob 50%:
                    if p >= 0.5:
                        pred_deviations.append(k)
                    else:
                        continue
                
            prefix = tgt_prefs[i] if (eval_purpose and tgt_prefs is not None) else (smpls[0].get("prefix") if len(smpls) > 0 else None)

            if eval_purpose:
                result = {"case_id": case_ids[i],
                          "label": int(labels[i]),
                          "prefix": prefix,
                          # suffixes
                          "tgt_suffix": tgt_sufs[i],
                          "pred_suffix": sampled_suffixes,
                          # alignments: sync, model and log moves
                          "tgt_cleaned_aligns": cleaned_tgt_alignments[i],
                          "pred_cleaned_aligns": cleaned_aligns,
                          # All suffix deviating only moves with probability across all samples
                          "tgt_deviations": tgt_deviations[i],
                          # dev if prob across samples is higher than threshold
                          "pred_deviations": pred_deviations,
                          # deviation per case and how often it appears across the samples
                          "deviations_prob_per_case": all_devs_with_prob}
            else:
                result = {"case_id": case_ids[i],
                          "label": int(labels[i]),
                          "prefix": prefix,
                          # suffixes
                          "pred_suffix": sampled_suffixes,
                          # alignments: sync, model and log moves
                          "pred_cleaned_aligns": cleaned_aligns,
                          # dev if prob across samples is higher than threshold
                          "pred_deviations": pred_deviations,
                          # deviation per case and how often it appears across the samples
                          "deviations_prob_per_case": all_devs_with_prob}
                
            # Only compute/store position info for risk cases and for deviations that are actually predicted (i.e., in pred_deviations) prob >= thresh.
            if include_positions and int(labels[i]) == 0:
                if eval_purpose:
                    tgt_model, tgt_log = _extract_positions_sequence_index(cleaned_tgt_alignments[i])
                    pred_model, pred_log, pred_model_pos, pred_log_pos = _aggregate_sample_positions_sequence_index(# T alignments for T samples of case
                                                                                                                    cleaned_aligns,
                                                                                                                    # deviations predicted for the case
                                                                                                                    pred_deviations=pred_deviations)
                    result["tgt_model_moves"] = tgt_model
                    result["tgt_log_moves"] = tgt_log
                    # List of deviation moves observed across samples
                    result["pred_model_moves"] = pred_model
                    result["pred_log_moves"] = pred_log
                    # Position frequencies across all samples (count/num_samples)
                    # Example: pred_log_pos[("Resolve ticket", ">>")] == [(0, 0.5), (4, 0.23), ...]
                    result["pred_model_moves_pos"] = pred_model_pos
                    result["pred_log_moves_pos"] = pred_log_pos
                    result["num_samples"] = int(total_samples)
                else:
                    pred_model, pred_log, pred_model_pos, pred_log_pos = _aggregate_sample_positions_sequence_index(# T alignments for T samples of case
                                                                                                                    cleaned_aligns,
                                                                                                                    # deviations predicted for the case
                                                                                                                    pred_deviations=pred_deviations)
                    # List of deviation moves observed across samples
                    result["pred_model_moves"] = pred_model
                    result["pred_log_moves"] = pred_log
                    # Position frequencies across all samples (count/num_samples)
                    # Example: pred_log_pos[("Resolve ticket", ">>")] == [(0, 0.5), (4, 0.23), ...]
                    result["pred_model_moves_pos"] = pred_model_pos
                    result["pred_log_moves_pos"] = pred_log_pos
                    result["num_samples"] = int(total_samples)
                    
            results.append(result)

        return results

    def get_probabilistic_deviations_with_positions(self, deviation_thresholds: dict, eval_purpose: bool=False):
        """
        Like get_probabilistic_deviations() but also adds deviation position info.
        Adds to each result dict: deviating moves with position across samples.
        Get position as the index of the cleaned alignment list where the deviation occurs.
        """
        return self.get_probabilistic_deviations(deviation_thresholds=deviation_thresholds,
                                                 include_positions=True,
                                                 eval_purpose=eval_purpose)
    
    # Use pickle
    def save(self, path: Union[str, Path], deviations: dict):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(deviations, f)
        return str(path)


# helpers for robust (sequence-index) deviation positions
def _is_real_event(x: Any) -> bool:
    return (x is not None) and (x != ">>")

def _extract_positions_sequence_index(align: List[Tuple[Any, Any]]):
    """
    Only for single list predictions -> target, most likely, aggregated probabilistic
    Returns:
    - model_positions[label] = list of model-seq indices where ('>>', label) occurs
    - log_positions[label]   = list of log-seq indices   where (label, '>>') occurs
    """
    model_positions: Dict[str, List[int]] = {}
    log_positions: Dict[str, List[int]] = {}

    log_i = -1
    model_i = -1

    for (log_move, model_move) in align:
        if _is_real_event(log_move):
            log_i += 1
        if _is_real_event(model_move):
            model_i += 1

        # model deviation: ('>>', x)
        if (log_move == ">>") and _is_real_event(model_move):
            key = ('>>', str(model_move))
            model_positions.setdefault(key, []).append(model_i)

        # log deviation: (x, '>>')
        elif (model_move == ">>") and _is_real_event(log_move):
            key = (str(log_move), '>>')
            log_positions.setdefault(key, []).append(log_i)

    return model_positions, log_positions

def _aggregate_sample_positions_sequence_index(align_samples: List[List[Tuple[Any, Any]]],
                                               pred_deviations: List[Tuple[Any, Any]] | None = None):
    """
    Aggregate deviation positions across alignment samples.
    Keeps deviations as *moves* (tuples) and aggregates their positions.
    - If pred_deviations is provided, only those moves are returned/aggregated.

    Returns:
    - pred_model: list of model-deviation moves (('>>', label)) in pred_deviations
    - pred_log:   list of log-deviation moves ((label, '>>')) in pred_deviations
    - pred_model_pos[move] = list[(pos, freq)] where freq = count(move at pos)/num_samples
    - pred_log_pos[move]   = list[(pos, freq)] where freq = count(move at pos)/num_samples
    """
    num_samples = len(align_samples)
    if num_samples == 0:
        return [], [], {}, {}

    allowed: set[Tuple[str, str]] | None = None
    if pred_deviations is not None:
        # Normalize to the same tuple-of-strings format used below.
        allowed = set((str(a), str(b)) for (a, b) in pred_deviations)

    counts_model_pos: Dict[Tuple[str, str], Dict[int, int]] = {}
    counts_log_pos: Dict[Tuple[str, str], Dict[int, int]] = {}

    for sample in align_samples:
        log_i = -1
        model_i = -1

        for (log_move, model_move) in sample:
            if _is_real_event(log_move):
                log_i += 1
            if _is_real_event(model_move):
                model_i += 1

            # model deviation: ('>>', x)
            if (log_move == ">>") and _is_real_event(model_move):
                move = (">>", str(model_move))
                if (allowed is not None) and (move not in allowed):
                    continue
                pos = int(model_i)
                d = counts_model_pos.setdefault(move, {})
                d[pos] = int(d.get(pos, 0)) + 1

            # log deviation: (x, '>>')
            elif (model_move == ">>") and _is_real_event(log_move):
                move = (str(log_move), ">>")
                if (allowed is not None) and (move not in allowed):
                    continue
                pos = int(log_i)
                d = counts_log_pos.setdefault(move, {})
                d[pos] = int(d.get(pos, 0)) + 1

    pred_model = sorted(counts_model_pos.keys())
    pred_log = sorted(counts_log_pos.keys())

    pred_model_pos: Dict[Tuple[str, str], List[Tuple[int, float]]] = {
        move: [(pos, cnt / num_samples) for pos, cnt in sorted(pos_counts.items())]
        for move, pos_counts in counts_model_pos.items()
    }
    pred_log_pos: Dict[Tuple[str, str], List[Tuple[int, float]]] = {
        move: [(pos, cnt / num_samples) for pos, cnt in sorted(pos_counts.items())]
        for move, pos_counts in counts_log_pos.items()
    }

    return pred_model, pred_log, pred_model_pos, pred_log_pos


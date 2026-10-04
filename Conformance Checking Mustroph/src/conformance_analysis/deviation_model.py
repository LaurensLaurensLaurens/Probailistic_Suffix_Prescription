import numpy as np
from sklearn.metrics import fbeta_score, precision_recall_curve
from typing import Union
from pathlib import Path
import json

class DeviationPredictionCalibration:
    """
    Methods to calibrate the binary threshold per deviation label.

    Goal:
    - Build y_true per case: whether each deviation label appears at least once in the target suffix.
    - Build probs per case: estimated probability that each deviation label appears in a sampled suffix.
    - For each label, pick the probability threshold that maximizes F-beta on the calibration set.
    """
    def __init__(self, risk_conformance_results):
        self.risk_conformance_results = risk_conformance_results
    
    def _get_y(self):
        # Get all tgt alignments
        tgt_aligns = [tgt['suffix_alignment'] for tgt in self.risk_conformance_results['target_conformance']]
        # Remove filler from target alignments
        cleaned_tgt_alignments = [[a for a in align if a != ('>>', None) and a != (None, '>>')] for align in tgt_aligns]
        tgt_deviations = [[(a, b) for (a, b) in align if a != b] for align in cleaned_tgt_alignments]
        
        # Dynamically collect all unique labels (deviations) from the sequences
        all_labels = set()
        for seq in tgt_deviations:
            all_labels.update(seq)

        # Convert to a sorted list for consistent ordering (optional, but good for reproducibility)
        labels = sorted(list(all_labels))

        num_cases = len(tgt_deviations)
        num_labels = len(labels)
        # Now create y_true as a binary matrix: rows = instances, columns = labels: 1 if the label (deviation) appears at least once in the sequence, else 0
        y_true = np.zeros((num_cases, num_labels), dtype=int)

        # Iterate through the target deviations: tuple of devs per case
        for i, seq in enumerate(tgt_deviations):
            # make deviation sequence unique
            seq_set = set(seq)
            for j, label in enumerate(labels):
                if label in seq_set:
                    y_true[i, j] = 1
                    
        return labels, y_true
    
    def _get_threshold_data(self):
        # Target data:
        labels, y_true = self._get_y()

        # Basic consistency check
        n_target = len(self.risk_conformance_results.get("target_conformance", []))
        n_samples = len(self.risk_conformance_results.get("samples_conformance", []))
        if n_target != n_samples:
            raise ValueError(f"Length mismatch: target_conformance={n_target} vs samples_conformance={n_samples}.")
        
        # number of cases:
        n = n_samples

        # Aggregate probabilistic deviations.
        # Probability is computed as fraction of sampled suffixes in which the deviation appears at least once
        # (binary per-sample presence), not raw count of occurrences across all samples.
        label_to_idx = {lbl: j for j, lbl in enumerate(labels)}
        probs = []
        for case_idx in range(n):
            smpls = self.risk_conformance_results["samples_conformance"][case_idx]
            total_samples = len(smpls)
            if total_samples == 0:
                raise ValueError(f"No samples available for case {case_idx}.")

            # Count in how many sampled suffixes each deviation label appears (at least once).
            present_counts = np.zeros(len(labels), dtype=float)
            for smpl in smpls:
                align = smpl["suffix_alignment"]
                cleaned_align = [a for a in align if a != ('>>', None) and a != (None, '>>')]

                # Use a set so each deviation counts at most once per sampled suffix
                devs_in_sample = {(a, b) for (a, b) in cleaned_align if a != b}
                for dev in devs_in_sample:
                    j = label_to_idx.get(dev)
                    if j is not None:
                        present_counts[j] += 1.0

            probs.append((present_counts / float(total_samples)).tolist())
        # numpy array:      
        probs = np.array(probs)  
        return probs, (labels, y_true)
            
    def find_optimal_thresholds(self,
                                beta: float = 1.0,
                                per_label: bool = True,
                                preference: str = "balanced"):
        """
        Find optimal threshold(s) by maximizing F-beta.

        Parameters
        - beta: F-beta parameter. beta < 1 emphasizes precision, beta > 1 emphasizes recall.
        - preference: convenience wrapper around beta:
            "precision": uses beta=0.5 unless you pass beta explicitly
            "balanced": uses beta=1.0
            "recall": uses beta=2.0
        - If you want full control, set preference="custom" and pass beta.
        """
        pref = (preference or "balanced").lower().strip()
        if pref not in {"precision", "balanced", "recall", "custom"}:
            raise ValueError("preference must be one of {'precision','balanced','recall','custom'}")
        if pref != "custom":
            # only override beta for non-custom convenience modes
            beta = {"precision": 0.5, "balanced": 1.0, "recall": 2.0}[pref]
        if beta <= 0:
            raise ValueError("beta must be > 0")
        
        # probs = np.array([[1.0, 1.0, 0.55, ...]])# Shape (N: cases, M: number of labels)
        # y_true = np.array([[1, 0, 1, ...]])      # Ground truths
        probs, (labels, y_true) = self._get_threshold_data()
        
        thresholds = {}
        if per_label:
            for j in range(probs.shape[1]):
                y_j = y_true[:, j]
                p_j = probs[:, j]

                # If there are no positives, PR curve is degenerate; choose a conservative threshold.
                if int(np.sum(y_j)) == 0:
                    thresholds[labels[j]] = 1.0
                    continue

                prec, rec, thresh = precision_recall_curve(y_j, p_j)

                # precision_recall_curve returns:
                # prec, rec of length len(thresh)+1
                # thresh of length n_unique_scores
                if thresh.size == 0:
                    thresholds[labels[j]] = 0.5
                    continue

                prec_t = prec[:-1]
                rec_t = rec[:-1]
                fbeta = (1 + beta**2) * (prec_t * rec_t) / ((beta**2 * prec_t) + rec_t + 1e-10)
                best_idx = int(np.argmax(fbeta))
                thresholds[labels[j]] = float(thresh[best_idx])
        # Global threshold via micro F-beta
        else:  
            candidates = np.linspace(0.01, 0.99, 99)
            fbetas = []
            for t in candidates:
                preds = (probs >= t).astype(int)
                fbetas.append(fbeta_score(y_true.ravel(), preds.ravel(), beta=beta, average='micro'))
            thresholds['global'] = candidates[np.argmax(fbetas)]
        return thresholds
    
    # Save the trained logistic regression model
    def save(self, path: Union[str, Path], thresholds: dict):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        def serialize_key(k):
            # deviation labels are typically tuples like (a,b); keep those as lists for JSON.
            if isinstance(k, tuple):
                return list(k)
            return k

        payload = [{"key": serialize_key(k), "value": float(v)} for k, v in thresholds.items()]
        with path.open("w") as f:
            json.dump(payload, f, indent=4)
        
        return str(path)

from typing import Optional

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

class PredictionResults:
    def __init__(self, model, test_set, mode: str = "collective"):
        self.model = model
        self.test_set = test_set
        self.mode = mode
        if self.mode not in {"collective", "separate"}:
            raise ValueError(f"Unsupported evaluation mode '{self.mode}'.")

    def get_predictions_targets(self,
                                model_mode: Optional[str] = 'lstm',
                                batch_size: int = 256,
                                shuffle: bool = False,
                                device: torch.device = torch.device('cpu')):
        if self.mode == "collective":
            return self._run_single(model_mode=model_mode,
                                    model=self.model,
                                    dataset=self.test_set,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    device=device,
                                    use_softmax=False)
        
        if self.mode == "separate":
            if isinstance(self.test_set, dict):
                if not isinstance(self.model, dict):
                    raise ValueError("Separate mode with multiple labels expects 'model' as a dict keyed by label.")
                results = {}
                for label, dataset in self.test_set.items():
                    if label not in self.model:
                        raise KeyError(f"No model provided for label '{label}'.")
                    results[label] = self._run_single(model_mode=model_mode,
                                                      model=self.model[label],
                                                      dataset=dataset,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      device=device,
                                                      use_softmax=True)
                return results
            
            return self._run_single(model_mode=model_mode,
                                    model=self.model,
                                    dataset=self.test_set,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    device=device,
                                    use_softmax=True)
        
        raise ValueError(f"Unsupported evaluation mode '{self.mode}'.")

    def _run_single(self, model_mode, model, dataset, batch_size, shuffle, device, use_softmax: bool):
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        model.eval()
        
        all_targets, all_probs, all_preds = [], [], []
        
        with torch.no_grad():
            for batch in test_loader:
                if model_mode == 'ffn':
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                else:
                    x_act, x_res, x_month, x_trace, y = batch
                    x_act = x_act.to(device)
                    x_res = x_res.to(device)
                    x_month = x_month.to(device)
                    x_trace = x_trace.to(device)
                    y = y.to(device)
                    logits = model(x_act, x_res, x_month, x_trace)
                
                if use_softmax:
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = torch.argmax(probs, dim=1, keepdim=True)
                    preds = torch.zeros_like(probs).scatter_(1, pred_idx, 1)
                
                else:
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).long()
                
                all_probs.append(probs.cpu())
                all_targets.append(y.cpu())
                all_preds.append(preds.cpu())
        
        probs = torch.cat(all_probs).numpy()
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        
        return probs, preds, targets
    
class Metrics:
    def __init__(self, preds, targets):
        """
        preds, targets: array-like with shape (N, m), binary {0,1} values.
        """
        self.preds = np.asarray(preds)
        self.targets = np.asarray(targets)
        if self.preds.shape != self.targets.shape:
            raise ValueError("preds and targets must have the same shape (N, m)")

    def _per_label_counts(self):
        # returns arrays of TP, FP, FN per label (for positive class==1)
        tp = np.sum((self.preds == 1) & (self.targets == 1), axis=0).astype(float)
        fp = np.sum((self.preds == 1) & (self.targets == 0), axis=0).astype(float)
        fn = np.sum((self.preds == 0) & (self.targets == 1), axis=0).astype(float)
        tn = np.sum((self.preds == 0) & (self.targets == 0), axis=0).astype(float)
        return tp, fp, fn, tn

    def macro_precision_recall_dev(self):
        """
        Compute macro-averaged precision and recall for deviation predictions.
        Deviation = label==1, TP = pred 1 and target 1.
        Returns dict with per-label arrays and macro averages.
        """
        tp, fp, fn, tn = self._per_label_counts()
        with np.errstate(divide='ignore', invalid='ignore'):
            precision_per_label = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
            recall_per_label = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)

        macro_precision = float(np.nanmean(precision_per_label))
        macro_recall = float(np.nanmean(recall_per_label))

        return {
            "precision_per_label": precision_per_label,
            "recall_per_label": recall_per_label,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall
        }

    def macro_precision_recall_no_dev(self):
        """
        Compute macro-averaged precision and recall for non-deviation predictions (treat 0 as positive).
        Non-deviation TP = pred 0 and target 0.
        """
        tp, fp, fn, tn = self._per_label_counts()
        
        # For class "0" as positive:
        tp0 = tn
        fp0 = fn  # predicted 0 but target 1 -> false positive for class 0
        fn0 = fp  # predicted 1 but target 0 -> false negative for class 0

        with np.errstate(divide='ignore', invalid='ignore'):
            precision_per_label_0 = np.divide(tp0, tp0 + fp0, out=np.zeros_like(tp0), where=(tp0 + fp0) != 0)
            recall_per_label_0 = np.divide(tp0, tp0 + fn0, out=np.zeros_like(tp0), where=(tp0 + fn0) != 0)

        macro_precision_0 = float(np.nanmean(precision_per_label_0))
        macro_recall_0 = float(np.nanmean(recall_per_label_0))

        return {
            "precision_per_label_non_deviation": precision_per_label_0,
            "recall_per_label_non_deviation": recall_per_label_0,
            "macro_precision_non_deviation": macro_precision_0,
            "macro_recall_non_deviation": macro_recall_0
        }

    def plot_macro_pr_auc(self, prob_scores, label_names=None, figsize=(8, 6)):
        """
        Compute per-label PR-AUC (Average Precision) and plot Precision-Recall curves.
        prob_scores: array-like shape (N, m) with predicted probabilities.
        label_names: optional list of m names for legend.
        Returns dict with per-label AP and macro_AP.
        """
        prob_scores = np.asarray(prob_scores)
        if prob_scores.shape != self.targets.shape:
            raise ValueError("prob_scores must match targets shape (N, m)")

        m = prob_scores.shape[1]
        per_label_ap = np.full(m, np.nan)

        plt.figure(figsize=figsize)
        ax = plt.gca()

        for i in range(m):
            y_true_i = self.targets[:, i]
            y_score_i = prob_scores[:, i]

            # skip labels with only one class
            if np.unique(y_true_i).size < 2:
                per_label_ap[i] = np.nan
                continue

            precision, recall, _ = precision_recall_curve(y_true_i, y_score_i)
            ap = average_precision_score(y_true_i, y_score_i)
            per_label_ap[i] = ap

            lab = label_names[i] if (label_names is not None and i < len(label_names)) else f"label_{i}"
            ax.plot(recall, precision, lw=1.5, label=f"{lab} (AP={ap:.3f})")

            # baseline = positive class frequency
            baseline = np.mean(y_true_i)
            ax.hlines(baseline, 0, 1, colors="gray", linestyles="--", linewidth=1)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curves per Deviation Label")
        ax.legend(loc="lower left", fontsize="small")
        plt.show()

        macro_ap = float(np.nanmean(per_label_ap))
        return {"per_label_ap": per_label_ap, "macro_ap": macro_ap}

class MetricsSep:
    """
    Separate-mode metrics for a single binary label.
    preds: array-like shape (N, 2) with per-class scores (logits or probs).
    targets: array-like shape (N,) with integer class indices {0,1}.
    """
    def __init__(self, preds, targets):
        preds = np.asarray(preds)
        targets = np.asarray(targets).reshape(-1).astype(int)
        if preds.ndim != 2 or preds.shape[1] != 2:
            raise ValueError("preds must have shape (N, 2).")
        if preds.shape[0] != targets.shape[0]:
            raise ValueError("preds and targets must share the same sample count.")
        self.targets = targets
        self.preds = np.argmax(preds, axis=1).astype(int)

    def _counts(self):
        tp = float(np.sum((self.preds == 1) & (self.targets == 1)))
        fp = float(np.sum((self.preds == 1) & (self.targets == 0)))
        fn = float(np.sum((self.preds == 0) & (self.targets == 1)))
        tn = float(np.sum((self.preds == 0) & (self.targets == 0)))
        return tp, fp, fn, tn

    def precision_recall_dev(self):
        tp, fp, fn, _ = self._counts()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return {"precision": precision, "recall": recall}

    def precision_recall_no_dev(self):
        tp, fp, fn, tn = self._counts()
        tp0, fp0, fn0 = tn, fn, fp
        precision0 = tp0 / (tp0 + fp0) if (tp0 + fp0) else 0.0
        recall0 = tp0 / (tp0 + fn0) if (tp0 + fn0) else 0.0
        return {"precision": precision0, "recall": recall0}

    def plot_pr_auc(self, prob_scores, figsize=(6, 4)):
        probs = np.asarray(prob_scores)
        if probs.ndim != 2 or probs.shape[1] != 2 or probs.shape[0] != self.targets.shape[0]:
            raise ValueError("prob_scores must be shape (N, 2) aligned with targets.")
        y_score = probs[:, 1]
        if np.unique(self.targets).size < 2:
            raise ValueError("Cannot draw PR curve with a single class present.")
        precision, recall, _ = precision_recall_curve(self.targets, y_score)
        ap = average_precision_score(self.targets, y_score)

        plt.figure(figsize=figsize)
        plt.plot(recall, precision, lw=1.5, label=f"AP={ap:.3f}")
        baseline = np.mean(self.targets)
        plt.hlines(baseline, 0, 1, colors="gray", linestyles="--", linewidth=1)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve (Separate Mode)")
        plt.legend(loc="lower left")
        plt.show()

        return {"average_precision": float(ap)}



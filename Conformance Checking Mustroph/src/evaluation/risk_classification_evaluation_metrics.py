import numpy as np
from sklearn.metrics import (precision_score, recall_score, roc_auc_score)

def _as_1d_int(a):
    return np.asarray(a).reshape(-1).astype(int)

def _aggregate_median_samples_fitness(samples_fitness):
    """
    Aggregate per-case sample fitness arrays into a single median score per case.
    """
    if samples_fitness is None:
        raise ValueError("samples_fitness is None")

    # Common formats in this repo:
    # list[list[float]]: one inner list per case
    # 2D np.ndarray: shape (n_cases, n_samples)
    # list[np.ndarray]
    if isinstance(samples_fitness, np.ndarray) and samples_fitness.ndim == 2:
        if samples_fitness.shape[1] == 0:
            raise ValueError("samples_fitness has zero samples per case")
        return np.nanmedian(samples_fitness.astype(float), axis=1)

    out = []
    for i, case_samples in enumerate(samples_fitness):
        arr = np.asarray(case_samples, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"Empty sample fitness array for case {i}")
        out.append(float(np.nanmedian(arr)))
    return np.asarray(out, dtype=float)

def risk_model_eval(labels, probs, y):
    """
    Evaluation of the risk-controlled conformance classifier RC2C:
    
    Parameters:
    - labels, probs: predictions
    - y: true labels (1 = safe, 0 = risk)
    """
    y_true = np.asarray(y).astype(int)

    # normalize probs to a 1-D array representing P(safe) if possible
    probs_arr = np.asarray(probs)
    # shape (n, 2) like sklearn.predict_proba -> take column 1 as P(safe)
    if probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
        y_score = probs_arr[:, 1]
    else:
        # shape (n,) or (n,1)
        y_score = np.squeeze(probs_arr)
        # ensure it's float
        try:
            y_score = y_score.astype(float)
        except Exception:
            y_score = None

    # labels: ensure 1-D int array
    y_pred = np.squeeze(np.asarray(labels)).astype(int)

    # Basic classification metrics (per-class)
    # TP: pred safe & true safe,  TN: pred risk & true risk, FP: pred risk & true safe, FN: pred safe & true risk 
    prec_safe = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_safe = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    # f1_safe = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # TP: pred risk & true risk, TN: pred safe & true safe, FP: pred safe & true risk, FN: pred risk & true safe
    prec_risk = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_risk = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    # f1_risk = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    # ROC AUC / Average Precision
    roc_auc_safe = None
    roc_auc_risk = None

    if y_score is not None:
        # ROC AUC for 'safe' as positive class
        try:
            roc_auc_safe = roc_auc_score(y_true, y_score)
        except Exception as e:
            print("Could not compute roc_auc_safe:", e)

        # For 'risk' class, use P(risk) = 1 - P(safe)
        score_risk = 1.0 - y_score
        try:
            roc_auc_risk = roc_auc_score((y_true == 0).astype(int), score_risk)
        except Exception as e:
            print("Could not compute roc_auc_risk:", e)

    else:
        print("No continuous prediction scores available (probs). AUC/PR AUC cannot be computed.")

    print("Metrics (class = SAFE, label=1)")
    print(f"Precision (safe): {prec_safe:.4f}")
    print(f"Recall    (safe): {recall_safe:.4f}")
    # print(f"F1        (safe): {f1_safe:.4f}")
    if roc_auc_safe is not None:
        print(f"ROC AUC   (safe): {roc_auc_safe:.4f}")

    print("\nMetrics (class = RISK, label=0)")
    print(f"Precision (risk): {prec_risk:.4f}")
    print(f"Recall    (risk): {recall_risk:.4f}")
    # print(f"F1        (risk): {f1_risk:.4f}")
    if roc_auc_risk is not None:
        print(f"ROC AUC   (risk): {roc_auc_risk:.4f}")

def risk_model_eval_median_fitness(samples_fitness, y, fitness_threshold: float):
    """
    Baseline evaluation using the *median* suffix fitness score across sampled suffixes.
    
    Decision rule:
    - predict safe (1) if median_fitness >= fitness_threshold else risk (0)
    """
    y_true = _as_1d_int(y)
    med = _aggregate_median_samples_fitness(samples_fitness)
    if med.shape[0] != y_true.shape[0]:
        raise ValueError("Length mismatch between samples_fitness and y")

    y_pred = (med >= float(fitness_threshold)).astype(int)

    prec_safe = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_safe = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec_risk = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_risk = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    roc_auc_safe = None
    roc_auc_risk = None
    try:
        roc_auc_safe = roc_auc_score(y_true, med)
    except Exception as e:
        print("Could not compute roc_auc_safe:", e)

    try:
        roc_auc_risk = roc_auc_score((y_true == 0).astype(int), -med)
    except Exception as e:
        print("Could not compute roc_auc_risk:", e)

    print("\nMetrics (class = SAFE, label=1)")
    print(f"Precision (safe): {prec_safe:.4f}")
    print(f"Recall    (safe): {recall_safe:.4f}")
    if roc_auc_safe is not None:
        print(f"ROC AUC   (safe): {roc_auc_safe:.4f}")

    print("\nMetrics (class = RISK, label=0)")
    print(f"Precision (risk): {prec_risk:.4f}")
    print(f"Recall    (risk): {recall_risk:.4f}")
    if roc_auc_risk is not None:
        print(f"ROC AUC   (risk): {roc_auc_risk:.4f}")

    return y_pred, med
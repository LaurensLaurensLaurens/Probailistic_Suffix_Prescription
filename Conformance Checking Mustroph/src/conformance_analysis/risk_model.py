from __future__ import annotations
import math
from typing import Optional, List, Union, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
               
class DataFrameConstruction:
    """
    Create dataframe for risk or safe classification.
    """
    def __init__(self, conformance_results: dict):
        self.res_target_conf = conformance_results['target_conformance']
        self.res_target_conf_suffix_fit = [res['suffix_fitness'] for res in self.res_target_conf]

        self.res_ml_conf = conformance_results['ml_conformance']
        self.res_ml_conf_suffix_fit = [res['suffix_fitness'] for res in self.res_ml_conf]

        self.res_smpl_conf = conformance_results['samples_conformance']
        self.res_smpl_conf_suffix_fit = [[r['suffix_fitness'] for r in res] for res in self.res_smpl_conf]
        
    # Die art auf die, die Smaples aggregiert werden.
    def __aggregate_samples_fitness(self, samples_fitness: np.ndarray, aggregation: str) -> Tuple[float, float]:
        """
        Helper method to aggregate the samples using various moment metrics.
        """
        samples_fitness = np.asarray(samples_fitness, dtype=float)
        if samples_fitness.size == 0:
            raise ValueError("samples_fitness must not be empty")
        
        if aggregation == 'mean':
            agg = float(np.mean(samples_fitness))
        elif aggregation == 'median':
            agg = float(np.median(samples_fitness))
        elif aggregation == 'min':
            agg = float(np.min(samples_fitness))
        elif aggregation == 'max':
            agg = float(np.max(samples_fitness))
        elif aggregation == 'variance':
            ddof = 1 if samples_fitness.size > 1 else 0
            agg = float(np.var(samples_fitness, ddof=ddof))
        elif aggregation == 'std':
            ddof = 1 if samples_fitness.size > 1 else 0
            agg = float(np.std(samples_fitness, ddof=ddof))
        elif aggregation == 'skewness':
            std = float(np.std(samples_fitness, ddof=1) if samples_fitness.size > 1 else 0.0)
            if std == 0.0:
                agg = 0.0
            else:
                mu = float(np.mean(samples_fitness))
                agg = float((np.mean((samples_fitness - mu) ** 3)) / (std ** 3))
        elif aggregation == 'kurtosis':
            std = float(np.std(samples_fitness, ddof=1) if samples_fitness.size > 1 else 0.0)
            if std == 0.0:
                agg = -3.0
            else:
                mu = float(np.mean(samples_fitness))
                agg = float((np.mean((samples_fitness - mu) ** 4)) / (std ** 4) - 3.0)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        
        # sample standard deviation (ddof=1 if >1 sample, else 0)
        std = float(np.std(samples_fitness, ddof=1) if samples_fitness.size > 1 else 0.0)

        return (agg, std)
        
    def __value_at_quantiles(self, values: list, alpha_risk: float) -> Dict[str, Any]:
        """
        Given an unsorted list of floats, return the lower-tail empirical values at q_risk and q_highrisk.
        """
        if not (0.0 <= alpha_risk <= 1.0):
            raise ValueError("alpha_risk must be in [0,1]")

        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return {'q_risk': None}

        # Conservative lower-tail quantile for thresholding.
        try:
            q_risk = float(np.quantile(arr, alpha_risk, method="lower"))
        except TypeError:
            q_risk = float(np.quantile(arr, alpha_risk))

        return {'q_risk': q_risk}
    
    def empirical_quantile_thresholds(self, alpha_risk: float, aggregation: str='mean') -> dict:
        """
        Compute one-sided lower-tail empirical thresholds for q_risk and q_highrisk.
        """
        # Target (suffix)
        target_fitness_scores = self.res_target_conf_suffix_fit
        # target_fitness_scores: [
        #     0.92,   # echter Suffix von Case 1
        #     0.61,   # echter Suffix von Case 2
        #     0.84    # echter Suffix von Case 3
        # ]

        # Get thresholds
        # alpha risk: 10%-Quantil der echten Fitnesswerte bestimmt.
        thresholds_target = self.__value_at_quantiles(target_fitness_scores, alpha_risk)
        # Beispiel:
        # alpha_risk = 0.10
        # Input:
        # [0.92, 0.61, 0.84]
            
        # Most likely (suffix)
        ml_fitness_scores = self.res_ml_conf_suffix_fit
        # Beispiel:
        # [
        #     0.88,   # most-likely prediction Case 1
        #     0.70,   # most-likely prediction Case 2
        #     0.79    # most-likely prediction Case 3
        # ]

        # Get thresholds
        thresholds_ml = self.__value_at_quantiles(ml_fitness_scores, alpha_risk)
        # Beispiel:
        # thresholds_ml =
        # {
        #     "q_risk": 0.70
        # }
            
        # Samples
        sampled_fitness_scores = self.res_smpl_conf_suffix_fit
        # [
        #     [0.95, 0.91, 0.87, 0.93],   # Case 1. Mehrere, da für jeden Case mehrere PSP-Samples enthält
        #     [0.40, 0.58, 0.65, 0.51],   # Case 2
        #     [0.71, 0.83, 0.79, 0.90]    # Case 3
        # ]
        #
        # sampled_fitness_scores[0]
        # -> [0.95, 0.91, 0.87, 0.93]
        
        # Aggreagate the fitness samples (per case): Add tuples (aggregated, std)
        aggragted_sampled_fitness_scores = [self.__aggregate_samples_fitness(samples_fitness=np.array(smp), aggregation=aggregation) for smp in sampled_fitness_scores]
        # Für Case 1 beispielsweise:
        # median([0.95, 0.91, 0.87, 0.93]) = 0.92
        # std(...) ≈ 0.03
        #
        # Ergebnis:
        # (0.92, 0.03)

        # Get thresholds
        thresholds_sampled = self.__value_at_quantiles([agg_smp[0] for agg_smp in aggragted_sampled_fitness_scores], alpha_risk)
        # Beispiel:
        # thresholds_sampled =
        # {
        #     "q_risk": 0.545
        # }

        mean_std_sampled = np.nanmean([agg_smp[1] for agg_smp in aggragted_sampled_fitness_scores])
        # Typ:
        # float
        #
        # Beispiel:
        # mean_std_sampled = 0.0633

        thresholds_sampled['mean_std'] = mean_std_sampled
        # thresholds_sampled =
        # {
        #     "q_risk": 0.545,
        #     "mean_std": 0.0633
        # }
            
        return {'target': thresholds_target,
                'most_likely': thresholds_ml,
                'samples': thresholds_sampled}

        # Beispiel des kompletten Returns:
        #
        # {
        #     "target": {
        #         "q_risk": 0.61
        #     },
        #
        #     "most_likely": {
        #         "q_risk": 0.70
        #     },
        #
        #     "samples": {
        #         "q_risk": 0.545,
        #         "mean_std": 0.0633
        #     }
        # }
        
    # ToDataframe
    def samples_to_dataframe(self,
                             q_risk: float = 0.0,
                             target_col: str = "y",
                             include_tail_features: bool = False):
        """
        Create dataframe for logistic model training, and predictions
        """
        # Target suffix fitness scores
        targets = self.res_target_conf_suffix_fit
        # Predicted samples fitness scores
        predicted_samples = self.res_smpl_conf_suffix_fit

        if len(targets) != len(predicted_samples):
            raise ValueError("Length mismatch between targets and predicted_samples.")

        rows = []
        # 1000 samples
        for i, samples in enumerate(predicted_samples):
            arr = np.asarray(samples, dtype=float)
            if arr.size == 0:
                raise ValueError(f"Empty sample array at index {i}.")
            mean = float(arr.mean())
            median = float(np.median(arr))
            var = float(arr.var(ddof=0))
            std = float(arr.std(ddof=0))
            mn = float(arr.min())
            mx = float(arr.max())
            q25 = float(np.percentile(arr, 25))
            q75 = float(np.percentile(arr, 75))
            iqr = q75 - q25
            cm2 = float(np.mean((arr - mean) ** 2))
            cm3 = float(np.mean((arr - mean) ** 3))
            cm4 = float(np.mean((arr - mean) ** 4))
            skew = (cm3 / (cm2 ** 1.5)) if cm2 > 0 else 0.0
            kurt = ((cm4 / (cm2 ** 2)) - 3.0) if cm2 > 0 else -3.0
            base = [mean, var, std, skew, kurt, median, mn, mx, q25, q75, iqr]

            if include_tail_features:
                q05 = float(np.percentile(arr, 5))
                q10 = float(np.percentile(arr, 10))
                q90 = float(np.percentile(arr, 90))
                q95 = float(np.percentile(arr, 95))

                below = arr[arr < q_risk]
                p_below = float(below.size / arr.size)
                mean_below = float(np.mean(below)) if below.size > 0 else float(q_risk)
                shortfall = float(max(0.0, q_risk - mean_below))
                base.extend([q05, q10, q90, q95, p_below, mean_below, shortfall])

            rows.append(base)

        columns = ['mean','variance','std','skewness','kurtosis_excess','median','min','max','q25','q75','iqr']
        if include_tail_features:
            columns.extend(['q05','q10','q90','q95','p_below_qrisk','mean_below_qrisk','shortfall_below_qrisk'])
        df = pd.DataFrame(rows, columns=columns)
        df[target_col] = [1 if t >= q_risk else 0 for t in targets]
        
        return df
    
# Visualisierung
class ConformalAnalysisVisualizations:
    """
    Visulaize distribution of target and aggregated samples fitness scores.
    """
    def __init__(self, sampled_fitness, target_fitness, ml_fitness):
        self.samples_fitness = sampled_fitness
        self.target_fitness = target_fitness
        self.ml_fitness = ml_fitness

    def __aggregate_samples_fitness(self, samples_fitness, aggregation: str) -> np.ndarray:
        """
        Aggregate each element in samples_fitness using aggregation.
        Accepts:
         - list/tuple of array-like (each inner element is an array of samples)
         - 2D numpy array shaped (n, m) -> aggregates across axis=1
         - 1D array (assumed already aggregated) -> returned as-is (cast to float array)
        """
        if samples_fitness is None:
            raise ValueError("samples_fitness is None")

        # If it's a numpy array and 1D numeric, assume already aggregated
        arr = np.asarray(samples_fitness, dtype=object)

        # helper map
        agg_funcs = {'mean': np.mean,
                     'median': np.median,
                     'min': np.min,
                     'max': np.max}
        
        if aggregation not in agg_funcs:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        agg_f = agg_funcs[aggregation]

        # If arr is object dtype but each element is array-like -> iterate
        if arr.dtype == object or arr.ndim == 1 and any(hasattr(x, '__iter__') for x in arr):
            out = []
            for x in arr:
                x_a = np.asarray(x)
                if x_a.size == 0:
                    raise ValueError("no calibration value exist!")
                out.append(float(agg_f(x_a)))
            return np.array(out, dtype=float)

        # If it's a 2D numeric array, aggregate per-row
        arr_num = np.asarray(samples_fitness, dtype=float)
        if arr_num.ndim == 2:
            if arr_num.shape[1] == 0:
                raise ValueError("no calibration value exist!")
            if aggregation == 'mean':
                return np.nanmean(arr_num, axis=1)
            elif aggregation == 'median':
                return np.nanmedian(arr_num, axis=1)
            elif aggregation == 'min':
                return np.min(arr_num, axis=1)
            elif aggregation == 'max':
                return np.max(arr_num, axis=1)

        # If it's 1D numeric, return it (ensure float dtype)
        if arr_num.ndim == 1:
            if arr_num.size == 0:
                raise ValueError("no calibration value exist!")
            return arr_num.astype(float)

        raise ValueError("Unsupported shape for samples_fitness")
    
    def plot_distribution(self,
                          aggregation: Optional[str] = None,
                          bins=30,
                          show_kde=True,
                          alpha_risk: float = 1.0):
        """
        1) Plot distribution of target fitness and (optional) aggregated sample fitness.
        2) Plot vertical lines for risk fitness score threshold based on (empirical) alpha level.
        """
        # aggregate samples
        smpls_fit = None
        if aggregation is not None:
            smpls_fit = self.__aggregate_samples_fitness(samples_fitness=self.samples_fitness, aggregation=aggregation)

        # handle target_fitness
        target_fit = None
        if self.target_fitness is not None:
            target_fit = np.asarray(self.target_fitness, dtype=float).flatten()
            if target_fit.size == 0:
                target_fit = None

        # Print summary statistics (safe casting to float for formatting)
        def print_stats(name, arr):
            if arr is None:
                print(f"{name}: None")
                return
            arr = np.asarray(arr, dtype=float).flatten()
            n = arr.size
            mean = float(np.mean(arr)) if n > 0 else float('nan')
            median = float(np.median(arr)) if n > 0 else float('nan')
            std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            if n > 1:
                q1, q3 = np.percentile(arr, [25, 75])
            elif n == 1:
                q1 = q3 = float(arr[0])
            else:
                q1 = q3 = float('nan')
            print(f"{name}: n={n}, mean={mean:.4f}, median={median:.4f}, std={std:.4f}, Q1={q1:.4f}, Q3={q3:.4f}")

        if aggregation is not None:
            print_stats("Aggregated samples fitness statistics", smpls_fit)
        print_stats("Target fitness statistics", target_fit)

        # compute requested empirical quantiles (on aggregated samples)
        q_risk = None
 
        # alpha risk
        if not (0.0 <= alpha_risk <= 1.0):
            raise ValueError("alpha_risk must be in [0,1]")
        
        q_risk = float(np.quantile(target_fit, alpha_risk))
        print("Risk threshold fitness score: ",q_risk)

        # Histogram (density) overlay
        plt.figure(figsize=(9, 5))
        if smpls_fit is not None:
            plt.hist(smpls_fit, bins=bins, density=True, alpha=0.6, edgecolor='black', linewidth=0.5, label='aggregated samples fitness', color='blue')
        if target_fit is not None:
            plt.hist(target_fit, bins=bins, density=True, alpha=0.5, edgecolor='black', linewidth=0.5, label='target fitness', color='green')

        # KDE overlays (only if >1 sample)
        if show_kde:
            try:
                valid_xmin, valid_xmax = float('inf'), float('-inf')

                if smpls_fit is not None and smpls_fit.size > 1:
                    valid_xmin = min(valid_xmin, float(smpls_fit.min()))
                    valid_xmax = max(valid_xmax, float(smpls_fit.max()))

                if target_fit is not None and target_fit.size > 1:
                    valid_xmin = min(valid_xmin, float(target_fit.min()))
                    valid_xmax = max(valid_xmax, float(target_fit.max()))

                if valid_xmin < valid_xmax:  # Ensure valid range for KDE
                    x = np.linspace(valid_xmin, valid_xmax, 400)

                if smpls_fit is not None and smpls_fit.size > 1:
                    kde_means = gaussian_kde(smpls_fit)
                    plt.plot(x, kde_means(x), lw=2, label='KDE (samples)', color='blue')

                if target_fit is not None and target_fit.size > 1:
                    kde_target = gaussian_kde(target_fit)
                    plt.plot(x, kde_target(x), lw=2, label='KDE (target)', color='green')

            except Exception as e:
                print(f"Error in KDE computation: {e}")
                # Silently skip KDE if it fails (e.g., scipy missing or KDE error)
                pass

        # draw quantile vertical lines + annotate their numeric values on the x-axis
        ax = plt.gca()
        ylim = ax.get_ylim()
        # small offset above bottom
        y_text_pos = ylim[0] + 0.03 * (ylim[1] - ylim[0])

        if q_risk is not None:
            ax.axvline(q_risk, color='red', linestyle='--', linewidth=2, label=f'risk fitness threshold (alpha-risk quantile={alpha_risk})')
            ax.text(q_risk, y_text_pos, f"{q_risk:.3f}", color='red', ha='center', va='bottom', fontsize=9, backgroundcolor='white')

        plt.xlabel('(aggregated) fitness score')
        plt.ylabel('density')
        if aggregation is not None:
            plt.title(f'Distribution of aggregated samples and target fitness scores')
        else:
            plt.title('Distribution of target fitness scores')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.35)
        plt.tight_layout()
        plt.show()
        
        return q_risk


# PSP-Samples
#     ↓
# Conformance-Fitnesswerte
#     ↓
# Features
# (mean, median, std, quantiles, ...)
#     ↓
# LogisticRegressionModel
#     ↓
# P(safe)
#     ↓
# kalibrierter Threshold
#     ↓
# safe / risk
class LogisticRegressionModel:
    """
    Logistic regression model for predicting "safe" (1) vs "risk" (0).

    This class is intentionally minimal: it supports
    - fitting a scaled logistic regression (optionally probability-calibrated),
    - calibrating a probability threshold using CP (conformal prediction) or CRC (risk control),
    - predicting with that stored threshold,
    - saving/loading.
    """
    def __init__(self, 
                 alpha_quantile_risk: float = 0.5,
                 risk_fitness_threshold: float = 1.0,
                 classifier: Optional[LogisticRegression] = None):
        # Stored for provenance (not used in fitting):
        self.alpha_quantile_risk = float(alpha_quantile_risk)
        self.risk_fitness_threshold = float(risk_fitness_threshold)

        self.classifier = classifier if classifier is not None else LogisticRegression(max_iter=1000,
                                                                                       class_weight="balanced")
        self.pipeline = None
        self.feature_names: Optional[List[str]] = None
        self.trained: bool = False

        # Set by calibrate_crc_safe_threshold
        self.calibration_info: Dict[str, Any] = {}

    def _build_pipeline(self, calibrate: bool):
        scaler = StandardScaler()
        if calibrate:
            base = self.classifier
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        else:
            clf = self.classifier
        self.pipeline = make_pipeline(scaler, clf)

    # Logistic regression fitting
    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            feature_names: Optional[List[str]] = None,
            calibrate: bool = False,
            **fit_kwargs) -> "LogisticRegressionModel":
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns) if feature_names is None else list(feature_names)
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise ValueError(f"X is missing required features: {missing}")
            X_fit = X[self.feature_names]
        else:
            if feature_names is None:
                raise ValueError("feature_names must be provided when X is a numpy array")
            self.feature_names = list(feature_names)
            X_fit = X

        y_arr = np.asarray(y)
        self._build_pipeline(calibrate=bool(calibrate))
        self.pipeline.fit(X_fit, y_arr, **fit_kwargs)
        self.trained = True
        return self

    def fit_from_dataframe(self,
                           df: pd.DataFrame,
                           target_col: str = "y",
                           features: Optional[List[str]] = None,
                           calibrate: bool = False,
                           **fit_kwargs) -> "LogisticRegressionModel":
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not in dataframe")
        features = features or [c for c in df.columns if c != target_col]
        X = df[features]
        y = df[target_col]
        return self.fit(X, y, feature_names=features, calibrate=calibrate, **fit_kwargs)

    def _ensure_feature_order(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                raise RuntimeError("Model has no stored feature names.")
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Input DataFrame missing features: {missing}")
            return X[self.feature_names]
        # numpy array: assume caller ensured correct column order
        return X

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        """Split-conformal quantile with (n+1) correction."""
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1].")
        s = np.asarray(scores, dtype=float).reshape(-1)
        if s.size == 0:
            raise ValueError("Empty calibration scores.")
        s_sorted = np.sort(s)
        k = int(math.ceil((s_sorted.size + 1) * (1.0 - alpha)))
        k = min(max(k, 1), s_sorted.size)
        return float(s_sorted[k - 1])
    
    def calibrate_conformal_threshold(self,
                                      X_cal: Union[np.ndarray, "pd.DataFrame"],
                                      y_cal: Union[np.ndarray, "pd.Series"],
                                      alpha: float = 0.05) -> Dict[str, Any]:
        """
        Split Conformal Prediction (CP) for *set coverage*.

        Uses nonconformity score s(x,y)= 1-p_hat(y|x).
        With binary p = p_hat(y=1|x), the calibration score is:
            s_i = 1 - p_i  if y_i=1
            s_i = p_i      if y_i=0
        Then q = conformal quantile of {s_i}.
        "Safe" label 1 is included in the conformal prediction set iff:
            p >= 1 - q
        We return threshold t = 1 - q: if p >= t -> safe
        It guarantees coverage of the *set predictor*.
        """
        if not getattr(self, "trained", False):
            raise RuntimeError("Model must be trained before calibration.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1].")

        # predictions (probabilities) as float
        p = np.asarray(self.predict_proba(X_cal), dtype=float).reshape(-1)
        # target labels as int
        y = np.asarray(y_cal).reshape(-1)
        y = y.astype(int)
        
        if p.shape[0] != y.shape[0]:
            raise RuntimeError("Calibration features and labels have different lengths.")
        
        # score of the TRUE label: s_i = 1 - p_true, where p_true = p if y=1 else (1-p)
        s = np.where(y == 1, 1.0 - p, p)

        q = self._conformal_quantile(s, alpha=alpha)
        t = float(np.clip(1.0 - q, 0.0, 1.0))

        self.calibration_info = {"method": "CP",
                                 "note": "Split conformal threshold for including label 'safe' (set-coverage, not risk-specific).",
                                 "alpha": float(alpha),
                                 "threshold": t,
                                 "n_cal": int(len(y))}
        return dict(self.calibration_info)

    # conformal risk control (CRC) for false-safe
    def calibrate_crc_safe_threshold(self,
                                     X_cal: Union[np.ndarray, "pd.DataFrame"],
                                     y_cal: Union[np.ndarray, "pd.Series"],
                                     alpha: float = 0.05,
                                     delta: float = 0.05,
                                     n_grid: int = 200,
                                     min_pred_safe: int = 25) -> Dict[str, Any]:
        """
        Conformal Risk Control (CRC) for bounding *false-safe among predicted-safe*: P(Y=0 | predict_safe) <= alpha (with confidence >= 1-delta)
        - Here predict_safe := [p_hat >= t]: We search thresholds t over a grid and pick the one that satisfies: UCB( false_safe_rate_among_safe ) <= alpha

        Implementation detail:
        - We test many thresholds; to keep overall confidence 1-delta we use a Bonferroni adjustment: delta_eff = delta / |grid|.
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before calibration.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0,1].")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0,1).")
        if n_grid < 2:
            raise ValueError("n_grid must be >= 2.")
        if min_pred_safe < 1:
            raise ValueError("min_pred_safe must be >= 1.")

        p = np.asarray(self.predict_proba(X_cal), dtype=float).reshape(-1)
        y = np.asarray(y_cal).reshape(-1)
        y = y.astype(int)
        
        if p.shape[0] != y.shape[0]:
            raise RuntimeError("Calibration features and labels have different lengths.")

        # threshold grid: quantiles of predicted probabilities
        grid = np.quantile(p, np.linspace(0.0, 1.0, n_grid))
        grid = np.unique(grid)
        grid.sort()

        # multiple-testing control across thresholds
        delta_eff = float(delta / max(1, len(grid)))

        def ucb_false_safe_rate(m: int, k: int) -> float:
            if k <= 0:
                return 1.0
            phat = m / k
            rad = math.sqrt(math.log(1.0 / delta_eff) / (2.0 * k))
            return float(min(1.0, phat + rad))

        feasible = []
        # search the grid: a list of sorted values between 0 and 1:
        for t in grid:
            mask = p >= t
            k = int(mask.sum())  # predicted safe
            if k < min_pred_safe:
                continue
            m = int((y[mask] == 0).sum())  # false safe among predicted safe
            ub = float(ucb_false_safe_rate(m, k))
            if ub <= alpha:
                emp = float(m / k)
                feasible.append((float(t), k, m, emp, ub))

        if not feasible:
            print("not feasible!")
            t_hat = 1.0
            self.calibration_info = {"method": "CRC",
                                     "note": "No feasible threshold found; fallback threshold=1.0 (predict safe never).",
                                     "alpha": float(alpha),
                                     "delta": float(delta),
                                     "delta_eff": float(delta_eff),
                                     "threshold": float(t_hat),
                                     "n_cal": int(len(y))}
            return dict(self.calibration_info)

        # choose the smallest feasible threshold -> largest "predicted-safe" set
        # NOTE: even though the predicted-safe sets are nested as t increases,
        # the conditional error P(Y=0 | p>=t) is not guaranteed to be monotone in t.
        # So we explicitly select a threshold from the tested grid.
        t_hat, k, m, emp, ub = min(feasible, key=lambda z: z[0])

        self.calibration_info = {"method": "CRC",
                                 "note": "CRC threshold with UCB on false-safe rate among predicted-safe <= alpha.",
                                 "alpha": float(alpha),
                                 "delta": float(delta),
                                 "delta_eff": float(delta_eff),
                                 "threshold": float(t_hat),
                                 "n_cal": int(len(y))
                                 }
        return dict(self.calibration_info)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model not trained.")
        X_in = self._ensure_feature_order(X)
        return self.pipeline.predict(X_in)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Returns probability for the *safe* class (label == 1).
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")
        X_in = self._ensure_feature_order(X)
        proba = np.asarray(self.pipeline.predict_proba(X_in))

        # robustly select P(class==1) even if class order is not [0,1]
        try:
            classes = self.pipeline[-1].classes_
        except Exception:
            classes = None
        if classes is not None and 1 in set(classes):
            idx = int(np.where(classes == 1)[0][0])
            return proba[:, idx]
        return proba[:, 1]
    
    # Predict with conformal threshold
    def predict_with_threshold(self,
                               X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels using stored conformal calibration threshold.
        Returns labels and optionally probabilities.
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")
        
        if not self.calibration_info:
            raise RuntimeError("No calibration info found. Call calibrate_crc_safe_threshold first.")

        # Predicted probabilities: Calls the fitted pipeline and predict proabilities.
        probs = self.predict_proba(X)
        t_hat = self.calibration_info.get("threshold", None)
        if t_hat is None:
            raise RuntimeError("Calibration info missing threshold.")
        
        labels = np.zeros_like(probs, dtype=int)
        # when using threshold: label is 1 if prob >= t_hat -> safe (1), else risk (0)
        labels[probs >= t_hat] = 1
        
        # Return labels and probabilities
        return (labels, probs)

    # Save the trained logistic regression model
    def save(self, path: Union[str, Path]):
        if not self.trained:
            raise RuntimeError("Train model before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha_quantile_risk": self.alpha_quantile_risk,
            "risk_fitness_threshold": self.risk_fitness_threshold,
            "pipeline": self.pipeline,
            "feature_names": self.feature_names,
            "calibration_info": self.calibration_info,
        }
        joblib.dump(payload, path)
        return str(path)
    
    # Load a trained logistic regression model
    @classmethod
    def load(cls, path: Union[str, Path]) -> "LogisticRegressionModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = joblib.load(path)
        #
        lm = cls(alpha_quantile_risk=float(payload.get("alpha_quantile_risk", 1.0)),
                 risk_fitness_threshold=float(payload.get("risk_fitness_threshold", 1.0)))
        lm.pipeline = payload.get("pipeline")
        lm.feature_names = payload.get("feature_names")
        lm.calibration_info = payload.get("calibration_info", {})
        lm.trained = True
        return lm
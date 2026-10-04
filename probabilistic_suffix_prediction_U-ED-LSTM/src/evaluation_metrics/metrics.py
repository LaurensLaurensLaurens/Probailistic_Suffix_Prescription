from abc import ABC, abstractmethod
import CRPS.CRPS as pscore
from Levenshtein import distance
from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
import numpy as np
from collections import Counter

class EvaluationMetric(ABC):
    def __init__(self, attribute_name, outlier_removal : float = 0):
        self.attribute_name = attribute_name
        self.outlier_removal = outlier_removal

    def remove_outliers_mad(self, data, threshold=3.5, scale_factor=1.4826):
        """
        Removes outliers based on the Median Absolute Deviation (MAD) method.

        Parameters:
        - data: list or numpy array of numerical values
        - threshold: float, default 3.5 (values beyond this * MAD are removed)
        - scale_factor: float, default 1.4826 (scales MAD to match std deviation for normal data)

        Returns:
        - numpy array: Data with outliers removed
        """
        # MAD Method on log-transformed data
        data = np.array(data)
        median = np.median(data)
        mad_value = np.median(np.abs(data - median)) * 1.4826  # Scaled MAD
        
        if mad_value == 0:
            return data  # Avoid division by zero if all values are similar
        
        modified_z_scores = 0.6745 * (data - median) / mad_value
        return data[np.abs(modified_z_scores) < threshold]


    def remove_outliers_log_mad(self, data, threshold=3.5, scale_factor=1.4826):
        """
        Removes outliers based on the Median Absolute Deviation (MAD) method.

        Parameters:
        - data: list or numpy array of numerical values
        - threshold: float, default 3.5 (values beyond this * MAD are removed)
        - scale_factor: float, default 1.4826 (scales MAD to match std deviation for normal data)

        Returns:
        - numpy array: Data with outliers removed
        """
        data = np.array(data)
        min_value = np.min(data)
        shifted_data = data - min_value + 1
        log_data = np.log(shifted_data)
        return self.remove_outliers_mad(log_data, threshold, scale_factor)

    @abstractmethod
    def evaluate(prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        pass

class SuffixCount(EvaluationMetric):
    def __init__(self):
        super().__init__(None)

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value = len(suffix)
        mean_value = len(mean_prediction)
        proba_values = [len(s) for s in predicted_suffixes]
        return true_value, mean_value, proba_values

class SuffixCountMAE(SuffixCount):
    def __init__(self,
                 percentile : float,
                 outlier_removal : float = 0.0):
        super().__init__()
        self.percentile = percentile
        self.outlier_removal = outlier_removal

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value, mean_value, proba_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        mean_mae = abs(true_value - mean_value)
        proba_mean_mae = abs(true_value - np.mean(proba_values))

        proba_maes = [abs(true_value - proba_value) for proba_value in proba_values]
        bottom_proba_mse = np.percentile(proba_maes, 100*self.percentile)
        top_proba_mse = np.percentile(proba_maes, 100*(1-self.percentile))
        return {'mean' : mean_mae, 'prob' : (proba_mean_mae, (bottom_proba_mse, top_proba_mse))} 

class SumValues(EvaluationMetric):
    def __init__(self, attribute_name : str, value_factor : float = 1.0,
                 outlier_removal : float = 0.0):
        super().__init__(attribute_name)
        self.value_factor = value_factor
        self.outlier_removal = outlier_removal

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value = sum(e[self.attribute_name] for e in suffix) / self.value_factor
        if len(mean_prediction):
            mean_value = sum(e[self.attribute_name] for e in mean_prediction) / self.value_factor
        else:
            #take last element of prefix
            mean_value = 0
        proba_values = [sum(e[self.attribute_name] for e in proba_suffix) / self.value_factor
                        if len(proba_suffix) else 0
                        for proba_suffix in predicted_suffixes]
        return true_value, mean_value, proba_values
    
class SumAbsValues(EvaluationMetric):
    def __init__(self, attribute_name : str, value_factor : float = 1.0,
                 outlier_removal : float = 0.0):
        super().__init__(attribute_name)
        self.value_factor = value_factor
        self.outlier_removal = outlier_removal

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value = sum(e[self.attribute_name] for e in suffix) / self.value_factor
        if len(mean_prediction):
            mean_value = sum(min(e[self.attribute_name], 0) for e in mean_prediction) / self.value_factor
        else:
            #take last element of prefix
            mean_value = 0
        proba_values = [sum(min(e[self.attribute_name], 0) for e in proba_suffix) / self.value_factor
                        if len(proba_suffix) else 0
                        for proba_suffix in predicted_suffixes]
        return true_value, mean_value, proba_values
    
class SumValueMeanMSE(SumValues):
    def __init__(self, attribute_name : str, percentile : float, value_factor : float = 1.0,
                 outlier_removal : float = 0.0):
        super().__init__(attribute_name, outlier_removal)
        self.percentile = percentile
        self.value_factor = value_factor
        self.outlier_removal = outlier_removal

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_sum, mean_sum, proba_sums = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        if self.outlier_removal:
            proba_sums = self.remove_outliers_mad(proba_sums)
        mean_mse = (mean_sum - true_sum)**2
        mean_proba_mse = (np.mean(proba_sums) - true_sum)**2
        proba_maes = [(proba_sum - true_sum)**2 for proba_sum in proba_sums]
        bottom_proba_mse = np.percentile(proba_maes, 100*self.percentile)
        top_proba_mse = np.percentile(proba_maes, 100*(1-self.percentile))
        return {'mean' : mean_mse, 'prob' : (mean_proba_mse, (bottom_proba_mse, top_proba_mse))} 


class SumValueMeanMAE(SumValues):
    def __init__(self, attribute_name: str, percentile: float, value_factor: float = 1.0,
                 outlier_removal: float = 0.0):
        super().__init__(attribute_name, outlier_removal)
        self.percentile = percentile
        self.value_factor = value_factor
        self.outlier_removal = outlier_removal

    def unwrap(self, x):
        if isinstance(x, (np.ndarray, list, tuple)) and len(x) == 1:
            return x[0]
        elif isinstance(x, np.ndarray) and x.shape == ():  # scalar array
            return x.item()
        return x

    def evaluate(self, prefix: list, suffix: list, mean_prediction: list, predicted_suffixes: list[list]):
        true_sum, mean_sum, proba_sums = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)

        # Unwrap values
        true_sum = self.unwrap(true_sum)
        mean_sum = self.unwrap(mean_sum)
        proba_sums = [self.unwrap(p) for p in proba_sums]

        if self.outlier_removal:
            proba_sums = self.remove_outliers_mad(proba_sums)

        # MAE calculations
        mean_mae = np.abs(mean_sum - true_sum)
        mean_proba_mae = np.abs(np.mean(proba_sums) - true_sum)
        proba_maes = [np.abs(proba_sum - true_sum) for proba_sum in proba_sums]
        bottom_proba_mse = np.percentile(proba_maes, 100 * self.percentile)
        top_proba_mse = np.percentile(proba_maes, 100 * (1 - self.percentile))

        return {'mean': mean_mae, 'prob': (mean_proba_mae, (bottom_proba_mse, top_proba_mse))}

class SumAbsValueMeanMAE(SumValues):
    '''
    mean := abs(sum(event_durations) - true_case_duration)
    proba := abs(mean(sum(proba_duration)) - true_case_duration)
    '''
    def __init__(self, attribute_name : str, percentile : float, value_factor : float = 1.0,
                 outlier_removal : float = 0.0):
        super().__init__(attribute_name, outlier_removal)
        self.percentile = percentile
        self.value_factor = value_factor
        self.outlier_removal = outlier_removal
        
    def unwrap(self, x):
        if isinstance(x, (np.ndarray, list, tuple)) and len(x) == 1:
            return x[0]
        elif isinstance(x, np.ndarray) and x.shape == ():  # scalar array
            return x.item()
        return x

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_sum, mean_sum, proba_sums = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        
        # Unwrap values
        true_sum = self.unwrap(true_sum)
        mean_sum = self.unwrap(mean_sum)
        proba_sums = [self.unwrap(p) for p in proba_sums]
        
        if self.outlier_removal:
            proba_sums = self.remove_outliers_mad(proba_sums)
        
        mean_mae = np.abs(mean_sum - true_sum)
        mean_proba_mae = np.abs(np.mean(proba_sums) - true_sum)
        proba_maes = [np.abs(proba_sum - true_sum) for proba_sum in proba_sums]
        bottom_proba_mse = np.percentile(proba_maes, 100*self.percentile)
        top_proba_mse = np.percentile(proba_maes, 100*(1-self.percentile))
        return {'mean' : mean_mae, 'prob' : (mean_proba_mae, (bottom_proba_mse, top_proba_mse))} 


class LastValue(EvaluationMetric):
    def __init__(self, attribute_name : str, value_factor : float = 1.0,
                 outlier_removal : float = 0.0):
        super().__init__(attribute_name, outlier_removal)
        self.value_factor = value_factor
        self.outlier_removal = outlier_removal

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value = suffix[-1][self.attribute_name] / self.value_factor
        if len(mean_prediction):
            mean_value = mean_prediction[-1][self.attribute_name] / self.value_factor
        else:
            #take last element of prefix
            mean_value = prefix[-1][self.attribute_name] / self.value_factor
        proba_values = [proba_suffix[-1][self.attribute_name] / self.value_factor
                        if len(proba_suffix) else prefix[-1][self.attribute_name] / self.value_factor
                        for proba_suffix in predicted_suffixes]
        return true_value, mean_value, proba_values


class LastValuePIT(LastValue):
    """
    Create Probability Integral Transformation
    """
    def __init__(self, attribute_name : str, value_factor = 1.0):
        super().__init__(attribute_name, value_factor)
        
    def unwrap(self, x):
        if isinstance(x, (np.ndarray, list, tuple)) and len(x) == 1:
            return x[0]
        elif isinstance(x, np.ndarray) and x.shape == ():  # zero-dimensional
            return x.item()
        return x
    
    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value, mean_last_value, proba_last_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        
        # --- Unwrap and print for debug ---
        true_value = self.unwrap(true_value)
        mean_last_value = self.unwrap(mean_last_value)
        proba_last_values = [self.unwrap(val) for val in proba_last_values]
        
        pit_value_prob = np.mean(np.array(proba_last_values) <= true_value)
        pit_value_mean = int(mean_last_value <= true_value)
        return {'mean' : pit_value_mean, 'prob' : pit_value_prob}


class SumValuesPIT(SumValues):
    """
    Create Probability Integral Transformation
    """
    def __init__(self, attribute_name : str, value_factor = 1.0):
        super().__init__(attribute_name, value_factor)
        
    def unwrap(self, x):
        if isinstance(x, (np.ndarray, list, tuple)) and len(x) == 1:
            return x[0]
        elif isinstance(x, np.ndarray) and x.shape == ():  # zero-dimensional
            return x.item()
        return x
    
    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value, sum_mean_value, sums_proba_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        
        true_value = self.unwrap(true_value)
        sum_mean_value = self.unwrap(sum_mean_value)
        sums_proba_values = [self.unwrap(val) for val in sums_proba_values]

        pit_value_prob = np.mean(np.array(sums_proba_values) <= true_value)
        pit_value_mean = int(sum_mean_value <= true_value)
        return {'mean' : pit_value_mean, 'prob' : pit_value_prob}


class LastValueCRPS(LastValue):
    def __init__(self, attribute_name : str, value_factor : float = 1.0):
        super().__init__(attribute_name)
        self.value_factor = value_factor
    
    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value, mean_last_value, proba_last_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        p_score_mean = pscore([mean_last_value], true_value).compute()[0]
        p_score_prob = pscore(proba_last_values, true_value).compute()[0]
        return {'mean' : p_score_mean, 'prob' : p_score_prob}


class LastValueMeanMSE(LastValue):
    def __init__(self, attribute_name : str, value_factor : float = 1.0):
        super().__init__(attribute_name)
        self.value_factor = value_factor
    
    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value, mean_last_value, proba_last_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        mean_mse = (mean_last_value - true_value)**2
        mean_proba_mse = np.mean([(proba_last_value - true_value)**2 for proba_last_value in proba_last_values])
        return {'mean' : mean_mse, 'prob' : mean_proba_mse}

class LastValueMeanVarMSE(LastValue):
    def __init__(self, attribute_name : str, percentile : float, value_factor : float = 1.0):
        super().__init__(attribute_name)
        self.percentile = percentile
        self.value_factor = value_factor
    
    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value, mean_last_value, proba_last_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        mean_mae = (mean_last_value - true_value)**2
        proba_maes = [(proba_last_value - true_value)**2 for proba_last_value in proba_last_values]
        mean_proba_mse = np.mean(proba_maes)
        bottom_proba_mse = np.percentile(proba_maes, 100*self.percentile)
        top_proba_mse = np.percentile(proba_maes, 100*(1-self.percentile))
        return {'mean' : mean_mae, 'prob' : (mean_proba_mse, (bottom_proba_mse, top_proba_mse))} 

class LastValueMean2VarMSE(LastValue):
    def __init__(self, attribute_name : str, percentile : float, value_factor : float = 1.0,
                 outlier_removal : float = 0.0):
        super().__init__(attribute_name, outlier_removal)
        self.percentile = percentile
        self.value_factor = value_factor
        self.outlier_removal = outlier_removal

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value, mean_last_value, proba_last_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        if self.outlier_removal:
            proba_last_values = self.remove_outliers_log_mad(proba_last_values)
        mean_mse = (mean_last_value - true_value)**2
        mean_proba_mse = (np.mean(proba_last_values) - true_value)**2
        proba_maes = [(proba_last_value - true_value)**2 for proba_last_value in proba_last_values]
        bottom_proba_mse = np.percentile(proba_maes, 100*self.percentile)
        top_proba_mse = np.percentile(proba_maes, 100*(1-self.percentile))
        return {'mean' : mean_mse, 'prob' : (mean_proba_mse, (bottom_proba_mse, top_proba_mse))} 


class LastValueMedianVarMSE(LastValue):
    def __init__(self, attribute_name : str, percentile : float, value_factor : float = 1.0):
        super().__init__(attribute_name)
        self.percentile = percentile
        self.value_factor = value_factor
    
    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value, mean_last_value, proba_last_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        mean_mae = (mean_last_value - true_value)**2
        proba_maes = [(proba_last_value - true_value)**2 for proba_last_value in proba_last_values]
        median_proba_mse = np.median(proba_maes)
        bottom_proba_mse = np.percentile(proba_maes, 100*self.percentile)
        top_proba_mse = np.percentile(proba_maes, 100*(1-self.percentile))
        return {'mean' : mean_mae, 'prob' : (median_proba_mse, (bottom_proba_mse, top_proba_mse))} 


class LastValueMedian2VarMSE(LastValue):
    def __init__(self, attribute_name : str, percentile : float, value_factor : float = 1.0):
        super().__init__(attribute_name)
        self.percentile = percentile
        self.value_factor = value_factor
    
    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value, mean_last_value, proba_last_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        mean_mae = (mean_last_value - true_value)**2
        proba_maes = [(proba_last_value - true_value)**2 for proba_last_value in proba_last_values]
        mean_proba_mse = (np.median(proba_last_values) - true_value)**2
        bottom_proba_mse = np.percentile(proba_maes, 100*self.percentile)
        top_proba_mse = np.percentile(proba_maes, 100*(1-self.percentile))
        return {'mean' : mean_mae, 'prob' : (mean_proba_mse, (bottom_proba_mse, top_proba_mse))} 


class LastValueMeanMAE(LastValue):
    def __init__(self, attribute_name : str, value_factor : float = 1.0):
        super().__init__(attribute_name)
        self.value_factor = value_factor
    
    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value, mean_last_value, proba_last_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        mean_mse = abs(mean_last_value - true_value)
        mean_proba_mse = np.mean([abs(proba_last_value - true_value) for proba_last_value in proba_last_values])
        return {'mean' : mean_mse, 'prob' : mean_proba_mse} 

class LastValueMedianVarMAE(LastValue):
    def __init__(self, attribute_name : str, percentile : float, value_factor : float = 1.0):
        super().__init__(attribute_name)
        self.percentile = percentile
        self.value_factor = value_factor
    
    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value, mean_last_value, proba_last_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        mean_mae = abs(mean_last_value - true_value)
        proba_maes = [abs(proba_last_value - true_value) for proba_last_value in proba_last_values]
        median_proba_mae = abs(np.median(proba_maes) - true_value)
        #mean_proba_mse = np.mean(proba_maes)
        bottom_proba_mse = np.percentile(proba_maes, 100*self.percentile)
        top_proba_mse = np.percentile(proba_maes, 100*(1-self.percentile))
        return {'mean' : mean_mae, 'prob' : (median_proba_mae, (bottom_proba_mse, top_proba_mse))} 
    

class LastValueMedian2VarMAE(LastValue):
    def __init__(self, attribute_name : str, percentile : float, value_factor : float = 1.0):
        super().__init__(attribute_name)
        self.percentile = percentile
        self.value_factor = value_factor
    
    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value, mean_last_value, proba_last_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        mean_mae = abs(mean_last_value - true_value)
        proba_maes = [abs(proba_last_value - true_value) for proba_last_value in proba_last_values]
        median_proba_mae = np.median(proba_maes)
        #mean_proba_mse = np.mean(proba_maes)
        bottom_proba_mse = np.percentile(proba_maes, 100*self.percentile)
        top_proba_mse = np.percentile(proba_maes, 100*(1-self.percentile))
        return {'mean' : mean_mae, 'prob' : (median_proba_mae, (bottom_proba_mse, top_proba_mse))} 


class LastValueMeanVarMAE(LastValue):
    def __init__(self, attribute_name : str, percentile : float, value_factor : float = 1.0):
        super().__init__(attribute_name)
        self.percentile = percentile
        self.value_factor = value_factor
    
    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_value, mean_last_value, proba_last_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        mean_mae = abs(mean_last_value - true_value)
        proba_maes = [abs(proba_last_value - true_value) for proba_last_value in proba_last_values]
        median_proba_mae = abs(np.mean(proba_maes) - true_value)
        #mean_proba_mse = np.mean(proba_maes)
        bottom_proba_mse = np.percentile(proba_maes, 100*self.percentile)
        top_proba_mse = np.percentile(proba_maes, 100*(1-self.percentile))
        return {'mean' : mean_mae, 'prob' : (median_proba_mae, (bottom_proba_mse, top_proba_mse))} 
    

class LastValueMean2VarMAE(LastValue):
    '''
    mean := abs(mean_last_value - true_value)
    proba := 
    '''
    def __init__(self, attribute_name: str, percentile: float, value_factor: float = 1.0,
                 outlier_removal: float = 0.0):
        super().__init__(attribute_name, outlier_removal)
        self.percentile = percentile
        self.value_factor = value_factor
        self.outlier_removal = outlier_removal

    def unwrap(self, x):
        if isinstance(x, (np.ndarray, list, tuple)) and len(x) == 1:
            return x[0]
        elif isinstance(x, np.ndarray) and x.shape == ():  # scalar array
            return x.item()
        return x

    def evaluate(self, prefix: list, suffix: list, mean_prediction: list, predicted_suffixes: list[list]):
        true_value, mean_last_value, proba_last_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)

        # Unwrap values
        true_value = self.unwrap(true_value)
        mean_last_value = self.unwrap(mean_last_value)
        proba_last_values = [self.unwrap(p) for p in proba_last_values]

        mean_mae = abs(mean_last_value - true_value)

        if self.outlier_removal:
            proba_last_values = self.remove_outliers_log_mad(proba_last_values)

        # Compute MAEs and percentiles
        proba_maes = [abs(proba_last_value - true_value) for proba_last_value in proba_last_values]
        median_proba_mae = np.median(proba_maes)
        bottom_proba_mse = np.percentile(proba_maes, 100 * self.percentile)
        top_proba_mse = np.percentile(proba_maes, 100 * (1 - self.percentile))

        return {'mean': mean_mae, 'prob': (median_proba_mae, (bottom_proba_mse, top_proba_mse))}


class RawAttribute(EvaluationMetric):
    def __init__(self, attribute_name : str):
        super().__init__(attribute_name)

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_suffix = [event[self.attribute_name] for event in suffix]
        mean_suffix = [event[self.attribute_name] for event in mean_prediction]
        proba_suffixes = [[event[self.attribute_name] for event in proba_suffix] for proba_suffix in predicted_suffixes]
        return {'true' : true_suffix, 'mean' : mean_suffix, 'prob' : proba_suffixes}

class LevenshteinDistance(EvaluationMetric):
    def __init__(self, attribute_name : str):
        super().__init__(attribute_name)

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_suffix = [event[self.attribute_name] for event in suffix]
        mean_suffix = [event[self.attribute_name] for event in mean_prediction]
        proba_suffixes = [[event[self.attribute_name] for event in proba_suffix] for proba_suffix in predicted_suffixes]
        mean_distance = distance(mean_suffix, true_suffix)
        proba_distances = [distance(proba_suffix, true_suffix) for proba_suffix in proba_suffixes]
        return mean_distance, proba_distances

class LevenshteinCRPS(LevenshteinDistance):
    def __init__(self, attribute_name : str):
        super().__init__(attribute_name)

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        mean_distance, proba_distances = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        p_score_mean = pscore([mean_distance], 0).compute()[0]
        p_score_prob = pscore(proba_distances, 0).compute()[0]
        return {'mean' : p_score_mean, 'prob' : p_score_prob}
    
class LevenshteinMean(LevenshteinDistance):
    def __init__(self, attribute_name : str):
        super().__init__(attribute_name)

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        mean_distance, proba_distances = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        proba_mean_distance = np.mean(proba_distances)
        return {'mean' : mean_distance, 'prob' : proba_mean_distance}
    
class LevenshteinMeanVar(LevenshteinDistance):
    def __init__(self, attribute_name : str, percentile : float):
        super().__init__(attribute_name)
        self.percentile = percentile

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        mean_distance, proba_distances = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        proba_mean_distance = np.mean(proba_distances)
        proba_bottom = np.percentile(proba_distances, 100*self.percentile)
        proba_top = np.percentile(proba_distances, 100*(1-self.percentile))
        return {'mean' : mean_distance, 'prob' : (proba_mean_distance, (proba_bottom, proba_top))}
    

class LevenshteinMedianVar(LevenshteinDistance):
    def __init__(self, attribute_name : str, percentile : float):
        super().__init__(attribute_name)
        self.percentile = percentile

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        mean_distance, proba_distances = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        proba_mean_distance = np.median(proba_distances)
        proba_bottom = np.percentile(proba_distances, 100*self.percentile)
        proba_top = np.percentile(proba_distances, 100*(1-self.percentile))
        return {'mean' : mean_distance, 'prob' : (proba_mean_distance, (proba_bottom, proba_top))}
    

class NormalizedDamerauLevenshteinDistance(EvaluationMetric):
    '''
    DLS Distance
    '''
    def __init__(self, attribute_name : str):
        super().__init__(attribute_name)

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_suffix = [event[self.attribute_name] for event in suffix]
        mean_suffix = [event[self.attribute_name] for event in mean_prediction]
        proba_suffixes = [[event[self.attribute_name] for event in proba_suffix] for proba_suffix in predicted_suffixes]
        #mean_distance = normalized_damerau_levenshtein_distance(mean_suffix, true_suffix)
        #proba_distances = [normalized_damerau_levenshtein_distance(proba_suffix, true_suffix) for proba_suffix in proba_suffixes]
        mean_distance = 1 - (damerau_levenshtein_distance(mean_suffix, true_suffix) / max(len(true_suffix),len(mean_suffix)))
        proba_distances = [1 - (damerau_levenshtein_distance(proba_suffix, true_suffix) / max(len(true_suffix),len(proba_suffix))) for proba_suffix in proba_suffixes]
        return mean_distance, proba_distances


class NormalizedDamerauLevenshteinDistanceCRPS(NormalizedDamerauLevenshteinDistance):
    def __init__(self, attribute_name : str):
        super().__init__(attribute_name)

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        mean_distance, proba_distances = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        p_score_mean = pscore([mean_distance], 0).compute()[0]
        p_score_prob = pscore(proba_distances, 0).compute()[0]
        return {'mean' : p_score_mean, 'prob' : p_score_prob}


class NormalizedDamerauLevenshteinDistanceMean(NormalizedDamerauLevenshteinDistance):
    def __init__(self, attribute_name : str):
        super().__init__(attribute_name)

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        mean_distance, proba_distances = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        proba_mean_distance = np.mean(proba_distances)
        return {'mean' : mean_distance, 'prob' : proba_mean_distance}


class NormalizedDamerauLevenshteinDistanceMeanVar(NormalizedDamerauLevenshteinDistance):
    def __init__(self, attribute_name : str, percentile : float):
        super().__init__(attribute_name)
        self.percentile = percentile

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        mean_distance, proba_distances = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        proba_mean_distance = np.mean(proba_distances)
        bottom_proba_distance = np.percentile(proba_distances, 100*self.percentile)
        top_proba_distance = np.percentile(proba_distances, 100*(1-self.percentile))
        return {'mean' : mean_distance, 'prob' : (proba_mean_distance, (bottom_proba_distance, top_proba_distance))} 
    
class NormalizedDamerauLevenshteinDistanceMedianVar(NormalizedDamerauLevenshteinDistance):
    def __init__(self, attribute_name : str, percentile : float):
        super().__init__(attribute_name)
        self.percentile = percentile

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        mean_distance, proba_distances = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        proba_mean_distance = np.median(proba_distances)
        bottom_proba_distance = np.percentile(proba_distances, 100*self.percentile)
        top_proba_distance = np.percentile(proba_distances, 100*(1-self.percentile))
        return {'mean' : mean_distance, 'prob' : (proba_mean_distance, (bottom_proba_distance, top_proba_distance))} 


class ClosestNormalizedDamerauLevenshteinDistance(EvaluationMetric):
    def __init__(self, attribute_name : str):
        super().__init__(attribute_name)

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_suffix = ''.join([event[self.attribute_name] for event in suffix])
        mean_suffix = ''.join([event[self.attribute_name] for event in mean_prediction])
        proba_suffixes = [''.join([event[self.attribute_name] if event[self.attribute_name] else '' for event in proba_suffix]) for proba_suffix in predicted_suffixes]
        mean_distance = normalized_damerau_levenshtein_distance(mean_suffix, true_suffix)
        proba_distances = [normalized_damerau_levenshtein_distance(proba_suffix, true_suffix) for proba_suffix in proba_suffixes]
        return mean_distance, proba_distances


class PercentileNormalizedDamerauLevenshteinDistanceMeanVar(ClosestNormalizedDamerauLevenshteinDistance):
    def __init__(self, attribute_name : str, percentile : float):
        super().__init__(attribute_name)
        self.percentile = percentile

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        mean_distance, proba_distances = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        proba_mean_distance = np.mean(proba_distances)
        bottom_proba_distance = np.percentile(proba_distances, 100*self.percentile)
        top_proba_distance = np.percentile(proba_distances, 100*(1-self.percentile))
        return {'mean' : mean_distance, 'prob' : (proba_mean_distance, (bottom_proba_distance, top_proba_distance))} 
    

class LastValueInterval(LastValue):
    '''
    mean := abs(mean_last_value - true_value)
    proba := 
    '''
    def __init__(self, attribute_name: str, percentile: float, value_factor: float = 1.0,
                 outlier_removal: float = 0.0):
        super().__init__(attribute_name, outlier_removal)
        self.percentile = percentile
        self.value_factor = value_factor
        self.outlier_removal = outlier_removal

    def unwrap(self, x):
        if isinstance(x, (np.ndarray, list, tuple)) and len(x) == 1:
            return x[0]
        elif isinstance(x, np.ndarray) and x.shape == ():  # scalar array
            return x.item()
        return x

    def evaluate(self, prefix: list, suffix: list, mean_prediction: list, predicted_suffixes: list[list]):
        true_value, mean_last_value, proba_last_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)

        # Unwrap values
        true_value = self.unwrap(true_value)
        proba_last_values = [self.unwrap(p) for p in proba_last_values]

        if self.outlier_removal:
            proba_last_values = self.remove_outliers_log_mad(proba_last_values)

        bottom_proba = np.percentile(proba_last_values, 50 - 100 * self.percentile/2)
        top_proba = np.percentile(proba_last_values, 50 + 100 * self.percentile/2)
        in_interval = (true_value >= bottom_proba) and  (true_value <= top_proba)

        return {'mean': 0, 'prob': (in_interval, (0, 0))}
    

class SumValuesInterval(SumValues):
    '''

    '''
    def __init__(self, attribute_name : str, percentile : float, value_factor : float = 1.0,
                 outlier_removal : float = 0.0):
        super().__init__(attribute_name, outlier_removal)
        self.percentile = percentile
        self.value_factor = value_factor
        self.outlier_removal = outlier_removal
        
    def unwrap(self, x):
        if isinstance(x, (np.ndarray, list, tuple)) and len(x) == 1:
            return x[0]
        elif isinstance(x, np.ndarray) and x.shape == ():  # scalar array
            return x.item()
        return x

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        true_sum, mean_sum, proba_sums = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)
        
        # Unwrap values
        true_sum = self.unwrap(true_sum)
        mean_sum = self.unwrap(mean_sum)
        proba_sums = [self.unwrap(p) for p in proba_sums]
        
        if self.outlier_removal:
            proba_sums = self.remove_outliers_mad(proba_sums)
        
        bottom_proba = np.percentile(proba_sums, 50 - 100 * self.percentile/2)
        top_proba = np.percentile(proba_sums, 50 + 100 * self.percentile/2)
        in_interval = (true_sum >= bottom_proba) and  (true_sum <= top_proba)

        return {'mean' : 0, 'prob' : (in_interval, (0, 0))} 
    

class SuffixCountInterval(SuffixCount):
    def __init__(self,
                 percentile : float,
                 outlier_removal : float = 0.0):
        super().__init__()
        self.percentile = percentile
        self.outlier_removal = outlier_removal

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        len_true_value, len_mean_value, len_proba_values = super().evaluate(prefix, suffix, mean_prediction, predicted_suffixes)

        bottom_proba = np.percentile(len_proba_values, 50 - 100 * self.percentile/2)
        top_proba = np.percentile(len_proba_values, 50 + 100 * self.percentile/2)
        in_interval = (len_true_value >= bottom_proba) and  (len_true_value <= top_proba)

        return {'mean' : 0, 'prob' : (in_interval, (0, 0))}
    

class EventLabelCountInterval(EvaluationMetric):
    def __init__(self,
                 attribute_name : str,
                 event_label_list : list[str],
                 percentile : float,
                 outlier_removal : float = 0.0):
        super().__init__(attribute_name)
        self.event_label_list = event_label_list
        self.percentile = percentile
        self.outlier_removal = outlier_removal

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        res = dict()
        for event_label in self.event_label_list:
            event_label_true_count = [e[self.attribute_name] for e in suffix].count(event_label)
            event_label_mean_count = [e[self.attribute_name] for e in mean_prediction].count(event_label)
            event_label_counts = [[e[self.attribute_name] for e in proba_suffix].count(event_label) for proba_suffix in predicted_suffixes]
            bottom_proba = np.percentile(event_label_counts, 50 - 100 * self.percentile/2)
            top_proba = np.percentile(event_label_counts, 50 + 100 * self.percentile/2)
            in_interval = (event_label_true_count >= bottom_proba) and (event_label_true_count <= top_proba)
            res[event_label] = in_interval

        return res


class EventLabelCount(EvaluationMetric):
    def __init__(self,
                 attribute_name : str,
                 event_label : str,
                 outlier_removal : float = 0.0):
        super().__init__(attribute_name)
        self.event_label = event_label
        self.outlier_removal = outlier_removal

    def evaluate(self, prefix : list, suffix : list, mean_prediction : list, predicted_suffixes : list[list]):
        event_label_true_count = [e[self.attribute_name] for e in suffix].count(self.event_label)
        event_label_mean_count = [e[self.attribute_name] for e in mean_prediction].count(self.event_label)
        event_label_counts = Counter([[e[self.attribute_name] for e in proba_suffix].count(self.event_label) for proba_suffix in predicted_suffixes])

        return {'true' : event_label_true_count, 'mean' : event_label_mean_count,
                'prob' : event_label_counts}
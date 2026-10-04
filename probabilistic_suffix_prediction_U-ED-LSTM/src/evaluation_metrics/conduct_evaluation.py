import os
import pickle
import concurrent.futures
import collections

def evaluate_metric(
    metric : object,
    prefix : list,
    suffix : list,
    mean_prediction : list,
    predicted_suffix : list[list]):
    
    return metric.evaluate(prefix, suffix, mean_prediction, predicted_suffix)

def evaluate_single(batch_folder : str, filename : str, metrics : dict):
    file_path = os.path.join(batch_folder, filename)
    results = {metric_name : {} for metric_name in metrics.keys()}
    pc, sc = collections.defaultdict(int), collections.defaultdict(int)
    try:
        with open(file_path, 'rb') as f:
            chunk = pickle.load(f)
        for (case_name, prefix_len), (prefix, suffix, mean_prediction, predicted_suffixes)\
        in chunk.items():
            suffix_len = len(suffix)
            pc[prefix_len] += 1
            sc[suffix_len] += 1
            # check if predicted_suffixes contain model state
            if isinstance(predicted_suffixes[0], tuple):
                predicted_suffixes = [p[0] for p in predicted_suffixes]
            for metric_name, metric_obj in metrics.items():
                result = evaluate_metric(metric_obj, prefix, suffix, mean_prediction, predicted_suffixes)
                results[metric_name][(case_name, prefix_len, suffix_len)] = result
        print(f'Loaded: {filename}')
    except Exception as e:
        print(f'Error: {filename}: {e}')
    return results, (pc, sc)

def evaluate_sequentially(
        batch_folder : str,
        metrics : dict[str, object]):
    
    pickle_files = sorted(f for f in os.listdir(batch_folder) if f.endswith('.pkl'))
    prefix_count, suffix_count = collections.defaultdict(lambda : 0), collections.defaultdict(lambda : 0)
    results = {metric_name : {} for metric_name in metrics.keys()}
    for filename in pickle_files:
        new_results, (pc, sc) = evaluate_single(batch_folder, filename, metrics)
        for metric_name, metric in results.items():
            metric.update(new_results[metric_name])
        for k,v in pc.items():
            prefix_count[k] += v
        for k,v in sc.items():
            suffix_count[k] += v
    return results, (dict(prefix_count), dict(suffix_count))

def batch_evaluate(
        batch_folder : str,
        metrics : dict[str, object],
        num_workers : int = 4):
    
    pickle_files = sorted(f for f in os.listdir(batch_folder) if f.endswith('.pkl'))
    results = {metric_name : {} for metric_name in metrics.keys()}
    prefix_count, suffix_count = collections.defaultdict(lambda : 0), collections.defaultdict(lambda : 0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Schedule the evaluate_single calls concurrently
        futures = {
            executor.submit(evaluate_single, batch_folder, filename, metrics): filename
            for filename in pickle_files
        }

        # Process each future as it completes
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            new_results, (pc, sc) = future.result()
            for metric_name, metric in results.items():
                metric.update(new_results[metric_name])
            # Update counts
            for k, v in pc.items():
                prefix_count[k] += v
            for k, v in sc.items():
                suffix_count[k] += v

    return results, (dict(prefix_count), dict(suffix_count))

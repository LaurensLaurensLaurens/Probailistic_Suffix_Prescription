from .evaluation import Evaluation

import concurrent.futures
import random
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm

class ProbabilisticEvaluation(Evaluation):
    def __init__(self, 
                 model,
                 dataset,
                 concept_name = 'concept:name',
                 eos_value = 'EOS',
                 growing_num_values = ['case_elapsed_time'],
                 all_cat = None,
                 all_num = None,
                 num_processes : int = 32,
                 samples_per_case : int = 100,
                 sample_argmax : bool = False,
                 use_variance_cat : bool = True,
                 use_variance_num : bool = True,
                 variational_dropout_sampling : bool = False):
        
        super().__init__(model, dataset, concept_name, eos_value, growing_num_values, all_cat, all_num)
        self.samples_per_case = samples_per_case
        self.model = model
        self.dataset = dataset
        self.num_processes = num_processes
        self.eos_value = eos_value
        self.growing_num_values = growing_num_values
        self.sample_argmax = sample_argmax
        self.use_variance_cat = use_variance_cat
        self.use_variance_num = use_variance_num
        self.variational_dropout_sampling = variational_dropout_sampling

        self.prefix_cat_attributes = self.model.enc_feat[0]
        self.prefix_num_attributes = self.model.enc_feat[1]

        # Change and make implementation more flexible.
        self.all_cat_attributes = all_cat if all_cat else [cat[0] for cat in self.dataset.all_categories[0]]
        self.all_num_attributes = all_num if all_num else [num[0] for num in self.dataset.all_categories[1]]

    def sample_cat_predictions(self, cat_means, cat_variances):
        return ProbabilisticEvaluation.sample_cat_predictions_optim(self.all_cat_attributes,
                                                                    self.use_variance_cat,
                                                                    self.sample_argmax,
                                                                    cat_means,
                                                                    cat_variances)
    @torch.jit.script
    def sample_cat_predictions_optim(
                               all_cat_attributes : list[str],
                               use_variance_cat : bool,
                               sample_argmax : bool,
                               cat_means : dict[str, torch.Tensor],
                               cat_variances : dict[str, torch.Tensor]
     ) -> dict[str, torch.Tensor]:
        result = {}
        for c in all_cat_attributes:
            if use_variance_cat:
                sampled_cats = torch.normal(cat_means[c+'_mean'], torch.exp(cat_variances[c+'_var']))
            else:
                sampled_cats = cat_means[c+'_mean']
            
            if sample_argmax:
                result[c+'_mean'] = torch.argmax(sampled_cats, keepdim=True)
            else:
                probs = F.softmax(sampled_cats, dim=-1)
                result[c+'_mean'] = torch.multinomial(probs, num_samples=1, replacement=True)
        
        return result
    
    def sample_num_predictions(self, num_means, num_variances, last_values):
        return ProbabilisticEvaluation.sample_num_predictions_optim(self.all_num_attributes,
                                                                    self.use_variance_num,
                                                                    self.growing_num_values,
                                                                    num_means,
                                                                    num_variances,
                                                                    last_values)

    @torch.jit.script
    def sample_num_predictions_optim(
                               all_num_attributes : list[str],
                               use_variance_num : bool,
                               growing_num_values : list[str],                        
                               num_means : dict[str, torch.Tensor],
                               num_variances : dict[str, torch.Tensor],
                               last_values : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = {}
        for c in all_num_attributes:
            if use_variance_num:
                mean = num_means[c + '_mean']
                std = torch.exp(0.5 * num_variances[c + '_var'])
                
                # Sample
                sample = torch.normal(mean, std) 
            else:
                sample = num_means[c + '_mean'] 
            
            if c in growing_num_values:
                result[c+'_mean'] = torch.max(last_values[c+'_mean'], sample)
            else:
                result[c+'_mean'] = sample
        
        return result
    
    def sample_suffix(self, prefix, prefix_len, include_model_states):
        prediction, (h, c), z = self.model.inference(prefix=prefix)
        suffix = []
        max_iteration = self.dataset.encoder_decoder.window_size - \
                        self.dataset.encoder_decoder.min_suffix_size - \
                        prefix_len
        last_means = {a+'_mean' : prefix[1][self.all_num_attributes.index(a)][:,-1].unsqueeze(1) for a in self.growing_num_values}
        
        cat_predictions = self.sample_cat_predictions(prediction[0][0], prediction[1][0])
        num_predictions = self.sample_num_predictions(prediction[0][1], prediction[1][1], last_means)
        
        if include_model_states:
            model_states = []

        i = 0
        eos_predicted = lambda : cat_predictions[self.concept_name+'_mean'] == self.eos_id
        while i <= max_iteration and not eos_predicted():
            
            readable_prediction = self.prediction_to_readable(cat_predictions, num_predictions)
            
            suffix.append(readable_prediction)
            
            if include_model_states:
                model_states.append((h, c))
            last_means = {key: tensor.clone() for key, tensor in num_predictions.items()} # clone last_means
            
            if self.variational_dropout_sampling:
                prediction, (h, c) = self.model.inference(last_event=(list(cat_predictions.values()), list(num_predictions.values())), hx=(h,c), z=z)
            else:
                prediction, (h, c) = self.model.inference(last_event=(list(cat_predictions.values()), list(num_predictions.values())),hx=(h,c), z=None)
            
            cat_predictions = self.sample_cat_predictions(prediction[0][0], prediction[1][0])
            num_predictions = self.sample_num_predictions(prediction[0][1], prediction[1][1], last_means)
            
            i += 1
        
        if include_model_states:
            return suffix, model_states
        else:
            return suffix

    def predict_probabilistic_suffix(self, prefix, prefix_len, include_model_states):
        suffixes = []
        for i in range(self.samples_per_case):
            suffix = self.sample_suffix(prefix, prefix_len, include_model_states)
            suffixes.append(suffix)
        return suffixes     
    
    
    def _evaluate_single(self, case_name, prefix_len, prefix, suffix, include_model_states):
        readable_prefix = self.case_to_readable(prefix, prune_eos=False)
        # print("Prefix: ", readable_prefix)
        
        readable_suffix = self.case_to_readable(suffix, prune_eos=True)
        # print("Suffix: ", readable_suffix)
        
        mean_prediction = self._predict_suffix_with_means(prefix, prefix_len)
        # print("Mean Pred: ", mean_prediction)
        
        predicted_suffixes = self.predict_probabilistic_suffix(prefix, prefix_len, include_model_states)

        return case_name, prefix_len, readable_prefix, predicted_suffixes, readable_suffix, mean_prediction
    
    
    def count_only(self, random_order=False):
        case_items = list(self.cases.items())
        if random_order:
            case_items = random.sample(case_items, len(case_items))
        for i, (case_name, full_case) in tqdm(enumerate(case_items), total=len(self.cases)):
            for j, (prefix_len, prefix, suffix) in enumerate(self._iterate_case(full_case)):
                yield case_name, prefix_len, None, None, None, None
                
                
    def evaluate(self, random_order=False, include_model_states=False):
        #compiled_evaluate_single = torch.compile(self._evaluate_single)
        case_items = list(self.cases.items())
        if random_order:
            case_items = random.sample(case_items, len(case_items))
        for i, (case_name, full_case) in tqdm(enumerate(case_items), total=len(self.cases)):
            for j, (prefix_len, prefix, suffix) in enumerate(self._iterate_case(full_case)):
                yield self._evaluate_single(case_name, prefix_len, prefix, suffix, include_model_states)

    def evaluate_multi_processing(self, random_order=False, include_model_states=False):
        case_items = list(self.cases.items())
        if random_order:
            case_items = random.sample(case_items, len(case_items))
        max_in_flight = self.num_processes
        futures = []
        multiprocessing.set_start_method('spawn', force=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            for i, (case_name, full_case) in tqdm(enumerate(case_items), total=len(self.cases)):
                for j, (prefix_len, prefix, suffix) in enumerate(self._iterate_case(full_case)):
                    # we need an explicit copy here - otherwise we run into very bad memory issues
                    prefix = [[t.clone() for t in i] for i in prefix]
                    suffix = [[t.clone() for t in i] for i in suffix]
                    future = executor.submit(self._evaluate_single, case_name, prefix_len, prefix, suffix, include_model_states)
                    futures.append(future)

                    if len(futures) >= max_in_flight:
                        done, pending = concurrent.futures.wait(
                            futures, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for completed in done:
                            yield completed.result()
                        futures = list(pending)
            
            for future in concurrent.futures.as_completed(futures):
                yield future.result()

    
    def evaluate_continue_multi_processing(self, processed_prefixes, random_order=False, include_model_states=False):
        case_items = list(self.cases.items())
        if random_order:
            case_items = random.sample(case_items, len(case_items))

        max_in_flight = self.num_processes
        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            for case_name, full_case in tqdm(case_items, total=len(self.cases)):
                for prefix_len, prefix, suffix in self._iterate_case(full_case):
                    if (case_name, prefix_len) in processed_prefixes:
                        print("skipped", case_name, prefix_len)
                    else:
                        # Copy to avoid memory issues
                        prefix = [[t.clone() for t in i] for i in prefix]
                        suffix = [[t.clone() for t in i] for i in suffix]
                        future = executor.submit(self._evaluate_single, case_name, prefix_len, prefix, suffix, include_model_states)
                        futures.append(future)
                    
                    if len(futures) >= max_in_flight:
                        done, pending = concurrent.futures.wait(
                            futures, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for completed in done:
                            yield completed.result()
                        futures = list(pending)
            for future in concurrent.futures.as_completed(futures):
                yield future.result()
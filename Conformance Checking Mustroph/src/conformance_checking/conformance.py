import pm4py
import pandas as pd

class ConformanceChecking():
    def __init__(self, path_process_model):
        self.path_process_model = path_process_model
    
    # BPMN to Petri Net
    def pre_proces_process_model(self):
        # Read BPMN model to pm4py object
        pm = pm4py.read.read_bpmn(self.path_process_model)
        # Convert BPMN to Petri Net: Return to Petri net, Initial Marking, Final Marking
        pn, im, fm = pm4py.convert.convert_to_petri_net(pm)
        return pn, im, fm
                
    # Berechnet die Alignment von samples suffixes
    # conformance_of_sampled_suffixes(...) berechnet die Conformance für alle gesampelten Suffixe, 
    # die in predicted_sampled_suffixes für genau einen (Case, Prefix-Länge)-Eintrag enthalten sind.
    def conformance_of_sampled_suffixes(self, log_name: str, result_values: list):        
        # To-Be Process model
        pn, im, fm = self.pre_proces_process_model()
            
        if log_name == 'Helpdesk':
            concept_name = 'Activity'
        else:
            concept_name = 'concept:name'

        #         prefix = [
        #     {
        #         "concept:name": "RECEPTION",
        #         "org:resource": "12",
        #         "case_elapsed_time": 0.0,
        #         "event_elapsed_time": 0.0,
        #         "day_in_week": 2.0,
        #         "seconds_in_day": 32400.0
        #     },
        #     {
        #         "concept:name": "ACKNOWLEDGEMENT",
        #         "org:resource": "7",
        #         "case_elapsed_time": 418.0,
        #         "event_elapsed_time": 418.0,
        #         "day_in_week": 2.0,
        #         "seconds_in_day": 32818.0
        #     }
        # ]

        # GT Suffix i guess
        # suffix = [
        #     {
        #         "concept:name": "DISASSEMBLY",
        #         "org:resource": "23",
        #         "case_elapsed_time": 2150.0,
        #         "event_elapsed_time": 1732.0,
        #         "day_in_week": 2.0,
        #         "seconds_in_day": 34550.0
        #     },
        #     {
        #         "concept:name": "REPAIR",
        #         "org:resource": "31",
        #         "case_elapsed_time": 5840.0,
        #         "event_elapsed_time": 3690.0,
        #         "day_in_week": 2.0,
        #         "seconds_in_day": 38240.0
        #     },
        #     {
        #         "concept:name": "QUALITY_CONTROL",
        #         "org:resource": "8",
        #         "case_elapsed_time": 8300.0,
        #         "event_elapsed_time": 2460.0,
        #         "day_in_week": 2.0,
        #         "seconds_in_day": 40700.0
        #     },
        #     {
        #         "concept:name": "SHIPPING",
        #         "org:resource": "4",
        #         "case_elapsed_time": 12000.0,
        #         "event_elapsed_time": 3700.0,
        #         "day_in_week": 2.0,
        #         "seconds_in_day": 44400.0
        #     }
        # ]

        # mostlikely_prediction = [
        #     {
        #         "concept:name": "DISASSEMBLY",
        #         "event_elapsed_time": 1604.37
        #     },
        #     {
        #         "concept:name": "REPAIR",
        #         "event_elapsed_time": 3521.81
        #     },
        #     {
        #         "concept:name": "QUALITY_CONTROL",
        #         "event_elapsed_time": 2388.16
        #     },
        #     {
        #         "concept:name": "SHIPPING",
        #         "event_elapsed_time": 3410.92
        #     }
        # ]

        # predicted_sampled_suffixes = [

        #     # Sample 1
        #     [
        #         {
        #             "concept:name": "DISASSEMBLY",
        #             "event_elapsed_time": 1498.52
        #         },
        #         {
        #             "concept:name": "REPAIR",
        #             "event_elapsed_time": 3812.71
        #         },
        #         {
        #             "concept:name": "QUALITY_CONTROL",
        #             "event_elapsed_time": 2291.43
        #         },
        #         {
        #             "concept:name": "SHIPPING",
        #             "event_elapsed_time": 3284.11
        #         }
        #     ],

        #     # Sample 2
        #     [
        #         {
        #             "concept:name": "DISASSEMBLY",
        #             "event_elapsed_time": 1783.14
        #         },
        #         {
        #             "concept:name": "REPAIR",
        #             "event_elapsed_time": 4011.22
        #         },
        #         {
        #             "concept:name": "SHIPPING",
        #             "event_elapsed_time": 3021.87
        #         }
        #     ],

        #     # Sample 3
        #     [
        #         {
        #             "concept:name": "REPAIR",
        #             "event_elapsed_time": 3255.93
        #         },
        #         {
        #             "concept:name": "QUALITY_CONTROL",
        #             "event_elapsed_time": 2449.21
        #         },
        #         {
        #             "concept:name": "SHIPPING",
        #             "event_elapsed_time": 3501.72
        #         }
        #     ],

        #     # ... weitere 97 Samples        
        prefix, suffix, mostlikely_prediction, predicted_sampled_suffixes = result_values
        
        prefix_activities = [p[concept_name] for p in prefix]
        
        # Target Cases
        suffix_activities = [s[concept_name] for s in suffix]

        # Konkatenation von Prefix und Suffix
        target_case = prefix_activities + suffix_activities
        # Df mit den target cases erstellen
        df_target_case = self.__create_df(cases=[target_case])
        
        # Conformance Checking der target cases
        cf_target_cases = pm4py.conformance.conformance_diagnostics_alignments(df_target_case, pn, im, fm, multi_processing=False)
        
        # Get costs and fitness split for prefix and suffic
        prefix_alignments, suffix_alignments, prefix_cost, suffix_cost, prefix_fitness, suffix_fitness = self.__get_prefix_suffix_fitness(bwc=cf_target_cases[0]['bwc'], cost=cf_target_cases[0]['cost'], prefix=prefix_activities, alignment=cf_target_cases[0]['alignment'])
        
        target_case_alignment = {"prefix": prefix_activities,
                                 "target_suffix": suffix_activities,
                                 "alignment": cf_target_cases[0]['alignment'],
                                 "prefix_alignment": prefix_alignments,
                                 "suffix_alignment": suffix_alignments,
                                 "cost": cf_target_cases[0]['cost'],
                                 "prefix_cost": prefix_cost,
                                 "suffix_cost": suffix_cost,
                                 "fitness": cf_target_cases[0]['fitness'],
                                 "prefix_fitness": prefix_fitness,
                                 "suffix_fitness": suffix_fitness}
            
        # Most likely case
        mostlikely_suffix_activities =  [s[concept_name] for s in mostlikely_prediction]
        mostlikely_case = prefix_activities + mostlikely_suffix_activities
        df_mostlikely_case = self.__create_df(cases=[mostlikely_case])
        
        cf_mostlikely_cases = pm4py.conformance.conformance_diagnostics_alignments(df_mostlikely_case, pn, im, fm, multi_processing=False)
        
        # HIER NOCHMAL NUR DER MOST LIKELY CASE
        # Get costs and fitness split for prefix and suffix
        prefix_alignments, suffix_alignments, prefix_cost, suffix_cost, prefix_fitness, suffix_fitness = self.__get_prefix_suffix_fitness(bwc=cf_mostlikely_cases[0]['bwc'], cost=cf_mostlikely_cases[0]['cost'], prefix=prefix_activities, alignment=cf_mostlikely_cases[0]['alignment'])
        
        mostlikely_case_alignment = {"prefix": prefix_activities,
                                     "ml_suffix": mostlikely_suffix_activities,
                                     "alignment": cf_mostlikely_cases[0]['alignment'],
                                     "prefix_alignment": prefix_alignments,
                                     "suffix_alignment": suffix_alignments,
                                     "cost": cf_mostlikely_cases[0]['cost'],
                                     "prefix_cost": prefix_cost,
                                     "suffix_cost": suffix_cost,
                                     "fitness": cf_mostlikely_cases[0]['fitness'],
                                     "prefix_fitness": prefix_fitness,
                                     "suffix_fitness": suffix_fitness}
            
        # Sampled cases
        predicted_sampled_suffixes_activities = [[s[concept_name] for s in predicted_suffix] for predicted_suffix in predicted_sampled_suffixes]
        predicted_sampled_cases = [prefix_activities + predicted_suffix for predicted_suffix in predicted_sampled_suffixes_activities]  
        df_sampled_cases = self.__create_df(cases=predicted_sampled_cases)

        cf_sampled_cases = pm4py.conformance.conformance_diagnostics_alignments(df_sampled_cases, pn, im, fm, multi_processing=False)
        
        predicted_sample_case_alignment = []
        for i, cf_res in enumerate(cf_sampled_cases):
            # Get costs and fitness split for prefix
            prefix_alignments, suffix_alignments, prefix_cost, suffix_cost, prefix_fitness, suffix_fitness = self.__get_prefix_suffix_fitness(bwc=cf_res['bwc'], cost=cf_res['cost'], prefix=prefix_activities, alignment=cf_res['alignment'])
            
            predicted_sample_case_alignment.append({"prefix": prefix_activities,
                                                    "sampled_suffix": predicted_sampled_suffixes_activities[i],
                                                    "alignment": cf_res['alignment'],
                                                    "prefix_alignment": prefix_alignments,
                                                    "suffix_alignment": suffix_alignments,
                                                    "cost": cf_res['cost'],
                                                    "prefix_cost": prefix_cost,
                                                    "suffix_cost": suffix_cost,
                                                    "fitness": cf_res['fitness'],
                                                    "prefix_fitness": prefix_fitness,
                                                    "suffix_fitness": suffix_fitness})
        
        return target_case_alignment, mostlikely_case_alignment, predicted_sample_case_alignment
    
    # Alignment kommt direkt aus pm4py
    # Berechnet die Alignemnt, Kosten und Fitness für Prefix und Suffix getrennt
    def __get_prefix_suffix_fitness(self, bwc, cost, prefix, alignment):        
        # Prefix und Suffix alignments
        prefix_alignments = []
        suffix_alignments = []

        log_events_encountered = []

        # Find the index where prefix ends
        prefix_end_index = None
        # (a,b) is move
        for i, (a, b) in enumerate(alignment):
        # Only count moves where the log side (a) is a real event
        # Check if it is log or model move then
            if a !=  None and a != '>>':
                log_events_encountered.append(a)

        # Stop when prefix fully matched
            if len(log_events_encountered) == len(prefix) and log_events_encountered == prefix:
                prefix_end_index = i
                break

        if prefix_end_index is None:
            raise ValueError("Prefix not found in alignment (sequence mismatch).")

        # Hier werden Prefix und suffix getrennt behandelt. damit kosten getrennt berechnet werden können
        # Prefix alignments for transperancy
        prefix_alignments = alignment[:prefix_end_index + 1]
        # Suffix alignments
        suffix_alignments = alignment[prefix_end_index + 1:]
        
        # prefix costs
        prefix_cost = 0
        for (a,b) in prefix_alignments:
            if (a == None and b == '>>') or (a == '>>' and b == None):
                prefix_cost += 1
            elif a == b:
                continue
            else:
                # Ist wohl eine Skalierungskonvention
                prefix_cost += 10000
        # suffix costs        
        suffix_cost = 0
        for (a,b) in suffix_alignments:
            if (a == None and b == '>>') or (a == '>>' and b == None):
                suffix_cost += 1
            elif a == b:
                continue
            else:
                # Ist wohl eine Skalierungskonvention
                suffix_cost += 10000
        
        if prefix_cost + suffix_cost != cost:
            raise ValueError("Cost mismatch.")
        
        # prefix fitness
        prefix_fitness = round(1.0 - float(prefix_cost) / float(bwc), 2)
        # suffix fitness
        suffix_fitness = round(1.0 - float(suffix_cost) / float(bwc), 2)

        return prefix_alignments, suffix_alignments, prefix_cost, suffix_cost, prefix_fitness, suffix_fitness
                
    # Erstellt ein DataFrame mit den Cases
    def __create_df(self, cases: list):            
        # Single, deterministic base timestamp
        base_time = pd.Timestamp("2025-01-01 00:00:00")
                
        df = pd.DataFrame({'case:concept:name': range(len(cases)),
                           'concept:name': cases})
                
        df['case:concept:name'] = df['case:concept:name'].astype(str)
                
        df = df.explode('concept:name', ignore_index=True)
        # Ensure concept:name is a string and not None
        df['concept:name'] = df['concept:name'].apply(lambda x: str(x) if pd.notnull(x) else 'UNKNOWN')
                    
        # Compute offset grouped by case
        df['offset'] = df.groupby('case:concept:name').cumcount()
                
        df['time:timestamp'] = base_time + pd.to_timedelta(df['offset'], unit='min')
        # Fill missing timestamps with the last valid one
        last_valid_timestamp = df['time:timestamp'].dropna().max()
        df['time:timestamp'] = df['time:timestamp'].fillna(last_valid_timestamp)
                
        return df
    
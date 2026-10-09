from src.utils.mini_tools import split_raw_data, save_data
from src.utils.prep_tools.tensor_prep import TensorPreprocessor
from src.utils.prep_tools.agg_prep import AggPreprocessor
from src.utils.prep_tools.kmeans_prep import KMeansPreprocessor
import os

# Rohdaten pro Stage in das benötigte Modellformat bringen
    # Prozessstart
    #     ↓
    # Stage 0 = erster Decision Point
    #     ↓
    # weitere Prozessereignisse
    #     ↓
    # Stage 1 = zweiter Decision Point
    #     ↓
    # weitere Prozessereignisse
    #     ↓
    # Stage 2 = dritter Decision Point
    #     ↓
    # Outcome
class ProcessPreprocessor():
    def __init__(self, args, raw_data, DATASET_PARAMS_LIST):
        to_add = ''
        if args.dataset == 'bpic17':
            conf_suffix = "_case" if args.confounding_type == "case" else ""
            to_add = os.path.join("bpic17" + conf_suffix, str(args.n_stages))
        elif args.dataset == "SimRepair":
            to_add = "SimRepair"
        self.DATA_FOLDER = os.path.join("data", to_add, str(args.train_size), str(int(100 * args.delta)))
        self.PATH_BEGIN = str(args.train_size) + "_" + str(int(100*args.delta)) + "_"

        self.args = args
        self.raw_data = raw_data
        self.DATASET_PARAMS_LIST = DATASET_PARAMS_LIST

        # Hier werden die Ereignisse jedes Falls chronologisch sortiert und zwei Werte für die spätere Datenaufbereitung bestimmt.
        if "elapsed_time" in self.raw_data.columns:
            self.data_ordered = raw_data.sort_values(by=["case_nr", "elapsed_time"])
        elif "event_nr" in self.raw_data.columns:
            self.data_ordered = raw_data.sort_values(by=["case_nr", "event_nr"])
        else:
            self.data_ordered = raw_data.sort_values(by=["case_nr"])

        # Dadurch stehen alle Ereignisse eines Falls zusammen und normalerweise in zeitlicher Reihenfolge.
        # Maximale Prozesslänge:
        # den längsten Fall auswählen
        max_process_len = self.data_ordered.groupby(["case_nr"]).size().max() - 1 # Minus one because the last activity is cancel or accept application, not useful for our inference
        missing_value = -100

        self.add_properties = {
            "max_process_len": max_process_len,
            "missing_value": missing_value
        }

    # Konkretes Beispiel: "agg"
    def preprocess(self, encoding, prep_utils_list=None, eval=False, action_combo=""):
        # "infer_prop": 1.0, "train_prop": 0.0 -> 100 % der Daten werden als Inferenz-/Testdaten verwendet.
        # 100 % der Daten werden als Inferenz-/Testdaten verwendet.
        if eval:
            self.PREP_PARAMS =  {"infer_prop": 1.0, "train_prop": 0.0, "filter_useless_cols": True}
            self.DATA_FOLDER_TOTAL = os.path.join(self.DATA_FOLDER, "eval")
            self.PATH_END_DICT = {"infer": "_preprocessed_data_eval", "utils": "_preprocessed_utils_eval"}
        else:
            self.PREP_PARAMS =  {"infer_prop": 0.2, "train_prop": 0.8, "filter_useless_cols": True}
            self.DATA_FOLDER_TOTAL = os.path.join(self.DATA_FOLDER, "training_and_tuning")
            self.PATH_END_DICT = {"train": "_preprocessed_data_train", "infer": "_preprocessed_data_infer", "utils": "_preprocessed_utils"}

        ## Splitten der Rohdaten in Trainings- und Inferenzdaten
        # In Mini_tools, werden die Daten gesplittet
        # Data train sind die rohen Daten
        # infer prop ist nur, wie train und test splits aufgespaltet werden
        self.data_train, self.data_infer = split_raw_data(data=self.data_ordered, infer_prop=self.PREP_PARAMS["infer_prop"])
        
        # KMeans-Q wird hier nur mit einem gemeinsamen Datensatz und einem gemeinsamen Modell vorbereitet:
        if "kmeans_q" in self.args.methods:
            n_stages = 1
        else: n_stages = self.args.n_stages

        self.data_train_prep_list = []
        self.data_infer_prep_list = []
        self.prep_utils_list = []

        # Prefixe auf die Stages verteilen
        for stage in range(n_stages):
        # DATASET_PARAMS_LIST for this SimRepair run (stage 0):
        # [
        #     {
        #         'train_size': 1000,
        #         'test_size': 20,
        #         'val_share': 0.5,
        #         'train_val_size': 20,
        #         'test_val_size': 10,
        #         'simulation_start': datetime.datetime(2024, 3, 20, 8, 0),
        #         'random_seed_train': 6724,
        #         'random_seed_test': 16900,
        #         'log_cols': [
        #             'case_nr', 'activity', 'timestamp', 'elapsed_time',
        #             'duration', 'bike_value', 'repair_severity',
        #             'quality_uncertainty', 'process_type',
        #             'employee_competence', 'material_quality',
        #             'customer_patience', 'customer_friendliness', 'outcome'
        #         ],
        #         'case_cols': ['bike_value', 'repair_severity'],
        #         'event_cols': [
        #             'activity', 'elapsed_time', 'quality_uncertainty',
        #             'process_type', 'employee_competence', 'material_quality'
        #         ],
        #         'cat_cols': ['activity', 'process_type'],
        #         'scale_cols': [
        #             'elapsed_time', 'bike_value', 'repair_severity',
        #             'quality_uncertainty', 'employee_competence',
        #             'material_quality', 'outcome'
        #         ],
        #         'last_state_cols': ['elapsed_time'],
        #         'intervention_info': {
        #             'name': 'choose_procedure',
        #             'data_impact': 'direct',
        #             'actions': ['start_standard', 'start_priority'],
        #             'action_width': 2,
        #             'action_depth': 1,
        #             'activities': ['start_standard', 'start_priority'],
        #             'column': 'activity',
        #             'start_control_activity': ['initiate_case'],
        #             'end_control_activity': ['initiate_case'],
        #             'retain_method': 'precise',
        #             'action_combinations': ('start_standard',),
        #             'action_width_combinations': 2,
        #             'action_depth_combinations': 1,
        #             'len': 1,
        #             'RCT': False,
        #             'flat_activities': 'start_standard'
        #         },
        #         'filename': "repair_log_['choose_procedure']",
        #         'policies_info': {
        #             'general': 'real',
        #             'take_employee': {'max_employee_waits': 3},
        #             'conduct_qc': {'max_qc': 3},
        #             'max_rerepairs': 3
        #         }
        #     }
        # ]

            #print(self.DATASET_PARAMS_LIST[stage])
            DATASET_PARAMS = self.DATASET_PARAMS_LIST[stage]
            # Brauchst du vor allem später in dem Preprocessing von bspw. Tensor, weil hier pro event markiert wird, ob ein Treatement vorlag
            nr_treatment_columns = DATASET_PARAMS["intervention_info"]["action_width"] if DATASET_PARAMS["intervention_info"]["action_width"] > 2 else 1
            # nr_treatment_columns = action_width if action_width > 2 else 1
            # 0 = wait
            # 1 = call
            #
            # Aktion 0 = [1, 0, 0]
            # Aktion 1 = [0, 1, 0]
            # Aktion 2 = [0, 0, 1]

            self.add_properties["nr_treatment_columns"] = nr_treatment_columns
            # Stage wird wichtig um zu sehen, welche Intervention, welche Ation und welche Kontroll Aktvitiäten gelten für diese Stage
            # Ist abe rnur wichtig bei agg apparently
            self.add_properties["stage"] = stage

            # Je nach Encoding eigentliche Transformation durchführen
            # Für PSP Ist Tensor processing besser
            if encoding == "tensor":
                self.preprocessor = TensorPreprocessor(self.data_train, self.data_infer, self.PREP_PARAMS, DATASET_PARAMS, self.add_properties, prep_utils=prep_utils_list[stage] if prep_utils_list is not None else None)
            elif encoding == "agg":
                self.preprocessor = AggPreprocessor(self.data_train, self.data_infer, self.PREP_PARAMS, DATASET_PARAMS, self.add_properties, prep_utils=prep_utils_list[stage] if prep_utils_list is not None else None)
            elif encoding == "kmeans":
                # NOTE: returns data_infer as None, because it is not used in the KMeansPreprocessor
                self.preprocessor = KMeansPreprocessor(self.data_train, self.data_infer, self.PREP_PARAMS, DATASET_PARAMS, prep_utils=prep_utils_list[stage] if prep_utils_list is not None else None, args=self.args)

            data_train_prep, data_infer_prep, prep_utils = self.preprocessor.preprocess()
            self.data_train_prep_list.append(data_train_prep)
            self.data_infer_prep_list.append(data_infer_prep)
            self.prep_utils_list.append(prep_utils)
            # DEBUG-STOPP: nach der ersten Stage direkt zurück, ohne zu speichern (zum Entfernen diese zwei Zeilen löschen)
            # return self.data_train_prep_list, self.data_infer_prep_list, self.prep_utils_list

            # Save
            if not eval:
                save_data(data_train_prep, os.path.join(os.getcwd(), self.DATA_FOLDER_TOTAL, self.PATH_BEGIN + encoding + "_" + str(stage) + "_" + action_combo + self.PATH_END_DICT["train"]))
            save_data(data_infer_prep, os.path.join(os.getcwd(), self.DATA_FOLDER_TOTAL, self.PATH_BEGIN + encoding + "_" + str(stage) + "_" + action_combo + self.PATH_END_DICT["infer"]))
            save_data(prep_utils, os.path.join(os.getcwd(), self.DATA_FOLDER_TOTAL, self.PATH_BEGIN + encoding + "_" + str(stage) + "_" + action_combo + self.PATH_END_DICT["utils"]))

        # 50000.0,0.35 da zwei prefixe 
        # X_case Fallattribute
        # X_event sind event_Sequezen
        # prefixlength ist klar
        # T is Treatment an Entscheidungspunkt
        # Outcome des Falls
        # ({"X_case":[[50000.0,0.35],[25000.0,0.60]], "X_event":[[[3,6,2],[4,1,0]],[[2,5,0],[0,0,0]]], "prefix_len":[2,1], "T":[1,0], "Y":[1200.0,-300.0]}, 
        # {"X_case":[[42000.0,0.45]], "X_event":[[[3,6,2],[4,1,0]]], "prefix_len":[2], "T":[1], "Y":[800.0]}, 
        # {"dim_x_case":2,"dim_x_event":3,"dim_t":1,"dim_output":1,"case_feature_names":["amount","quality_scaled"],"event_feature_names":["activity_code","resource_code","time_scaled"]})
        return self.data_train_prep_list, self.data_infer_prep_list, self.prep_utils_list
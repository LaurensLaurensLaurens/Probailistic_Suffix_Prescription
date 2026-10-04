import warnings
import pandas as pd
from copy import copy, deepcopy
import random
import simpy
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.utils.petri_utils import get_transition_by_name
from SimRepair.extra_flow_conditions import ExtraFlowConditioner
from SimRepair.activity_execution import ActivityExecutioner
import SimRepair.petri_net_generator as petri_net_generator

class PresProcessGenerator():
    def __init__(self, dataset_params, seed=82):
        #GENERAL params
        self.log_cols = dataset_params["log_cols"]
        self.random_seed = seed
        self.simulation_start = dataset_params["simulation_start"]
        #POLICY params
        self.policies_info = dataset_params["policies_info"]
        #INTERVENTION params
        self.intervention_info = dataset_params["intervention_info"]
        self.intervention_info["flat_activities"] = [act for sublist in self.intervention_info["activities"] for act in sublist]
        self.get_petri_net()

    warnings.filterwarnings('ignore')


    def find_activity_index(self, target_string):
        for index, sublist in enumerate(self.intervention_info["activities"]):
            if target_string in sublist:
                return index
        return -1


    def get_petri_net(self):
        self.net = petri_net_generator.generate_petri_net()

    # Simulationsumgebung erzeugen
    # → setup() als Simulationsprozess registrieren
    # → alle geplanten Ereignisse ausführen
    # → stoppen, wenn keine Ereignisse mehr vorhanden sind

    # env	gemeinsame virtuelle Simulationsumgebung
    # net	Petri-Netz mit möglichen Prozessaktivitäten
    # initial_marking	anfängliche Markenverteilung im Petri-Netz
    # n_cases	Anzahl zu simulierender Prozessfälle
    # simulation_state	optionaler gespeicherter Zustand für eine Fortsetzung
    # timeout	optionale Wartezeit vor der Fortsetzung
    def simulation_of_events(self, net, initial_marking, n_cases, simulation_state=None, timeout=None):
        env = simpy.Environment()
        env.process(self.setup(env, net, initial_marking, n_cases, simulation_state, timeout))
        env.run()

    # setup() organisiert die Simulation auf Fallebene. Die Methode erzeugt nicht direkt einzelne Eventzeilen, sondern:
    # durchläuft alle zu simulierenden Fälle,
    # setzt den Zufallszustand,
    # baut den Simulationszustand des Falls auf,
    # startet simulate_trace() für diesen Fall,
    # wartet auf dessen Abschluss,
    # startet anschließend den nächsten Fal
    def setup(self, env, net, initial_marking, n_cases, simulation_state=None, timeout=None):
        for i in range(0, n_cases):
            # Set random seed for each case if normal simulation (not under each action)
            if self.normal_simulation:
                self.random_seed = self.random_seed + i
                self.random_obj.seed(self.random_seed)
                self.activity_executioner.random_obj = self.random_obj
                self.extra_flow_conditioner.random_obj = deepcopy(self.random_obj)
            # Set simulation state
            if not simulation_state:
                simulation_state = {"trace": [], "net": copy(net), "marking": copy(initial_marking), "parallel_executions": False, "parallel_timestamps": {"HQ": [self.simulation_start, self.simulation_start], "val": [self.simulation_start, self.simulation_start]}, "case_nr": i, "env": env}
            else:
                simulation_state["env"] = env
                simulation_state["net"] = copy(net)
                simulation_state["marking"] = copy(initial_marking)
                simulation_state["case_nr"] = i
            # Do timeout if necessary
            if timeout:
                self.do_timeout(env, timeout)
            # Simulate trace
            proc = env.process(self.simulate_trace(simulation_state))
            yield proc
            # Set simulation start and end
            if not self.generate_under_each_action and i == self.n_cases - 1:
                self.simulation_start, self.simulation_end = self.activity_executioner.set_simulation_end_and_start(simulation_start=self.simulation_start, last_event=self.log[-1])
        

    def simulate_trace(self, simulation_state):
        case_num = simulation_state["case_nr"]
        env = simulation_state["env"]
        trace = deepcopy(simulation_state["trace"])
        marking = copy(simulation_state["marking"])
        net = copy(simulation_state["net"])
        parallel_executions = simulation_state["parallel_executions"]
        parallel_timestamps = simulation_state["parallel_timestamps"]

        if self.intervention_info["RCT"]:
            int_timing_list = [1000] * len(self.intervention_info["name"])
        int_enabled_list = [0] * len(self.intervention_info["name"])

        while True:
            # GET PREVIOUS EVENT
            prev_event = trace[-1] if len(trace) > 0 else None
            # BREAK IF NO TRANSITIONS ARE ENABLED (END OF TRACE)
            if (not semantics.enabled_transitions(net, marking)):
                for event in trace:
                    event["outcome"] = trace[-1]["outcome"]
                    self.log.append(event)
                    if self.generate_under_each_action:
                       self.int_points_available  = False
                break

            # GET THE ENABLED ACTIVITIES BASED ON (EXTRA) FLOW CONDITIONS (MOSTLY POLICIES)
            control_flow_enabled_trans = list(semantics.enabled_transitions(net, marking))
            control_flow_enabled_trans = sorted(control_flow_enabled_trans, key=lambda x: x.label)
            all_enabled_trans = self.extra_flow_conditioner.filter_enabled_trans(net, marking, control_flow_enabled_trans, trace, self.policies_info, self.intervention_info)
            
            # 
            # Die beiden Blöcke behandeln zwei verschiedene Simulationsarten: Der erste erzeugt bei der Entscheidungsfindung mehrere Gegen alternative „Was-wäre-wenn?“-Zweige; der zweite erzeugt für Trainingsdaten nur einen tatsächlich ausgeführten, randomisierten Verlauf.
            # GENERATE UNDER EACH ACTION IF ENABLED
            # DAS IST SEHR RELEVNAT, WEIL HIER NACHGESEHEN WIRD, OB ES SICH UM EINEN INTERVENTIONSPUNKT HANDELT
            if self.generate_under_each_action:
                all_enabled_trans_cf = self.extra_flow_conditioner.filter_enabled_trans(net, marking, control_flow_enabled_trans, trace, self.policies_info, self.intervention_info, ignore_intervention_policy=True)
                int_activities = [act for act in self.intervention_info["flat_activities"] if act != "do_nothing"]
                enabled_int_activities = [act for act in int_activities if get_transition_by_name(net, act) in all_enabled_trans_cf]
                # Is it an intervention point?
                if len(enabled_int_activities) > 0:
                    first_activity = enabled_int_activities[0]
                    self.current_int_index = self.find_activity_index(first_activity)
                    # Is there a choice for intervention?
                    choice_for_int_unavailable = (len(all_enabled_trans_cf) <= 1 and self.intervention_info["data_impact"][self.current_int_index] == "direct")
                    # If generate under each action is enabled, generate all intervention events
                    if not choice_for_int_unavailable and self.generate_under_each_action:
                        simulation_state = {"trace": deepcopy(trace), "net": copy(net), "marking": copy(marking), "parallel_executions": parallel_executions, "parallel_timestamps": parallel_timestamps, "case_nr": case_num, "env": env}
                        # Was erzeugt generate_all_int_events()?
                        #  z.b. "set_ir_3_levels"
                        # actions = [0.07, 0.08, 0.09]
                        # ERZEUGT NUR DIE ZUSTÄNDE 
                        self.generate_all_int_events(self.current_int_index, simulation_state)
                        break
                    
            # CHECK FOR RCT IF ENABLED
            if self.intervention_info["RCT"]:
                int_activities = [act for act in self.intervention_info["flat_activities"] if act != "do_nothing"]
                enabled_int_activities = [act for act in int_activities if get_transition_by_name(net, act) in all_enabled_trans]
                if len(enabled_int_activities) > 0:
                    first_activity = enabled_int_activities[0]
                    self.current_int_index = self.find_activity_index(first_activity)
                    int_enabled_list[self.current_int_index] += 1
                    # Is there a choice for intervention?
                    choice_for_int_unavailable = (len(all_enabled_trans) <= 1 and self.intervention_info["data_impact"][self.current_int_index] == "direct")
                    if (int_enabled_list[self.current_int_index] - 1) != int_timing_list[self.current_int_index] and not choice_for_int_unavailable:
                        all_enabled_trans = [act for act in all_enabled_trans if act.label not in self.intervention_info["flat_activities"]]
                    elif (int_enabled_list[self.current_int_index] -1) == int_timing_list[self.current_int_index] and not choice_for_int_unavailable:
                        all_enabled_trans = [act for act in all_enabled_trans if act.label in self.intervention_info["flat_activities"]]
           
            self.random_obj.shuffle(all_enabled_trans)
            trans = all_enabled_trans[0]
            
            # EXECUTE THE ACTIVITY
            if trans.label is not None and 'ghost' not in trans.label:
                # Set basic attributes
                event = {}
                event["case_nr"] = case_num
                event["activity"] = trans.label

                # Set other attributes
                event = self.activity_executioner.set_event_variables(event, prev_event, intervention_info=self.intervention_info)
                # Set timestamp, elapsed time and timeout env
                event["timestamp"], parallel_executions, parallel_timestamps, timeout = self.activity_executioner.set_event_timestamp(event["activity"], prev_event, env, parallel_executions, parallel_timestamps, self.simulation_start)
                event["elapsed_time"] = ((event["timestamp"] - trace[0]["timestamp"]).total_seconds() / 86400) if len(trace) > 0 else 0
                yield env.timeout(timeout)

                if self.intervention_info["RCT"]:
                    for timing in int_timing_list:
                        if timing == 1000:
                            int_timing_list = self.sample_timing(event)

                trace.append(event)
            
            marking = semantics.execute(trans, net, marking)

    # generate_all_int_events(...) erzeugt an einem Entscheidungspunkt für jede mögliche Aktion genau eine alternative Fortsetzung.
        #     log_per_action_list = [
        #     [...Präfix..., {"activity": "calculate_offer",
        #                     "interest_rate": 0.07}],

        #     [...Präfix..., {"activity": "calculate_offer",
        #                     "interest_rate": 0.08}],

        #     [...Präfix..., {"activity": "calculate_offer",
        #                     "interest_rate": 0.09}]
        # ]
    def generate_all_int_events(self, int_index, simulation_state):
        log_per_action_list = [] # Präfix inklusive des nächsten Ereignisses für jede Aktion
        self.simulation_state_list = [] # Simulationszustand nach jeder Aktion
        self.timeout_list = [] # benötigte SimPy-Zeit pro Aktion
        log_current_state = []
        self.activity_executioner_list = [] # benötigte SimPy-Zeit pro Aktion
        # random_state = self.random_obj.getstate() ist der Zufallszustand der für die Simulation verwendet wird
        # Aktueller Zufallszustand wird gespeichert
        # Warum? Ohne das Speichern könnten die Alternativen beispielsweise unterschiedliche zufällige Bearbeitungszeiten bekommen:
        random_state = self.random_obj.getstate()
        
        # Sagen wir vorher
        # [
        #     {
        #         "activity": "start",
        #         "outcome": 0
        #     },
        #     {
        #         "activity": "initiate_application",
        #         "outcome": -250
        #     }
        # ]
        # Die Methode setzt für alle bisherigen Ereignisse den momentan letzten bekannten Outcome:
        # [
        #     {
        #         "activity": "start",
        #         "outcome": 0
        #     },
        #     {
        #         "activity": "initiate_application",
        #         "outcome": -250
        #     }
        # ]
        for event in simulation_state["trace"]:
            event["outcome"] = simulation_state["trace"][-1]["outcome"]
            log_current_state.append(event)
        # Das geht über alle Aktionen, z.b. ir level setzen, das wäre int_index = 1 oder Prozedur wählen, das wäre int_index = 0
            # int_index ist der Index des Entscheidungspunkts
            #    intervention_info["name"] = [
            #     "choose_procedure",
            #     "set_ir_3_levels"
            # ]
        for action in self.intervention_info["actions"][int_index]:
            # Set each a new random state, to stay consistent (same changes of event variables under each action)
            # Jede Aktion erhält denselben Zufalls-Ausgangspunkt
            # Für jede Aktion wird ein separater ActivityExecutioner erzeugt.
            # Warum ein eigener ActivityExecutioner pro Aktion? Nachdem ein Ereignis erzeugt wurde, hat sich dessen Zufallszustand weiterbewegt. 
            # Wenn diese Alternative später ausgewählt wird, muss genau mit diesem fortgeschrittenen Zustand weitergerechnet werden.
            random_obj = deepcopy(self.random_obj)
            activity_executioner = ActivityExecutioner(random_obj)
            activity_executioner.set_state(random_state)
            self.activity_executioner_list.append(activity_executioner)
            #Generate
            action_log = deepcopy(log_current_state)
            # Die jeweils anderen Aktionen werden bestimmt
            # actions = ["start_standard", "start_priority"]
            #action = "start_standard"
            other_actions = [act for act in self.intervention_info["actions"][int_index] if act != action]
            
            # Ein Ereignis für diese Aktion wird erzeugt. Für die direkte Aktion "start_priority" beispielsweise:
            # z.B.
            # {
            #     "case_nr": 0,
            #     "activity": "start_priority",
            #     "timestamp": datetime(...),
            #     "elapsed_time": 1.5,
            #     "cum_cost": 75,
            #     "outcome": -75
            # }
            # action_simulation_state. Der State direk nach dem Ereignis.
            # timeout_to_be_done. Die SimPy-Zeit, die mit dem Ereignis verbunden ist,
            # 
            int_event, action_simulation_state, timeout_to_be_done = self.generate_one_int_event(int_index, simulation_state, action, other_actions, activity_executioner, random_obj)
            #Append
            # self.log_per_action_list = [
            #     [
            #         {"activity": "start"},
            #         {"activity": "initiate_application"},
            #         {"activity": "start_standard"}
            #     ],
            #     [
            #         {"activity": "start"},
            #         {"activity": "initiate_application"},
            #         {"activity": "start_priority"}
            #     ]
            # ]
            action_log.append(int_event)
            log_per_action_list.append(action_log)
            # ERZEIGT NUR DIE ZUSTÄNDE
            # self.simulation_state_list = [
            #     Zustand_nach_7_Prozent,
            #     Zustand_nach_8_Prozent,
            #     Zustand_nach_9_Prozent
            # ]
            self.simulation_state_list.append(action_simulation_state)
            self.timeout_list.append(timeout_to_be_done)
        self.log_per_action_list = log_per_action_list


    # Erzeugt ein Ereignis für eine bestimmte Aktion.
    def generate_one_int_event(self, int_index, simulation_state, action, other_actions, activity_executioner, random_obj):
        case_num = simulation_state["case_nr"]
        env = simulation_state["env"]
        trace = deepcopy(simulation_state["trace"])
        marking = copy(simulation_state["marking"])
        net = copy(simulation_state["net"])
        parallel_executions = deepcopy(simulation_state["parallel_executions"])
        parallel_timestamps = deepcopy(simulation_state["parallel_timestamps"])

        prev_event = trace[-1] if len(trace) > 0 else None

        control_flow_enabled_trans = list(semantics.enabled_transitions(net, marking))
        control_flow_enabled_trans = sorted(control_flow_enabled_trans, key=lambda x: x.label)
        all_enabled_trans = self.extra_flow_conditioner.filter_enabled_trans(net, marking, control_flow_enabled_trans, trace, self.policies_info, self.intervention_info)

        if self.intervention_info["data_impact"][int_index] == "direct":
            if action != "do_nothing":
                all_enabled_trans = self.extra_flow_conditioner.filter_enabled_trans(net, marking, control_flow_enabled_trans, trace, self.policies_info, self.intervention_info, action_to_be_taken=action, ignore_intervention_policy=True)
            else:
                all_enabled_trans = self.extra_flow_conditioner.filter_enabled_trans(net, marking, control_flow_enabled_trans, trace, self.policies_info, self.intervention_info, ignore_intervention_policy=True)
                transition_action_list = [get_transition_by_name(net, other_act) for other_act in other_actions]
                all_enabled_trans = [act for act in all_enabled_trans if act not in transition_action_list]

        random_obj.shuffle(all_enabled_trans)
        trans = all_enabled_trans[0]
        
        # EXECUTE THE ACTIVITY
        if trans.label is not None and 'ghost' not in trans.label:
            # Set basic attributes
            event = {}
            event["case_nr"] = case_num
            event["activity"] = trans.label

            # Set other attributes
            # Die Methode füllt praktisch alle fachlichen Attribute des neuen Events aus.
            #amount
            # quality
            # unc_quality
            # est_quality
            # cum_cost
            # interest_rate
            # discount_factor
            # outcome
            # noc
            # nor
            # min_interest_rate
            if self.intervention_info["data_impact"][int_index] == "indirect":
                event = activity_executioner.set_event_variables(event, prev_event, action, self.intervention_info)
            else:
                event = activity_executioner.set_event_variables(event, prev_event, intervention_info=self.intervention_info)

            # Set timestamp, elapsed time and timeout env
            event["timestamp"], new_parallel_executions, new_parallel_timestamps, timeout_to_be_done = activity_executioner.set_event_timestamp(event["activity"], prev_event, env, parallel_executions, parallel_timestamps, self.simulation_start)
            event["elapsed_time"] = ((event["timestamp"] - trace[0]["timestamp"]).total_seconds() / 86400) if len(trace) > 0 else 0
            # WE NEED TO REMEMBER THE TIMEOUT FOR THE REST OF THE CASE GENERATION
        trace.append(event)
        marking = semantics.execute(trans, net, marking)
        new_simulation_state = {"trace": deepcopy(trace), "net": copy(net), "marking": copy(marking), "parallel_executions": new_parallel_executions, "parallel_timestamps": new_parallel_timestamps, "case_nr": case_num, "env": env}
        return event, new_simulation_state, timeout_to_be_done
    
    # Die gewählte Aktivität benötigt 7.200 simulierte Sekunden. Damit die Intervention etwas kostet
    def do_timeout(self, env, timeout):
        yield env.timeout(timeout)


    # INFERENCE IS ALWAYS ONE CASE
    def end_simulation_inference(self):
        return deepcopy(self.log)


    def continue_simulation_inference(self, action_index):
        simulation_state = self.simulation_state_list[action_index]
        timeout_to_be_done = self.timeout_list[action_index]
        self.activity_executioner = self.activity_executioner_list[action_index]
        self.random_obj = self.activity_executioner.random_obj
        self.log_per_action_list = []
        self.simulation_of_events(net=simulation_state["net"], initial_marking=simulation_state["marking"], n_cases=self.n_cases, simulation_state=simulation_state, timeout = timeout_to_be_done)
        return deepcopy(self.log_per_action_list)
    

    def start_simulation_inference(self, seed_to_add=0, seed_to_add_conditioner=0):
        self.normal_simulation = False
        self.random_seed = self.random_seed + seed_to_add
        self.random_obj = random.Random(self.random_seed)
        self.generate_under_each_action = True
        self.int_points_available = True
        self.n_cases = 1
        self.current_int_index = None
        self.activity_executioner = ActivityExecutioner(self.random_obj)
        random_obj_conditioner = deepcopy(self.random_obj)
        random_obj_conditioner.seed(self.random_seed + seed_to_add + seed_to_add_conditioner)
        self.extra_flow_conditioner = ExtraFlowConditioner(random_obj_conditioner)
        self.log_per_action_list = []
        self.log = []
        self.simulation_of_events(self.net, self.net.initial_marking, n_cases=self.n_cases)
        return deepcopy(self.log_per_action_list)


    # FOR NORMAL YOU CAN SPECIFY THE N_CASES
    def run_simulation_normal(self, n_cases, seed_to_add=0):
        self.normal_simulation = True
        self.random_seed = self.random_seed + seed_to_add
        self.random_obj = random.Random(self.random_seed)
        self.random_obj_for_timing = random.Random(self.random_seed)
        self.generate_under_each_action = False
        self.int_points_available = True
        self.n_cases = n_cases
        self.current_int_index = None
        self.activity_executioner = ActivityExecutioner(self.random_obj)
        self.extra_flow_conditioner = ExtraFlowConditioner(deepcopy(self.random_obj))
        self.log = []
        self.simulation_of_events(self.net, self.net.initial_marking, n_cases=self.n_cases)
        self.log = pd.DataFrame(self.log)
        return deepcopy(self.log)
    

    def sample_with_weighted_probability(self, integer_set, decrease_rate):
        weights = [1/(i ** decrease_rate) for i in integer_set]
        return weights
    

    def sample_timing(self, event=None):
        timing_list = []
        for int_index in range(len(self.intervention_info["name"])):
            lower_bound = 0
            upper_bound = self.intervention_info["action_depth"][int_index] - 1
            if "do_nothing" in self.intervention_info["activities"][int_index]:
                upper_bound += 1
            
            timing = self.random_obj_for_timing.randint(lower_bound, upper_bound)
            timing_list.append(timing)
        return timing_list
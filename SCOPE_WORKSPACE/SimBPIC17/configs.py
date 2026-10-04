"""
Below are each activiy's mean and standard deviation.

activity          mean           std   count
3      W_Complete application    477.697515   1629.458322  148025
7      W_Validate application    481.713791   1588.498384  193440
2     W_Call incomplete files    816.740904   3210.512418  161627
6      W_Shortened completion    926.202876   3448.351758     233
1         W_Call after offers   3808.815381   9074.362537  181120

We adjust slightly for simplicity (rounding, and keeping the elapsed time of treatment and wait less variable). Please also note that the durations are in minutes. 
The duration of the treatment activity = (the duration of the control activity - the effect on the duration due to treatment).

"""

shortened_prefix_chance = 0.05

# treatment_point_activities bezeichnet Aktivitäten, an denen eine neue Entscheidung getroffen werden darf
treatment_point_activities = ["validate_application", "wait_incomplete_files", "call_incomplete_files"]

# treatment_activities enthält dagegen nur die ausgeführten Behandlungsaktionen:
treatment_activities = ["wait_incomplete_files", "call_incomplete_files"]

duration_data_stats = {
    "complete_application": (4000, 50, 'lognormal'),
    "call_after_offers": (4000, 50, 'lognormal'),
    "shortened_completion": (10, 0, 'lognormal'),
    "validate_application": (4000, 50, "lognormal"),
    "wait_incomplete_files": (10000, 0, "constant"),
    "end": (0, 0, "constant"),
}

stages_to_sum = ["complete_application", "call_after_offers", "validate_application"]
# Calculate average duration (expected value)
average_duration = sum(duration_data_stats[stage][0] for stage in stages_to_sum)
average_duration = average_duration / len(stages_to_sum)

standard_row = {
    "case_nr": -100,
    "activity": "complete_application",
    "elapsed_time": -100,
    "duration": -100,
    "monthlycost": -100,
    "loangoal": "Car",
    "outcome": None,
    "treatment": 0,
    "treatment_effect": -100,
    "event_nr": -100,
}

standard_prefix = {
    "case_nr": [-100, -100, -100],
    "activity": ["complete_application", "call_after_offers", "validate_application"],
    "elapsed_time": [-100, -100, -100],
    "duration": [-100, -100, -100],
    "monthlycost": [-100, -100, -100],
    "loangoal": ["Car", "Car", "Car"],
    "outcome": [None, None, None],
    "treatment": [0, 0, 0],
    "treatment_effect": [-100, -100, -100],
    "event_nr": [-100, -100, -100],
}

shortened_prefix = {
    "case_nr": [-100, -100, -100, -100],
    "activity": ["complete_application", "call_after_offers", "shortened_completion", "validate_application"],
    "elapsed_time": [-100, -100, -100, -100],
    "duration": [-100, -100, -100, -100],
    "monthlycost": [-100, -100, -100, -100],
    "loangoal": ["Car", "Car", "Car", "Car"],
    "outcome": [None, None, None, None],
    "treatment": [0, 0, 0, 0],
    "treatment_effect": [-100, -100, -100, -100],
    "event_nr": [-100, -100, -100, -100],
}

# Treatment effect calculation parameters
treatment_function = 'linear'
# treatment_function = 'discrete'
constant_term_min = 0
constant_term_max = 2000
exponent_term_effect = 4
average_duration_point_effect_graph = 0.75
divider_term_effect = (1 / average_duration_point_effect_graph) * average_duration

# Outcome calculation parameters
treatmen_doubt_point_intersect_effect_graph = 0.25
cost_treatment_baseline = duration_data_stats["wait_incomplete_files"][0] / 2
if treatment_function == 'exponential':
    cost_treatment_baseline = ( ( (average_duration_point_effect_graph) ** exponent_term_effect ) ) * divider_term_effect
cost_time_baseline = 1

# Treatment assignment parameters
monthly_cost_level_assign = 200
loan_assign = ['Existing loan takeover', 'Car']
avg_duration_level_assign = 4025
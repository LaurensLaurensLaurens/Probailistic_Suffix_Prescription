path = "c:\\Users\\u0166838\\OneDrive - KU Leuven\\Documents\\Doc\\Code\\DTR-Pro"
import sys
sys.path.append(path)

from SimBPIC17.configs import duration_data_stats, treatment_function, constant_term_max, constant_term_min, exponent_term_effect, cost_time_baseline, cost_treatment_baseline, loan_assign, avg_duration_level_assign, treatment_point_activities, treatment_activities, divider_term_effect
import numpy as np

random_seed = 42
np.random.seed(random_seed)

# Diese Funktion berechnet die Behandlungsentscheidung für ein gegebenes Event.
def calculate_treatment_assignment(row, delta=0.99, treatment_to_do=None, random_object_random_policy=None):
    # Das ist eine Schutzbedingung: Eine neue Behandlungsentscheidung wird nur getroffen, wenn beide Bedingungen erfüllt sind:
    # Die aktuelle Aktivität ist ein erlaubter Entscheidungspunkt.
    # Das Event ist das aktuell letzte Event des Falls (event_nr == length).
    # Bei jeder Stufe enthält der Fall bereits mehrere vergangene Events. Die nächste Aktion (call oder wait) darf nur am aktuellen Ende des Prozessverlaufs angehängt werden. Deshalb wird nur beim letzten Event (event_nr == length) eine neue Entscheidung berechnet.
    # Ohne diese Bedingung würde der Code auch für frühere Events erneut Entscheidungen treffen und möglicherweise bereits erzeugte Prozessschritte rückwirkend verändern.

    if row["activity"] not in treatment_point_activities or row["event_nr"] < row["length"]:
        return row["treatment"]  
    
    if treatment_to_do is not None:
        return treatment_to_do

    avg_duration = row["avg_duration"]
    assign = 0

    # Die Zeile prüft, ob der Kreditzweck des aktuellen Falls zu den in loan_assign definierten Kategorien gehört
    loan_condition = row["loangoal"] in loan_assign
    duration_condition = avg_duration >= avg_duration_level_assign

    if duration_condition and loan_condition:
        assign = 1

    # Now, either do the original treatment policy, or by a chance delta, choose randomly
    if np.random.rand() >= delta:
        if random_object_random_policy is None:
            assign = np.random.choice([0, 1])
        else:
            assign = random_object_random_policy.choice([0, 1])

    return assign

# Diese Funktion berechnet den Behandlungseffekt auf die Dauer des nächsten Events.
# loangoal_mapping nicht benutzt
def calculate_treatment_effect_on_duration(row, monthlycost_avg=100, loangoal_mapping={"A":0, "B":1, "C":2, "D":3}):
    if row["activity"] not in treatment_point_activities or row["event_nr"] < row["length"]:
        return row["treatment_effect"]
    monthly_cost = float(row["monthlycost"])
    loangoal = row["loangoal"]
    avg_duration = row["avg_duration"]
    non_treat_duration = duration_data_stats["wait_incomplete_files"][0]

    if treatment_function == 'linear':
        if monthly_cost > monthlycost_avg:
            if "Car" in loangoal or "loan" in loangoal:
                intercept = non_treat_duration * (1 / 3)
            else:
                intercept = non_treat_duration * (1 / 6)
        else:
            if "Car" in loangoal or "loan" in loangoal:
                intercept = non_treat_duration * (1 / 12)
            else:
                intercept = 0

        slope = 1

        treatment_effect = intercept + slope * avg_duration
    elif treatment_function == 'discrete':
        # Base level from loangoal & monthly cost
        if monthly_cost > monthlycost_avg:
            if "Car" in loangoal or "loan" in loangoal:
                base = 3000
            else:
                base = 2000
        else:
            if "Car" in loangoal or "loan" in loangoal:
                base = 1000
            else:
                base = 0

        # Duration tiers (example thresholds)
        if avg_duration < 3:
            duration_bonus = 500
        elif avg_duration < 6:
            duration_bonus = 2000
        elif avg_duration < 9:
            duration_bonus = 5000
        else:
            duration_bonus = 8000

        treatment_effect = base + duration_bonus

        # Clamp between 50 and 10 000
    elif treatment_function == 'exponential':
        constant_term = 0
        if monthly_cost > monthlycost_avg:
            if 'Car' in loangoal or 'loan' in loangoal:
                constant_term = 0.15
            else:
                constant_term = 0.1
        else:
            if 'Car' in loangoal or 'loan' in loangoal:
                constant_term = 0.05

        avg_duration_divided = avg_duration / divider_term_effect

        # b + x^a
        treatment_effect_divided = constant_term + avg_duration_divided ** (exponent_term_effect)
        treatment_effect = treatment_effect_divided * divider_term_effect

    treatment_effect = max(0, min(non_treat_duration, treatment_effect))
    
    return treatment_effect

def calculate_duration_t(row, rng):
    # Damit nur die neu angehängte call- oder wait-Aktivität eine neue Dauer erhält.
    # Ohne diese Prüfung würde calculate_duration_t() in jeder Stufe auch frühere Events erneut berechnen. Wegen des Zufallsanteils könnten sich bereits festgelegte Dauern nachträglich verändern.
    if row["activity"] not in treatment_activities or row["event_nr"] < row["length"]:
        return row["duration"]
    
    # Vorheriges Event: validate_application
    # treatment = 1
    # treatment_effect = 500
    #
    # Aktuelles Event: call_incomplete_files
    # prev_treatment = 1
    # prev_treatment_effect = 500
    #
    # duration = normale Wartezeit - (1 × 500)
    prev_treatment = row["prev_treatment"]
    prev_treatment_effect = row["prev_treatment_effect"]
    # In configs.py ist für wait_incomplete_files eine Basisdauer von 10.000 Minuten definiert. rng.uniform(-10, 10) fügt eine kleine gleichverteilte Zufallsschwankung hinzu.
    non_treat_duration = duration_data_stats["wait_incomplete_files"][0] + rng.uniform(-10, 10)

    duration = non_treat_duration - (prev_treatment * prev_treatment_effect)
    duration = max(50, duration)  # duration cannot be less than 50
    return duration

def calculate_outcome(row):
    if row["activity"] != "end":
        return row["outcome"]  # Only calculate outcome at the end activity
    
    total_elapsed_time = row["elapsed_time_incomplete_files"]
    total_treatments = row["total_nr_treatments"]
    outcome = (total_elapsed_time * cost_time_baseline) + (total_treatments * cost_treatment_baseline)
    return outcome


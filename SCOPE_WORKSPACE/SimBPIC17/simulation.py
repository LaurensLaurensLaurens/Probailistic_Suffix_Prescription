path = "c:\\Users\\u0166838\\OneDrive - KU Leuven\\Documents\\Doc\\Code\\SCOPE"

import sys
sys.path.append(path)

import numpy as np
import pandas as pd
from SimBPIC17.configs import standard_prefix, shortened_prefix, standard_row, shortened_prefix_chance
from copy import deepcopy
from SimBPIC17.tools import get_duration, get_duration_samples, get_params_loangoal_monthlycost
from SimBPIC17.dag_functions import calculate_treatment_assignment, calculate_treatment_effect_on_duration, calculate_duration_t, calculate_outcome


def run_simulation(random_seed=42, n_cases=1000, n_stages=3, delta=0.99, policy=None, random_object_random_policy=None):
    np.random.seed(random_seed)
    rng_duration_t = np.random.default_rng(random_seed)

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

    duration_samples = get_duration_samples(n_cases=n_cases, random_seed=random_seed)

    # load the unique monthly costs and loan goals from the original BPIC17 dataset
    unique_loangoal_monthlycost, monthlycost_avg, loangoal_mapping = get_params_loangoal_monthlycost()

    # Make a df with n_cases times standard_prefix and shortened_prefix (5% chance of shortened prefix)
    rows = []
    for case_nr in range(n_cases):
        if np.random.rand() < shortened_prefix_chance:
            prefix = shortened_prefix
        else:
            prefix = standard_prefix

        # Damit jeder simulierte Fall ein realistisches Paar aus:
        # Kreditzweck (loangoal)
        # monatlichen Kosten (monthlycost)
        # aus dem ursprünglichen BPIC17-Datensatz erhäl
        loangoal_monthlycost_sample = unique_loangoal_monthlycost.sample(n=1, random_state=random_seed + case_nr).iloc[0]
        while pd.isnull(loangoal_monthlycost_sample["monthlycost"]):
            loangoal_monthlycost_sample = unique_loangoal_monthlycost.dropna().sample(n=1, random_state=random_seed + case_nr).iloc[0]

        # Explode each list into separate rows
        # Dieser Block wandelt einen Prozess-Prefix aus mehreren Listen in einzelne Event-Zeilen um.
        n_rows = len(prefix['activity'])
        for i in range(n_rows):
            row = {key: prefix[key][i] for key in prefix}
            row['case_nr'] = case_nr
            row["event_nr"] = i+1
            # Assign loangoal and monthlycost. If activity is complete_application, do not assign mnonthlycost yet. Also make sure that monthlycost does not missing values.
            row["loangoal"] = loangoal_monthlycost_sample["loangoal"]
            row["monthlycost"] = loangoal_monthlycost_sample["monthlycost"] if row["activity"] != "complete_application" else -100
            
            rows.append(row)

    # Die DataFrame df enthält nun alle Event-Zeilen für alle simulierten Fälle.
    df = pd.DataFrame(rows)
    df_expanded = df.sort_values(by=["case_nr"]).reset_index(drop=True)
    # Zeile 70 ergänzt für jede Zeile eine simulierte duration.
    df_expanded["duration"] = df_expanded.apply(lambda row: get_duration(row=row, duration_samples=duration_samples), axis=1)
    df_expanded = df_expanded.sort_values(by=["case_nr", "event_nr"]).reset_index(drop=True)

    # Loop through decision points:
    for stage in range(n_stages):
        current_treatment_to_do = None
        # z.b. Anrufen, warten, anrufen policy = [1, 0, 1]
        # Hier haben wir 6 Stages: policy = [0, 0, 1, 1, 0, 1]
        if policy is not None:
            current_treatment_to_do = policy[stage]

        # for every case, set the current length of the prefix
        # Die Zeile berechnet für jeden Fall seine aktuell höchste Eventnummer
        # Dieser Maximalwert wird in jede Zeile desselben Falls geschrieben.
        df_expanded["length"] = df_expanded.groupby("case_nr")["event_nr"].transform("max")
        # calculate average cummulative activity duration per case
        # Die Zeile berechnet für jedes Event die durchschnittliche bisherige Aktivitätsdauer innerhalb desselben Falls.
        # Beeinflusst später den Behandlungseffekt
        df_expanded["avg_duration"] = (df_expanded.groupby("case_nr")["duration"].expanding().mean().reset_index(level=0, drop=True))
        # Get treatment decision for every prefix. 
        df_expanded["treatment"] = df_expanded.apply(lambda row: calculate_treatment_assignment(row=row, delta=delta, treatment_to_do=current_treatment_to_do, random_object_random_policy=random_object_random_policy), axis=1)
        # Calculate treatment effect on duration
        # Der Treatment-Effkt wird ausgerechnet
        df_expanded["treatment_effect"] = df_expanded.apply(lambda row: calculate_treatment_effect_on_duration(row=row, monthlycost_avg=monthlycost_avg, loangoal_mapping=loangoal_mapping), axis=1)

        # Add a row to every prefix based on treatment in previous row (if treatment = 1, add call_incomplete_files, else add wait_incomplete_files)
        # z.b. df_expanded
        # case_nr  event_nr  activity               duration  length  avg_duration  treatment
        # 0        1         complete_application    3900      3       3900          0
        # 0        2         call_after_offers       4100      3       4000          0
        # 0        3         validate_application    3800      3       3933          1
        # 1        1         complete_application    4200      3       4200          0
        # 1        2         call_after_offers       4000      3       4100          0
        # 1        3         validate_application    4300      3       4167          0
        rows_to_add = []
        for case_nr, group in df_expanded.groupby("case_nr"):
            last_row = group.iloc[-1]
            treatment = last_row["treatment"]
            new_row = deepcopy(standard_row)
            new_row["event_nr"] = last_row["event_nr"] + 1
            new_row["case_nr"] = last_row["case_nr"]
            
            if treatment == 1:
                new_row["activity"] = "call_incomplete_files"
                new_row["monthlycost"] = last_row["monthlycost"]
                new_row["loangoal"] = last_row["loangoal"]
            else:
                new_row["activity"] = "wait_incomplete_files"
                new_row["duration"] = duration_samples["wait_incomplete_files"][case_nr]
                new_row["monthlycost"] = last_row["monthlycost"]
                new_row["loangoal"] = last_row["loangoal"]

            rows_to_add.append(new_row)
        
            if stage == n_stages - 1:
                # Add validate application 
                validate_new_row = deepcopy(standard_row)
                validate_new_row["event_nr"] = last_row["event_nr"] + 2
                validate_new_row["case_nr"] = last_row["case_nr"]
                validate_new_row["activity"] = "validate_application"
                validate_new_row["duration"] = duration_samples["validate_application"][case_nr]
                validate_new_row["monthlycost"] = last_row["monthlycost"]
                validate_new_row["loangoal"] = last_row["loangoal"]
                rows_to_add.append(validate_new_row)

                # we are at the last stage, add the end event
                end_new_row = deepcopy(standard_row)
                end_new_row["event_nr"] = last_row["event_nr"] + 3
                end_new_row["case_nr"] = last_row["case_nr"]
                end_new_row["activity"] = "end"
                end_new_row["duration"] = duration_samples["end"][case_nr]
                end_new_row["monthlycost"] = last_row["monthlycost"]
                end_new_row["loangoal"] = last_row["loangoal"]
                rows_to_add.append(end_new_row)

        df_additional = pd.DataFrame(rows_to_add)
        df_expanded = pd.concat([df_expanded, df_additional], ignore_index=True)
        df_expanded = df_expanded.sort_values(by=["case_nr", "event_nr"]).reset_index(drop=True)

        # Beide Zeilen übertragen Informationen des vorherigen Events auf das aktuelle Event desselben Falls.
        # Vorher trifft validate_application die Entscheidung:
        # Event 3: validate_application
        # treatment = 1
        # treatment_effect = 500
        # Danach wird der Anruf als neue Zeile angehängt:
        # Event 4: call_incomplete_files
        # shift(1) kopiert die Entscheidung von Event 3 zu Event 4:
        # Event 4:
        # prev_treatment = 1
        # prev_treatment_effect = 500
        # Damit weiß Event 4: „Vor mir wurde Anrufen gewählt, mit einem Effekt von 500.“ Daraus wird anschließend seine Dauer berechnet. shift(1) bedeutet hier einfach: Nimm den Wert aus der vorherigen Zeile.
        df_expanded["prev_treatment"] = (df_expanded.groupby("case_nr")["treatment"].shift(1)).fillna(-100)
        df_expanded["prev_treatment_effect"] = (df_expanded.groupby("case_nr")["treatment_effect"].shift(1)).fillna(-100)

        # Calculate the duration_t based on treatment effect and treatment
        # Random duration t. 
        df_expanded["duration"] = df_expanded.apply(lambda row: calculate_duration_t(row=row, rng=rng_duration_t), axis=1)

    # Berechnet elapsed time
    df_expanded["elapsed_time"] = df_expanded.groupby("case_nr")["duration"].cumsum() - df_expanded["duration"]

    # Berechnet elapsed time incomplete files für spätere Berechnung des Outcomes.
    # Aktivität               Dauer   elapsed_time_incomplete_files
    # validate_application    50      0
    # wait_incomplete_files   100     0
    # call_incomplete_files   80      100
    # end                     0       180
    df_expanded["elapsed_time_incomplete_files"] = (
        df_expanded.assign(
            filtered_duration=lambda d: d["duration"].where(
                d["activity"].isin(["wait_incomplete_files", "call_incomplete_files"]), 0
            )
        )
        .groupby("case_nr")["filtered_duration"]
        .cumsum()
        - df_expanded["duration"].where(
            df_expanded["activity"].isin(["wait_incomplete_files", "call_incomplete_files"]), 0
        )
    )
    # NOTE: "elapsed_time_incomplete_files" is eventually not used in models
    # length: höchste Eventnummer und damit endgültige Falllänge. Derselbe Wert steht in allen Zeilen des Falls.
    df_expanded["length"] = df_expanded.groupby("case_nr")["event_nr"].transform("max")

    #total_nr_treatments: zählt pro Fall, wie oft treatment == 1 gewählt wurde, also wie oft angerufen wurde. Auch dieser Gesamtwert wird in jede Fallzeile geschrieben.
    df_expanded["total_nr_treatments"] = (
        df_expanded.groupby("case_nr")["treatment"]
        .transform(lambda x: (x == 1).sum())
    )
    # avg_duration: berechnet erneut den fortlaufenden Durchschnitt der bisherigen Event-Dauern
    df_expanded["avg_duration"] = (df_expanded.groupby("case_nr")["duration"].expanding().mean().reset_index(level=0, drop=True))
    df_expanded = df_expanded.sort_values(by=["case_nr", "event_nr"]).reset_index(drop=True)

    # Berwchnet kosten auf Zeit und die Kosten des Treatments
    df_expanded["outcome"] = df_expanded.apply(calculate_outcome, axis=1)
    # now make sure that for each case, the outcome is set on all rows (in the function, this only happened to rows where 'activity' == 'end')
    df_expanded['outcome'] = (df_expanded.groupby('case_nr')['outcome'].transform(lambda x: x.ffill().bfill()))
    df_expanded['outcome'] = -df_expanded['outcome']

    total_outcome = df_expanded[df_expanded["activity"] == "end"].groupby("case_nr")["outcome"].sum().sum()
    print(f"Total outcome: {total_outcome}", "\n")

    return df_expanded
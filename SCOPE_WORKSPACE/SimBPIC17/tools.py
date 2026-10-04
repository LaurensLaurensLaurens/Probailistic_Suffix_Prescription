path = "c:\\Users\\u0166838\\OneDrive - KU Leuven\\Documents\\Doc\\Code\\SCOPE"

import os
import sys
sys.path.append(path)

import numpy as np
from SimBPIC17.configs import duration_data_stats
import pandas as pd

# Map activity to duration samples
def get_duration(row, duration_samples):
    activity = row["activity"]
    case_nr = row["case_nr"]
    if activity == "complete_application":
        return duration_samples["complete_application"][case_nr]
    elif activity == "call_after_offers":
        return duration_samples["call_after_offers"][case_nr]
    elif activity == "shortened_completion":
        return duration_samples["shortened_completion"][case_nr]
    elif activity == "validate_application":
        return duration_samples["validate_application"][case_nr]
    else:
        return 0
    
def get_duration_samples_activity(mean=1, std=1, distr='lognormal', n_cases=10, random_seed=42):
    np.random.seed(random_seed)
    mu = 0
    sigma = 0
    duration_samples_activity = [0]*n_cases
    if distr == "lognormal":
        sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
        mu = np.log(mean) - (sigma**2) / 2

        duration_samples_activity = np.random.lognormal(mean=mu, sigma=sigma, size=n_cases)

    elif distr == "constant":
        mu = mean
        std = 0
    
        duration_samples_activity = [mu]*n_cases

    return duration_samples_activity

def get_duration_samples(n_cases, random_seed=42):
    duration_samples = {activity: get_duration_samples_activity(mean=mean, std=std, distr=distr, n_cases=n_cases, random_seed=random_seed) for activity, (mean, std, distr) in duration_data_stats.items()}
    return duration_samples

# Diese Zeile definiert eine Funktion, die die wichtigsten Parameter für die Datengenerierung aus den BPIC17-Daten extrahiert.
# Nur die drei häufigsten Kreditziele bleiben separat.
# Seltene Ziele werden zu Other zusammengefasst.
# Das reduziert viele seltene Kategorien, stabilisiert das Training und macht Interventionseffekte leichter kontrollierbar.
# Der durchschnittliche monthlycost dient als Schwellenwert für die simulierte Wirkung eines Anrufs.
def get_params_loangoal_monthlycost():
    # load the unique monthly costs and loan goals from the original BPIC17 dataset
    unique_loangoal_monthlycost = pd.read_csv(
        os.path.join(
            os.getcwd(),
            "data", "bpic17",
            "bpic2017_unique_loangoal_monthlycost.csv"
        )
    )
    # get the top 3 most frequent loangoals, and only keep those, replace with 'Other' otherwise
    top_loangoals = unique_loangoal_monthlycost['case:loangoal'].value_counts().nlargest(3).index.tolist()
    unique_loangoal_monthlycost['case:loangoal'] = unique_loangoal_monthlycost['case:loangoal'].apply(lambda x: x if x in top_loangoals else 'Other')

    # make a mapping of loangoal, where each loangoal corresponds to 0, 0.25, 0.5, 0.75
    loangoal_mapping = {loangoal: (idx / 8) for idx, loangoal in enumerate(top_loangoals + ['Other'])}

    # get the min, and max of monthlycost
    monthlycost_avg = unique_loangoal_monthlycost['case:monthlycost'].mean()

    # rename monthly cost and loangoal columns to not have event: and case:
    unique_loangoal_monthlycost = unique_loangoal_monthlycost.rename(columns={"case:monthlycost": "monthlycost", "case:loangoal": "loangoal"})

    return unique_loangoal_monthlycost, monthlycost_avg, loangoal_mapping
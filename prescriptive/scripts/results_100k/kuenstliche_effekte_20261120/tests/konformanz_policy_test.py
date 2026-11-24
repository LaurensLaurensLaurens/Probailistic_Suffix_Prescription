"""Funktionstest KonformanzPolicy (2026-11-20): Alignment-Hilfe gegen bekannte Spuren, dann ein kleiner Simulatorlauf (Effekte v3) mit
der Policy (PSP svk, echter Log – nur Mechanik und Laufzeit, das Modell kennt die künstlichen Abweichungen nicht)."""
import os, sys, time, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
import wahrheit as w  # Pfade, Katalog, simulator()
import numpy as np
from prescriptive.policies import konformanz_policy as kp

konform = ("A_Create Application", "A_Submitted", "W_Handle leads", "W_Complete application", "A_Concept", "A_Accepted", "O_Create Offer",
           "O_Created", "O_Sent (mail and online)", "W_Call after offers", "A_Complete", "W_Validate application", "A_Validating",
           "O_Returned", "O_Accepted", "A_Pending")
print("konforme Spur:", kp.abweichungen(konform))
print("Prüfung ausgelassen:", kp.abweichungen(tuple(a for a in konform if a != "W_Validate application")))
print("A_Complete doppelt:", kp.abweichungen(konform[:11] + ("A_Complete",) + konform[11:]))
print("Token-Projektion:", kp.projektion_tokens(["W_Validate application|schedule", "W_Validate application|start", "A_Validating|complete", "EOS"]))

import kuenstliche_effekte as ke
ke.aktivieren(ke.laden("kuenstliche_effekte/effekte_v3.yaml"))
modell = str(w.ROOT/"prescriptive/models/psp_model_durchlaufzeit_t_learner_svk_crn.pkl")
policy = kp.KonformanzPolicy(modell, samples=50, first_only=("angebot_nachlegen",))
start = time.time()
events, choices = w.simulator().run(40, policy=policy, seed=3)
dauer = time.time() - start
import pandas as pd
d = pd.DataFrame(policy.entscheidungen)
print(f"{len(d)} Entscheidungen in {dauer:.0f} s, Zwischenspeicher {kp.abweichungen.cache_info()}")
print(d.groupby("point").wahl.value_counts(dropna=False).to_string())
print(d.drop(columns=["case", "time"]).head(8).round(2).to_string())

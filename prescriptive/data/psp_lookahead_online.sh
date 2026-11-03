#!/bin/bash
# Wartet, bis das originalgetreue PSP fertig trainiert und bewertet ist (psp_orig_run.log), dann die Online-Auswertung am
# Wahlpunkt W_Call incomplete files: Bank, Zufall, PSP (Mittelwert) und PSP mit Vorausschau (policies/psp_lookahead_policy.py)
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
until grep -q "######## Fertig\|Traceback" $ROOT/prescriptive/data/psp_orig_run.log; do sleep 60; done
grep -q "Traceback" $ROOT/prescriptive/data/psp_orig_run.log && { echo "PSP-Training mit Fehler beendet, keine Online-Auswertung"; exit 1; }
cd $ROOT/process-simulation-engine
.venv/bin/python -u ../prescriptive/scripts/main.py --outcome bpic17=durchlaufzeit --steps online \
    --online-policies bank random psp psp_lookahead --online-points "bpic17=Mitarbeiter: W_Call incomplete files" --online-workers 12 \
    2>&1 | grep --line-buffered -v -E 'FutureWarning|applied = df'

#!/bin/bash
# Rollout mit Causal Forest an späteren Wahlpunkten, Conformance (2026-11-23, Nutzer: „Setze eine Version auf, die mittels Causal Forest
# die späteren Entscheidungen in den Suffixen korrigiert“). Forest models/cate_model_cf_konformanz_simv3.pkl (rollout_cf_20261123/
# train_cf_konformanz.py), Rollout policies/ev_policy/rollout.py (geprüft: test_rollout.py vergleich = bitgenau ohne Rollout, rollout = OK).
# 1. Diagnose je Teilgruppe mit Rollout (k3x4ol, keyed_noise, 100 Ziehungen; vergleichbar mit konformanz_unsicherheit_crn.md)
# 2. Online-Arm konformanz_ol_ro (300 Ziehungen, 12 Seeds), erst nach der Kontrolle konformanz_ol300 (gleiches Modell, gleiche Ziehungen,
#    nur ohne Rollout → sauberer Paarvergleich); 8 Prozesse (Speicher: nachts laden die Netze der Simulator-Basis).
# Aufruf (tmux): bash bpic17_psp_rollout_cf.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
K=$P/scripts/results_100k/kuenstliche_effekte_20261120
RO=$P/scripts/results_100k/rollout_cf_20261123
SB=$P/scripts/results_100k/sim_basis_20261123
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $RO/fortschritt.log; }
note "Diagnose mit Rollout beginnt"
KU_MODELL=$P/models/psp_model_durchlaufzeit_option_learner_k3x4_schlank_crn.pkl KU_ROLLOUT=$P/models/cate_model_cf_konformanz_simv3.pkl \
KU_NAME=konformanz_unsicherheit_ro .venv/bin/python -u $K/diagnose/konformanz_unsicherheit.py 6 60 > $RO/diagnose.log 2>&1 \
    && note "Diagnose fertig: $(grep -E '^Gesamt' $K/diagnose/konformanz_unsicherheit_ro.md)" || note "Diagnose FEHLER"
note "wartet auf das Ende der Kontrolle konformanz_ol300"
until grep -q "Kontrolle konformanz_ol300 beendet" $SB/fortschritt.log; do sleep 120; done
note "Online konformanz_ol_ro beginnt (12 Seeds, 300 Ziehungen, 8 Prozesse)"
.venv/bin/python -u $K/online_konformanz.py konformanz_ol_ro 0-11 8 > $K/online_konformanz_ol_ro.log 2>&1
note "Online beendet (Code $?): $(grep -E '^FERTIG' $K/online_konformanz_ol_ro.log | head -1)"
note "Alles fertig"

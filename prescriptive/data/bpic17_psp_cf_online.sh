#!/bin/bash
# Online-Test PSP + Causal Forest bei Unsicherheit (2026-11-23, Nutzer: „mach es so, dass wenn dieser Fall eintritt, dann soll ein Causal
# Forest entscheiden: nicht unterscheidbar, wenn |Δ| < 1,96 · SE“). Arm olsvcf_fest300 = olsv (8 Netze, keyed_noise, 300 Ziehungen, first_only)
# + policies/ev_policy/psp_cf_policy.py mit dem Causal Forest models/cate_model_cf_durchlaufzeit_real_alle.pkl. Aufbau F, 12 Seeds; danach
# Auswertung (f_zwoelf_auswerten, abweichungen_regel), Register (online_sammeln, register_erstellen).
# Geprüft (psp_cf_20261123/test_gleichheit.py): mit z = 0 dieselben 33 Wahlen wie PSPPolicy/olsv.
# Aufruf (tmux): bash bpic17_psp_cf_online.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
OUT=$P/scripts/results_100k/psp_cf_20261123
V=$P/scripts/results_100k/modellvergleich_20261116
R=$P/scripts/results_100k/psp_register
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $OUT/fortschritt.log; }
note "Online-Test 12 Seeds beginnt (olsvcf_fest300)"
.venv/bin/python -u $V/f_zwoelf_seeds.py --arms olsvcf_fest300 --seeds 0-11 --workers 12 > $OUT/online_olsvcf.log 2>&1
note "Online-Test beendet (Code $?, $(grep -c COMPLETE $OUT/online_olsvcf.log) Seeds)"
.venv/bin/python -u $V/f_zwoelf_auswerten.py > $OUT/online_auswertung.log 2>&1
.venv/bin/python -u $V/abweichungen_regel.py > $OUT/abweichungen.log 2>&1
(cd $R && $ROOT/process-simulation-engine/.venv/bin/python online_sammeln.py > /dev/null 2>&1 && $ROOT/process-simulation-engine/.venv/bin/python register_erstellen.py > $OUT/register.log 2>&1)
note "$(grep -E '^ *olsvcf_fest300 ' $V/f_zwoelf_auswertung.md | head -1)"
note "$(grep -E 'als olsvcf_fest300' $V/f_zwoelf_auswertung.md | tr -s ' ' | tr '\n' ';')"
note "Alles fertig"

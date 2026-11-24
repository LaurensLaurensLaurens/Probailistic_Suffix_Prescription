#!/bin/bash
# Online-Test T-Learner mit Zufallszahlen je Ereignis (2026-11-19, Nutzer: „ja mach das“). Wie tl_fest300 (Aufbau F, 12 Seeds × 200 Fälle,
# 300 Ziehungen fest, first_only angebot_nachlegen), einziger Unterschied params keyed_noise in der Hüllen-Kopie
# models/psp_model_durchlaufzeit_t_learner_zztf1_crn.pkl (Referenz- und Aktionsmodell). Diagnose vorher: crn_schluessel.py
# (ρ 0,15 → 0,7, Rauschen je Entscheidung ≈ 1,0 → 0,6 d bei 300 Ziehungen). Danach Auswertung und Abweichungen von der Regel.
# Aufruf (tmux): bash bpic17_psp_tl_crn_online.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
V=$ROOT/prescriptive/scripts/results_100k/modellvergleich_20261116
OUT=$V/crn_schluessel
export PSP_CATALOG=$ROOT/prescriptive/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') tlcrn: $1" >> $OUT/fortschritt.log; }
note "Online-Test 12 Seeds beginnt (tlcrn_fest300)"
.venv/bin/python -u $V/f_zwoelf_seeds.py --arms tlcrn_fest300 --seeds 0-11 --workers 6 > $OUT/online_tlcrn.log 2>&1
note "Online-Test beendet (Code $?, $(grep -c COMPLETE $OUT/online_tlcrn.log) Seeds)"
.venv/bin/python -u $V/f_zwoelf_auswerten.py > $OUT/online_auswertung.log 2>&1
.venv/bin/python -u $V/abweichungen_regel.py > $OUT/abweichungen.log 2>&1
note "$(grep -E '^ *tlcrn_fest300 ' $V/f_zwoelf_auswertung.md | head -1)"
note "$(grep -E 'tl_fest300 schneller als tlcrn_fest300|regel schneller als tlcrn_fest300|c_fest300 schneller als tlcrn_fest300' $V/f_zwoelf_auswertung.md | tr -s ' ' | tr '\n' ';')"
note "Alles fertig"

#!/bin/bash
# Online-Test T-Learner mit Suffix-Verlust (2026-11-19, Nutzer: „ja“). Wie tlcrn_fest300 (Aufbau F, 12 Seeds × 200 Fälle, 300 Ziehungen
# fest, first_only angebot_nachlegen, keyed_noise), einziger Unterschied: beide Arme mit sequence_loss nachtrainiert
# (data/bpic17_psp_suffix_verlust.sh), Modell models/psp_model_durchlaufzeit_t_learner_sv_crn.pkl. 12 Prozesse parallel (Rechner frei).
# Danach Auswertung und Abweichungen von der Regel. Aufruf (tmux): bash bpic17_psp_tl_sv_online.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
V=$ROOT/prescriptive/scripts/results_100k/modellvergleich_20261116
OUT=$ROOT/prescriptive/scripts/results_100k/suffix_verlust_20261119
export PSP_CATALOG=$ROOT/prescriptive/data/catalog_neue_punkte_zuteilung_random.yaml
cd $ROOT/process-simulation-engine
note() { echo "$(date '+%Y-%m-%d %H:%M') tlsv: $1" >> $OUT/fortschritt.log; }
note "Online-Test 12 Seeds beginnt (tlsv_fest300)"
.venv/bin/python -u $V/f_zwoelf_seeds.py --arms tlsv_fest300 --seeds 0-11 --workers 12 > $OUT/online_tlsv.log 2>&1
note "Online-Test beendet (Code $?, $(grep -c COMPLETE $OUT/online_tlsv.log) Seeds)"
.venv/bin/python -u $V/f_zwoelf_auswerten.py > $OUT/online_auswertung.log 2>&1
.venv/bin/python -u $V/abweichungen_regel.py > $OUT/abweichungen.log 2>&1
note "$(grep -E '^ *tlsv_fest300 ' $V/f_zwoelf_auswertung.md | head -1)"
note "$(grep -E 'schneller als tlsv_fest300' $V/f_zwoelf_auswertung.md | tr -s ' ' | tr '\n' ';')"
note "Alles fertig"

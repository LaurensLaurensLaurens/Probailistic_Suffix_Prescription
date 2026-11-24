#!/bin/bash
# Causal Forest mit Rückwärtsinduktion für die Konformität (2026-11-24, Nutzer: „ja mach beides“).
#   1. results_100k/rueckwaerts_cf_20261124/rueckwaerts.py: 4 Stufen (nachlegen → nach V. → konzept → leads), je 8 × 6.000 Fälle im
#      Simulator mit Effekten v3, Stufenpunkt zufällig, spätere Punkte ihr Forest; Forests models/cate_model_cf_konformanz_rw_<kurz>.pkl
#   2. Online-Arm cf_rueckwaerts (kuenstliche_effekte_20261120/online_konformanz.py, Aufbau F, 12 Seeds)
#   3. Wahlen je Teilgruppe (rollout_cf_20261123/forest_wahlen.py) für cf_rueckwaerts, cf_konformanz, orakel, regel, konformanz_ol300
# Nur CPU; oom_score_adj 1000 (bei Speichermangel zuerst diese Läufe). Aufruf (tmux): bash bpic17_cf_rueckwaerts.sh
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
RW=$P/scripts/results_100k/rueckwaerts_cf_20261124
K=$P/scripts/results_100k/kuenstliche_effekte_20261120
cd $ROOT/process-simulation-engine
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
note() { echo "$(date '+%Y-%m-%d %H:%M') $1" >> $RW/fortschritt.log; }
note "Pipeline beginnt (frei $(free -g | awk '/Mem:/ {print $7}') GB)"
.venv/bin/python -u $RW/rueckwaerts.py 8 6000 8 > $RW/rueckwaerts.log 2>&1 || { note "FEHLER: rueckwaerts.py"; exit 1; }
.venv/bin/python -u $K/online_konformanz.py cf_rueckwaerts 0-11 6 > $K/online_konformanz_cf_rueckwaerts.log 2>&1 \
    && note "Online cf_rueckwaerts fertig" || { note "FEHLER: Online cf_rueckwaerts"; exit 1; }
.venv/bin/python $P/scripts/results_100k/rollout_cf_20261123/forest_wahlen.py cf_rueckwaerts,cf_konformanz,orakel,regel,konformanz_ol300 \
    > $RW/forest_wahlen.log 2>&1 && note "Wahlen je Teilgruppe fertig" || note "FEHLER: forest_wahlen"
note "Alles fertig"

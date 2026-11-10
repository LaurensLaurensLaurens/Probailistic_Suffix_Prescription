#!/bin/bash
# Warteschlange 2026-11-10 (2): nach dem Test mit erster Entscheidung das Training (d) starten (Speicher: daneben läuft (c)),
# 40 Minuten später (nach der Vorverarbeitung von (d)) den zurückgestellten Lauf 3 mit 6 Arbeitsprozessen.
P=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive
L=$P/data/PSP_neue_punkte_online_real_np_s40_ohne_zukunft_abstaende_erste_entscheidung.log
until [ -f $L ] && grep -q -E "######## Alles fertig|######## Abbruch" $L && ! tmux has-session -t psp_warteschlange 2>/dev/null; do sleep 30; done
tmux new-session -d -s psp_stufe2d "bash $P/data/bpic17_psp_real_stufe2d.sh"
sleep 2400
WORKERS=6 bash $P/data/bpic17_psp_stufe2_online_lauf3.sh

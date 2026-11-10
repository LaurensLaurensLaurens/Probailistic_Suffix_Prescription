#!/bin/bash
# Warteschlange 2026-11-10: Lauf 3 von bpic17_psp_stufe2_online.sh ((a) über Suffixe) zurückstellen, sobald er startet,
# stattdessen den Test mit erster Entscheidung (bpic17_psp_erste_entscheidung.sh). Lauf 3 kommt später einzeln.
P=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive
L=$P/data/PSP_stufe2_online.log
until grep -q "######## Online neue_punkte_online_stufe2_ohne_kopf_suffixe100" $L; do sleep 3; done
MAIN=$(pgrep -f "scripts/main.py.*stufe2_ohne_kopf_suffixe100" | head -1)
tmux kill-session -t psp_online2
[ -n "$MAIN" ] && { pkill -TERM -P "$MAIN"; kill "$MAIN" 2>/dev/null; }
sleep 20
echo "######## Lauf 3 zurückgestellt ($(date +%H:%M:%S)), stattdessen erste Entscheidung" >> $L
bash $P/data/bpic17_psp_erste_entscheidung.sh

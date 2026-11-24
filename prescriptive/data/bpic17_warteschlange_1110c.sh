#!/bin/bash
# Ersatz für bpic17_warteschlange_1110b.sh (2026-11-10 16:40): die alte Schleife wartete auf sich selbst (tmux has-session -t
# psp_warteschlange passt als Präfix auch auf psp_warteschlange2). Training (d) sofort, Lauf 3 mit 6 Arbeitsprozessen 40 Minuten danach.
P=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive
tmux new-session -d -s psp_stufe2d "bash $P/data/bpic17_psp_real_stufe2d.sh"
sleep 2400
WORKERS=6 bash $P/data/bpic17_psp_stufe2_online_lauf3.sh

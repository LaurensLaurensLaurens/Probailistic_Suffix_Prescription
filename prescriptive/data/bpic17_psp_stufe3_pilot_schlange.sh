#!/bin/bash
# Warteschlange für die Stufe-3-Pilotläufe (2026-11-11): die GPU teilen wir mit anderen Nutzern; ein Lauf startet erst, wenn
# genug Speicher frei ist (sonst "CUDA out of memory" beim ersten Batch). Bei viel freiem Speicher laufen mehrere nebeneinander.
# Zweiter Durchgang für Läufe, die ohne Modell enden. Protokoll: data/PSP_pilot_schlange.log
# Aufruf: tmux new-session -d -s pilot_q "bash prescriptive/data/bpic17_psp_stufe3_pilot_schlange.sh"
P=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription/prescriptive
S=$P/data/bpic17_psp_stufe3_pilot.sh
Q=$P/data/PSP_pilot_schlange.log
JOBS=(
    "pilot_token_tf05|3000|7000|--gap-tokens --teacher-forcing 0.5 --option-in-decoder"
    "pilot_token_tf0|3000|7000|--gap-tokens --option-in-decoder"
    "pilot_token_tf1_c6000|6000|9000|--gap-tokens --teacher-forcing 1.0 --option-in-decoder"
    "pilot_token_tf08|3000|7000|--gap-tokens --teacher-forcing 0.8 --option-in-decoder"
)
free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
on_gpu() {
    local pid
    pid=$(pgrep -f "train_psp_real.py --tag $1 " | head -1)
    [ -n "$pid" ] && nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -qx "$pid"
}
launch_all() {
    for job in "${JOBS[@]}"; do
        IFS='|' read -r tag cases need extra <<< "$job"
        [ -e "$P/models/psp_model_durchlaufzeit_$tag.pkl" ] && continue
        while [ "$(free)" -lt "$need" ]; do sleep 60; done
        # Rest eines abgebrochenen Versuchs (nur model.pkl ohne Netz): das Training bricht sonst mit "gibt es schon" ab
        case "$tag" in pilot_*) [ -d "$P/models/psp_training_$tag" ] && rm -r "$P/models/psp_training_$tag" ;; esac
        echo "$(date +%H:%M) Start $tag (frei $(free) MiB)" >> "$Q"
        TAG=$tag CASES=$cases EXTRA="$extra" bash "$S" > /dev/null 2>&1 &
        pid=$!
        while kill -0 $pid 2>/dev/null && ! on_gpu "$tag"; do sleep 20; done
        sleep 90  # der Lauf belegt seinen Speicher, bevor der nächste misst
    done
    wait
}
launch_all
echo "$(date +%H:%M) erster Durchgang fertig" >> "$Q"
launch_all
echo "$(date +%H:%M) Schlange fertig: $(ls $P/models | grep -c 'pilot_token_tf0\|pilot_token_tf05\|pilot_token_tf08\|pilot_token_tf1_c6000') von 4 Modellen" >> "$Q"

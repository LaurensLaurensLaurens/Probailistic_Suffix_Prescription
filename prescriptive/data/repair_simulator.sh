#!/bin/bash
# Werkstatt (Datensatz repair, eventlog/repair/repair_workshop.xes.gz): den Simulator (Next-Event-Modell, Engine mit Lifecycles) auf dem Log trainieren
# und gegen das Log prüfen (10.000 simulierte Fälle). Es bleiben: process-simulation-engine/models/next_event/repair, scripts/results_repair_pipeline,
# data/repair_simulator.log.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
LOG=$P/data/repair_simulator.log
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory'
cd $ROOT/process-simulation-engine
echo "######## Simulator und Check ($(date +%H:%M:%S))" | tee -a "$LOG"
.venv/bin/python -u ../prescriptive/scripts/main.py --datasets repair --steps check --n-cases 10000 --sim-threads 16 --out $P/scripts/results_repair_pipeline 2>&1 | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"

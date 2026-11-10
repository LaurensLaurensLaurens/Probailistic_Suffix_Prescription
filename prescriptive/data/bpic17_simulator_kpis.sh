#!/bin/bash
# Evaluation des Simulators (main.py --steps kpis): die allgemeinen KPIs und die Prescriptive-KPIs je Wahlpunkt
# (helpers/simulator_prescriptive.py), mit den Wahlpunkten eines Laufs:
# MODE=gesamt: 12 Punkte mit channel+angebot_nachlegen (wie PSP_real_w3_10k_online, xgb_real_10k)
# MODE=channel: am O_Created nur der Kanal (wie PSP_real_w3_10k_online_channel, xgb_real_10k_channel)
# Der Katalog des Laufs ist eine Kopie (data/catalog_simulator_kpis_<MODE>.yaml über PSP_CATALOG); prescriptive/catalog.yaml
# bleibt unberührt. Ergebnisse in scripts/results_100k/simulator_kpis_<MODE> (simulator/), Log data/simulator_kpis_<MODE>.log.
# Der Prozess räumt bei Speichermangel als erster (oom_score_adj 1000), damit laufende Trainings bleiben.
ROOT=/home/chair/henryks_students/laurens_ohl/Probabilistic_Suffix_Prescription
P=$ROOT/prescriptive
MODE=${MODE:-gesamt}; CASES=${CASES:-10000}
case $MODE in
    gesamt) CONTROL='["handle_leads", "annehmen_nach_vervollstaendigung", "angebot_nachlegen", "neues_angebot_beim_nachfassen"]' ;;
    channel) CONTROL='["handle_leads", "annehmen_nach_vervollstaendigung", "neues_angebot_beim_nachfassen"]' ;;
    *) echo "MODE gesamt oder channel"; exit 1 ;;
esac
TAG=simulator_kpis_$MODE
OUT=$P/scripts/results_100k/$TAG
LOG=$P/data/$TAG.log
CATALOG=$P/data/catalog_$TAG.yaml
FILTER='FutureWarning|applied = df|cuda|CUDA|hasattr|pin_memory|super\(\).__init__'
python3 - "$P/catalog.yaml" "$CATALOG" "$CONTROL" <<'PY' || { echo "######## Katalog nicht erstellt, Abbruch" | tee -a "$LOG"; exit 1; }
import re, sys
source, target, control = sys.argv[1:4]
text = open(source, encoding="utf-8").read()
start = text.index("\nbpic17:\n") + len("\nbpic17:\n")
end = re.compile(r"^\S", re.M).search(text, start).start()
values = {"mitarbeiterwahl": '["Mitarbeiter: W_Call incomplete files", "Mitarbeiter: W_Complete application", "Mitarbeiter: W_Handle leads", '
                             '"Mitarbeiter: W_Validate application", "Mitarbeiter: W_Call after offers (resume)", "Mitarbeiter: W_Call incomplete files (resume)", '
                             '"Mitarbeiter: W_Complete application (resume)", "Mitarbeiter: W_Validate application (resume)"]',
          "control_flow": control}
part = text[start:end]
for key, value in values.items():
    part, n = re.subn(rf"^    {key}: \[.*\]$", lambda m: f"    {key}: {value}", part, flags=re.M)
    assert n == 1, key
open(target, "w", encoding="utf-8").write(text[:start] + part + text[end:])
PY
export PSP_CATALOG=$CATALOG
cd $ROOT/process-simulation-engine
echo "######## Evaluation des Simulators, $MODE, $CASES Fälle ($(date +%H:%M:%S))" | tee -a "$LOG"
echo 1000 > /proc/self/oom_score_adj
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u ../prescriptive/scripts/main.py --steps kpis --n-cases $CASES --out "$OUT" "$@" 2>&1 \
    | grep --line-buffered -v -E "$FILTER" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "######## Abbruch ($(date +%H:%M:%S))" | tee -a "$LOG"; exit 1; }
echo "######## Alles fertig ($(date +%H:%M:%S))" | tee -a "$LOG"

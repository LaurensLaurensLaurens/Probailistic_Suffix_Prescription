"""
Next-event model for the simulation: generates the control flow of a case event by event.

An event is one token made of activity and lifecycle transition (e.g. "W_Call after offers|suspend"),
so the model cannot produce a lifecycle the activity never has.

datasets.py            je Datensatz (NEXT_EVENT): welche Fallattribute, Angebotsattribute und welcher Anrufer Eingang
                       sind, Arbeitsschritte, Grenzen (max_events, max_gap_days), festes Modell (model_folder, trainable)
data_prep.py           Log -> Traces (load_traces), Kodierung der Eingänge (Encoder), Zeitklassen; für alle Datensätze gleich
event_time_model.py    das Modell (EventTimeModel): nächstes Event und Zeit bis dahin
train.py               Training und Kennzahlen (python -m next_event_prediction.train)
models.py              Vergleichsmodelle für train.py: n-gram, LSTM und Transformer ohne Zeiten
BPIC2017/              nur BPIC17: Kalibrierung des Anrufer-Effekts, Kennzahlen am Entscheidungspunkt (train.py)
Die Simulatoren, die das Modell nutzen, liegen in simulation/ (case_engine.py, case_simulator.py, dataset_simulator.py);
die Interventionspunkte (Aktion als Eingang) in interventions/.
"""

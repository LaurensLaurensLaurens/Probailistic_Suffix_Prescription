"""
Das Next-Event-Modell je Datensatz: welche Eingänge es hat und wo es liegt. data_prep.py (load_traces, Encoder) und
event_time_model.py lesen hier nach, der Code selbst ist für alle Datensätze gleich. prescriptive/profiles.py übernimmt die
Einträge ins Profil.

Je Datensatz die Attribute als Eingang, statisch (je Fall) oder dynamisch (je Event), kategorial oder numerisch:
- static_categorical, static_numeric: Fallattribute, je Fall ein Wert vom ersten Event (case_attributes/datasets.py
  zieht für die Simulation genau diese)
- dynamic_numeric: numerische Attribute eines Events, die ab diesem Event Eingang sind (der zuletzt gesetzte Wert gilt
  weiter; activity: an deren complete-Event sie stehen, attributes: welche), None: keine. BPIC17: das aktuelle Angebot
- dynamic_categorical: kategoriale Attribute eines Events nach demselben Prinzip; noch nicht als Eingang des Modells
  umgesetzt (data_prep.Encoder bricht ab, wenn gesetzt), None: keine
- work_item_prefix: Aktivitäten mit diesem Präfix sind Arbeitsschritte mit Sitzungen eines Mitarbeiters (start/resume
  öffnen, suspend/complete/ate_abort/withdraw schließen), None: keine. Mit Arbeitsschritten ist der Mitarbeiter, der ein
  Event ausführt, Eingang für die Zeit bis zu diesem Event, für jede Aktivität gleich und jeder mit eigenem Index: Das
  Modell zieht erst das nächste Event ohne den Mitarbeiter, dann die Zeit abhängig von ihm (event_time_model.py)
- performers: nur complete-Events, aber der Mitarbeiter jedes Events ist Eingang (im Prefix und für die Zeit), und das
  Modell zieht ihn selbst: erst das nächste Event, dann wer es ausführt, dann die Zeit (simulation/case_engine_lifecycle_unaware.py).
  Fehlt der Eintrag: nein
- max_events: höchstens so viele Events je Fall minus 1 (längere Fälle lässt der Simulator weg)
- max_gap_days: die Zeitklassen reichen bis zu so vielen Tagen (für die Zeit bis zum nächsten Event und seit Fallbeginn;
  längere fallen in die letzte Klasse). Gilt für neu trainierte Modelle; ein gespeichertes Modell behält seine Klassen
- model_folder, trainable: festes Simulationsmodell (BPIC17: das der Pipeline, nicht neu trainiert); sonst
  models/next_event/<datensatz>_lstm_time_act_<intervention> (simulation/dataset_simulator.py)
"""

NEXT_EVENT = {
    "bpic17": {
        "static_categorical": ["case:LoanGoal", "case:ApplicationType"], 
        "static_numeric": ["case:RequestedAmount"],
        # Das aktuelle Angebot. CreditScore, Selected und Accepted nicht: sie verraten im echten Log den Ausgang des Falls
        "dynamic_numeric": {"activity": "O_Create Offer", "attributes": ["OfferedAmount", "NumberOfTerms", "MonthlyCost", "FirstWithdrawalAmount"]},
        "dynamic_categorical": None,
        "work_item_prefix": "W_",
        # Längster Fall 180 Events; Abbrüche kommen 30,5 bis 30,9 Tage nach dem letzten Angebot
        "max_events": 200, "max_gap_days": 180,
        # Auf den Lifecycles des Logs trainiert (schedule/start/suspend/resume/complete/ate_abort/withdraw)
        "model_folder": "models/next_event/bpic17", "trainable": False,
    },
    "bpic13": {
        "static_categorical": ["case:impact", "case:product", "case:org_line"], "static_numeric": [],
        "dynamic_numeric": None, "dynamic_categorical": None, "work_item_prefix": None,
        # Längster Fall 123 Events; 0,6 % der Fälle dauern länger als 180 Tage
        "max_events": 200, "max_gap_days": 180, "model_folder": "models/next_event/bpic13",
    },
    "bpic19": {
        "static_categorical": ["case:Spend area text", "case:Sub spend area text", "case:Item Type", "case:Item Category", "case:Document Type",
                               "case:Company", "case:Vendor"],
        "static_numeric": [], "dynamic_numeric": None, "dynamic_categorical": None, "work_item_prefix": None, "performers": True,
        # 0,13 % der Fälle haben mehr als 199 Events (bis 923), 2,7 % dauern länger als 180 Tage (99 %-Quantil 218 Tage)
        "max_events": 200, "max_gap_days": 180, "model_folder": "models/next_event/bpic19",
    },
    # Die synthetische Werkstatt (simulation/repair_workshop.py): Arbeitsschritte W_... mit start und complete, der Gerätetyp als Fallattribut.
    # Längster Fall rund 40 Events, Teile kommen nach höchstens 20 Tagen
    "repair": {
        "static_categorical": ["case:device"], "static_numeric": [], "dynamic_numeric": None, "dynamic_categorical": None,
        "work_item_prefix": "W_", "max_events": 200, "max_gap_days": 180, "model_folder": "models/next_event/repair",
    },
}


def model_spec(dataset):
    if dataset not in NEXT_EVENT:
        raise ValueError(f"Kein Next-Event-Modell für '{dataset}' festgelegt, möglich: {', '.join(NEXT_EVENT)}")
    return NEXT_EVENT[dataset]

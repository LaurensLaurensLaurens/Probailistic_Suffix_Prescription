┌──────────────────────────┐
│   ProcessPreprocessor    │
│                          │
│ Bereitet Prozessdaten    │
│ pro Decision Point auf   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│         Method           │
│                          │
│ Orchestriert SCOPE und   │
│ führt Backward Induction │
│ über die Stages aus      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│     SCOPEFunctions       │
│                          │
│ • Targets berechnen      │
│ • Q-Werte bestimmen      │
│ • Causal Estimates       │
│ • optimale Aktionen      │
│ • Values zurückgeben     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      ModelTrainer        │
│                          │
│ Trainiert Outcome-,      │
│ Effect- und PS-Modelle   │
└──────────────────────────┘
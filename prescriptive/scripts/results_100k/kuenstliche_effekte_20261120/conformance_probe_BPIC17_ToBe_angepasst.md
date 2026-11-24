# Conformance gegen BPIC17_ToBe_angepasst.bpmn (Projektion A_/O_ complete, W_ start)

## Echter Log (BPIC17) (1000 Fälle)

- Fitness Mittel 0.981, Median 1.000; konform (keine Log- oder Modellzüge; stille Übergänge kosten in pm4py 1, Abweichungen 10.000): 80.6%; Abweichungen je Fall 0.51
- häufigste Abweichungen (je Fall):
  - Logzug (unerlaubt): O_Returned: 0.098
  - Logzug (unerlaubt): O_Create Offer: 0.061
  - Logzug (unerlaubt): O_Created: 0.061
  - Modellzug (fehlt): O_Returned: 0.056
  - Logzug (unerlaubt): O_Cancelled: 0.047
  - Logzug (unerlaubt): O_Sent (online only): 0.038
  - Logzug (unerlaubt): W_Validate application: 0.025
  - Logzug (unerlaubt): O_Sent (mail and online): 0.024
  - Logzug (unerlaubt): W_Call incomplete files: 0.014
  - Modellzug (fehlt): A_Cancelled: 0.008
  - Logzug (unerlaubt): A_Incomplete: 0.006
  - Modellzug (fehlt): W_Validate application: 0.006

## Simulierter Log (rct_tlearner_20261118/rct_log_raw.pkl) (1000 Fälle)

- Fitness Mittel 0.978, Median 1.000; konform (keine Log- oder Modellzüge; stille Übergänge kosten in pm4py 1, Abweichungen 10.000): 72.5%; Abweichungen je Fall 0.60
- häufigste Abweichungen (je Fall):
  - Logzug (unerlaubt): O_Cancelled: 0.133
  - Modellzug (fehlt): O_Returned: 0.084
  - Logzug (unerlaubt): O_Returned: 0.083
  - Logzug (unerlaubt): O_Create Offer: 0.048
  - Logzug (unerlaubt): O_Created: 0.048
  - Logzug (unerlaubt): W_Validate application: 0.040
  - Logzug (unerlaubt): O_Sent (online only): 0.025
  - Logzug (unerlaubt): O_Sent (mail and online): 0.025
  - Logzug (unerlaubt): A_Accepted: 0.022
  - Logzug (unerlaubt): W_Call incomplete files: 0.020
  - Modellzug (fehlt): A_Cancelled: 0.010
  - Modellzug (fehlt): O_Sent (online only): 0.009

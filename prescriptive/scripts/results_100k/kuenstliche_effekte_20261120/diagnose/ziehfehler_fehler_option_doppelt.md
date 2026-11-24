# Woher die Abweichungen der gezogenen Suffixe kommen (k3, Diagnose 2026-11-21)

20 Ziehungen je Option, 3 × 40 Fälle; „echt“ = tatsächlich simulierter Verlauf mit der Option der Bank.

```
                                                                                 spuren  abweichungen  mit_ungueltigem_uebergang  ohne_eos  laenge
punkt                            option                        quelle                                                                             
angebot_nachlegen                O_Create Offer                echt (Simulator)       9         1.667                      0.000       0.0  30.000
                                                               gezogen             2300         2.352                      0.238       1.0  35.732
                                 O_Sent (mail and online)      echt (Simulator)     106         0.802                      0.009       0.0  28.189
                                                               gezogen             2300         1.009                      0.152       1.0  29.840
annehmen_bei_konzept             A_Accepted                    echt (Simulator)       2         0.000                      0.000       0.0  16.000
                                                               gezogen             1160         1.911                      1.000       1.0  35.177
                                 W_Complete application|start  echt (Simulator)      56         1.071                      0.000       0.0  36.286
                                                               gezogen             1160         1.084                      0.180       1.0  36.462
annehmen_nach_vervollstaendigung A_Accepted                    echt (Simulator)      15         0.867                      0.000       0.0  30.933
                                                               gezogen             1520         2.245                      1.000       1.0  32.884
                                 W_Complete application|resume echt (Simulator)      61         1.016                      0.016       0.0  35.967
                                                               gezogen             1520         1.221                      1.000       1.0  37.313
handle_leads                     W_Handle leads|start          echt (Simulator)      17         0.882                      0.059       0.0  38.059
                                                               gezogen             1460         2.266                      0.288       1.0  45.178
                                 W_Handle leads|withdraw       echt (Simulator)      56         1.071                      0.000       0.0  39.107
                                                               gezogen             1460         1.058                      1.000       1.0  39.964
```


## Häufigste Züge, echt (Simulator) (je Spur)

- fehlt W_Call after offers: 0.298
- unerlaubt A_Complete: 0.127
- fehlt O_Returned: 0.071
- unerlaubt O_Create Offer: 0.068
- unerlaubt O_Created: 0.068
- unerlaubt O_Cancelled: 0.068
- unerlaubt O_Returned: 0.065
- unerlaubt W_Validate application: 0.056
- unerlaubt O_Sent (mail and online): 0.043
- unerlaubt O_Sent (online only): 0.037
- unerlaubt W_Call incomplete files: 0.025
- fehlt W_Validate application: 0.012
- unerlaubt A_Incomplete: 0.009
- fehlt A_Validating: 0.009
- unerlaubt A_Accepted: 0.003

Abweichungen je Spur mit / ohne ungültigen Übergang: 2.00 / 0.95 (Anteil Spuren mit ungültigem Übergang 0.9%)

## Häufigste Züge, gezogen (je Spur)

- unerlaubt O_Create Offer: 0.269
- unerlaubt A_Accepted: 0.221
- fehlt W_Call after offers: 0.149
- unerlaubt W_Handle leads: 0.115
- unerlaubt O_Cancelled: 0.104
- unerlaubt O_Returned: 0.103
- unerlaubt A_Cancelled: 0.101
- unerlaubt A_Complete: 0.091
- fehlt O_Returned: 0.087
- unerlaubt O_Created: 0.061
- unerlaubt O_Sent (mail and online): 0.046
- unerlaubt W_Validate application: 0.045
- unerlaubt W_Call incomplete files: 0.035
- unerlaubt O_Sent (online only): 0.030
- fehlt O_Cancelled: 0.027

Abweichungen je Spur mit / ohne ungültigen Übergang: 1.92 / 1.32 (Anteil Spuren mit ungültigem Übergang 55.8%)

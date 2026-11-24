# Causal Forest Conformance (simv3), 2026-11-23 14:14

τ = Outcome(Aktion) − Outcome(Referenz), Outcome = −Abweichungen; τ > 0: Aktion hat weniger Abweichungen. Testfälle.

```
                       Wahlpunkt                      Referenz               Aktion  Entscheidungen  τ Mittel (Abw. weniger mit Aktion)  τ SD  Anteil Aktion besser
               angebot_nachlegen      O_Sent (mail and online)       O_Create Offer            8139                              -0.258 0.464                 0.389
            annehmen_bei_konzept  W_Complete application|start           A_Accepted            4571                               0.281 0.249                 0.808
annehmen_nach_vervollstaendigung W_Complete application|resume           A_Accepted            7459                               0.027 0.043                 0.717
                    handle_leads       W_Handle leads|withdraw W_Handle leads|start            4063                               0.132 0.210                 0.495
```

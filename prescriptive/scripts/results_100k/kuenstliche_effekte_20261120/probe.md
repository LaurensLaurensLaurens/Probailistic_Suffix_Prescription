# Probe künstliche Effekte (4000 Fälle, Saat 7, Bank-Policy)

Effektdatei: kuenstliche_effekte/effekte_v3.yaml

- Event-Typen ohne / mit: 59 / 60, nur ohne: [], nur mit: ['W_Shortened completion|suspend']
- Events je Fall ohne / mit: 42.70 / 40.12
- Durchlaufzeit (Tage) ohne / mit: Mittel 21.83 / 21.99, Median 18.07 / 18.22, P90 37.04 / 36.90
- je Fall gepaart (gleiche Fall-ID): mit − ohne Mittel +0.16 d, Fälle mit Effekt > 0: 1201

Event-Typen je Fall ohne / mit: A_Incomplete 0.718 / 0.655, W_Call incomplete files 0.726 / 0.665, A_Cancelled 0.307 / 0.332, O_Accepted 0.590 / 0.582, A_Pending 0.590 / 0.582, A_Denied 0.100 / 0.084, O_Create Offer 1.387 / 1.377, W_Validate application 1.369 / 1.254

Conformance (BPIC17_ToBe_angepasst) ohne / mit: konform 79.6% / 59.9%, Abweichungen je Fall 0.485 / 0.716

```
                           point                            option  ausgefuehrt  mit_effekt  tage_soll  tage_angewandt  abweichungen  Anteil mit Effekt  angewandt / soll  Tage je Fall (angewandt)
               angebot_nachlegen           O_Create Offer|complete          389         330        0.0           0.000            59              0.848               NaN                     0.000
               angebot_nachlegen O_Sent (mail and online)|complete         4652        2169        0.0           0.000          4407              0.466               NaN                     0.000
            annehmen_bei_konzept               A_Accepted|complete          976         418        0.0           0.000           648              0.428               NaN                     0.000
            annehmen_bei_konzept      W_Complete application|start         1941        1941        0.0           0.000           268              1.000               NaN                     0.000
annehmen_nach_vervollstaendigung               A_Accepted|complete         1369         706     2047.4        2044.500             0              0.516             0.999                     0.513
annehmen_nach_vervollstaendigung     W_Complete application|resume         3312        1825    -2007.5       -1818.154             0              0.551             0.906                    -0.456
                    handle_leads              W_Handle leads|start          508         248     -595.2        -587.218             0              0.488             0.987                    -0.147
                    handle_leads           W_Handle leads|withdraw         2102         998      598.8         598.800             0              0.475             1.000                     0.150
```


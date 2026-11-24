# Online Conformance, Simulator mit Effekten v3 (Aufbau F, 200 bewertete Fälle je Seed)

Abweichungen je Fall gegen BPIC17_ToBe_angepasst; konform = Anteil Fälle ohne Abweichung.

```
                  seeds  abweichungen  konform  dlz_tage  stunden
arm                                                              
bank                 12         0.666    0.654    20.450    0.008
cf_konformanz        12         0.595    0.716    20.506    0.011
cf_rueckwaerts       12         0.536    0.764    19.613    0.011
konformanz           12         0.642    0.695    21.444    0.717
konformanz_ol        12         0.610    0.709    21.357    0.628
konformanz_ol300     12         0.605    0.728    20.697    1.759
konformanz_ol_ro     12         0.623    0.697    20.703    2.009
konformanz_x4        12         0.602    0.710    21.210    0.718
orakel               12         0.577    0.731    20.492    0.009
regel                12         0.586    0.720    20.838    0.009
```

Gepaart je Seed (erste minus zweite; bei abweichungen positiv = zweite besser):

```
                          vergleich        größe  seeds  mittel    se
                  bank − konformanz abweichungen     12   0.024 0.039
                  bank − konformanz      konform     12  -0.041 0.013
                  bank − konformanz     dlz_tage     12  -0.994 0.324
                 regel − konformanz abweichungen     12  -0.057 0.027
                 regel − konformanz      konform     12   0.024 0.006
                 regel − konformanz     dlz_tage     12  -0.606 0.325
                orakel − konformanz abweichungen     12  -0.065 0.039
                orakel − konformanz      konform     12   0.035 0.013
                orakel − konformanz     dlz_tage     12  -0.952 0.352
                      bank − orakel abweichungen     12   0.089 0.027
                      bank − orakel      konform     12  -0.077 0.008
                      bank − orakel     dlz_tage     12  -0.042 0.389
                       bank − regel abweichungen     12   0.080 0.028
                       bank − regel      konform     12  -0.065 0.013
                       bank − regel     dlz_tage     12  -0.388 0.323
               bank − konformanz_x4 abweichungen     12   0.064 0.022
               bank − konformanz_x4      konform     12  -0.056 0.012
               bank − konformanz_x4     dlz_tage     12  -0.760 0.325
              regel − konformanz_x4 abweichungen     12  -0.016 0.020
              regel − konformanz_x4      konform     12   0.009 0.009
              regel − konformanz_x4     dlz_tage     12  -0.372 0.293
             orakel − konformanz_x4 abweichungen     12  -0.025 0.023
             orakel − konformanz_x4      konform     12   0.020 0.008
             orakel − konformanz_x4     dlz_tage     12  -0.719 0.275
         konformanz − konformanz_x4 abweichungen     12   0.040 0.025
         konformanz − konformanz_x4      konform     12  -0.015 0.011
         konformanz − konformanz_x4     dlz_tage     12   0.233 0.385
               bank − konformanz_ol abweichungen     12   0.056 0.040
               bank − konformanz_ol      konform     12  -0.055 0.012
               bank − konformanz_ol     dlz_tage     12  -0.907 0.415
              regel − konformanz_ol abweichungen     12  -0.025 0.030
              regel − konformanz_ol      konform     12   0.011 0.013
              regel − konformanz_ol     dlz_tage     12  -0.519 0.423
             orakel − konformanz_ol abweichungen     12  -0.033 0.041
             orakel − konformanz_ol      konform     12   0.022 0.013
             orakel − konformanz_ol     dlz_tage     12  -0.865 0.370
      konformanz_x4 − konformanz_ol abweichungen     12  -0.008 0.036
      konformanz_x4 − konformanz_ol      konform     12   0.002 0.010
      konformanz_x4 − konformanz_ol     dlz_tage     12  -0.146 0.366
            bank − konformanz_ol300 abweichungen     12   0.061 0.035
            bank − konformanz_ol300      konform     12  -0.074 0.014
            bank − konformanz_ol300     dlz_tage     12  -0.247 0.435
           regel − konformanz_ol300 abweichungen     12  -0.019 0.029
           regel − konformanz_ol300      konform     12  -0.008 0.011
           regel − konformanz_ol300     dlz_tage     12   0.141 0.367
          orakel − konformanz_ol300 abweichungen     12  -0.028 0.033
          orakel − konformanz_ol300      konform     12   0.003 0.013
          orakel − konformanz_ol300     dlz_tage     12  -0.205 0.282
   konformanz_ol − konformanz_ol300 abweichungen     12   0.005 0.027
   konformanz_ol − konformanz_ol300      konform     12  -0.019 0.009
   konformanz_ol − konformanz_ol300     dlz_tage     12   0.660 0.427
            bank − konformanz_ol_ro abweichungen     12   0.043 0.039
            bank − konformanz_ol_ro      konform     12  -0.043 0.014
            bank − konformanz_ol_ro     dlz_tage     12  -0.253 0.359
           regel − konformanz_ol_ro abweichungen     12  -0.038 0.029
           regel − konformanz_ol_ro      konform     12   0.022 0.012
           regel − konformanz_ol_ro     dlz_tage     12   0.135 0.341
          orakel − konformanz_ol_ro abweichungen     12  -0.046 0.033
          orakel − konformanz_ol_ro      konform     12   0.033 0.010
          orakel − konformanz_ol_ro     dlz_tage     12  -0.211 0.387
konformanz_ol300 − konformanz_ol_ro abweichungen     12  -0.018 0.029
konformanz_ol300 − konformanz_ol_ro      konform     12   0.030 0.010
konformanz_ol300 − konformanz_ol_ro     dlz_tage     12  -0.006 0.482
               bank − cf_konformanz abweichungen     12   0.071 0.034
               bank − cf_konformanz      konform     12  -0.062 0.014
               bank − cf_konformanz     dlz_tage     12  -0.056 0.399
              regel − cf_konformanz abweichungen     12  -0.010 0.027
              regel − cf_konformanz      konform     12   0.004 0.006
              regel − cf_konformanz     dlz_tage     12   0.332 0.234
             orakel − cf_konformanz abweichungen     12  -0.018 0.034
             orakel − cf_konformanz      konform     12   0.015 0.012
             orakel − cf_konformanz     dlz_tage     12  -0.014 0.365
   konformanz_ol300 − cf_konformanz abweichungen     12   0.010 0.020
   konformanz_ol300 − cf_konformanz      konform     12   0.012 0.010
   konformanz_ol300 − cf_konformanz     dlz_tage     12   0.191 0.396
              bank − cf_rueckwaerts abweichungen     12   0.130 0.029
              bank − cf_rueckwaerts      konform     12  -0.110 0.013
              bank − cf_rueckwaerts     dlz_tage     12   0.837 0.425
             regel − cf_rueckwaerts abweichungen     12   0.050 0.026
             regel − cf_rueckwaerts      konform     12  -0.044 0.009
             regel − cf_rueckwaerts     dlz_tage     12   1.225 0.337
            orakel − cf_rueckwaerts abweichungen     12   0.041 0.035
            orakel − cf_rueckwaerts      konform     12  -0.033 0.011
            orakel − cf_rueckwaerts     dlz_tage     12   0.879 0.282
  konformanz_ol300 − cf_rueckwaerts abweichungen     12   0.069 0.036
  konformanz_ol300 − cf_rueckwaerts      konform     12  -0.036 0.013
  konformanz_ol300 − cf_rueckwaerts     dlz_tage     12   1.084 0.289
     cf_konformanz − cf_rueckwaerts abweichungen     12   0.060 0.035
     cf_konformanz − cf_rueckwaerts      konform     12  -0.048 0.011
     cf_konformanz − cf_rueckwaerts     dlz_tage     12   0.893 0.298
```

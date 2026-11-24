# (c) in Aufbau F: mit und ohne first_only = angebot_nachlegen

Bank 21.14 d, feste Regel 18.53 d (3 Seeds × 200 Fälle, gepaart je Seed).

```
                                        lauf          arm  seeds  dlz_tage  bank_tage  gewinn_tage  gewinn_se        gewinn_je_seed  rueckstand_regel_tage  rueckstand_regel_se  angebote_je_fall  anteil_ge4_angebote  max_angebote  kosten_eur
                    ohne first_only (16.11.)  adaptiv_mit      3    21.473     21.142       -0.331      0.423 +0.33 / -1.12 / -0.20                  2.946                0.667             1.865                0.110            32     182.590
                    ohne first_only (16.11.) adaptiv_ohne      3    20.265     21.142        0.877      0.534 +0.31 / +0.38 / +1.94                  1.738                0.086             2.313                0.152            48     190.704
                    ohne first_only (16.11.)      fest100      3    21.117     21.142        0.025      0.743 -0.68 / -0.75 / +1.51                  2.590                0.138             2.437                0.167            30     200.293
               first_only Nachlegen (17.11.)  adaptiv_mit      3    21.424     21.142       -0.282      0.761 -0.31 / -1.58 / +1.05                  2.897                0.345             1.338                0.008             6     218.455
               first_only Nachlegen (17.11.) adaptiv_ohne      3    19.847     21.142        1.296      0.717 +0.19 / +1.06 / +2.64                  1.320                0.287             1.503                0.028             7     185.702
               first_only Nachlegen (17.11.)      fest100      3    19.414     21.142        1.728      1.044 +1.32 / +0.16 / +3.71                  0.887                0.493             1.533                0.027             8     183.804
first_only Nachlegen, 300 Ziehungen (17.11.)      fest300      3    19.360     21.142        1.782      0.995 +0.68 / +0.89 / +3.77                  0.833                0.399             1.402                0.022             5     211.479
```

## Anteil der Wahlen mit 2 Optionen, bei denen die Bank entscheidet (Rückfall oder first_only)

```
                                                                                            wahlen  anteil_bank
lauf                                         arm          punkt                                                
first_only Nachlegen (17.11.)                adaptiv_mit  angebot_nachlegen                  758.0        0.856
                                                          annehmen_bei_konzept               308.0        0.718
                                                          annehmen_nach_vervollstaendigung   418.0        0.560
                                                          handle_leads                       369.0        0.821
                                             adaptiv_ohne angebot_nachlegen                  823.0        0.301
                                                          annehmen_bei_konzept               326.0        0.000
                                                          annehmen_nach_vervollstaendigung   186.0        0.000
                                                          handle_leads                       369.0        0.000
                                             fest100      angebot_nachlegen                  837.0        0.307
                                                          annehmen_bei_konzept               317.0        0.000
                                                          annehmen_nach_vervollstaendigung   200.0        0.000
                                                          handle_leads                       369.0        0.000
first_only Nachlegen, 300 Ziehungen (17.11.) fest300      angebot_nachlegen                  771.0        0.249
                                                          annehmen_bei_konzept               360.0        0.000
                                                          annehmen_nach_vervollstaendigung   117.0        0.000
                                                          handle_leads                       369.0        0.000
ohne first_only (16.11.)                     adaptiv_mit  angebot_nachlegen                 1015.0        0.618
                                                          annehmen_bei_konzept               308.0        0.724
                                                          annehmen_nach_vervollstaendigung   423.0        0.572
                                                          handle_leads                       369.0        0.791
                                             adaptiv_ohne angebot_nachlegen                 1209.0        0.000
                                                          annehmen_bei_konzept               317.0        0.000
                                                          annehmen_nach_vervollstaendigung   188.0        0.000
                                                          handle_leads                       369.0        0.000
                                             fest100      angebot_nachlegen                 1285.0        0.000
                                                          annehmen_bei_konzept               311.0        0.000
                                                          annehmen_nach_vervollstaendigung   184.0        0.000
                                                          handle_leads                       369.0        0.000
```

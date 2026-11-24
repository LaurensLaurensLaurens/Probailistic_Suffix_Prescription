# Online-Wahlen je Teilgruppe: Anteil „andere Option“ (nicht Regel), erste Besuche, bewertete Fälle, 12 Seeds

kombi / orakel_bank = Ziel-Wahl (True = andere Option) der besten Kombination mit Regel sonst (wahrheit_f_regel) bzw. des Orakels (Wahrheit mit Bank sonst).

```
                                                                                                                          n_je_arm  kombi  orakel_bank  cf_rueckwaerts  cf_konformanz  orakel  regel  konformanz_ol300
punkt                            gruppe                                                                                                                                                                               
angebot_nachlegen                dauer_ab_tage=0.84, nur_erster_besuch=True                                                    451  False        False           0.000          0.003   0.000    0.0             0.067
                                 dauer_ab_tage=0.84, nur_erster_besuch=True | dauer_ab_tage=1.71, nur_erster_besuch=True        92   True         True           0.882          0.779   0.809    0.0             0.674
                                 dauer_unter_tage=0.84, nur_erster_besuch=True                                                1785  False        False           0.000          0.000   0.000    0.0             0.002
annehmen_bei_konzept             betrag_ab=11000                                                                               969   True         True           0.671          0.299   0.808    0.0             0.343
                                 betrag_unter=11000                                                                            676  False        False           0.008          0.006   0.000    0.0             0.249
annehmen_nach_vervollstaendigung keine                                                                                         481   True        False           0.149          0.121   0.000    0.0             0.399
                                 ziel_in=['Car', 'Existing loan takeover']                                                     511   True        False           0.161          0.174   0.000    0.0             0.384
handle_leads                     betrag_unter=11000                                                                            679  False         True           0.001          1.000   1.000    0.0             0.328
                                 keine                                                                                         780  False        False           0.024          0.015   0.000    0.0             0.071
```

Anteil Entscheidungen wie die Kombination:
```
arm
cf_konformanz       0.616
cf_rueckwaerts      0.838
konformanz_ol300    0.724
orakel              0.635
regel               0.773
```

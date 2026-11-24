# Rückfall von (c) auf die Bank im Aufbau F (adaptiv mit Rückfall, 3 Seeds × 200 Fälle)

## je Punkt

```
                                     n  rueckfall
point                                            
angebot_nachlegen                 1015      0.618
annehmen_bei_konzept               308      0.724
annehmen_nach_vervollstaendigung   423      0.572
handle_leads                       369      0.791
```

## je Punkt und Wiederholung im Fall

```
                                                 n  rueckfall
point                            wiederholung                
angebot_nachlegen                1             586      0.817
                                 2             136      0.463
                                 3              80      0.338
                                 4             213      0.272
annehmen_bei_konzept             1             308      0.724
annehmen_nach_vervollstaendigung 1             299      0.555
                                 2              86      0.616
                                 3              24      0.625
                                 4              14      0.571
handle_leads                     1             369      0.791
```

## je Punkt und Angebotszahl bisher

```
                                             n  rueckfall
point                            angebote                
angebot_nachlegen                1         576      0.819
                                 2         140      0.493
                                 3          76      0.329
                                 4         223      0.274
annehmen_bei_konzept             0         308      0.724
annehmen_nach_vervollstaendigung 0         418      0.567
                                 1           4      1.000
                                 2           1      1.000
handle_leads                     0         369      0.791
```

## je Punkt und Fallalter

```
                                                 n  rueckfall
point                            alter_klasse                
angebot_nachlegen                <1 d          531      0.821
                                 1-7           152      0.776
                                 7-14           78      0.397
                                 14-30         166      0.217
                                 >30            88      0.068
annehmen_bei_konzept             <1 d          308      0.724
                                 1-7             0        NaN
                                 7-14            0        NaN
                                 14-30           0        NaN
                                 >30             0        NaN
annehmen_nach_vervollstaendigung <1 d          357      0.563
                                 1-7            66      0.621
                                 7-14            0        NaN
                                 14-30           0        NaN
                                 >30             0        NaN
handle_leads                     <1 d          369      0.791
                                 1-7             0        NaN
                                 7-14            0        NaN
                                 14-30           0        NaN
                                 >30             0        NaN
```

## je Punkt und Antragsart

```
                                                n  rueckfall
point                            typ                        
angebot_nachlegen                Limit raise  130      0.738
                                 New credit   885      0.600
annehmen_bei_konzept             Limit raise    6      0.833
                                 New credit   302      0.722
annehmen_nach_vervollstaendigung Limit raise  103      0.670
                                 New credit   320      0.541
handle_leads                     New credit   369      0.791
```

## je Punkt und beantragtem Betrag (Quartile)

```
                                                        n  rueckfall
point                            betrag_klasse                      
angebot_nachlegen                (-0.001, 6000.0]     261      0.705
                                 (6000.0, 11500.0]    245      0.551
                                 (11500.0, 20000.0]   299      0.599
                                 (20000.0, 250000.0]  210      0.614
annehmen_bei_konzept             (-0.001, 6000.0]      91      0.747
                                 (6000.0, 11500.0]     70      0.671
                                 (11500.0, 20000.0]    85      0.718
                                 (20000.0, 250000.0]   62      0.758
annehmen_nach_vervollstaendigung (-0.001, 6000.0]      89      0.539
                                 (6000.0, 11500.0]    108      0.537
                                 (11500.0, 20000.0]   126      0.587
                                 (20000.0, 250000.0]  100      0.620
handle_leads                     (-0.001, 6000.0]     100      0.690
                                 (6000.0, 11500.0]     95      0.874
                                 (11500.0, 20000.0]   103      0.825
                                 (20000.0, 250000.0]   71      0.775
```

## Wahl, wenn (c) sicher ist

```
                                                                  n
point                            choice                            
angebot_nachlegen                O_Create Offer                 282
                                 O_Sent (mail and online)       106
annehmen_bei_konzept             A_Accepted                      83
                                 W_Complete application|start     2
annehmen_nach_vervollstaendigung A_Accepted                     176
                                 W_Complete application|resume    5
handle_leads                     W_Handle leads|start             7
                                 W_Handle leads|withdraw         70
```

## Wahl der festen Regel (gleicher Simulator)

```
                                                             n
point                            choice                       
angebot_nachlegen                O_Sent (mail and online)  701
annehmen_bei_konzept             A_Accepted                376
annehmen_nach_vervollstaendigung A_Accepted                 82
handle_leads                     W_Handle leads|withdraw   369
```

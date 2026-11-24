# Abweichungen von der festen Regel in Aufbau F (2026-11-19)

Regel: konzept → A_Accepted, nach V. → A_Accepted, nachlegen → O_Sent (kein Zusatzangebot), leads → withdraw. Differenz = Durchlaufzeit Arm − Regel in Tagen (positiv = Arm langsamer), gepaart je Seed und Fall-ID. Wahrer Nachteil der Nicht-Regel-Option (Simulator, randomisiert): konzept +2.79, nach V. +2.40, nachlegen (1.) +1.55, leads +1.32 d.

## Anteil Abweichungen je Punkt (an echten Wahlen; nachlegen nur 1. Besuch)

| arm | konzept | nach V. | nachlegen (1.) | leads |
|---|---|---|---|---|
| c_fest100 | 10% | 13% | 22% | 17% |
| c_fest300 | 1% | 5% | 9% | 6% |
| zztf1_fest100 | 24% | 12% | 52% | 47% |
| tl_fest300 | 8% | 10% | 39% | 20% |
| tl_fest500 | 4% | 6% | 36% | 13% |
| tlks_fest300 | 2% | 2% | 43% | 7% |
| tlrct_fest300 | 6% | 4% | 19% | 11% |
| tlcrn_fest300 | 1% | 3% | 34% | 8% |
| tlsv_fest300 | 9% | 20% | 10% | 1% |
| tlsvk_fest300 | 0% | 6% | 1% | 1% |
| tlsvp_fest300 | 0% | 4% | 84% | 83% |
| tlsvh_fest300 | 1% | 6% | 21% | 2% |
| olsv_fest300 | 0% | 3% | 0% | 0% |
| plsv_fest300 | 1% | 0% | 46% | 12% |
| olsvcf_fest300 | 0% | 0% | 0% | 0% |

## Nachteil gegen Regel: beobachtet vs. erwartet bei zufälliger Auswahl der abweichenden Fälle

| arm | seeds | nachteil_gegen_regel | se | erwartet_bei_zufaelliger_auswahl | ohne_abweichung | ohne_abweichung_se |
|---|---|---|---|---|---|---|
| c_fest100 | 12 | 1.06 | 0.26 | 0.72 | 0.34 | 0.37 |
| c_fest300 | 12 | 0.66 | 0.32 | 0.22 | -0.03 | 0.32 |
| zztf1_fest100 | 12 | 2.7 | 0.35 | 1.54 | 0.55 | 0.51 |
| tl_fest300 | 12 | 1.11 | 0.25 | 0.93 | 0.02 | 0.39 |
| tl_fest500 | 12 | 1.24 | 0.34 | 0.73 | 0.25 | 0.4 |
| tlks_fest300 | 12 | 1.22 | 0.28 | 0.74 | -0.34 | 0.4 |
| tlrct_fest300 | 12 | 0.65 | 0.19 | 0.49 | -0.06 | 0.34 |
| tlcrn_fest300 | 12 | 0.35 | 0.37 | 0.61 | -0.77 | 0.37 |
| tlsv_fest300 | 12 | 0.4 | 0.32 | 0.42 | 0.11 | 0.34 |
| tlsvk_fest300 | 12 | 0.02 | 0.29 | 0.06 | -0.1 | 0.3 |
| tlsvp_fest300 | 12 | 3.11 | 0.48 | 1.98 | -0.86 | 0.98 |
| tlsvh_fest300 | 12 | 0.48 | 0.25 | 0.38 | -0.14 | 0.34 |
| olsv_fest300 | 12 | 0.17 | 0.18 | 0.02 | 0.12 | 0.27 |
| plsv_fest300 | 12 | 1.14 | 0.26 | 0.79 | -0.15 | 0.39 |
| olsvcf_fest300 | 12 | -0.01 | 0.13 | 0.0 | -0.01 | 0.19 |

## Je Fallgruppe (nach Punkt der Abweichung)

| arm | abweichung_an | faelle | anteil_faelle | differenz_je_fall | se | se_seeds | dlz_unter_regel | wahr_bei_zufall | beitrag_je_fall_insgesamt |
|---|---|---|---|---|---|---|---|---|---|
| c_fest100 | keine | 1559 | 0.65 | 0.34 | 0.37 | 0.31 | 18.5 | – | 0.22 |
| c_fest100 | angebot_nachlegen | 423 | 0.18 | 1.77 | 0.94 | 0.6 | 18.51 | 1.55 | 0.31 |
| c_fest100 | annehmen_bei_konzept | 87 | 0.04 | 2.83 | 2.2 | 2.62 | 19.19 | 2.79 | 0.1 |
| c_fest100 | annehmen_nach_vervollstaendigung | 34 | 0.01 | 0.09 | 2.65 | 2.84 | 17.37 | 2.4 | 0.0 |
| c_fest100 | handle_leads | 175 | 0.07 | 3.32 | 1.58 | 1.34 | 18.43 | 1.32 | 0.24 |
| c_fest100 | mehrere | 122 | 0.05 | 3.49 | 1.97 | 1.49 | 20.77 | – | 0.18 |
| c_fest300 | keine | 2099 | 0.87 | -0.03 | 0.32 | 0.29 | 18.66 | – | -0.03 |
| c_fest300 | angebot_nachlegen | 193 | 0.08 | 4.61 | 1.53 | 1.16 | 19.34 | 1.55 | 0.37 |
| c_fest300 | annehmen_bei_konzept | 16 | 0.01 | 9.18 | 4.92 | 6.11 | 16.31 | 2.79 | 0.06 |
| c_fest300 | annehmen_nach_vervollstaendigung | 6 | 0.0 | 4.83 | 7.71 | 8.25 | 20.6 | 2.4 | 0.01 |
| c_fest300 | handle_leads | 69 | 0.03 | 6.51 | 1.9 | 2.01 | 16.31 | 1.32 | 0.19 |
| c_fest300 | mehrere | 17 | 0.01 | 8.3 | 3.79 | 3.71 | 16.07 | – | 0.06 |
| zztf1_fest100 | keine | 750 | 0.31 | 0.55 | 0.51 | 0.38 | 18.39 | – | 0.17 |
| zztf1_fest100 | angebot_nachlegen | 724 | 0.3 | 2.95 | 0.73 | 0.7 | 18.4 | 1.55 | 0.89 |
| zztf1_fest100 | annehmen_bei_konzept | 92 | 0.04 | 0.91 | 1.69 | 1.57 | 17.48 | 2.79 | 0.03 |
| zztf1_fest100 | annehmen_nach_vervollstaendigung | 22 | 0.01 | -0.29 | 4.03 | 3.92 | 18.07 | 2.4 | -0.0 |
| zztf1_fest100 | handle_leads | 287 | 0.12 | 4.57 | 1.15 | 1.4 | 18.57 | 1.32 | 0.55 |
| zztf1_fest100 | mehrere | 525 | 0.22 | 4.86 | 0.96 | 0.93 | 19.51 | – | 1.06 |
| tl_fest300 | keine | 1224 | 0.51 | 0.02 | 0.39 | 0.25 | 18.31 | – | 0.01 |
| tl_fest300 | angebot_nachlegen | 739 | 0.31 | 2.47 | 0.71 | 0.64 | 19.03 | 1.55 | 0.76 |
| tl_fest300 | annehmen_bei_konzept | 52 | 0.02 | 2.41 | 2.06 | 1.42 | 18.68 | 2.79 | 0.05 |
| tl_fest300 | annehmen_nach_vervollstaendigung | 33 | 0.01 | 1.47 | 3.37 | 3.14 | 17.58 | 2.4 | 0.02 |
| tl_fest300 | handle_leads | 179 | 0.07 | 0.89 | 1.44 | 1.44 | 18.4 | 1.32 | 0.07 |
| tl_fest300 | mehrere | 173 | 0.07 | 2.81 | 1.59 | 1.11 | 19.47 | – | 0.2 |
| tl_fest500 | keine | 1389 | 0.58 | 0.25 | 0.4 | 0.35 | 18.39 | – | 0.14 |
| tl_fest500 | angebot_nachlegen | 739 | 0.31 | 1.91 | 0.73 | 0.72 | 19.11 | 1.55 | 0.59 |
| tl_fest500 | annehmen_bei_konzept | 27 | 0.01 | 1.16 | 2.72 | 2.68 | 19.81 | 2.79 | 0.01 |
| tl_fest500 | annehmen_nach_vervollstaendigung | 20 | 0.01 | 7.44 | 4.86 | 4.47 | 14.47 | 2.4 | 0.06 |
| tl_fest500 | handle_leads | 137 | 0.06 | 2.3 | 1.61 | 2.24 | 19.53 | 1.32 | 0.13 |
| tl_fest500 | mehrere | 88 | 0.04 | 8.28 | 2.3 | 1.33 | 17.26 | – | 0.3 |
| tlks_fest300 | keine | 1316 | 0.55 | -0.34 | 0.4 | 0.29 | 18.62 | – | -0.19 |
| tlks_fest300 | angebot_nachlegen | 954 | 0.4 | 3.0 | 0.61 | 0.45 | 18.53 | 1.55 | 1.19 |
| tlks_fest300 | annehmen_bei_konzept | 14 | 0.01 | -1.72 | 4.73 | 3.5 | 19.57 | 2.79 | -0.01 |
| tlks_fest300 | annehmen_nach_vervollstaendigung | 4 | 0.0 | 8.53 | 8.15 | 4.53 | 12.48 | 2.4 | 0.01 |
| tlks_fest300 | handle_leads | 67 | 0.03 | 4.98 | 2.07 | 2.27 | 18.92 | 1.32 | 0.14 |
| tlks_fest300 | mehrere | 45 | 0.02 | 3.96 | 4.06 | 4.53 | 20.4 | – | 0.07 |
| tlrct_fest300 | keine | 1769 | 0.74 | -0.06 | 0.34 | 0.23 | 18.55 | – | -0.04 |
| tlrct_fest300 | angebot_nachlegen | 376 | 0.16 | 3.14 | 0.99 | 0.83 | 18.37 | 1.55 | 0.49 |
| tlrct_fest300 | annehmen_bei_konzept | 62 | 0.03 | 4.05 | 2.55 | 2.6 | 17.91 | 2.79 | 0.1 |
| tlrct_fest300 | annehmen_nach_vervollstaendigung | 14 | 0.01 | 2.32 | 4.03 | 3.64 | 14.63 | 2.4 | 0.01 |
| tlrct_fest300 | handle_leads | 120 | 0.05 | 0.42 | 1.69 | 1.61 | 20.79 | 1.32 | 0.02 |
| tlrct_fest300 | mehrere | 59 | 0.02 | 2.43 | 2.28 | 2.3 | 19.73 | – | 0.06 |
| tlcrn_fest300 | keine | 1502 | 0.63 | -0.77 | 0.37 | 0.36 | 18.73 | – | -0.48 |
| tlcrn_fest300 | angebot_nachlegen | 752 | 0.31 | 2.49 | 0.68 | 0.97 | 18.33 | 1.55 | 0.78 |
| tlcrn_fest300 | annehmen_bei_konzept | 10 | 0.0 | -16.25 | 8.89 | 5.58 | 27.23 | 2.79 | -0.07 |
| tlcrn_fest300 | annehmen_nach_vervollstaendigung | 7 | 0.0 | 5.46 | 4.58 | 4.85 | 17.26 | 2.4 | 0.02 |
| tlcrn_fest300 | handle_leads | 90 | 0.04 | 0.69 | 2.03 | 2.36 | 19.77 | 1.32 | 0.03 |
| tlcrn_fest300 | mehrere | 39 | 0.02 | 4.48 | 2.72 | 2.57 | 15.33 | – | 0.07 |
| tlsv_fest300 | keine | 1949 | 0.81 | 0.11 | 0.34 | 0.3 | 18.55 | – | 0.09 |
| tlsv_fest300 | angebot_nachlegen | 208 | 0.09 | 1.4 | 1.37 | 1.58 | 19.46 | 1.55 | 0.12 |
| tlsv_fest300 | annehmen_bei_konzept | 106 | 0.04 | 1.98 | 2.13 | 2.39 | 19.63 | 2.79 | 0.09 |
| tlsv_fest300 | annehmen_nach_vervollstaendigung | 97 | 0.04 | 0.45 | 1.9 | 2.13 | 17.17 | 2.4 | 0.02 |
| tlsv_fest300 | handle_leads | 7 | 0.0 | -3.48 | 5.94 | 6.95 | 22.74 | 1.32 | -0.01 |
| tlsv_fest300 | mehrere | 33 | 0.01 | 6.47 | 3.78 | 4.51 | 17.9 | – | 0.09 |
| tlsvk_fest300 | keine | 2330 | 0.97 | -0.1 | 0.3 | 0.31 | 18.68 | – | -0.09 |
| tlsvk_fest300 | angebot_nachlegen | 34 | 0.01 | 0.89 | 3.07 | 4.0 | 17.23 | 1.55 | 0.01 |
| tlsvk_fest300 | annehmen_bei_konzept | 1 | 0.0 | -21.2 | – | – | 39.97 | 2.79 | -0.01 |
| tlsvk_fest300 | annehmen_nach_vervollstaendigung | 18 | 0.01 | 6.24 | 3.12 | 3.85 | 18.79 | 2.4 | 0.05 |
| tlsvk_fest300 | handle_leads | 17 | 0.01 | 9.53 | 4.19 | 3.43 | 12.53 | 1.32 | 0.07 |
| tlsvp_fest300 | keine | 258 | 0.11 | -0.86 | 0.98 | 0.81 | 17.54 | – | -0.09 |
| tlsvp_fest300 | angebot_nachlegen | 924 | 0.38 | 1.79 | 0.63 | 0.64 | 18.83 | 1.55 | 0.69 |
| tlsvp_fest300 | handle_leads | 213 | 0.09 | 3.77 | 1.39 | 1.27 | 18.86 | 1.32 | 0.33 |
| tlsvp_fest300 | mehrere | 1005 | 0.42 | 5.2 | 0.67 | 0.68 | 18.66 | – | 2.18 |
| tlsvh_fest300 | keine | 1851 | 0.77 | -0.14 | 0.34 | 0.36 | 18.7 | – | -0.11 |
| tlsvh_fest300 | angebot_nachlegen | 485 | 0.2 | 2.74 | 0.92 | 1.11 | 18.32 | 1.55 | 0.55 |
| tlsvh_fest300 | annehmen_bei_konzept | 16 | 0.01 | 1.22 | 3.31 | 3.25 | 13.94 | 2.79 | 0.01 |
| tlsvh_fest300 | annehmen_nach_vervollstaendigung | 13 | 0.01 | -3.28 | 4.37 | 7.41 | 22.03 | 2.4 | -0.02 |
| tlsvh_fest300 | handle_leads | 21 | 0.01 | 0.03 | 5.07 | 4.82 | 21.04 | 1.32 | 0.0 |
| tlsvh_fest300 | mehrere | 14 | 0.01 | 6.42 | 4.89 | 6.72 | 17.02 | – | 0.04 |
| olsv_fest300 | keine | 2383 | 0.99 | 0.12 | 0.27 | 0.17 | 18.64 | – | 0.12 |
| olsv_fest300 | angebot_nachlegen | 3 | 0.0 | 9.02 | 8.26 | 8.26 | 13.14 | 1.55 | 0.01 |
| olsv_fest300 | annehmen_bei_konzept | 2 | 0.0 | 12.82 | 18.79 | 18.79 | 27.96 | 2.79 | 0.01 |
| olsv_fest300 | annehmen_nach_vervollstaendigung | 11 | 0.0 | 7.29 | 4.8 | 4.29 | 13.76 | 2.4 | 0.03 |
| olsv_fest300 | handle_leads | 1 | 0.0 | 0.42 | – | – | 31.47 | 1.32 | 0.0 |
| plsv_fest300 | keine | 1249 | 0.52 | -0.15 | 0.39 | 0.25 | 18.43 | – | -0.08 |
| plsv_fest300 | angebot_nachlegen | 967 | 0.4 | 2.3 | 0.63 | 0.41 | 18.89 | 1.55 | 0.93 |
| plsv_fest300 | annehmen_bei_konzept | 3 | 0.0 | -15.1 | 6.94 | 6.94 | 26.87 | 2.79 | -0.02 |
| plsv_fest300 | handle_leads | 91 | 0.04 | 4.13 | 1.87 | 1.74 | 16.38 | 1.32 | 0.16 |
| plsv_fest300 | mehrere | 90 | 0.04 | 4.13 | 2.22 | 2.2 | 20.44 | – | 0.16 |
| olsvcf_fest300 | keine | 2400 | 1.0 | -0.01 | 0.19 | 0.13 | 18.62 | – | -0.01 |

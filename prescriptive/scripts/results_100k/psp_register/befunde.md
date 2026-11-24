# Befunde zu den BPIC17-PSP-Konfigurationen (von Hand geführt, neueste oben)

Zahlen stehen in `register.md` / `register.csv` (erzeugt von `register_erstellen.py`). Hier stehen nur Deutungen,
Schlussfolgerungen und offene Fragen, jeweils mit Datum. Bei jedem neuen Ergebnis: Register neu erzeugen und hier einen
Eintrag ergänzen. Feste Regeln für PSP-Änderungen: Loss Attenuation, Dropout-Maske und im Decoder gezogene Suffixe mit
Event und Zeit je Schritt bleiben; Bewertung über einen Restzeitkopf ist verboten.

## Aktueller Stand: beste Konfiguration

- **Stand 2026-11-20 01:45 (Aufbau F, korrigierter Simulator, 12 × 200): bestes PSP online ist der T-Learner mit Suffix-Verlust
  + Kalibrierung je (Punkt, Option) und Zufallszahlen je Ereignis (tlsvk_fest300): +2,61 ± 0,47 gegen die Bank, Regel +2,63
  (Abstand 0,02 ± 0,29) – gleichauf mit der Regel.** Abweichungen 0 / 6 / 1 / 1 %. Vorher tlsv +2,23, tlcrn +2,29, (c) 300 +1,97
  (Einträge 01:45, 00:55, 20:10). Vorbehalt: Kalibrierung in der jetzigen Form schrumpft fallabhängige Unterschiede (Eintrag 01:45).
- **Stand 2026-11-19 20:10 (Aufbau F, korrigierter Simulator, 12 × 200): bestes PSP online ist der T-Learner mit Suffix-Verlust
  und Zufallszahlen je Ereignis (tlsv_fest300): +2,23 ± 0,50 gegen die Bank, bereinigt ≈ +2,2; Regel +2,63 (Abstand 0,40 ± 0,32, n.s.).**
  tlcrn (ohne Suffix-Verlust) +2,29, bereinigt ≈ +2,0; (c) 300 +1,97. Fehler an nachlegen 34 → 10 %, Rest-Rückstand jetzt aus konzept /
  nach V. (Einträge 20:10, 18:30, 12:55).
- **Offline-Entscheidungssignal: (c)** (`real_alle_s60_normal`): AIPW-Vorzeichen 84/96/80/80 %, kleinstes Rauschen
  (gepaarter SE 1,29 Tage, Optionskorrelation 0,64), unter den fertigen Modellen an 4 von 5 Gruppen am nächsten an der Simulator-Wahrheit.
- **Online:** im älteren Simulator (c) +1,84 Tage gegen die Bank (10 Seeds); im korrigierten Simulator (3 × 200) (c) adaptiv
  ohne Rückfall +0,88 und D ohne Rückfall +0,82, beide innerhalb des Rauschens (SE 0,5–0,9).
- **Kein PSP erreicht Causal Forest / X-Learner / feste AIPW-Regel** (+2,85 / +2,71 / +2,97 im älteren Simulator; Forest
  +2,62 im korrigierten). Grund: kaum fallindividuelle Effekte (Screening: kein Policy-Gewinn an allen vier Punkten), die
  Verfahren gewinnen über einen präzisen Mittelwert je Punkt, PSP verliert über 100 verrauschte Ziehungen je Entscheidung
  plus fehlendes Langzeitgedächtnis.
- **Suffixgüte** (Abläufe) ist mit Teacher Forcing 1 am besten (DLS 0,39–0,40), das verbessert die Entscheidungen aber nicht.
- **Mehr Ziehungen helfen (c) deutlich**: mit 300 statt 100 Ziehungen +2,45 Tage (5 fertige Seeds, gerundete Werte) gegen
  Causal Forest +2,69 auf denselben Seeds; mit 100 Ziehungen +1,84. Für OE bringen 300 Ziehungen nur +1,12 (Forest +2,94).
  → Für die beste PSP-Konfiguration ist die Ziehungszahl ein Hebel mindestens so groß wie die Architektur.
  Gepaart auf denselben Seeds 0–4 (nachgerechnet 2026-11-17; 300er-Werte nur auf 0,1 Tage gerundet im Log, Lauf brach nach 5 Seeds ab):
  (c) 300 minus (c) 100 +0,72 ± 0,26 Tage (4 von 5 Seeds besser); Causal Forest minus (c) 300 +0,24 ± 0,13; Regel minus (c) 300 +0,39 ± 0,17.
  Mittel Seeds 0–4: Regel 2,83, Forest 2,69, X-Learner 2,46, (c) 300 2,45, XGB 1,92, (c) 100 1,73.

## Nächste Schritte (Empfehlung 2026-11-16, nach erwartetem Outcome-Gewinn je Aufwand)

Woher die Lücke von (c) zur Regel (−1,14 Tage, 10 Seeds) kommt: (1) Rauschen – das Signal von (c) stimmt, wo es sicher ist
(97–100 % AIPW-Vorzeichen), ist aber bei 100 Ziehungen nur in 15–32 % der Entscheidungen sicher; 300 Ziehungen schlossen den Abstand
zum Forest weitgehend (+2,45 vs. +2,69). (2) Zu kleine Effekte – Konzept nur 58 % der Simulator-Wahrheit; der Endzustand-Kanal
(Option ändert, wie der Fall endet), der im Simulator 15–33 % der Effekte trägt (Konzept 0,42, nach V. 0,29, Nachlegen 0,70 Tage),
fehlt in allen Modellen (≈ 0), weil 47 % der (c)-Ziehungen gar keinen Endzustand erreichen (A_Pending 15 % statt ~55 %).
(3) Nachlegen ab dem 2. Besuch falsches Vorzeichen (−1,9 statt +0,4) → zu viele Angebote (2,3–2,4 je Fall statt 1,3).

1. **Rauschen senken bei (c)** (kein Training nötig): (a) offline aus `psp_rauschen_*/entscheidungen.csv` durchrechnen, was
   mehr Ziehungen (Poolen je Punkt vom Nutzer gestrichen) an Vorzeichen bringen (Minuten); (b) Kopplung der Optionen verbessern
   (gemeinsames Gumbel-Rauschen je Schritt und Token für die Eventwahl, gemeinsames Normalrauschen je Schritt für die Zeiten),
   messen mit psp_rauschen (Optionskorrelation 0,64 → ?, gepaarter SE); (c) online (c) adaptiv bis |z| > 2 (max. 300–500) im
   Standardaufbau, 10 Seeds gepaart mit Bank/Regel/Forest.
2. **Endzustände in den Ziehungen reparieren** (die Ziehgüte, die für das Outcome zählt): erst Ursache der 47 % ohne Ende
   bestimmen (EOS zu früh / Schrittgrenze 60 / Schleifen; aus `psp_zerlegung/ziehungen.csv.gz`), dann gezielt beheben (z. B.
   EOS beim Ziehen erst nach einem Endzustand zulassen, Wahrscheinlichkeiten des Netzes renormiert; oder längerer Decoder);
   prüfen mit Zerlegung (Endzustand-Anteile, Endzustand-Kanal), psp_rauschen, punkte_alle. TF1 zeigt, dass realistische Enden
   allein nicht reichen (Kopplung bricht dort ein) → mit Punkt 1 (b) kombinieren.
3. **Konzept: Dauer-Anteil verfolgen** – Simulator 2,36 Tage aus der Dauer, (c) 1,48: Abschnitt für Abschnitt gegen die
   Simulator-Wahrheit wie bei leads_meilensteine (≈ 30 min).
4. **ZZ abschließen** (Ep. 9 / final), danach Ablation „nur Zähler“ ohne Zeit je Event, falls das Gedächtnis hält, aber die
   Streuung (SD 26 Tage) bleibt – Zeit je Event ist der wahrscheinlichste Grund der Streuung.
5. **Simulator neu vermessen** (KPIs und Screening-Validierung sind vom 06.11., vor gelernter Zuteilung und korrigierter
   Optionsprüfung), damit die Simulator-Spalten aktuell sind.
6. Später/niedriger: first_only beim Nachlegen (bei s40 online ohne Gewinn: 1,30 vs. 1,39), Simulator-Verbesserungen.

Nicht weiter verfolgen: Varianten nur für bessere Suffixähnlichkeit (TF1: beste DLS, schwächstes Signal); Modellauswahl über den
Validierungsverlust; absolute Zeitgenauigkeit an Wahlpunkten (Grenze: naiv 10,3–10,9 Tage MAE, selbst mit bekanntem Endzustand
7,1–7,7; alle Modelle ≈ naiv); 3 × 200-Onlineläufe für Unterschiede < 1,5 Tage.

**Simulator als Ursache?** Für die Lücke von PSP zur Regel nein: Forest, X-Learner und Regel (alle aus dem echten Log) erreichen im
selben Simulator +2,7 bis +3,0 Tage; XGB aus dem echten Log +5,2 vs. PSP +0,6 (06.11.). Schwächen des Simulators (Abschnitt
„Simulator“ in register.md): Effekte nur ~64 % der echten Größe bei Konzept, nach Vervollständigung, Leads (z 2–5; Nachlegen
gleich groß); Zustände an Konzept/Nachlegen/Leads von echten unterscheidbar (AUC 0,76–0,80, v. a. Fallalter); nur 47 % der
simulierten Fälle haben eine echte Variante; Wartezeit auf den Start 70 % zu kurz; Mitarbeitereffekte nicht nachgebildet. Folge:
absolute Online-Gewinne sind kleiner als in der Wirklichkeit, die Rangfolge der Verfahren bleibt fair, Mitarbeiterpunkte sind
online nicht aussagekräftig.

## 2026-11-16: Ist die Optionsprüfung des Simulators (15.11.) sinnvoll? Prüfung am echten Log

- Regel (`interventions/work_state.py`): W_-Arbeit darf nicht „start“en, wenn sie aktiv oder pausiert ist, und nur „resume“n, wenn
  sie pausiert oder unbeobachtet ist. Im echten Log verletzen das nur 1.017 von 128.227 Starts (0,8 %, alle „start“ bei aktiver
  Arbeit) und 0 von 127.160 Resumes → die Regel beschreibt die Wirklichkeit; der alte Simulator ließ unmögliche Aktionen zu.
- Wirkung an den Wahlpunkten (echter Log): nach Vervollständigung, Nachlegen, Leads 100 % beider Optionen zulässig → **die Prüfung
  kann den Unterschied beim Nachlegen nicht erklären** (O_-Events sind immer zulässig). Bei **Konzept** ist W_Complete application
  in **35 %** der Besuche schon aktiv → „W_Complete application|start“ unzulässig, der neue Simulator lässt dann nur A_Accepted zu
  (siehe Korrektur unten: das betrifft nur die Besuche, an denen der Wahlpunkt überhaupt feuert).
- Echter Log in genau diesen Zuständen: danach 66 % A_Accepted, 34 % W_Complete application|suspend.
- **Korrektur (gleicher Tag, nach Blick in den Code):** Der Wahlpunkt feuert nur, wenn das Simulator-Modell eine der Optionen als
  nächstes Event zieht (`drawn in point_control_flow[1]`). Zieht es bei aktiver Arbeit „suspend“ (echt 34 %), gibt es gar keine
  Entscheidung, der Fall läuft normal weiter → die Bank wird NICHT zum Annehmen gezwungen. Betroffen sind nur Entscheidungen, bei
  denen das Modell A_Accepted (oder selten einen doppelten Start) zieht. Im echten Log sind das ≈ 30 % der Konzept-Entscheidungen
  (≈ 7.300 von 24.300) und dort wurde immer angenommen – eine echte Alternative wurde nie beobachtet (passt zur niedrigen
  Überlappung 0,56 bei Konzept). → **Die Korrektur entfernt nur eine unrealistische Option und ist sinnvoll.**
- Warum es sie braucht: Der Simulator lässt nur Übergänge zu, die im echten Log direkt aufeinander folgten (`allowed[a, b]` in
  `next_event_prediction/event_time_model.py`, nur das vorige Token). A_Concept → W_Complete application|start ist ein häufiger echter
  Übergang (wenn die Arbeit geplant ist), die Maske kennt aber den Arbeitszustand nicht; die Wahl der Policy ersetzt das gezogene
  Event ohne weitere Prüfung → im alten Simulator konnten Policy und RCT-Zufallswahl einen doppelten Start erzeugen.
- Folge: Die RCT-Wahrheit vom 14.11. (Konzept +2,79) enthält im Zufallsarm „start“ doppelte Starts in den ≈ 30 % aktiven
  Entscheidungen → Konzept-Wahrheit mit dem korrigierten Simulator neu messen. Zustandsabhängige Optionen ({A_Accepted, suspend} bei
  aktiver Arbeit) wären eine neue Intervention, keine Fehlerbehebung.
- Die Simulatoränderungen sind uncommittet und stammen aus einer anderen Sitzung → vor Änderungen mit dem Nutzer abstimmen.

## 2026-11-16: Warum (c) im korrigierten Simulator nur +0,02 statt +1,84 Tage (Indizien, nicht bewiesen)

- Nachtrag: Die Optionsprüfung betrifft nur Konzept (siehe oben), nicht das Nachlegen. Die stärkere Angebotsschleife muss also aus der
  gelernten Zuteilung, dem kleinen Laufumfang oder Zufall kommen; offen.

- Unterschiedliche Aufbauten: der 10-Seed-Lauf nutzte die Katalogkopie mit zufälliger Zuteilung und den Simulator vor dem 15.11.;
  der korrigierte Lauf gelernte Bank-Zuteilung (Standard seit 08.11.) und die Optionsprüfung vom 15.11. (nicht zulässige Optionen
  werden für Bank und Policy vor dem Ziehen entfernt, wiederholtes Öffnen gilt nicht als neue Aufgabe; uncommittete Änderungen in
  `case_engine_lifecycle_aware.py`, `interventions/__init__.py`, `resources/case_allocation.py`). Bank 21,14 statt 22,33 Tage.
- Rauschen: 3 × 200 Fälle, SE 0,74; je Seed −0,68 / −0,75 / +1,51; zwei fast gleiche (c)-Policies (fest 100 vs. adaptiv ohne)
  liegen 0,86 Tage auseinander. Der Abstand 1,84 → 0,02 ist ≈ 2,4 SE, also vermutlich nicht nur Rauschen.
- Muster: wer beim Nachlegen in die Angebotsschleife gerät, verliert im korrigierten Simulator: Causal Forest (1,27 Angebote je
  Fall, Bank 1,31) 2,85 → 2,62; XGB (2,23 Angebote) 1,92 → 1,18; (c) (2,44 Angebote) 1,84 → 0,02 bzw. 0,88 (adaptiv). Im alten
  Lauf erzeugte (c) nur +1,6 Events je Fall gegenüber der Bank (≈ +0,4–0,5 Angebote), XGB +0,45 Events; im korrigierten +1,13 bzw.
  +0,92 Angebote je Fall → die Schleife ist dort mehrfach stärker. Wahlanteile sonst gleich ((c) alt: A_Accepted 89 / 92 %,
  Create Offer 50 %, Leads withdraw 82 %; korrigiert regelkonform 90 / 88 / 52 / 82 %).
- Prüfung (billig): (c) im korrigierten Simulator mit `first_only` beim Nachlegen (spätere Besuche folgen der Bank); springt der
  Gewinn deutlich, ist die Schleife bestätigt. Danach gehört das falsche Vorzeichen bei späteren Besuchen (−1,9 statt +0,4) in die
  Liste der Modellfehler mit höchster Priorität.

## 2026-11-16: Querschnitt aus dem Register (alle vorhandenen Messungen zusammengeführt)

- **Effektgrößen offline** (Optionsunterschied / AIPW-Effekt auf dem echten Log 5,37 / 2,94 / −1,46 / −2,22 Tage): (c) erreicht
  27 / 71 / 83 / 56 %, s40 45 / 89 / 102 / 36 %, alle anderen fertigen Modelle höchstens 20 % bei Konzept. Konzept wird von allen
  Modellen stark unterschätzt (auch gegen die Simulator-Wahrheit 2,79: (c) 1,61). AIPW-Effekte sind auf dem echten Log größer als im
  Simulator (Konzept 5,37 vs. 2,79).
- **Entscheidungssicherheit**: bei (c) sind mit 100 Ziehungen nur 15–32 % der Entscheidungen sicher (|z| > 2), diese aber zu
  97–100 % mit AIPW-Vorzeichen; nötige Ziehungen für |z| = 2 im Median 160–390 (s40 230–850, die übrigen meist 500–1.000). Passt zum Online-Gewinn
  mit 300 Ziehungen. → Bei (c) ist das Signal richtig und nur verrauscht; bei den übrigen Modellen fehlt eher das Signal selbst.
- **AIPW-Details**: KIs eng, Policy-Gewinn einer Einzelfall-Policy an allen vier Punkten −0,5 bis −0,7 Tage (feste Regel besser),
  BLP-p 0,05–0,50, E-Werte 1,45–2,25.

- **Zerlegung Pfad/Zeit**: In allen Modellen kommt der Optionsvorteil fast nur aus den Abständen (Pfadanteil |≤ 0,26| Tage,
  Ausnahme OE beim Nachlegen −0,51). Im echten Log hat der beobachtete Unterschied deutliche Pfadanteile (Konzept 1,41 von
  2,41 Tagen, Nachlegen −0,97 von −4,56; beobachtet, nicht kausal). → Die Modelle lassen die Option kaum andere Abläufe erzeugen.
- **Ziehungen ohne Endzustand**: (c) 47 %, (d) 39 %, OE 36 %, (c)-TF1 1 %; normale Abstände mit TF1 (Piloten) 15 %. Teacher
  Forcing 1 beendet die Fälle sauber, nimmt aber das Optionssignal (siehe TF1-Befund).
- **Diagnose an echten Entscheidungen**: Fehler der gezogenen Durchlaufzeit ≈ 10 Tage bei (c), (d), s40, 10k (≈ naive
  Vorhersage), Korrelation mit der echten Durchlaufzeit nur 0,02–0,11; Log-Abstände ((a), (b)) 21–23 Tage.
- **Online-Übersicht je Aufbau** steht jetzt vollständig in register.md / online.csv (7 Aufbauten A–G, 49 Policy-Varianten,
  inkl. ERM-Ziel und Eventkosten-Ziel der anderen Sitzung). Innerhalb jedes Aufbaus ist (c) bzw. D die beste PSP-Variante;
  OE liegt online überall darunter.
- Lücken im Register (nicht gemessen): Suffixgüte für (b) und 10k; Diagnose für OE, (e), TF1-Modelle; Zerlegung für D, (e),
  D-TF1, s40; Online für (d) 1 ms, (e) und TF1 nur 1 × 200; Simulator-Wahrheit/Gedächtnistests nur für (c), D, OE, OE-TF1, ZZ.

## 2026-11-17 03:05: ZZ Epoche 11 + Suffixgüte Ep. 9 / 11; ZZ-TF1 Epoche 1

- ZZ Ep. 11 (freier Lauf): Entscheidungsrichtung wie Ep. 9 (AIPW-Vorzeichen 74 / 96 / 82 / 78 %; Simulator richtig 80 / 92 / 72 / 15 /
  77 %), Lead-Gedächtnis 2,77 (echt 2,83), eingefügtes Angebot erstmals richtige Richtung (+0,29). Aber die Zeitskala bläht weiter
  auf: Effekte 3,5 / 8,7 / 3,8 / −7,6 / 2,8 (Wahrheit 2,8 / 2,4 / 2,1 / 0,4 / 1,3), SD der Durchlaufzeit 41 Tage, Rauschen 4,3,
  ungültige Übergänge 44 + 25 je 100. Kein Trend zur Besserung im freien Lauf.
- Suffixgüte ZZ Ep. 9 / 11: DLS 0,27 / 0,27 (Ep. 3 noch 0,34; (c) 0,25), Schritt 10 10 / 11 %, Restzeit-MAE 8,92 / 9,10 ((c) 8,13),
  Bias +3,6 / +4,2 Tage, erster Abstand 14,5 / 15,4 h ((c) 13,1) → mit dem freien Lauf verliert ZZ den Vorsprung bei den Abläufen und
  wird bei den Zeiten schlechter als (c).
- ZZ-TF1: Epoche 1 fertig 02:34 (~75 min je Epoche bei zwei Läufen auf der GPU → Ende ≈ 20 Uhr; ZZ Ende ≈ 7:15). Verluste Ep. 1
  identisch mit ZZ Ep. 1 (1,4038 / 9,4796 / 323,36) – erwartet (gleicher Seed, Ep. 1 hat in beiden TF 1) und bestätigt gleichen Aufbau.

## 2026-11-17 11:15: Lernrate? – kein Hinweis auf falsche Lernrate, aber eine Auffälligkeit bei ZZ / ZZ-TF1

- Alle Vollmodelle: AdamW, Lernrate 1e-4, Gradient-Clipping 1,0, ReduceLROnPlateau (Faktor 0,1, patience 15 → greift in 15 Epochen
  nie; die letzten Epochen laufen mit voller Lernrate). Trainingsverluste fallen glatt: (c)-TF1 −0,73 → −4,24, D-TF1 0,11 → −5,90,
  OE-TF1 0,28 → −5,08 (26 Ep.). Mit 1e-5 (alte 10k-Modelle) lernte das Netz viel langsamer (Validierung nach 100 Ep. noch fallend).
  ZZ-TF1 erreicht nach 6 Epochen die Suffixgüte von OE-TF1 nach 26 → 1e-4 lernt schnell genug.
- Auffällig: Trainingsverlust (attenuiert) von **ZZ-TF1 fällt nicht**: 1,40 / 0,86 / 1,33 / 1,59 / 1,64 / 1,59 / 1,43 / 1,44 (Ep. 1–8),
  bei (c)-TF1 im selben Zeitraum −0,73 → −3,79. Auch ZZ blieb in der TF-Phase bei 1,4–1,7 und fiel erst im freien Lauf. Also hängt es an
  Zählern und/oder Zeit je Event unter Teacher Forcing – möglich: Varianzköpfe je Event schwingen (Lernrate für die neuen Gewichte zu
  hoch) oder die Verlustzusammensetzung (GradNorm-Gewichte) ändert sich. Die Prüfmaße (Suffixgüte, Zeiten, Gültigkeit) sind trotzdem
  gut. Offen; Diagnose: Verlust je Ausgabe (Event, Fallzeit, Abstand; Mittel und Varianz) für ZZ-TF1 ep6 vs. (c)-TF1 auf echten Prefixen
  mit TF 1. Betrifft auch ZZU-TF1.
- **Diagnose 11:35 (`verlust_diagnose_20261117/verlust_je_ausgabe.py`, 300 Testfälle, Teacher Forcing 1, nächste 10 echte Events, rohe
  Verluste ohne GradNorm-Gewichte):** ZZ-TF1 ep6 lernt normal – Aktivität Kreuzentropie 0,355 / Treffer 87,6 % ((c)-TF1 0,339 / 87,7 %),
  Fallzeit attenuiert −2,28 (−2,30), Abstand attenuiert −2,16 (−2,08), MAE Abstand 0,53 Tage (0,53). Der stehende Wert im Log ist also ein
  Artefakt der Verlustzusammensetzung (GradNorm-Gewichte je Lauf verschieden; Logit-Varianz der Aktivität bei ZZ-TF1 höher, −6,6 statt −7,3),
  kein Lernproblem. Kein Neustart von ZZU-TF1 nötig; die Lernrate ist nicht das Problem.
- Nebenbefund (passt zur Deutung „Randverteilung je Position“): Bei echter Vorgeschichte sagen die frei trainierten Modelle das nächste Event
  nur zu 73 % richtig voraus ((c) 73,5 %, ZZ final 73,2 %), die TF1-Modelle zu 86–88 % (ZZ ep3 mit TF 0,73: 85 %) – im freien Lauf verlernt
  das Netz, die tatsächliche Vorgeschichte zu nutzen.
- Mögliche kleine Verbesserung unabhängig davon: Lernrate am Ende absenken (z. B. ×0,1 ab Epoche 12 oder Kosinus), da der Plateau-
  Scheduler in 15 Epochen nie greift.

## 2026-11-17 11:08: Frage „TF langsamer abfallen lassen?“ – Zwischenwerte von ZZ werden nachgemessen

- Überlegung: Der Versatz entsteht an jedem Schritt mit eigener Eingabe (Anteil 1 − TF); entscheidend ist der TF-Wert am Ende des
  Trainings, nicht die Geschwindigkeit. Endet der Plan bei 0, landet man wieder bei (c)/ZZ. Bisher monotoner Zielkonflikt: ZZ ep3 / ep7 /
  ep9 (TF 0,73 / 0,20 / 0) ungültig 7+5 / 23+15 / 42+24 je 100, Lead-Gedächtnis 0,51 / 1,49 / 3,70; Piloten mit fester Untergrenze
  0 / 0,5 / 0,8 / 1,0: Konzept-Vorzeichen 85 / 71 / 63 / 59 %, ohne Ende 40 / 40 / 34 / 24 %.
- Prüfreihe für die fehlenden ZZ-Zwischenstände ep4 / ep5 / ep6 (TF 0,60 / 0,47 / 0,33) gestartet (tmux psp_zz_ep456, ~35 min) – Frage:
  gibt es einen Wert mit wenig Ungültigem und schon brauchbarem Fernsignal? Dann wäre ein langsamer Plan mit Stopp dort ein Training.
- **Ergebnis 11:45 (ZZ ep3–ep7, TF 0,73 / 0,60 / 0,47 / 0,33 / 0,20): kein günstiger Zwischenwert.** Ungültig (zwischen Events) 7 / 10 / 13 /
  17 / 23 je 100, steigt gleichmäßig; das Entscheidungssignal ist in der Übergangsphase schlechter als an beiden Enden: Konzept 69 / 58 / 57 /
  59 / 55 % AIPW-Vorzeichen, gegen Simulator 78 / 60 / 70 / 50 / 52 %; Nachlegen 55 / 60 / 58 / 53 / 56 %; Leads 40 / 42 / 56 / 57 / 76 %;
  Lead-Gedächtnis 0,51 / 0,25 / 0,53 / 0,98 / 1,49; Nachfassen nach Extra-Angebot 0,24 / 0,18 / −0,16 / −0,29 / −1,18 (Simulator +1,92).
  Das Fernsignal kommt sprunghaft erst bei TF 0 (ep9: Konzept 82 %, Nachlegen 87 %, Gedächtnis 3,70, Nachfassen +2,37). Einschränkung: die
  Zwischenstände sind nicht bei festem TF ausgelernt; die Piloten mit fester Untergrenze zeigen aber denselben monotonen Zielkonflikt.
  → Langsamer Plan mit Stopp bei einem Zwischenwert nicht weiterverfolgen.

## 2026-11-17 10:20: Uhren umgesetzt und Training ZZU-TF1 gestartet

- Uhren (params `clocks`, Flag `--clocks`, nur mit `--counters` und stetigen Abständen): je Aktivität|Lifecycle-Token die Tage seit
  seinem letzten Auftreten, log(1 + Tage) / 4, 0 vor dem ersten Auftreten; aus den Abständen (Feature event_elapsed_time, Standardisierung
  aus dem Training), im Decoder je Schritt fortgeschrieben (Training: echte Abstände; Ziehen: gezogene). Code: `clock_days` und
  `clock_weight` im Encoder, Zustand (Zähler, Tage, aufgetreten) in `dropout_uncertainty_model.py` (`_state_of_prefix`, `_advance`,
  `_count_kwargs`: Zähler und Uhren als ein Vektor für den Decoder, count_size dort 2K), `psp_model.py` (`clocks()`, clock_stats aus dem
  StandardScaler), `train_psp_real.py --clocks`.
- Prüfungen: alte Modelle ziehen bitgleich (Referenz 24 / 24), ZZ Ep. 9 bitgleich (Schnelltest); Mechanik an einem Probemodell
  (300 Fälle, 1 Epoche, gelöscht): 10 / 10 OK (`zaehler_uhren_tf1_20261117/uhren_pruefen.py`).
- Training 10:19: `data/bpic17_psp_real_zaehler_uhren_tf1.sh`, TAG `real_alle_s60_normal_zaehler_uhren_tf1` = (c)-Einstellungen +
  Zähler + Uhren + Zeit je Event + Teacher Forcing 1; tmux psp_zzu / psp_zzu_begleiter (Prüfreihe nach Ep. 3 / 6 / 9 / 12 / final) /
  psp_zzu_sg (Suffixgüte); Register-Beschriftung „ZZU-TF1 <Stand>“. Parallel zu ZZ-TF1 auf der GPU.
- Erfolgskriterien zusätzlich zu ZZ-TF1: Abstand vor A_Cancelled ≈ 21 Tage (echt 20,7; ZZ-TF1 14,0), Restzeit storniert ≈ 28 Tage,
  Nachlegen und Leads deutlich über 50 %, Lead-Gedächtnis und Nachfassen nach Extra-Angebot in richtiger Größe.

## 2026-11-18 20:05: RCT-Basis Ep. 5 geprüft – Frist steckt in den Trainingsdaten, wird aber nicht gelernt; Fernpunkte Richtung besser, Größe weit zu klein

Prüfreihe „ZZ-TF1 (RCT) Ep. 5“ (Training auf 32.000 randomisierten Simulatorfällen, Prüfung auf echten Testprefixen bzw. Simulatorzuständen;
Vergleich ZZ-TF1 ep6 / final vom echten Log; Wahrheit = Simulator):
- **Storno-Frist:** in den RCT-Trainingsdaten selbst Median 30,7 d nach letztem O_Sent, 67 % in 29–32 d (echter Log 30,7 / 73 %). Das
  RCT-Modell zieht Median 11,2 d, 3,8 % im Fenster – wie alle Modelle vom echten Log. → Die Frist fehlt nicht wegen Confounding oder Daten,
  das Netz lernt sie aus vorhandenen Daten nicht (passt zur Zeitkopf-Sonde 16:50: Varianz schluckt den Fehler).
- Gegen Simulator (Effekt / richtig): Konzept +1,01 / 73 % (ep6 1,75 / 87 %, final 1,36 / 80 %; Wahrheit 2,79), nach V. +1,69 / 78 %
  (3,43 / 97 %; 2,29 / 77 %; 2,40), **Nachlegen 1. Besuch +0,38 / 60 %** (−0,27 / 43 %; 0,12 / 52 %; 2,11), später +1,84 / 77 % (0,42),
  **Leads +0,22 / 58 %** (−0,96 / 27 %; −0,03 / 52 %; 1,32). Richtung an den Fernpunkten besser als ZZ-TF1, an den Nahpunkten schlechter;
  die Größe der Fernwirkungen bleibt weit unter der Wahrheit (Nachlegen 18 %, Leads 17 %). 60 Zustände je Punkt → ±6 Prozentpunkte.
- AIPW-Vorzeichen am echten Log 68 / 80 / 56 / 56 % (ZZ-TF1 final 74 / 96 / 49 / 51) – am echten Log gemessen, für ein Simulatormodell nur
  eingeschränkt aussagekräftig (10 % unbekannte Mitarbeiter, s. u.).
- Eingefügtes Angebot +0,63 d Restzeit (richtiges Vorzeichen wie (c) +0,51; ZZ-TF1 −0,01); nach Extra-Angebot versendet 0,77 (Simulator 1,07),
  Nachfassen 0,43 (1,92); Lead-Gedächtnis 0,23. Gültig 0,38 + 0,31 je 100. Rauschen SE 1,88, Kopplung 0,21 (ZZ-TF1 1,85 / 0,16).
- → Zwischenfazit: unverzerrte Daten aus der Bewertungsumgebung helfen bei der Richtung an Nachlegen/Leads etwas, lösen aber das
  Hauptproblem (lange Wartezeiten, Fristen) nicht. Epoche 5 von 15, ZZ-TF1 bewegte sich zwischen ep6 und final noch stark.

## 2026-11-18 19:50: RCT-Lauf Zwischenstand – Basis Epoche 5/15; Prüfreihe auf Checkpoint Ep. 5 gestartet; Vokabular-Lücke

- Basis „ZZ-TF1 (RCT)“ (32.000 Simulatorfälle, 1,41 Mio. Events): Trainingsverlust (attenuiert) −1,46 / −3,11 / −3,28 / −3,52 / −3,73, Standard-
  Validierungsverlust 9,52 / 10,45 / 11,21 / 11,59 / 11,89 (Ep. 1–5) – steigt wie bei ZZ-TF1 auf dem echten Log (9,48 → 12,87), dort am 17.11.
  als Artefakt der Verlustzusammensetzung erkannt (Verlust je Ausgabe normal). ~1,9 h je Epoche mit zwei Trainings auf der GPU.
- Checkpoint Ep. 5 als `rct_tlearner_20261118/modelle/psp_zzrct_ep5.pkl` gebaut (Testfälle wie (c), echte Prefixe), Prüfreihe + Storno laufen
  (tmux psp_rct_ep5, Register-Label „ZZ-TF1 (RCT) Ep. 5“).
- **Vokabular-Lücke bei der Offline-Prüfung auf echten Prefixen:** der Simulator nutzt nur 106 der 149 Mitarbeiter; 10,2 % der Events in
  den echten Testfällen haben einen dem RCT-Modell unbekannten Mitarbeiter (2.440 von 6.301 Testfällen betroffen) → kodiert als 0 (Padding).
  Event-Typ fehlt nur W_Personal Loan collection|resume (7 Events, kein Testfall). Die Offline-Maße des RCT-Modells sind daher leicht
  verzerrt; online im Simulator kommen nur bekannte Mitarbeiter vor.

## 2026-11-24 12:05: Lernbarkeitstest mit festen Aufgabengewichten gestartet (Hebel a)

- Nutzer: „Ja mach a“. Neu: psp_model `task_weights` {"cat", "num"} (`--task-weights EVENT,ZAHLEN`): setzt die GradNorm-Gewichte des
  Trainers fest (Lernrate 0; Verhältnis bleibt nach der Normierung), Abschwächung und Dropout-Masken unverändert; ohne Schalter wie bisher.
  Sicherungen `statik_decoder_20261124/psp_model_vor_gewichte.py.bak`, `train_psp_real_vor_gewichte.py.bak`. Probelauf 2.000 Fälle:
  Gewichte [2,5, 0,25, 0,25] (Event, case_elapsed_time, event_elapsed_time), über die Schritte konstant.
- `data/bpic17_psp_lernbarkeit_gewichte.sh` (tmux psp_gewichte, Log `statik_decoder_20261124/gewichte.log`, Beginn 12:03): wie sn
  (nur Statik-Abbildung, LR 1e-2, 2 Epochen, warm von k3x4ol_konzept_bearbeiten), Varianten sw10 (10,1) und sw0 (1,0, obere Schranke);
  danach Teacher Forcing (k3x4ol, sn, sw10, sw0 → `gewichte_uebergang.md`) und gezogene Raten. Ende etwa 14:30–15:00.

## 2026-11-24 12:00: Causal Forest mit Rückwärtsinduktion – 76,4 % konform, signifikant besser als Regel, Orakel und PSP

- Nutzer: „ja mach beides“. `policies/ev_policy/rueckwaerts_forest.py` (StufenPolitik, PunktForestPolicy),
  `results_100k/rueckwaerts_cf_20261124/rueckwaerts.py`, Pipeline `data/bpic17_cf_rueckwaerts.sh` (tmux psp_cf_rw, 11:35–11:53, nur CPU).
  Stufen nachlegen → nach V. → konzept → leads, je 8 × 6.000 Fälle im Simulator v3: Stufenpunkt zufällig (nur Anfragen mit ≥ 2 zulässigen
  Optionen, nachlegen 1. Besuch), spätere Punkte ihr Forest, frühere Bank; Forest je Punkt (Outcome −Abweichungen), 20–37k Proben je Stufe.
  Forests `models/cate_model_cf_konformanz_rw_{nachlegen,nachv,konzept,leads}.pkl`.
- Wirkung je Teilgruppe im zufälligen Log (Abw. Aktion − Referenz ± SE; Forest −τ): nachlegen ≥ 1,71 d weiteres Angebot −0,30 ± 0,02 (−0,25),
  0,84–1,71 d +0,17 (+0,19), < 0,84 d +1,16 (+1,16); nach V. annehmen +0,007 / +0,015 ± 0,017 (≈ 0, mit dem nachlegen-Forest danach);
  konzept annehmen ≥ 11k +0,20 ± 0,02, < 11k −0,40 ± 0,02; leads Rückruf < 11k +0,08 ± 0,02 (mit Bank danach war Zurückziehen hier besser), sonst +0,03.
- **Online (Aufbau F, 12 Seeds): cf_rueckwaerts 0,536 Abweichungen je Fall, 76,4 % konform, DLZ 19,61 d** – Regel 0,586 / 72,0 / 20,84,
  Orakel 0,577 / 73,1, ol300 0,605 / 72,8, einfacher Forest 0,595 / 71,6. Gepaart: vs Regel +4,4 ± 0,9 pp konform, −0,050 ± 0,026 Abw.,
  −1,23 ± 0,34 d DLZ; vs Orakel +3,3 ± 1,1 pp; vs ol300 +3,6 ± 1,3 pp, −0,069 ± 0,036 Abw.; vs einfacher Forest +4,8 ± 1,1 pp.
  → erreicht die beste Kombination (≈ 76,3 %, wahrheit_f_regel) und ist dabei 1,2 Tage schneller als die Regel.
- Wahlen je Teilgruppe (`rollout_cf_20261123/forest_wahlen.md`, Anteil andere Option): leads < 11k 0,001 (einfacher Forest 1,00), konzept ≥ 11k
  0,67 (0,30), nachlegen ≥ 1,71 0,88; Anteil Entscheidungen wie die Kombination 0,84 (Regel 0,77, ol300 0,72, einfacher Forest 0,62).
  Genau die Überlappung (leads zurückziehen umgeht nur den schlechten Bank-Entscheid an konzept) ist behoben.

## 2026-11-24 11:55: Gradienten-Zerlegung – der Zeitverlust überdeckt den Effekt-Übergang; Abschwächung dämpft nicht

- `statik_decoder_20261124/gradienten.py` (Nutzer: „ja mach beides“): Datenweg von PSPModel.fit mit k3x4ol_konzept_bearbeiten + static_in_decoder
  (Abbildung 0), statt Optimierung Gradienten auf 20 Batches (5.120 Proben, Teacher Forcing 1, train-Modus, GradNorm-Gewichte 1). `gradienten.md`.
- Fokus-Schritte (Ziel direkt nach dem ersten A_Complete) = 3,0 % der Token; P(AC) im Training 0,163 / 0,128 (< 11k / ≥ 11k; wahr 0,296 / 0).
  Möglicher Gewinn mit der Gruppenrate 0,10 / 0,14 nats je Fokus-Token = 0,0037 nats je Token = **≈ 1 % des Event-Verlusts** (0,384).
- **Abschwächung ist nicht der Grund:** gelernte Logit-Varianz an den Fokus-Schritten am unteren Anschlag (logvar −6, σ 0,05), Gradient mit
  Rauschen = ohne (Verhältnis 1,00, Kosinus 1,00).
- **Gradient an der Betragsspalte der Abbildung:** gesamt |g| 1,49, davon case_elapsed_time 1,38 (Kosinus zum Gesamtgradienten 0,95),
  event_elapsed_time 0,36, übrige Event-Token 0,14, Fokus 0,19 (Kosinus zum Gesamtgradienten 0,08). Ganze Abbildung: Zeit 10,6 gegen Fokus 0,21.
  Projektion auf die Fokus-Richtung: Fokus +1,57, case_elapsed_time **−1,45** (zieht entgegen), event_elapsed_time +0,53, übrige Events +0,36.
  → Der kumulierte Zeitverlust bestimmt, wohin die Betrags-Gewichte gehen (Adam folgt ihm), und arbeitet dabei sogar gegen den Übergang.
- Hebel (erlaubt, brauchen OK): Gewichtung Event- gegen Zeitverlust (feste Aufgabengewichte statt GradNorm-Start 1), oder ein Statik-Pfad,
  den nur der Event-Verlust trainiert (Zeitverlust erreicht ihn nicht); Test wie sn (2 Epochen) und Teacher Forcing.

## 2026-11-24 11:20: Lernbarkeitstest abgeschlossen – auch mit trainierbarem Event-Kopf (sno) kein betragsabhängiger Übergang

- `k3x4sno_konzept_bearbeiten` (Statik-Abbildung LR 1e-2 + Event-Kopf des Decoders 1e-3, Rest eingefroren, 2 Epochen): Val-Verlust
  16,31 → 16,51; Abbildung Norm 0,81, Betragsspalte 0,40 (größte).
- Teacher Forcing (`statik_decoder_20261124/lernbarkeit_uebergang.md`): P(A_Complete direkt nach dem ersten) < 11k / ≥ 11k:
  k3x4ol 0,109 / 0,107, sn 0,117 / 0,120, **sno 0,118 / 0,109** (Wahrheit 0,275 / 0). W_Validate fehlt nach annehmen: unverändert
  0,369 / 0,361 (annehmen-Netz nicht neu trainiert; wahr 0,267 / 0,708).
- Gezogene Raten (`kuenstliche_effekte_20261120/diagnose/konzept_effekt_gezogen_sn/_sno.md`): doppeltes A_Complete sn 10,8 / 9,3 %,
  sno 10,9 / 8,4 % (k3x4ol 10,1 / 8,1; Log 29,9 / 0).
- Kontrolle, ob der Betrag ankommt (Hülle sno, 30 Fälle): die statischen Eingaben des Encoders tragen den richtig skalierten Betrag
  (6.000 → −0,931, 17.700 → 0,251, 50.000 → 3,514 = (b − 15.217) / 9.898), und der Hook gibt sie an die Abbildung weiter.
  (Beim Mitschneiden über encoder.forward liegt kept_static einen Aufruf zurück, weil der Hook erst nach forward läuft – kein Modellfehler.)
- → Eingaben und Weg stimmen; selbst ein direkter linearer Pfad vom Betrag in jeden Decoder-Schritt plus trainierbarer Kopf trennt die
  Gruppen nicht. Das Trainingsziel drückt diesen Übergang offenbar nicht: ein Schritt von ~20 im Suffix, ~0,1 nats Gewinn je
  betroffenem Fall, neben den Zeit- und übrigen Event-Verlusten (Abschwächung + GradNorm). Nächste Prüfung (braucht OK): Gradienten-
  Zerlegung ohne Training (Anteil dieses Schritts am Gradienten der Abbildung) bzw. Kurztraining nur mit dem Event-Verlust als Diagnose.

## 2026-11-24 10:30: Reiner Causal-Forest-Arm bei der Konformität (cf_konformanz) – Regel-Niveau, aber die Teilgruppen richtig

- Nutzer: „teste parallel mal, wie gut der Causal Forest bei der Konformitätsaufgabe performt“. Arm `cf_konformanz` in
  `kuenstliche_effekte_20261120/online_konformanz.py`: nur der Conformance-Forest (`models/cate_model_cf_konformanz_simv3.pkl`, Outcome
  −Abweichungen, trainiert auf log_effekte_v3 **1-fach**, 32k Fälle) entscheidet (CATEPolicy, Schwelle 0, first_only nachlegen); Aufbau F,
  12 Seeds, 2 Minuten.
- Ergebnis: **0,595 Abweichungen je Fall, 71,6 % konform** (Bank 0,666 / 65,4; Regel 0,586 / 72,0; Orakel 0,577 / 73,1; ol300 0,605 / 72,8).
  Gepaart: vs Regel −0,4 ± 0,6 pp konform (n.s.), vs ol300 −1,2 ± 1,0 pp (n.s.; Abweichungen 0,010 ± 0,020 weniger als ol300), vs Orakel
  −1,5 ± 1,2 pp (n.s.); Durchlaufzeit 0,33 ± 0,23 d kürzer als Regel (n.s.).
- Wahlen je Teilgruppe (`rollout_cf_20261123/forest_wahlen.md`, Anteil andere Option, Forest / Orakel / ol300): nachlegen ≥ 1,71 d
  0,78 / 0,81 / 0,67, übrige nachlegen 0,00–0,003 / 0 / 0,002–0,07; konzept < 11k 0,006 / 0 / 0,25; konzept ≥ 11k 0,30 / 0,81 / 0,34;
  leads < 11k 1,00 / 1,00 / 0,33, übrige leads 0,015 / 0 / 0,07; nach V. 0,12–0,17 / 0 / 0,38–0,40.
  → **Der Forest lernt die Heterogenität** (trennt die Teilgruppen fast wie das Orakel, Ausnahme konzept ≥ 11k mit kleinem Effekt), das PSP
  nicht (gleichmäßig 7–40 % andere Option). Online landen trotzdem beide auf Regel-Niveau: der Forest schätzt wie das Orakel die Wirkung
  **mit der Bank an den übrigen Punkten** (z. B. leads < 11k zurückziehen hilft nur, weil es den schlechten Bank-Entscheid an konzept
  umgeht), und die Punkt-Effekte überlappen (11-23, wahrheit_f / wahrheit_f_regel). Die beste Kombination mit der Regel sonst (≈ 76,3 %)
  verlangt eine Politik, die ihre eigenen späteren Entscheidungen berücksichtigt; weder Forest noch PSP tun das. Anteil Wahlen wie die
  Kombination: Regel 0,77, ol300 0,72, Orakel 0,64, Forest 0,62.
- Folgerung: Online-Konformität allein trennt die Lerner hier nicht; die Wahl je Teilgruppe zeigt, dass der Forest die fallweisen Effekte
  trifft und das PSP nicht.

## 2026-11-24 10:05: Lernbarkeitstest, Variante sn (nur Statik-Abbildung) – keine Trennung der Gruppen

- `k3x4sn_konzept_bearbeiten` (warm von k3x4ol, Statik-Abbildung mit 0 begonnen, nur sie trainiert, LR 1e-2, 2 Epochen, 39.154 Proben):
  Trainingsverlust −3,70 → −3,79, Val-Verlust 16,50 → 16,54. Abbildung wächst auf Norm 0,69 (k3x4sd nach 5 Epochen mit 1e-3: 0,23);
  Betragsspalte ist mit 0,30 die größte (übrige 0,17–0,27).
- Teacher Forcing (`statik_decoder_20261124/lernbarkeit_uebergang_sn_vorab.md`, 120 Fälle je Gruppe, 10 Ziehungen):
  P(A_Complete direkt nach dem ersten) **0,117 / 0,120** (< 11k / ≥ 11k; k3x4ol 0,109 / 0,107; Wahrheit 0,275 / 0); gezogen 12,1 / 11,6 %.
  → Beide Gruppen steigen um ~0,01, kein Unterschied. Selbst ein eigener Pfad mit dem Betrag in jedem Decoder-Schritt und hoher Lernrate
  bringt den Betrag nicht in diesen Übergang; der Gradient der Abbildung dient offenbar anderen Zielen (Zeiten / übrige Schritte).
- Variante sno (Abbildung + Event-Kopf) läuft (Beginn 09:46); danach volle Prüfung `lernbarkeit_uebergang.md` und gezogene Raten.

## 2026-11-24 08:50: Simulator-Basis lernt den betragsabhängigen Übergang auch nicht; Pipeline nach Filter-Fehler fortgesetzt; Lernbarkeitstest läuft

- Simulator-Basis `simv3_s60_zaehler_zeit_tf1` (von Grund auf, alle Prefixe, log_effekte_v3, 10 Epochen, Val-Verlust 12,27) fertig 03:50.
  Die Pipeline (teil2) brach 03:51 im Log-Filter ab: `PSPModel.view` liefert CSV-Text (StringIO), der Filter las ihn als Tabelle.
  Korrigiert in `sim_basis_20261123/log_filter.py` (read_csv; Probelauf: 13 von 128.000 Fällen entfernt – 4 mit seltenen Events wie
  W_Personal Loan collection, 9 mit unbekannten Mitarbeitern), fortgesetzt ab Schritt 2 mit `data/bpic17_psp_konformanz_sim_basis_teil3.sh`
  (tmux psp_simbasis3, 08:35; Schritte 3–4 unverändert aus teil2: 8 Netze simv3ol, Hülle, Diagnose, Online konformanz_sim).
- Teacher Forcing (`statik_decoder_20261124/uebergang_lehrer_simbasis.md`) auf der Simulator-Basis: P(A_Complete direkt nach dem ersten)
  0,113 / 0,108 (< 11k / ≥ 11k; Wahrheit 0,275 / 0), W_Validate fehlt nach annehmen 0,25 / 0,26 (wahr 0,27 / 0,71). → Auch von Grund auf
  auf Simulator-Daten mit allen Prefixen lernt das Netz den Übergang nicht betragsabhängig – **nicht der Warmstart vom echten Log**,
  sondern eine Eigenschaft von Architektur/Training. (Randrate gut getroffen; annehmen-Netz der Basis zieht 2–3 % doppelte A_Complete, wahr 0.)
- Lernbarkeitstest (Nutzer: „ja können wir machen“) läuft: `data/bpic17_psp_lernbarkeit_statik.sh` (tmux psp_lernbar, Log
  `statik_decoder_20261124/lernbarkeit.log`), psp_model `static_only` / `static_only_output` (`--static-only`, `--static-only-output`;
  ohne Schalter unverändert; Probelauf 2.000 Fälle: nur Statik-Abbildung und Event-Kopf geändert, alles andere bitgleich zu k3x4ol).
  Netz vervollständigen warm von k3x4ol, Statik-Abbildung mit 0 begonnen, Rest eingefroren: sn = nur Abbildung (LR 1e-2), sno = Abbildung
  + Event-Kopf (1e-2 / 1e-3), je 2 Epochen; danach Teacher Forcing (lernbarkeit_uebergang.md) und gezogene Raten je Hülle.

## 2026-11-24 01:40: static_in_decoder – Tor an konzept nicht bestanden; das Netz lernt den Effekt-Übergang selbst nicht (auch mit dem Betrag in jedem Decoder-Schritt)

- Pipeline psp_statik: beide konzept-Netze k3x4sd fertig (00:54 / 01:11, 5 Epochen, Verlaufskurven wie bei k3x4ol), Zwischenhülle
  gebaut. `diagnose/konzept_effekt_gezogen_sd.md`: gezogen A_Complete doppelt 10,4 / 8,4 % (< 11k / ≥ 11k; k3x4ol 10,1 / 8,1; Log 29,9 / 0),
  W_Validate fehlt 32,4 / 33,3 % (k3x4ol 33,8 / 34,5; Log 27,6 / 68,2) → **Tor nicht bestanden** (+2,0 / +0,9 Punkte statt ≥ 10),
  Pipeline wie vorgesehen beendet; keine weiteren Netze, kein Online-Lauf. Die Sonde (sonde_betrag_sd) brach ab: Hook-Nummern-Kollision –
  die gepickelte Hülle bringt den Hook `_keep_static` mit Nummer 0 mit, der neue Hook der Sonde bekommt im neuen Prozess wieder Nummer 0
  und überschreibt ihn (Fehler der Sonde, nicht des Modells; ohne fremde Hooks läuft `_keep_static` bei jedem Encoder-Aufruf, die
  Zwischenhülle hätte sonst wegen fehlender Merkmale abgebrochen).
- Gelernte Gewichte (`psp_model_durchlaufzeit_k3x4sd_konzept_*.pkl`, letzte Epoche): statische Abbildung Norm 0,23 / 0,25 gegenüber 5,6
  der Event-Eingangsprojektion, alle 11 Spalten gleich klein (0,04–0,11, die Betragsspalte 0,08) – Größe einer Zufallswanderung, kein
  gerichtetes Signal. Betrag kodiert per StandardScaler (Mittel 15.217, SD 9.898): 11.000 € → −0,43, 10.000 → −0,53, 11.500 → −0,38; die
  Betragsspalte verschiebt die Decoder-Eingabe zwischen 5.000 und 20.000 € nur um 0,12.
- **Teacher Forcing** (`statik_decoder_20261124/uebergang_lehrer.py`/.md, 5 min, keine Hooks): an 4 × 120 Bank-Zuständen die wahre
  Fortsetzung nach der Option bis einschließlich zum ersten A_Complete erzwungen. P(nächstes = A_Complete) aus den mittleren Logits:
  k3x4ol 0,109 / 0,107, k3x4sd 0,102 / 0,098 (< 11k / ≥ 11k); Wahrheit im Bank-Lauf 0,275 / 0. Gezogen genauso (0,11 / 0,10). Annehmen-Netz,
  freier Rest nach erzwungenem A_Complete: W_Validate fehlt 0,37 / 0,36 (k3x4ol), 0,34 / 0,35 (k3x4sd); wahr 0,27 / 0,71.
  → Der Betrag geht nicht erst beim Ziehen über viele Schritte verloren: **das Netz hat den Übergang nie betragsabhängig gelernt**, nur die
  Randrate. Die Sonde von 23:30 (Verblassen) war eine Folge, nicht die Ursache.
- Trainingsdaten im Netz geprüft (encoder_decoder.event_log von k3x4sd_konzept_bearbeiten, 83.201 Fälle): nach „vervollständigen“ folgt dem
  ersten A_Complete direkt ein zweiter (1 ms) bei < 11k in 29,8 %, bei ≥ 11k in 0,0 % (18.830 / 20.162 Fälle) → Daten und Eingang stimmen.
- Folgerung: ein Optimierungsproblem (warm von Netzen, die den Betrag ignorieren; LR 1e-4; ein Übergang je Probe unter ~60 Schritten und
  den Zeitverlusten, GradNorm-Gewichtung unbekannt – nicht protokolliert), keine fehlende Information. Vorschlag (braucht OK): Lernbarkeits-
  test – nur die statische Abbildung (ggf. + Ausgabeschicht) trainieren, Rest eingefroren, hohe LR, 1–2 Epochen am vervollständigen-Netz,
  dann uebergang_lehrer.py; trennt „Signal blockiert“ von „zu schwach / zu langsam“.

## 2026-11-23 23:45: static_in_decoder gebaut, geprüft, Training gestartet (Nutzer: „ja“)

- psp_model `static_in_decoder` (Schalter in params, `--static-in-decoder` in train_psp_real.py): Forward-Hook `_keep_static` am Encoder
  hält die kodierten statischen Merkmale (11 Werte: Einbettungen LoanGoal / ApplicationType + Betrag) je Batch fest, `_StaticProjection`
  addiert sie linear abgebildet (128 × 11, mit 0 begonnen) in jedem Decoder-Schritt auf die Eingangsprojektion (vor LayerNorm), wie
  decoder_context. Fehlen sie oder passt der Batch nicht → Abbruch. Eigene Lernrate `static_learning_rate` (`--static-learning-rate`).
  `_load_net` erkennt die Abbildung im Checkpoint. Sicherung der Dateien vorher: `statik_decoder_20261124/*.bak`.
- Prüfungen: test_rollout.py vergleich (Ziehungen bisheriger Modelle wie vor allen Änderungen) **bitgenau gleich**; test_statik.py: k3x4ol
  mit Null-Gewichten bitgenau gleich, mit Gewichten ≠ 0 verschieden, Speichern/Laden gleich → OK. Probelauf (2.000 Fälle, 1 Epoche,
  warm von k3x4ol_konzept_bearbeiten): Training, Checkpoint-Laden, Ziehen ok; neue Gewichte mit LR 1e-4 nur |w| ≤ 4e-4, mit 1e-3 ≤ 4e-3.
- Pipeline `data/bpic17_psp_statik_decoder.sh` (tmux psp_statik, Fortschritt `statik_decoder_20261124/fortschritt.log`): 8 Netze k3x4sd
  warm von ihren k3x4ol-Netzen, Rezept wie k3x4ol, LR 1e-4 / Statik 1e-3, 5 Epochen; zuerst die 2 konzept-Netze, dann Prüfung an konzept
  (konzept_effekt_gezogen_sd, sonde_betrag_sd) mit Tor ≥ 10 Punkte Gruppenunterschied (Wahrheit 30 / 41, k3x4ol 2 / 1); nur bei
  bestandenem Tor die übrigen 6 Netze (ein Strang, Speicher-/Trainingsgrenze), Hülle k3x4sd, Online-Arm konformanz_sd (12 Seeds, 300
  Ziehungen, gepaart mit konformanz_ol300) nach dem Ende der Simulator-Basis-Pipeline.

## 2026-11-23 23:30: Sonde – der Betrag verblasst im Decoder, bevor die Effekt-Events gezogen werden

- `diagnose/sonde_betrag.py` (Nutzer: „ja, mach die Sonde“; k3x4ol schlank, keyed_noise; 800 frische konzept-Zustände, 400 je Gruppe,
  5 Ziehungen je Fall und Netz; Zustände h + c über Forward-Hooks, logistische Regression, 5-fach GroupKFold nach Fall; 8 min;
  `sonde_betrag.md`). AUC für „Betrag < 11k“ (0,5 = nicht ablesbar):
  - Netz „vervollständigen“: Encoder nach Option **0,69**, Decoder Schritt 0 / 2 / 4 / 6 / 10 / 20: 0,65 / 0,58 / 0,54 / **0,50** / 0,51 / 0,49.
    Erster gezogener A_Complete im Median bei Schritt 8 → dort ist der Betrag nicht mehr ablesbar.
  - Netz „annehmen“: Encoder **0,93**, Decoder Schritt 0 / 4 / 8 / 10 / 12 / 15 / 20: 0,92 / 0,90 / 0,86 / 0,74 / 0,63 / **0,51** / 0,50.
    Erster A_Complete bei Schritt 6; W_Validate application kommt danach → der Betrag verblasst ungefähr dort, wo der Effekt wirken müsste.
- Deutung: Hypothese 1 gestützt. Der Decoder trägt den Betrag nur wenige Schritte (ohne Statik im Decoder muss er im Zustand überleben);
  beim Netz „vervollständigen“ ist er schon im Encoder-Zustand schwach. Lineare Sonde = untere Grenze der enthaltenen Information, aber
  der Verlauf über die Schritte ist eindeutig. Gleiches Muster wie das Lead-Gedächtnis (16.11.).
- Abhilfe (braucht OK): statische Merkmale in jeden Decoder-Schritt (Schalter in params, neue Eingänge mit Null-Gewichten → Netz anfangs
  unverändert; 8 Netze von k3x4ol aus nachtrainieren), danach konzept_effekt_gezogen.py + sonde_betrag.py erneut, erst dann online.

## 2026-11-23 23:10: Hypothese 1 bestätigt – k3x4ol zieht beide konzept-Effekte unabhängig vom Betrag (gemittelt über die Gruppen)

- `diagnose/konzept_effekt_gezogen.py` (k3x4ol schlank, keyed_noise; 240 frische konzept-Zustände aus den Bank-Läufen, 120 je Gruppe,
  30 Ziehungen je Option, 4 min; `konzept_effekt_gezogen.md`). Anteil je Fall ab dem Wahlpunkt, < 11k / ≥ 11k:
  - Netz „vervollständigen“, A_Complete doppelt: gezogen **10,1 / 8,1 %**; Trainings-Log x4 29,9 / 0,0 %; Bank-Lauf simuliert 29,7 / 0,0 %.
  - Netz „annehmen“, W_Validate application fehlt: gezogen **33,8 / 34,5 %**; Trainings-Log 27,6 / 68,2 %; Bank-Lauf 24,5 / 70,2 %.
  - Nebenbei: „vervollständigen“, W_Validate fehlt: gezogen 16,0 / 15,0 %, Log 32,8 / 16,6 % – auch dort flach.
- → Die gezogenen Suffixe hängen nicht vom Betrag ab; beide Netze ziehen ungefähr eine über die Gruppen gemittelte Rate (doppelter
  Abschluss sogar darunter: 9 % statt gepoolt ≈ 14 %). Das erklärt die flache PSP-Differenz bei konzept (+0,02 statt +0,53 / −0,06).
- Offen: ob der Betrag schon im Encoder-Zustand fehlt oder erst im Decoder verloren geht (lineare Sonde auf den Encoder-Zustand würde das
  trennen). Mögliche Abhilfe (nicht umgesetzt, braucht OK): Statik in jeden Decoder-Schritt (neue Eingänge mit Null-Gewichten
  initialisieren, 8 Netze nachtrainieren).

## 2026-11-23 23:00: Warum entscheidet das PSP bei Conformance schlecht? – Effekt steht klar im Trainings-Log, das PSP lernt ihn nicht

- Einfacher Gruppenvergleich im Trainings-Log (log_effekte_v3, 32k Fälle, Bank-Entscheidungen, Abweichungen aus
  rollout_cf_20261123/abweichungen_je_fall.csv; andere − Regel-Option) gegen Wahrheit (wahrheit_effekte_v3) und PSP online (ol300,
  Mittel der geschätzten Differenz aus entscheidungen.csv):
  konzept < 11k **+0,54** / +0,53 / **+0,02**; konzept ≥ 11k +0,01 / −0,06 / +0,02; leads < 11k **−0,35** / −0,43 / **+0,05**;
  leads ≥ 11k +0,08 / +0,08 / +0,15; nach V. −0,02 bis −0,01 / +0,05 bis +0,09 / ≈ 0; nachlegen < 0,84 d – / +1,38 / +0,49.
- Kein Confounding: die Bank wählt fast unabhängig vom Betrag (konzept vervollständigen 62,5 % bei < 11k, 70 % bei ≥ 11k; leads
  Rückruf 20 / 20 %). Der Effekt ist also ohne Adjustierung direkt aus den Daten ablesbar – das PSP bildet ihn nicht ab.
- PSP-Differenz bei konzept ist über alle Betragsklassen flach (0,01–0,03 von 0–5k bis > 20k): keine Wechselwirkung Option × Betrag
  gelernt. Bei leads steigt sie mit dem Betrag (0,02 → 0,17), aber auf falschem Niveau.
- Niveau zu hoch: geschätzte Abweichungen der gewählten Option 0,72–1,09 je Fall, tatsächlich 0,52–0,72 (passt zu ziehfehler.md vom
  21.11.: 15–30 % der gezogenen Spuren mit nie beobachtetem Übergang, ≈ +0,35 Abweichungen je Spur).
- Architektur: statische Merkmale (Betrag, Ziel, Antragsart) gehen nur in den Encoder (an jeden Encoder-Schritt angehängt,
  dropout_uncertainty_encoder.py:211–221); der Decoder bekommt sie nicht (forward: Event, hx, condition_event, counts). Der doppelte
  A_Complete (30 % bei < 11k nach „vervollständigen“) entsteht viele Decoder-Schritte später → der Betrag muss im versteckten Zustand
  überleben; dasselbe Muster wie das fehlende Lead-Gedächtnis (16.11.).
- **Nachtrag 23:05 (Modellbezug)**: Gruppenmittel auch auf dem Trainings-Log von k3x4ol (log_effekte_v3_x4, 128k Fälle,
  `rollout_cf_20261123/gruppenmittel_log_effekte_v3_x4.md`): konzept < 11k +0,515 ± 0,012, ≥ 11k +0,04; leads < 11k −0,38, ≥ 11k +0,12;
  nach V. −0,05 / 0,00; Bank-Anteile 62 / 70 % bzw. 21 / 20 % → gleich wie auf dem 1-fach-Log. Architektur an den geladenen k3x4ol-Netzen
  geprüft (Encoder static_input_size 11, Decoder ohne Statik). Alles in diesem Eintrag bezieht sich auf k3x4ol (ol300) bzw. dessen
  Trainings-Log, **außer** den 15–30 % Spuren mit nie beobachtetem Übergang (ziehfehler.md 21.11., Modell k3) und dem Lead-Gedächtnis
  (16.11., Modelle vom echten Log).
- Hypothesen (nicht geprüft): (1) Statik nur im Encoder + seltenes fernes Event → nicht gelernt; (2) Next-Event-Verlust gewichtet ein
  Event in 30 % einer Teilgruppe kaum; (3) Ziehartefakte erhöhen Niveau und Streuung → mehr „nicht unterscheidbar“; (4) Entscheidung
  nach mittleren Abweichungen, Bewertung nach Anteil konformer Fälle.

## 2026-11-23 22:40: Rollout online fertig (konformanz_ol_ro, 12 Seeds) – schlechter als die Kontrolle: −3,0 ± 1,0 pp konform

- 12 Seeds × 200 Fälle (konform / Abweichungen je Fall / DLZ): **ol_ro 69,7 % / 0,623 / 20,70**; Kontrolle ol300 72,8 / 0,605 / 20,70;
  Regel 72,0 / 0,586; Orakel 73,1 / 0,577; ol 70,9 / 0,610; Bank 65,4 / 0,666.
- Gepaart ro − ol300 **−3,0 ± 1,0 pp** (9/12 Seeds negativ, signifikant), Abweichungen +0,018 ± 0,029 (n. s.), DLZ +0,01 ± 0,48;
  − Regel −2,2 ± 1,2; − Orakel −3,3 ± 1,0; − ol −1,1 ± 1,0; − Bank +4,3 ± 1,4. Zwischenstand 22:15 (8 Seeds, −3,9) bestätigt.
- Wahlen (`diagnose/wahl_je_gruppe.md`, 12 Seeds): leads < 11k andere 33 → 1,6 % (Verlust je Entscheidung gegen „Bank sonst“
  0,29 → 0,43); konzept < 11k andere 40 → 59 % (0,21 → 0,31); konzept ≥ 11k 43 → 59 % (leicht besser); nach V. keine 40 → 57 %, Car
  38 → 29 %; nachlegen unverändert. Verlustsumme 377 → 586. Übereinstimmung mit dem Forest sinkt sogar: 74 → **65,5 %**
  (`rollout_cf_20261123/uebereinstimmung_forest_konformanz_ol_ro.md`; leads 52 %, konzept 45 %, nach V. 51 %, nachlegen 93 %).
- Fazit: Rollout mit Forest an späteren Punkten, während online das PSP selbst später entscheidet, schadet (inkonsistente Politiken,
  s. 22:15/22:30). Bei leads < 11k entfernt sich das PSP vom Forest (Forest 100 % andere, PSP 1,6 %).

## 2026-11-23 22:30: Entscheidet das PSP wie der Forest? – nur in 74 % der Online-Zustände; der Forest selbst wählt fast wie das Orakel

- `rollout_cf_20261123/uebereinstimmung_forest.py` (7 s): an jedem Online-Zustand der Kontrolle konformanz_ol300 (5.940 erste Besuche)
  die Wahl des Conformance-Forests berechnet und mit der PSP-Wahl verglichen. Ergebnis `uebereinstimmung_forest_konformanz_ol300.md`.
- Gleiche Wahl gesamt **74 %**: nachlegen 96 %, leads 65 %, nach V. 58 %, konzept 55 %. Nach Abstand der PSP-Schätzungen (|Δ erwartete
  Abweichungen|): < 0,02 → 49 %, 0,02–0,05 → 56 %, 0,05–0,1 → 61 %, 0,1–0,2 → 67 %, ≥ 0,2 → 93 %. Die Abweichungen vom Forest sind also
  überwiegend Rauschen (kleine Abstände ≈ Münzwurf), dazu eine Verzerrung bei leads < 11k (PSP andere 33 %, Forest 100 %).
- Anteil „andere Option“ Forest / PSP / Orakel (Wahrheitstabelle Bank sonst) je Teilgruppe: konzept < 11k 0 / 40 / 0 %; konzept ≥ 11k
  38 / 44 / 81 % (Wahrheit nur −0,06); nach V. keine 13 / 41 / 0, Car 18 / 39 / 0; nachlegen < 0,84 d 0 / 0 / 0, 0,84–1,71 1 / 7 / 0,
  ≥ 1,71 99 / 71 / 81; leads < 11k 100 / 33 / 100, sonst 1,5 / 7 / 0. → **Der Forest trifft die Wahrheitstabelle in fast jeder
  Teilgruppe** (Orakel-Arm 73,1 % geprüft: wählt wie die Tabelle).
- Folgerung für den Rollout: Im Rollout entscheidet später der Forest ≈ Orakel, online entscheidet das PSP und stimmt an konzept / nach V.
  nur zu 55–58 % mit ihm überein – passt zur Erklärung des schlechteren Rollout-Arms (22:15). Damit der Rollout konsistent ist, müsste das
  PSP an den späteren Punkten ≥ ~90 % wie der Forest wählen; mit dem Rauschen (≈ 75 % der Entscheidungen unter der Rauschgrenze) ist das
  auch mit der Simulator-Basis unwahrscheinlich. Prüfbar mit demselben Skript auf konformanz_sim, sobald der Lauf da ist.
- Nicht gemessen: der Forest allein als Online-Politik (wäre nach dieser Übereinstimmung ≈ Orakel zu erwarten).

## 2026-11-23 22:15: Rollout online (konformanz_ol_ro), Zwischenstand 8 von 12 Seeds – schlechter als die Kontrolle (−3,9 ± 1,5 pp)

- Seeds 0–7 (gleiche Seeds bei allen Armen), konform: **ol_ro 68,8 %**, Kontrolle ol300 72,7, Regel 72,5, Orakel 72,6, ol 70,9.
  Gepaart ro − ol300 **−3,9 ± 1,5 pp** (6 von 8 Seeds negativ), − Regel −3,7 ± 1,3, − Orakel −3,8 ± 1,0; Abweichungen je Fall ro − ol300
  +0,012 ± 0,030 (n. s.). Vorläufig (nach dem 6-Seed-Artefakt von 18:00 Vorsicht), aber 2,7 SE.
- Wahlen ro gegen ol300 (`diagnose/wahl_je_gruppe.md`, jetzt mit x4, ol, ol300, ol_ro; Wahrheit „Bank sonst“ = wahrheit_effekte_v3):
  leads < 11k andere Option 33 → **1,5 %** (fast immer zurückziehen wie die Regel; Wahrheit Bank sonst: andere besser um 0,43, Regel sonst:
  Regel besser um 0,15); konzept < 11k andere 40 → **58 %** (Wahrheit: Regel/annehmen besser um 0,53 → schlechter); konzept ≥ 11k andere
  43 → 61 % (leicht besser, Wahrheit −0,06); nach V. keine andere 40 → 56 % (Wahrheit +0,05, leicht schlechter); nachlegen kaum verändert.
  Verlust je Entscheidung gegen „Bank sonst“: ol300 0,063, ol_ro 0,095; Hauptanteile leads < 11k und konzept < 11k.
- Deutung (Indiz, nicht bewiesen): Der Rollout bewertet eine Option so, als ob später der Forest entscheidet; online entscheidet später
  aber das PSP mit Rollout selbst, und das wählt anders (z. B. konzept < 11k nur 42 % annehmen, Forest ~80 %). Bei konzept profitiert
  der Zweig „Antrag vervollständigen“ im Rollout vom Forest an nach V./nachlegen, den es online so nicht gibt. Folgerung für echte
  Backward Induction: Rollout-Politik und ausgeführte Politik an späteren Punkten müssen gleich sein (z. B. online an späteren Punkten
  ebenfalls den Forest entscheiden lassen, oder Rollout mit der Politik, die tatsächlich danach entscheidet).
- Seeds 8–11 laufen (Ende etwa 23 Uhr).

## 2026-11-23 19:05: Kontrolle konformanz_ol300 fertig (12 Seeds) – 72,8 % konform, gleichauf mit Regel und Orakel; Kopplung + 300 Ziehungen helfen

- 12 Seeds × 200 Fälle, Mittel (konform / Abweichungen je Fall / DLZ Tage): **ol300 72,8 % / 0,605 / 20,70**; Regel 72,0 / 0,586 / 20,84;
  Orakel 73,1 / 0,577 / 20,49; x4 71,0 / 0,602 / 21,21; ol (100 Ziehungen, ohne Kopplung) 70,9 / 0,610 / 21,36; Bank 65,4 / 0,666 / 20,45.
- Paarweise ol300 − …: ol **+1,9 ± 0,9 pp** (≈ 2 SE; Kopplung + 300 Ziehungen bringen etwas), x4 +1,8 ± 1,2, Regel +0,8 ± 1,1 (n. s.),
  Orakel −0,3 ± 1,3 (n. s.), Bank +7,4 ± 1,4. Abweichungen gegen Regel +0,019 ± 0,029 (n. s.; die „signifikant schlechter“-Aussage von
  18:00 war ein Artefakt der 6 Seeds).
- Wahlen (Anteil Aktion bzw. Option): nachlegen senden 93 % (ol 89, Regel 100, Orakel 97); konzept annehmen 70 % (Regel 100, Orakel 7);
  nach V. annehmen 58 % (ol 52, Regel/Orakel 100); leads zurückziehen 81 % (ol 68, Regel 100, Orakel 53).
- Abstand zur besten Kombination gegen die Regel (≈ 76,3 %, wahrheit_f_regel) bleibt ~3,5 pp. Kontrolle für den Rollout-Arm
  konformanz_ol_ro (läuft seit 18:35, 8 Prozesse, 2 Durchgänge).

## 2026-11-23 18:00: Zwischenstand Kontrolle konformanz_ol300 (6 von 12 Seeds) – konform wie Regel, aber mehr Abweichungen je Fall

- Arm konformanz_ol300 = k3x4ol, schlank, keyed_noise, 300 Ziehungen, ohne Rollout (Kontrolle für konformanz_ol_ro). Seeds 0–5 fertig
  (je ~1,8 h), Seeds 6–11 laufen bis etwa 18:50.
- Seeds 0–5, gleiche Seeds bei allen Armen: konform ol300 **72,2 %**, Regel 71,8, Orakel 72,5, x4 71,3, ol (100 Ziehungen, ohne Kopplung) 71,2,
  Bank 65,4. Paarweise ol300 − Regel +0,5 ± 1,5 pp (n. s.), − ol +1,1 ± 1,0 pp (n. s.).
- Abweichungen je Fall dagegen: ol300 0,618, Regel 0,553, ol 0,577, x4 0,608, Bank 0,646 → ol300 − Regel **+0,064 ± 0,013 (signifikant
  schlechter)** – **FALSCH, Artefakt der 6 Seeds; mit 12 Seeds +0,019 ± 0,029, n. s., s. 19:05**, − ol +0,041 ± 0,032. Mehr konforme Fälle, aber die nicht konformen haben mehr Abweichungen.
- Vorläufig; Endergebnis mit 12 Seeds und danach der Paarvergleich mit Rollout (konformanz_ol_ro, Start nach ~18:50, 8 Prozesse, 2 Durchgänge,
  Ende etwa 23–24 Uhr).

## 2026-11-23 10:45: PSP + Causal Forest bei Unsicherheit (olsvcf) – gebaut, geprüft, Online-Test gestartet

- Nutzer: bei nicht unterscheidbaren Optionen (|Δ| < 1,96 · SE, gepaart über den Ziehungsindex) entscheidet ein Causal Forest.
  `policies/ev_policy/psp_cf_policy.py` (PSPCausalForestPolicy): dieselben Ziehungen wie PSPPolicy (outcome_samples), Δ und SE beste gegen
  zweitbeste Option; unsicher → Forest `models/cate_model_cf_durchlaufzeit_real_alle.pkl` (12.11., echter Log, kennt alle 4 Punkte und Optionen).
- Prüfung `psp_cf_20261123/test_gleichheit.py` (Seed 3, 12 Fälle): z = 0 → dieselben 33 Wahlen wie olsv; z = 1,96 → 5 von 28 Entscheidungen
  unsicher (Δ 0,4–1,5 d bei SE 0,6–0,8), der Forest wählt dort jedes Mal dieselbe Option wie das PSP (die Regel-Option).
- Online-Arm olsvcf_fest300 (Aufbau F, 12 Seeds, tmux psp_cf, `data/bpic17_psp_cf_online.sh`), gepaart mit olsv: bis zur ersten abweichenden
  Wahl identische Ziehungen. Erwartung: ≈ olsv, weil der Forest auf dem echten Log fast immer die Regel-Option wählt.

## 2026-11-23 14:35: Diagnose mit Rollout – leads < 11k kehrt sich wie erwartet um; Richtung gegen „Regel sonst“ 63 → 69 %

- `diagnose/konformanz_unsicherheit_ro.md` (k3x4ol, keyed_noise, 100 Ziehungen, gleiche 967 Zustände wie _crn), Δ je Teilgruppe ohne → mit Rollout
  (Wahrheit Bank sonst / Regel sonst): **leads < 11k −0,06 → +0,17 (−0,25 / +0,15)**; leads sonst +0,11 → +0,23 (+0,14 / +0,15); konzept ≥ 11k +0,07 → 0,00
  (−0,18 / −0,17); konzept < 11k +0,09 → +0,02 (+0,32 / +0,38, schlechter); nach V. keine −0,01 → −0,03 (+0,08 / −0,17); nach V. Car +0,00 → +0,10
  (+0,03 / −0,08, falsche Richtung); nachlegen-Gruppen fast unverändert.
- Gesamt: Vorzeichen richtig gegen Wahrheit „Regel sonst“ 63 → **69 %**, gegen „Bank sonst“ 68 → 68 %; nicht unterscheidbar 77 → 73 %.
- Deutung: Der Rollout tut, wofür er gedacht ist – bei leads < 11k schätzt das PSP jetzt den Effekt, der gilt, wenn später gut entschieden wird.
  Verzerrung bei konzept < 11k und nach V. Car bleibt bzw. verschiebt sich (Forest ≠ Regel; derselbe Netz zieht nach dem eingesetzten Event weiter).
  Online-Test konformanz_ol_ro folgt nach der Kontrolle.

## 2026-11-23 14:20: Rollout mit Causal Forest an späteren Wahlpunkten – gebaut, geprüft, Diagnose und Online gestartet

- Nutzer: „Setze eine Version auf, die mittels Causal Forest die späteren Entscheidungen in den Suffixen korrigiert (Rollout-Weg)“.
- Forest für Conformance `models/cate_model_cf_konformanz_simv3.pkl` (`rollout_cf_20261123/train_cf_konformanz.py`): log_effekte_v3 (32.000 Fälle,
  0,73 Abweichungen je Fall, 60 % konform), Outcome = −Abweichungen, CausalForestDML wie am 12.11. Testfälle, τ = Aktion − Referenz (> 0: Aktion weniger
  Abweichungen): nachlegen (Aktion weiteres Angebot) −0,26, konzept (Aktion annehmen) +0,28, nach V. (Aktion annehmen) +0,03, leads (Aktion Rückruf) +0,13.
- Rollout `policies/ev_policy/rollout.py` (ForestRollout): erreicht ein gezogener Suffix den Auslöser eines Wahlpunkts, setzt draw_cases (neuer
  Parameter force, psp_model.py) im nächsten Schritt die Option des Forests (Merkmale: Prefix + Options-Event + gezogene Events); Zeit und Rest zieht
  der Decoder; nachlegen nur beim 1. Besuch. Eingeschaltet über model.rollout (Hülle und alle Netze); ohne bleibt alles gleich.
- Prüfungen (`rollout_cf_20261123/test_rollout.py`): ohne Rollout **bitgenau gleiche Ziehungen** wie vor der Änderung (Sicherung psp_model_vor_rollout.py.bak);
  mit Rollout 478 Forest-Eingriffe in 320 Suffixen, nach jedem Auslöser die Option (zusätzlich prüft draw_cases jede erzwungene Zeile).
- Läuft (tmux psp_rollout, `data/bpic17_psp_rollout_cf.sh`): Diagnose je Teilgruppe mit Rollout (`diagnose/konformanz_unsicherheit_ro.md`), danach nach der
  Kontrolle konformanz_ol300 der Online-Arm **konformanz_ol_ro** (k3x4ol, keyed_noise, 300 Ziehungen + Rollout) – gepaart mit konformanz_ol300 unterscheidet
  sich nur der Rollout.

## 2026-11-23 14:00: Wie realistisch zieht olsv online? – Abläufe gültig, Häufigkeiten stimmen, aber keine fallgenaue Vorhersage

- `suffix_realismus_20261123/olsv_realismus.py` (400 Zustände = erste Besuche in Bank-Läufen Aufbau F, Seeds 0–11, je 30 Ziehungen mit der
  Option der Bank, keyed_noise wie online; echt = was der Simulator danach tat). Erster Lauf mit falscher Token-Trennung archiviert (fehler_trennung/).
- Über alle Punkte gezogen / echt: Restzeit 22,1 / 21,7 d (SD 14,7 / 13,5; Q90 41,4 / 36,6 – Schwanz etwas zu lang), Events 34,8 / 35,4,
  Angebote 1,15 / 1,14, O_Sent 1,30 / 1,30, Anrufe nach Angeboten 1,00 / 1,00. Unbekannte Übergänge (nie im echten Log) 0,66 / 0,00 je 100 Events.
- Ausgänge gezogen / Simulator: angenommen 57 / 59 %, **abgelehnt 15 / 6,5 %, abgebrochen 27 / 34,5 %**, ohne Ende 1,4 / 0,25 %. Der echte Log hat
  ≈ 55 / 12 / 33 % – olsv liegt bei den Ablehnungen näher am echten Log als der Simulator (Simulator lehnt zu selten ab).
- Je Fall: Fehler der Restzeit olsv = naiver Mittelwert (MAE 9,8–11,5 d), Korrelation mit der echten Restzeit −0,01 / 0,04 / 0,14 / 0,23
  (nachlegen / konzept / nach V. / leads) → die Ziehungen treffen die Verteilung, nicht den einzelnen Fall (wie bei allen Modellen auf BPIC17).

## 2026-11-23 13:25: Basis nur auf dem Simulator gestartet (Nutzer: „ja“) – Pipeline bpic17_psp_konformanz_sim_basis.sh

- tmux psp_simbasis, Fortschritt `results_100k/sim_basis_20261123/fortschritt.log`. Basis ZZ-TF1 auf log_effekte_v3 (32.000 Fälle, alle Prefixe,
  10 Epochen statt 15 – auf dem echten Log Plateau ab Epoche 9–12; ≈ 9 h), dann 4-facher Log ohne die ≈ 10 Fälle mit Kategorien außerhalb des
  Basisvokabulars (sonst bricht vocab_from ab), 8 Netze warm von der Basis (≈ 6–7 h), Hülle schlank + keyed_noise, Diagnose, Online konformanz_sim
  (300 Ziehungen). Kontrolle parallel auf der CPU: konformanz_ol300 (bisherige Netze, keyed_noise + 300 Ziehungen), damit sich der Effekt der Basis
  von Kopplung und Ziehungszahl trennen lässt. Ende ≈ 23.11. nachts / 24.11. Vormittag.
- 13:31: Fehler in Schritt 2 gefunden (Filter verglich Rohnamen mit Tokens Aktivität|Lifecycle → hätte alle Fälle entfernt). Steuerskript beendet,
  Basistraining und Kontrolle laufen weiter; Fortsetzung `data/bpic17_psp_konformanz_sim_basis_teil2.sh` (tmux psp_simbasis2) mit Filter über
  PSPModel.view und Sicherung (höchstens 1 % der Fälle).
- 14:56: Epoche 1 der Basis fertig (Laden: Spitze 32 GB; ≈ 78 min je Epoche) → Basis ≈ 03:00 am 24.11., Netze bis ≈ 10:00, Online konformanz_sim ≈ mittags.

## 2026-11-23 13:20: Conformance mit eingeschalteter Kopplung (keyed_noise) – ρ 0,08 → 0,41, aber kaum mehr unterscheidbar

- Kopie `models/psp_model_durchlaufzeit_option_learner_k3x4_schlank_crn.pkl`, Diagnose `diagnose/konformanz_unsicherheit_crn.md` (gleiche 967 Zustände,
  100 Ziehungen): ρ (Median) 0,08 → **0,41** (Zeit auf dem echten Log ≈ 0,7), SE 0,20 → 0,16 Abweichungen (−19 %), nicht unterscheidbar 81 → **77 %**;
  Wahl nach dem Vorzeichen von Δ richtig (gegen Wahrheit Regel sonst) unverändert 63 %. Hochgerechnet mit 300 Ziehungen: 55 % nicht unterscheidbar,
  35 % sicher richtig, 10 % sicher falsch.
- Mittleres Δ je Teilgruppe praktisch wie ohne Kopplung (Verzerrung bleibt: konzept ≥ 11k +0,07 statt −0,17, konzept < 11k +0,09 statt +0,38, leads und nach V. ≈ 0).
- Deutung: Die Kopplung war ein Versäumnis, erklärt aber nur einen kleinen Teil. Die Event-Wahl direkt nach der Option (Schlüssel = voriges Event)
  bleibt entkoppelt, und dort entstehen die Abweichungen. Hauptproblem: Effekte von 0,2–0,4 Abweichungen gegen SE 0,16 und die Verzerrung des Modells.
  Ein neuer Online-Lauf nur mit Kopplung lohnt kaum.

## 2026-11-23 13:03: Conformance-Ansatz (k3x4ol, olsv-Rezept) – 81 % der Entscheidungen nicht unterscheidbar, dazu Verzerrung

- `kuenstliche_effekte_20261120/diagnose/konformanz_unsicherheit.py` (967 Zustände wie konformanz_signal, 100 Ziehungen je Option wie online):
  Δ = Abweichungen andere − Regel-Option, gepaart je Ziehung; **nicht unterscheidbar 81 %** (echter Log, olsv, Durchlaufzeit: 23 % bei 300).
  SE je Entscheidung (Median) 0,20 Abweichungen, **ρ der Optionen nur 0,08** [Ursache FALSCH gedeutet: keyed_noise war in den Conformance-Hüllen
  k3x4 / k3x4ol gar nicht eingeschaltet (13:05 geprüft) – ρ 0,08 entspricht der Bedingung „schritt“ (Zeit: 0,09–0,15)].
- Mittleres Δ je Teilgruppe gegen die Wahrheit (Bank sonst / Regel sonst): nachlegen ≥ 1,71 −0,25 (−0,17 / −0,24) ✓; 0,84–1,71 +0,20 (+0,39 / +0,37);
  < 0,84 +0,50 (+1,34 / +0,84); **konzept ≥ 11k +0,10 (−0,18 / −0,17) falsches Vorzeichen**; konzept < 11k +0,07 (+0,32 / +0,38) zu klein;
  nach V. −0,04 / +0,01 (Regel sonst −0,17 / −0,08); leads < 11k +0,01 (−0,25 / +0,15).
- Hochrechnung (SE ∝ 1/√n, Δ aus 100 Ziehungen): nicht unterscheidbar 100 / 300 / 1000 → 81 / 58 / 33 %; von den sicheren Entscheidungen zeigen
  87 / 76 / 70 % in die Richtung der Wahrheit (Regel sonst). Wahl nach dem Vorzeichen von Δ (wie online) richtig in 63 %.
- **Deutung: beides.** Rauschen ist hier das größere Problem als auf dem echten Log (Effekte 0,2–0,4 Abweichungen bei SE 0,2; kaum Kopplung),
  aber auch mit unendlich vielen Ziehungen bliebe die Verzerrung (konzept ≥ 11k falsch, konzept < 11k / leads / nach V. ≈ 0). Hebel: mehr
  Ziehungen (300: SE 0,12) oder Kopplung der Event-Wahl über die Netze (ρ 0,08 → 0,7 wirkte wie dreifache Ziehungen); gegen die Verzerrung
  helfen sie nicht.
- **Korrektur 13:05:** Die Conformance-Hüllen hatten keyed_noise aus (bpic17_psp_konformanz_option_learner.sh und v3x4c bauten keine _crn-Kopie
  wie olsv). Die schwache Kopplung ist also vor allem ein Versäumnis, kein Wesensmerkmal von Conformance.

## 2026-11-23 12:05: Online PSP + Causal Forest (olsvcf_fest300) +2,64 – deckungsgleich mit der Regel

- Aufbau F, 12 × 200: **olsvcf +2,64 ± 0,35 d gegen die Bank**; gegen die Regel −0,01 ± 0,13 (gleich), gegen olsv +0,18 ± 0,24 (n.s.),
  gegen tlsvk +0,03 ± 0,32. Abweichungen von der Regel 0 / 0 / 0 / 0 % (olsv 0 / 3 / 0 / 0 %). Laufzeit wie olsv (≈ 69 min je Seed).
- Unsicher (|Δ| < 1,96 · SE) waren online 10,4 % der Entscheidungen mit Ziehen (688 von 6.632): nachlegen 13 %, konzept 8 %, nach V. 36 %,
  leads 8 % – weniger als die Hochrechnung aus der Diagnose (23 %). Der Forest entschied dort und wählte in 669 Fällen dasselbe wie das PSP;
  nur 19 Entscheidungen (0,3 %) änderte er, alle zur Regel-Option (12 davon nach V., Δ dort 0,02–0,45 d bei SE 0,5–0,8).
- Deutung: Der Forest beseitigt genau die restlichen Abweichungen von olsv (nahe-0-Fälle bei nach V.); auf dem echten Log ist das Ergebnis
  damit die Regel. Mehrwert wäre nur dort möglich, wo es fallabhängige Effekte gibt (Simulator mit Effekten; dafür fehlt ein Forest auf Conformance).

## 2026-11-23 11:00: Wie oft kann das PSP die Optionen nicht unterscheiden? (echter Log, Aufbau F)

- `suffix_verlust_20261119/unterscheidbarkeit.py` auf den Ziehungen der Wirkungsdiagnosen (300 Zustände, je Option 100 Suffixe, keyed_noise):
  nicht unterscheidbar = |Δ andere − Regel| < 1,96 · SE der gepaarten Differenz; 300 Ziehungen hochgerechnet (SE / √3); gewichtet nach Wahlen je Fall.
- Anteil nicht unterscheidbar bei 300 (100) Ziehungen: **tlsvk 29 % (56 %)**, olsv 23 % (49 %), tlsv 45 % (79 %), plsv 53 % (80 %),
  T-Learner ZZ-TF1 56 % (81 %). Je Punkt tlsvk: konzept 8 %, nach V. 3 %, nachlegen 1. Besuch 43 %, leads 37 %; olsv 22 / 23 / 27 / 18 %.
- „Sicher andere Option besser“ bei tlsvk und olsv 0 %; sonst ist die Regel-Option sicher besser. Auch die nicht unterscheidbaren Fälle zeigen
  meist in Richtung Regel (Δ > 0, nur unter der Schwelle) – daher online trotzdem nur 0–6 % Abweichungen.
- SE der Differenz je Entscheidung bei 300 Ziehungen 0,6–0,9 d → Effekte unter ≈ 1,2–1,8 d je Fall löst das PSP einzeln nicht auf.
- Für einen Rückfall (z. B. Causal Forest) bei Unsicherheit: betroffen wären ≈ 23–29 % der Entscheidungen der besten PSPs; auf dem echten Log
  wählt der Forest dort die Regel-Option, das Ergebnis bliebe ≈ gleich.
- Hochrechnung mehr Ziehungen (gleiche Daten, SE ∝ 1/√n): nicht unterscheidbar tlsvk 100 / 300 / 500 / 1000 → 56 / 29 / 23 / 14 %, olsv 49 / 23 / 16 / 9 %.
  Erwarteter Rückstand zur Regel bei 300 nur ≈ 0,03 d (prognose.md) → 500 Ziehungen brächten ≈ 0,01–0,02 d, online nicht messbar (SE ≈ 0,3);
  frühere Messung tl_fest500 gegen tl_fest300: −0,13 ± 0,28 d trotz weniger Abweichungen (8/10/39/20 → 4/6/36/13 %), Laufzeit ≈ 2,75×.

## 2026-11-23 10:30: Wahrheit mit der Regel als Basis – echter Spielraum über die Regel +4,3 pp konform; leads kehrt das Vorzeichen um

- `kuenstliche_effekte_20261120/wahrheit_f_regel.py` (72 Läufe, Aufbau F, Effekte v3, 12 Seeds). basis = Regel überall, reproduziert den
  Online-Arm regel exakt (Seed 0: 0,550 = 0,550; Mittel 0,586 Abw., 72,0 % konform).
- Stufe 1, andere − Regel-Option an einem Punkt bei Regel sonst (in Klammern: bei Bank sonst, wahrheit_f): konzept ≥ 11k −0,17 (−0,18),
  < 11k +0,38 (+0,32); **nach V. keine −0,17 (+0,08), Car/Übernahme −0,08 (+0,03)**; nachlegen ≥ 1,71 −0,24 (−0,17; nur 34 Fälle, SE 0,22),
  0,84–1,71 +0,37, < 0,84 +0,84; **leads < 11k +0,15 (−0,25)**, leads sonst +0,15. → Der leads-Vorteil bestand nur gegen die Bank-Wahl bei
  konzept (bestätigt 10:05); dafür lohnt nach V. jetzt die andere Option.
- Stufe 2 (kombi: konzept ≥ 11k, nach V. beide Gruppen, nachlegen ≥ 1,71 umgeschaltet, Regel sonst), direkt gemessen gegen basis:
  **−0,059 ± 0,028 Abweichungen je Fall, +4,3 ± 0,9 pp konform (≈ 76,3 % statt 72,0 %), Durchlaufzeit −0,76 ± 0,38 d**; Summe der Einzeleffekte −0,081.
  Auswahl und Messung auf denselben Seeds (leicht optimistisch).
- **Folgen:** (1) Das Orakel aus der Bank-Wahrheit (73,1 %) war nicht optimal – es schaltete leads < 11k um (schadet bei Regel sonst) und nach V.
  nicht. (2) Der echte Spielraum über die Regel ist ≈ 4 pp konform; konformanz_ol (70,9 %) liegt 5,4 pp unter diesem Ziel.
  (3) Die optimale Wahl an einem Punkt hängt davon ab, wer die späteren Punkte entscheidet. Das PSP zieht die späteren Punkte wie im Log
  (Bank) und schätzt damit die Bank-Wahrheit – es würde wie das alte Orakel leads < 11k umschalten. Für eine Policy, die alle Punkte selbst
  entscheidet, müsste die Bewertung die eigenen späteren Entscheidungen einbeziehen (z. B. beim Ziehen an späteren Wahlpunkten die
  Policy-Option erzwingen statt das Modell ziehen zu lassen).

## 2026-11-23 10:05: Wahrheit in Aufbau F gemessen – die Tabelle stimmt, aber die Effekte der Punkte überlappen (Orakel nur +0,009)

- `kuenstliche_effekte_20261120/wahrheit_f.py` (96 Läufe, Option nur für die 200 bewerteten Fälle an einem Punkt erzwungen, Bank an den anderen):
  andere − Regel-Option in Abweichungen je Fall (F / alte Tabelle): konzept ≥ 11k −0,18 / −0,06, < 11k +0,32 / +0,53; nach V. +0,08 / +0,05 und
  +0,03 / +0,09; nachlegen 0,84–1,71 +0,39 / +0,28, ≥ 1,71 −0,17 / −0,26, < 0,84 +1,34 / +1,38; leads < 11k −0,25 / −0,43, sonst +0,14 / +0,08
  (SE 0,04–0,10). **Gleiche Vorzeichen überall** → meine Vermutung 09:20 („Tabelle gilt in F nicht“) war FALSCH. Erwarteter Vorsprung
  eines Orakels nach F-Wahrheit 0,15 Abweichungen je Fall (alte Tabelle 0,19), online 0,009.
- **Ursache: die Punkte wirken nicht unabhängig.** Jede Wahrheit gilt bei Bank an den anderen Punkten. Beispiel Application_701 (6.000 €):
  Regel → leads zurückziehen, konzept A_Accepted; Orakel → leads Rückruf, danach erreicht der Fall den Konzept-Punkt gar nicht mehr mit zwei
  Optionen (W_Complete suspend, später nach V.). Online erreichen im Orakel-Lauf nur 56 statt 685 Fälle < 11k den Konzept-Punkt, dafür
  1.537 statt 371 den Punkt nach V. Der leads-Gewinn (Rückruf) in der Einzelwahrheit besteht offenbar großenteils darin, der schlechten
  Bank-Wahl bei konzept < 11k auszuweichen – entscheidet die Regel konzept schon richtig, bleibt davon kaum etwas übrig.
- Das Orakel entscheidet je Gruppe genau wie seine Tabelle (konzept ≥ 11k und leads < 11k 100 % andere, nachlegen ≥ 1,71 81 %); die Regel wählt
  nachlegen bei jedem Besuch O_Sent, Orakel und Konformanz-Arme nur beim 1. Besuch (danach Bank).
- Folge für das PSP: KonformanzPolicy (wie die Einzelwahrheit) bewertet jede Wahl mit Suffixen, in denen die späteren Punkte wie im Log (Bank)
  entschieden werden – nicht so, wie die Policy selbst später entscheidet. Bei überlappenden Effekten ist das kurzsichtig.
- Nächster Schritt (vorgeschlagen, nicht gestartet): Wahrheit gegen die Regel als Basis (Regel an allen anderen Punkten, eine Teilgruppe
  umgeschaltet) – zeigt den echten Spielraum über die Regel hinaus.

## 2026-11-23 09:20: Online Conformance-Option-Learner (konformanz_ol) 70,9 % konform = konformanz_x4 – und das Orakel zeigt: die Wahrheitstabelle gilt in Aufbau F kaum [Deutung der Wahrheitstabelle FALSCH, s. 10:05]

- Aufbau F mit Effekten v3, 12 × 200 (`kuenstliche_effekte_20261120/online_konformanz.md`): Abweichungen je Fall / konform / Durchlaufzeit
  bank 0,666 / 65,4 % / 20,45 d; konformanz (k3) 0,642 / 69,5 % / 21,44; **konformanz_ol 0,610 / 70,9 % / 21,36**; konformanz_x4 0,602 / 71,0 % / 21,21;
  regel 0,586 / 72,0 % / 20,84; orakel 0,577 / 73,1 % / 20,49. Gepaart: ol − x4 0,008 ± 0,036 Abw., konform −0,2 ± 1,0 pp (gleich);
  Regel besser als ol um 0,025 ± 0,030 Abw. / 1,1 ± 1,3 pp (n.s.); Orakel besser als ol um 0,033 ± 0,041 / 2,2 ± 1,3 pp.
- Wahlen je Teilgruppe (`diagnose/wahl_je_gruppe.md`, erste Besuche der bewerteten Fälle; Anteil andere Option ol / x4, richtig wäre):
  nachlegen ≥ 1,71 d 64 / 34 % (andere) → ol jetzt meist richtig; leads < 11k 41 / 21 % (andere) → besser, aber halb zufällig;
  konzept < 11k 45 / 56 % (Regel), konzept ≥ 11k 45 / 66 % (andere), nach V. 44–45 / 57–67 % (Regel) → nahe Münzwurf, weil die geschätzten
  Unterschiede dort ≈ 0 sind. Erwarteter Verlust gegen das Orakel laut Wahrheitstabelle: ol 450, x4 545 Abweichungen auf 2.400 Fälle (−18 %).
- **Widerspruch:** Laut Wahrheitstabelle müsste das Orakel die Regel um ≈ 0,16 Abweichungen je Fall schlagen (vor allem leads < 11k: −0,43 je
  Entscheidung bei ≈ 680 Entscheidungen), online sind es 0,009. Die Wahrheit (`wahrheit.py`) stammt aus Läufen, in denen die Option für ALLE
  Fälle erzwungen wurde (4.000 Fälle, 3 Saaten, Bank an den anderen Punkten); in Aufbau F entscheidet die Policy nur für 200 bewertete Fälle.
  Systemeffekte und Paarungsrauschen machen die Tabelle für F offenbar ungültig – sie taugt weder als Maßstab für die Diagnose noch zeigt F
  Spielraum (Orakel − Regel ≈ 1 pp konform). Der Gewinn von ol in der Diagnose (nachlegen ≥ 1,71) lässt sich so online nicht nachweisen.
- Durchlaufzeit: die Konformanz-Arme sind 0,5–0,9 d langsamer als Regel und Bank (nicht ihre Zielgröße).
- Vorschlag (nicht gestartet): Wahrheit direkt in Aufbau F messen (Option nur für die bewerteten Fälle an einem Punkt erzwingen, je Punkt × Option
  × 12 Seeds = 96 kurze Läufe, CPU, Minuten); damit Orakel und Diagnose neu bewerten.

## 2026-11-23 08:35: Conformance-Option-Learner (k3x4ol) – Diagnose je Teilgruppe: nachlegen ≥ 1,71 jetzt gelernt, konzept < 11k und leads < 11k weiter nicht

- 8 Netze (je Wahlpunkt und Option) auf dem 4-fachen v3-Log, warm von den k3x4-Armen, nur Stufe A (kein Suffix-Verlust), keine Kalibrierung;
  Training 19:11–01:49. Diagnose `kuenstliche_effekte_20261120/diagnose/konformanz_signal_ol.md`: 967 Zustände (Bank entscheidet), je Option 50
  Ziehungen, k3x4 und k3x4ol auf denselben Zuständen. Erwartete Abweichungen je Fall, andere − Regel-Option (Wahrheit / k3x4 / k3x4ol, SE ≈ 0,03):
  - nachlegen ≥ 1,71 d: −0,26 / +0,06 / **−0,24** → jetzt richtig (k3x4 falsches Vorzeichen)
  - nachlegen 0,84–1,71 d: +0,28 / +0,40 / +0,20; nachlegen < 0,84 d: +1,38 / +0,73 / +0,45 → Richtung beide richtig, Betrag zu klein
  - konzept ≥ 11k: −0,06 / −0,03 / +0,11 (k3x4ol knapp falsch); konzept < 11k: +0,53 / +0,09 / +0,11 → Richtung richtig, Effekt kaum gelernt
  - nach V. keine: +0,05 / −0,03 / 0,00; nach V. Car/Übernahme: +0,09 / −0,11 / +0,01 (k3x4 falsch, k3x4ol ≈ 0)
  - leads < 11k: −0,43 / +0,12 / **0,00** (nicht mehr falsch, aber nicht gelernt); leads keine: +0,08 / +0,12 / +0,16
- Deutung: getrennte Netze lernen den zeitabhängigen Effekt beim Nachlegen (Prefix-Dauer) und beseitigen zwei falsche Vorzeichen; der Effekt
  „W_Complete doppelt → A_Complete doppelt“ bei kleinen Beträgen und der leads-Effekt bleiben ungelernt. Gezogene Spuren haben weiter zu viele
  Abweichungen (Regel-Option ≈ 0,9–1,25 je Fall, echt ≈ 0,6).
- Hinweis: Die Zustände weichen von der Diagnose 11-22 01:25 ab (andere Gruppengrößen trotz gleicher Saaten); gepaart ist nur innerhalb eines Laufs.
- Online: erster Versuch 02:04 vom OOM-Killer beendet (12 Prozesse × 10 GB: jedes Netz trug den kodierten 4-fach-Log mit, 0,6 GB je Netz;
  der tmux-Scope wurde mit beendet). Schlanke Hülle ohne den Log (`huelle_schlank.py`, 46 MB statt 5 GB, 95 von 95 Entscheidungen gleich);
  Online-Auswertung konformanz_ol neu ab 08:29 (12 Prozesse, tmux psp_k3ol_online).

## 2026-11-22 21:50: S-Learner je Wahlpunkt (plsv_fest300) +1,49 – Optionseffekt geschrumpft, nachlegen falsch; Diagnosefehler bei Hüllen korrigiert

- Aufbau F, 12 × 200, 300 Ziehungen, keyed_noise, first_only: **plsv +1,49 ± 0,46 gegen die Bank**; Regel schneller um 1,14 ± 0,26,
  tlsvk um 1,12 ± 0,29, olsv um 0,97 ± 0,20 (alle signifikant). Abweichungen von der Regel konzept / nach V. / nachlegen / leads
  1 / 0 / 46 / 12 % (nachlegen: 967 zusätzliche Angebote, die Angebotsschleife ist zurück).
- Diagnose (Bedingung event, korrigiert): Vorteil der Regel-Option 1,76 / 3,20 / **0,02** / 0,75 d (Wahrheit 2,79 / 2,40 / 2,11 / 1,32),
  ρ der Optionen 0,68–0,76; Prognose falsch 0,8 / 0,7 / 49 / 11,5 %, erwarteter Rückstand 0,84 d – passt zu online (46 / 12 %, 1,14 ± 0,26).
- **Deutung:** Die Kopplung ist nicht das Problem (ρ ≈ 0,7 wie tlsvk). Mit beiden Optionen in einem Netz geht die Option nur als Token ein,
  und der Optionseffekt schrumpft (klassische S-Learner-Verzerrung); der Suffix-Verlust je Wahlpunkt korrigiert das Mittel des Punkts, nicht die
  Aufteilung auf die Optionen. Getrennte Netze je Option (olsv, T-Learner-Arme) haben das nicht. **S-Learner je Wahlpunkt verworfen.**
- **Diagnosefehler (gefunden 21:40):** `modellvergleich_20261116/crn_schluessel.py` set_keyed schaltete keyed_noise nur in der Hülle und im
  Aktionsarm ein, nicht in den Netzen (`netze`) von Option-/Punkt-Learner-Hüllen → dort war Bedingung „event“ = „schritt“. Korrigiert, Diagnose
  für beide Hüllen neu (alte Ausgaben in `suffix_verlust_20261119/wirkung_*_learner_fehler_keyed_aus/`). Online war nicht betroffen (Hüllen _crn).
- **Korrektur für olsv:** mit keyed_noise ρ 0,67–0,71 (nicht 0,1), Rauschen bei 300 Ziehungen 0,6–0,7 d; Prognose falsch 0,1 / 3,5 / 0,2 / 0,0 %,
  erwarteter Rückstand 0,02 d – stimmt jetzt mit online überein (0 / 3 / 0 / 0 %). Die Aussagen „Kopplung verloren“ (16:55) und „Optionen kaum
  gekoppelt (ρ ≈ 0,1)“ (17:40) sind FALSCH; der Option-Learner hat keinen Nachteil gegenüber tlsvk außer der Netzzahl.

## 2026-11-22 17:40: Online Option-Learner (olsv_fest300) +2,46 – ohne Kalibrierung gleichauf mit tlsvk und der Regel

- Aufbau F, 12 × 200, 300 Ziehungen, keyed_noise, first_only: **olsv +2,46 ± 0,33 gegen die Bank**; Regel schneller um 0,17 ± 0,18 (n.s.),
  tlsvk um 0,15 ± 0,30 (n.s.); olsv besser als tlsv um 0,23 ± 0,34. Anteil Regel-Option 99,7 %, Angebote je Fall 1,27.
- Abweichungen von der Regel konzept / nach V. / nachlegen / leads 0 / 3 / 0 / 0 % (tlsvk 0 / 6 / 1 / 1, tlsv 9 / 20 / 10 / 1); erwarteter Rückstand
  aus den Abweichungsraten 0,02 d, beobachtet 0,17 ± 0,18, Fälle ohne Abweichung +0,12 ± 0,27 → der Rest ist Paarungsrauschen.
  (Prognose aus der Diagnose: 5,6 / 3,8 / 6,3 / 6,3 % falsch – online weicht olsv seltener ab als vorhergesagt.)
- **Ergebnis: Mit einem Netz je (Wahlpunkt, Option) ist die Kalibrierung überflüssig** – die Gruppenmittel stimmen von selbst (Diagnose 16:55),
  und es gibt keinen Schrumpfdruck gegen Unterschiede zwischen Fällen. Preis: 8 statt 2 Netze, die kleinen Netze passen sich an
  (Validierung steigt), Optionen kaum gekoppelt (ρ ≈ 0,1) [FALSCH, Diagnosefehler: mit keyed_noise ρ ≈ 0,7, s. 21:50].
- Läuft: S-Learner je Wahlpunkt (seit 17:36, tmux psp_pl).

## 2026-11-22 17:10: S-Learner je Wahlpunkt vorbereitet (startet nach dem Option-Learner-Online-Test)

Nutzer: „das wäre doch dann wieder ein S-Learner?“ – ja, je Wahlpunkt; „ja, bereite das so vor“. Gedanke: ein Netz für beide Optionen eines
Punkts → Kopplung der Optionen bleibt (Option-Learner verlor sie: ρ ≈ 0,1), eine Verschiebung durch den Suffix-Verlust trifft beide Optionen
gleich und hebt sich im Vergleich auf (keine Kalibrierung nötig), mehr Proben je Netz (≈ 13.000–25.000). Risiko: typische S-Learner-
Schwäche (Option als ein Token im langen Prefix wird unterschätzt, wie beim globalen ZZ-TF1 online −0,08).
- Code: option_learner_policy zieht gemeinsam, wenn alle Optionen eines Punkts dasselbe Netz haben; bauen lädt gleiche Pfade nur einmal.
- `data/bpic17_psp_punkt_learner.sh` (tmux psp_pl, wartet seit 17:08 auf das Ende des olsv-Online-Tests): 4 Netze `pl_<punkt>[_sv]` warm vom
  ZZ-TF1, --point ohne --option, Stufe A 20 / Stufe B 5 Epochen, LR 3e-5, keine Kalibrierung, Stufe B nacheinander (flock); Hülle
  `punkt_learner_sv(_crn).pkl`; Wirkung (`suffix_verlust_20261119/wirkung_punkt_learner/`) + Prognose; Online-Arm plsv_fest300 wird erst
  nach dem olsv-Test eingetragen (`punkt_learner_20261122/arm_eintragen.py`, laufende Skripte nicht ändern).

## 2026-11-22 16:55: Diagnose Option-Learner – Gruppenmittel ohne Kalibrierung richtig, aber Kopplung der Optionen verloren [Kopplung: FALSCH, Diagnosefehler, s. 21:50]

`suffix_verlust_20261119/wirkung_option_learner/` (300 Simulatorzustände, Bedingung event; Vergleich tlsv / tlsvk):
- Vorteil der Regel-Option konzept 2,26 (Wahrheit 2,79; tlsv 1,04, tlsvk 2,84), nach V. 3,06 (2,40; 1,79 / 3,42), nachlegen 1. Besuch 2,18 (2,11;
  0,98 / 2,11), leads 2,27 (1,32; 1,86 / 1,58) → **die Verschiebung je Arm ist weg, die Gruppenmittel stimmen ohne Kalibrierung** (an allen
  vier Punkten richtige Richtung und Größe nahe an tlsvk).
- Aber: Korrelation der Optionen ρ 0,09–0,15 (tlsv / tlsvk ≈ 0,6–0,7) → Rauschen bei 300 Ziehungen 1,09–1,17 d statt ≈ 0,6. Die acht Netze
  sind getrennt trainiert (40 / 20 Epochen auf wenigen Proben, Überanpassung) und ziehen trotz Zufallszahlen je Event kaum noch gekoppelt.
- Rauschfreie Streuung des Vorteils zwischen Zuständen 0,85–1,30 d (tlsvk 0–0,41) – mehr Unterschied zwischen Fällen, ob Signal oder
  Überanpassung, zeigt erst der Simulator mit Effekten.
- Prognose (Φ-Formel): falsch 5,6 / 3,8 / 6,3 / 6,3 %, Rückstand zur Regel 0,25 d (tlsv 0,19, tlsvk 0,03) → erwartet online ≈ +2,35–2,4.
- Hebel: mehr Ziehungen (Rauschen ∝ 1/√n, ≈ 1.000 für 0,6 d), weniger Epochen / beste Epoche (gegen Überanpassung und Auseinanderlaufen der
  Netze), oder ein Netz je Wahlpunkt für beide Optionen (gemeinsames Netz → Kopplung).

## 2026-11-22 11:55: Ein PSP je Wahlpunkt und Option (Option-Learner) ohne Kalibrierung – Lauf gestartet

Nutzer (Frage, warum nicht 4 T-Learner je Wahlpunkt; dann „Basismodell ZZ-TF1 und dann die T-Learner jeweils aufteilen pro Option?
Kalibrierung weglassen?“ – ja). Gedanke: beim T-Learner verschiebt der Suffix-Verlust jeden Arm als Ganzes (Befund 11-19 20:45), die
Kalibrierung flickt das je (Wahlpunkt, Option) und drückt dabei Unterschiede zwischen Fällen zusammen. Mit einem Netz je Gruppe trifft
die Verschiebung genau die Gruppe; die Kalibrierung kann entfallen.
- Code: `psp_model` only_point_names (Filter auf Wahlpunkte in _weight_points), `train_psp_real.py --point / --option`;
  `policies/ev_policy/option_learner_policy.py` (Hülle OptionLearnerPSP, netze {(Wahlpunkt, Token): PSPModel}, gleiche Startsaat je Option;
  Test `option_learner_20261122/tests/huelle_test.py` mit Platzhaltern: bauen / speichern / laden / Ziehen an allen 4 Punkten ok).
- Lauf `data/bpic17_psp_option_learner.sh` (tmux psp_ol, ab 11:43): 8 Netze warm vom ZZ-TF1, Stufe A wie die T-Learner-Arme (Aktions-
  Optionen 40, Referenz-Optionen 20 Epochen, LR 3e-5), Stufe B Suffix-Verlust (10 / 5), keine Kalibrierung; zwei Stränge parallel auf der
  GPU, --load-chunks 4 (Speicherspitze 19 GB); Proben z. B. konzept annehmen ≈ 6.000, konzept bearbeiten ≈ 9.800. Danach Hülle + keyed_noise,
  Wirkungsdiagnose + Prognose (`suffix_verlust_20261119/wirkung_option_learner/`), Online Aufbau F 12 Seeds (Arm olsv_fest300). Ende ≈ 17:30.
- 12:28 Abbruch in Stufe B (konzept annehmen): zwei Suffix-Verlust-Trainings gleichzeitig sprengten die 24 GB GPU (ein solches belegt
  ≈ 12 GB; die Fehlermeldung hatte der Log-Filter „CUDA“ verschluckt). Skript: Stufe B nur eine zur Zeit (flock), Fehlerzeilen bleiben im
  Log, Checkpoint-Ordner abgebrochener Läufe werden entfernt; Neustart 12:31, Stufe B läuft (konzept annehmen Ep. 1 Validierung 11,10).
  Ende jetzt eher ≈ 19 Uhr.
- Stand 15:05: 13 von 16 Trainings fertig (alle 4 Aktions-Netze mit Suffix-Verlust; offen nachlegen senden Stufe B, leads zurückziehen A + B).
  Validierungsverlust je Netz erste … letzte Epoche: konzept annehmen 11,08 … 11,23 (sv 11,10 … 11,20), konzept bearbeiten 16,81 … 17,19
  (sv 17,16 … 17,35), nach V. annehmen 11,23 … 11,33 (sv 11,29 … 11,24), nach V. weiter 16,36 … 16,46 (sv 16,68 … 16,44), nachlegen weiteres
  14,04 … 13,72 (sv 13,76 … 13,30), nachlegen senden 12,35 … 12,21, leads Rückruf 15,05 … 15,96 (sv 15,05 … 15,86) → bei den kleinen Netzen
  steigt die Validierung über die Epochen (Überanpassung bei 2.400–6.000 Proben, gespeichert wird die letzte Epoche), nur nachlegen fällt.
- Stand 16:20: alle 16 Trainings fertig (16:18), Hülle + keyed_noise-Kopie gebaut, Wirkungsdiagnose läuft, danach Online. Rest der Kurven:
  nachlegen senden sv 12,16 … 12,20 (flach), leads zurückziehen 15,32 … 15,94 (sv 16,00 … 16,21) → auch dieses Netz (10.889 Proben) passt
  sich zunehmend an.


## 2026-11-22 01:25: Diagnose k3x4 – mehr Daten verbessern nur konzept bei hohen Beträgen; die übrigen Effekte bleiben ungelernt

`diagnose/konformanz_signal.py` (8 × 45 frische Fälle, 50 Ziehungen; andere − Regel-Option, Wahrheit / k3 / k3x4 / svk):
- konzept ≥ 11.000 €: −0,06 / +0,08 / **−0,02** / −0,18; < 11.000 €: +0,53 / +0,14 / +0,02 / −0,12 → x4 trifft die Gruppe mit hohen Beträgen
  (daher öfter „vervollständigen“ online), verliert aber den kleinen Unterschied bei kleinen Beträgen; der doppelte Abschluss (A_Complete, 30 %)
  wird nicht gelernt, obwohl er im Trainings-Log steht (Vorverarbeitung entfernt keine Events: 5.115.485 vorher wie nachher).
- nachlegen 0,84–1,71 d / ≥ 1,71 d / < 0,84 d: Wahrheit +0,28 / −0,26 / +1,38; k3 +0,31 / +0,03 / +0,62; k3x4 +0,41 / +0,10 / +0,65 → Reihenfolge
  weiter richtig, Spannweite weiter etwa halb so groß, ≥ 1,71 d weiter falsches Vorzeichen; keine Verbesserung durch mehr Daten.
- nach V. ≈ −0,06 bis −0,09 (Wahrheit +0,05 / +0,09); leads +0,12 / +0,19 (Wahrheit −0,43 / +0,08, dort Nebenwirkung des Zeiteffekts).
- Deutung: der Online-Gewinn von x4 (Abstand zur Regel 0,057 → 0,016) kommt fast ganz aus konzept bei hohen Beträgen. Weit entfernte,
  seltene Folgen einer Option (doppelter Abschluss viele Schritte später, nur 30 % der Teilgruppe) lernt der PSP auch mit 4-facher Menge kaum;
  der Referenzarm mischt vier Optionen, die Bedingung „Option × Betrag“ geht darin unter.

## 2026-11-22 01:00: 4-fache Datenmenge (k3x4) – Conformance-Policy jetzt gleichauf mit der Regel, Orakel noch knapp voraus

- Training (GPU, load_chunks 8): Aktionsarm 13:00–14:46 (67.052 Proben = 4,0 × k3; Validierung 12,10 … 12,12, flach ab Epoche 1), Referenzarm
  bis 17:15 (247.935 Proben; 14,75 … 14,84, flach); Hülle `t_learner_k3x4.pkl`; Online konformanz_x4 17:15–18:05.
- **Online** (12 × 200, v3): Abweichungen je Fall / konform / Durchlaufzeit: bank 0,666 / 65,4 % / 20,45; konformanz (1-fach) 0,642 / 69,5 % /
  21,44; **konformanz_x4 0,602 / 71,0 % / 21,21**; regel 0,586 / 72,0 % / 20,84; orakel 0,577 / 73,1 % / 20,49.
- Gepaart: x4 gegen bank −0,064 ± 0,022 Abweichungen, +5,6 ± 1,2 Pp. konform (beides signifikant), +0,76 ± 0,33 d; gegen regel −0,016 ± 0,020 /
  +0,9 ± 0,9 Pp. zugunsten der Regel (n.s. → gleichauf); gegen orakel −0,025 ± 0,023 / +2,0 ± 0,8 Pp. zugunsten des Orakels; gegen 1-fach
  −0,040 ± 0,025 Abweichungen / +1,5 ± 1,1 Pp. besser (z ≈ 1,6, knapp n.s.).
- Wahlen x4 (2 Optionen) konzept vervollständigen 729 / annehmen 446 (1-fach 326 / 719 – näher am Orakel, das bei hohen Beträgen
  vervollständigt), nach V. weiter 1.362 / annehmen 727, nachlegen senden 2.122 / weiteres Angebot 165, leads zurückziehen 1.137 / Rückruf 301.
- Mehr Daten helfen also messbar (Abstand zur Regel 0,057 → 0,016), obwohl der Validierungsverlust je Schritt sich nicht ändert; der Rest bis
  zum Orakel (2 Pp.) und die längere Durchlaufzeit (die Policy beachtet die Zeit nicht) bleiben. Diagnose der gelernten Effekte (k3x4) läuft.

## 2026-11-21 13:10: Mehr Daten für die Arme – Speicherlimit, Laden in Stücken (load_chunks), Lauf mit 4-facher Menge gestartet

Nutzer: „mach das“ (4-fache Menge, 128.000 Fälle, Training auf der GPU).
- Log `log_effekte_v3_x4/` (32 × 4.000, Saaten 2000–2031, die ersten 8 Läufe = bisheriger v3-Log; 5.115.485 Events, 3 min).
- Training scheiterte beim Laden mit MemoryError, auch mit 3- und 2-facher Menge (Teil-Logs `log_effekte_v3_x3`, `_x2`): Prozess-Speicher-
  limit des Servers `ulimit -v` 56 GB (hart = weich, nur Admin), der Loader des U-ED-LSTM legt jeden Prefix als verschachtelte Python-Liste an
  und filtert erst danach auf die ≈ 2 % Wahlpunkt-Proben.
- **Lösung** `psp_model` load_chunks / `train_psp_real.py --load-chunks N` (nur mit --only-points und --only-points-validation): ein Loader
  (gleiche Encoder, Fensterbreite, Aufteilung), Kodieren in N Stücken von Fällen, je Stück sofort filtern, Stücke zusammenfügen. Test am
  1-fach-Log (4 Stücke): 16.650 / 3.813 Proben wie ohne Stücke. 4-fach mit 8 Stücken: Speicherspitze 25,5 GB, Stück 1: 10.401 Fälle →
  8.317 Proben.
- Lauf `data/bpic17_psp_konformanz_v3x4c.sh` (tmux psp_k3x4c, ab 13:00): Arme `k3x4_zaehler_zeit_tf1_t_{aktion,referenz}` warm von den Armen
  des echten Logs, LR 1e-4, Aktion 10 / Referenz 5 Epochen, GPU; Hülle `t_learner_k3x4.pkl`; danach Online-Arm konformanz_x4 (12 Seeds,
  gepaart mit bank / regel / orakel / konformanz). Erwartetes Ende ≈ 21 Uhr.

## 2026-11-21 01:10: Online Conformance (Simulator v3, Aufbau F, 12 × 200) – PSP-Policy besser als die Bank, schlechter als Regel und Orakel

`kuenstliche_effekte_20261120/online_konformanz.md` (Lauf 00:17–01:03, KonformanzPolicy korrigiert, k3, 100 Ziehungen, first_only nachlegen):
- Abweichungen je Fall / konforme Fälle / Durchlaufzeit: bank 0,666 / 65,4 % / 20,45 d; **konformanz 0,642 / 69,5 % / 21,44 d**;
  regel (Durchlaufzeit-Regel) 0,586 / 72,0 % / 20,84 d; orakel 0,577 / 73,1 % / 20,49 d.
- Gepaart: konformanz gegen bank +4,1 ± 1,3 Prozentpunkte konforme Fälle (signifikant), Abweichungen −0,024 ± 0,039 (n.s.), aber
  +0,99 ± 0,32 d Durchlaufzeit (die Policy beachtet die Zeit nicht); regel besser als konformanz um 0,057 ± 0,027 Abweichungen / 2,4 ± 0,6
  Pp.; orakel um 0,065 ± 0,039 / 3,5 ± 1,3 Pp.; orakel gegen bank −0,089 ± 0,027 / +7,7 ± 0,8 Pp.
- Die Durchlaufzeit-Regel ist für die Conformance fast so gut wie das Orakel: ihre festen Optionen (annehmen, senden, zurückziehen) sind in
  den meisten Teilgruppen auch conformance-günstig; das Orakel gewinnt nur an konzept ≥ 11.000 € und leads < 11.000 €.
- Wahlen der konformanz-Policy (2 Optionen): konzept annehmen 719 / vervollständigen 326, nach V. weiter 1.043 / annehmen 701, nachlegen
  senden 2.104 / weiteres Angebot 208, leads zurückziehen 1.001 / Rückruf 438. An den Punkten mit kleinen wahren Unterschieden (nach V.,
  leads, konzept) streuen die Wahlen stark → Rauschen der Ziehungen (Unterschiede ≈ 0,1 Abweichungen bei ≈ 1 je Spur) entscheidet oft.
  Gleichstand → Bank kam nur 80-mal vor (Toleranz 0).
- Hebel: nur bei deutlichem Unterschied abweichen (Toleranz ≈ 2 gepaarte SE, sonst Bank); Zeit mitbewerten; mehr Ziehungen; schwach gelernte
  Effekte an konzept / leads (Aktionsarm hat wenige Proben je Option).

## 2026-11-21 00:35: Korrigierte Diagnosen – gezogene Suffixe realistisch konform; k3 lernt den Conformance-Effekt an nachlegen, kaum an konzept / leads

- **Ziehqualität** (`diagnose/ziehfehler.md`, 3 × 40 Fälle, 20 Ziehungen je Option, k3): Abweichungen je Spur gezogen / tatsächlich simuliert
  nachlegen senden 1,00 / 0,80, weiteres Angebot 1,35 / 1,67 (9 echte), konzept vervollständigen 1,08 / 1,07, nach V. weiter 1,22 / 1,02,
  leads zurückziehen 1,06 / 1,07, Rückruf 1,27 / 0,88 → nahe am echten Verlauf, leicht zu hoch. Spuren mit nie beobachtetem Übergang
  gezogen 15–30 % (echt 1 %), mit 2,44 statt 0,80 Abweichungen (Anteil 22 % → ≈ +0,35 je Spur). Fast alle Suffixe enden ordentlich (1–3 %
  an der Schrittgrenze). Häufigste Züge gezogen wie echt (Nachfassen fehlt, O_Cancelled / O_Returned außer der Reihe, A_Complete doppelt).
- **Gelernte Effekte** (`diagnose/konformanz_signal.md`, 6 × 60 Fälle, 50 Ziehungen; andere − Regel-Option, Wahrheit / k3 / svk):
  nachlegen 0,84–1,71 d +0,28 / +0,35 / +2,24, ab 1,71 d −0,26 / −0,03 / +2,29, unter 0,84 d +1,38 / +0,63 / +2,29 → **k3 trifft die
  Reihenfolge der Teilgruppen**, Spannweite etwa halb so groß, Vorzeichen ab 1,71 d knapp nicht; svk (echter Log) ohne Signal.
  konzept ≥ 11.000 € −0,06 / +0,11 / −0,19, < 11.000 € +0,53 / +0,16 / −0,10 → Richtung des Unterschieds richtig, aber nur 0,06 statt 0,59.
  nach V. +0,05 / −0,08, +0,09 / −0,05 (beide ≈ 0). leads < 11.000 € −0,43 / +0,13, sonst +0,08 / +0,15 → kein Unterschied gelernt (dort
  ist der wahre Unterschied eine Nebenwirkung des Zeiteffekts).
- Erwartung online: Conformance-Policy ≈ Regel-Optionen an konzept / leads, an nachlegen teils „weiteres Angebot“ bei langer Dauer.

## 2026-11-21 00:20: Korrektur – Fehler in der Conformance-Policy (Option doppelt), Online-Auswertung neu gestartet

Nutzer: „Ich dachte, beim Ziehen macht das PSP durch TF wenige Fehler.“ Richtig für einzelne Übergänge (Register: ZZ-TF1 / svk 0,6–0,85
nie beobachtete Übergänge je 100 Events, 15–18 % der Suffixe mit mindestens einem, 1–2 % ohne Ende; (c) ohne TF1: 48 je 100).
Die Aufschlüsselung (`diagnose/ziehfehler.py`) zeigte, dass die häufigsten „Abweichungen“ der gezogenen Spuren die Aktions-Optionen selbst
waren (unerlaubt O_Create Offer / A_Accepted / W_Handle leads in 100 % der Spuren dieser Option). Ursache: **die gezogenen Suffixe beginnen
schon mit dem Options-Event** (option_as encoder), KonformanzPolicy.bewerten setzte es zusätzlich davor → Option doppelt. Bei Referenz-
Optionen fällt das im angepassten Modell nicht auf (doppelte W_Complete-Sitzung bzw. doppelter Versand sind erlaubt, withdraw wird nicht
abgebildet), bei Aktions-Optionen kostet es je Spur eine Abweichung → die scheinbare „+1 je Arm“ (Einträge 12:00 und 23:55 ungültig).
- Korrektur `policies/konformanz_policy.py` mit_option (nur voranstellen, wenn der Suffix nicht damit beginnt); ebenso in ziehfehler.py.
- Online-Auswertung (Arm konformanz damit ungültig, noch ohne Ergebnisse) gestoppt und 00:17 neu gestartet (tmux psp_k3_online); falsche
  Ausgaben als *_fehler_option_doppelt.* aufbewahrt; Diagnosen konformanz_signal.py und ziehfehler.py laufen neu.

## 2026-11-20 23:55: Diagnose – der k3-PSP lernt die eingebauten Conformance-Effekte kaum; Ziehfehler je Arm dominieren

**FALSCH (Korrektur 2026-11-21 00:20): Fehler in KonformanzPolicy – das Options-Event stand doppelt in der Spur; die „+1 je Arm“ ist dieser Fehler, kein Netzunterschied. Zahlen unten ungültig.**

`kuenstliche_effekte_20261120/diagnose/konformanz_signal.py`: 6 × 60 frische Fälle (v3, Bank), erster Besuch je Punkt, KonformanzPolicy.bewerten
mit 50 Ziehungen je Option; geschätzte Abweichungen andere − Regel-Option je Teilgruppe, k3 (v3-Log) und svk (echter Log), gegen die Wahrheit:
- konzept: Wahrheit −0,06 (≥ 11.000 €) / +0,53 (< 11.000 €); k3 −0,89 / −0,84, svk −1,19 / −1,10 → kein Unterschied zwischen den
  Teilgruppen gelernt, beide sagen für „vervollständigen“ (Referenzarm) ≈ 1 Abweichung weniger voraus.
- nachlegen: Wahrheit +0,28 (0,84–1,71 d) / −0,26 (≥ 1,71 d) / +1,38 (< 0,84 d); k3 +1,35 / +0,97 / +1,63 (Reihenfolge teils richtig,
  Spannweite 0,66 statt 1,64, Niveau ≈ +1 zu hoch), svk ≈ +3,3 überall (kein Signal).
- nach V.: Wahrheit +0,05 / +0,09; k3 −1,08 / −1,05, svk ≈ −1,1. leads: Wahrheit −0,43 / +0,08; k3 +1,13 / +1,15, svk ≈ +1,2.
- Gezogene Abweichungen für die Regel-Option 0,8–2,2 je Suffix (echte Fälle ≈ 0,5–0,7) → der PSP zieht 2–4-mal zu viele Abweichungen.
- **Deutung:** das Vorzeichen folgt dem Arm – Optionen des Aktionsarms (A_Accepted, O_Create Offer, W_Handle leads start) bekommen
  ≈ 1 Abweichung mehr als die des Referenzarms, an jedem Punkt, bei beiden Modellen. Wie die Arm-Verschiebung der Restzeit beim Suffix-
  Verlust (Befund 11-19 20:45) ist das ein Unterschied der Netze, kein Effekt der Option. Die Conformance-Policy wird damit praktisch
  „immer die Referenz-Option“ wählen. Gelernt hat k3 nur an nachlegen etwas von der Teilgruppe.
- Mögliche Hebel (nicht umgesetzt): Kalibrierung der gezogenen Abweichungen je (Punkt, Option) wie svk; ein Modell für beide Optionen
  (S-Learner: Netzfehler heben sich zwischen den Optionen auf); beschränktes Ziehen nur erlaubter Übergänge (Befund 11-17).

## 2026-11-20 23:40: Arme auf dem v3-Log trainiert; Online-Auswertung nach Server-Neustart neu gestartet

- Training (CPU): Aktionsarm 13:01–13:55 (10 Epochen, Validierung 11,89 → 11,84, bester 11,74 in Ep. 9), Referenzarm bis 15:26 (5 Epochen),
  Hülle `models/psp_model_durchlaufzeit_t_learner_k3.pkl`.
- Der Server wurde um 15:41 neu gestartet (neuer Kernel 7.3.4, GPU wieder nutzbar). Dabei endeten alle tmux-Sitzungen; die Online-Auswertung
  (Start 15:26) lief nur 15 min und hinterließ keine Ergebnisse (Log `online_konformanz_abgebrochen_1526.log`). Neu gestartet 23:35 in tmux
  psp_k3_online, dazu die Diagnose `diagnose/konformanz_signal.py` (geschätzte Abweichungen andere − Regel-Option je Teilgruppe für k3 und
  svk gegen die Wahrheit).

## 2026-11-20 13:20: Conformance-Test gestartet – Bank-Log v3, Training der T-Learner-Arme (CPU), Online-Auswertung vorbereitet

Nutzer: „okay mach das“ (Bank-Log v3 erzeugen, PSP trainieren, Online-Auswertung Conformance).
- **Log** `kuenstliche_effekte_20261120/log_effekte_v3/` (`log_erzeugen.py`): Bank, Effekte v3, 8 × 4.000 Fälle (Saaten 2000–2007,
  Fall-IDs K<r>_), 1.282.870 Events, effekte.csv je ausgeführter Option; 1 min.
- **GPU nicht nutzbar** seit Mittag: NVIDIA-Kernelmodul 580.178, Bibliotheken 580.196 („Driver/library version mismatch“, torch.cuda
  nicht verfügbar) – braucht Neustart / Neuladen des Treibers durch einen Admin. Training läuft auf der CPU (16 Threads), Aktionsarm
  ≈ 7 min je Epoche (GPU ≈ 3,3).
- **Warmstart auf anderem Vokabular:** der Simulator-Log kennt 5 Event-Typen und 41 Mitarbeiter weniger; die Netzform (auch die Breite
  der Einbettungen) hängt vom Vokabular ab. Lösung `psp_model` vocab_from / `train_psp_real.py --vocab-from`: der EventLogLoader
  übernimmt nach dem Anpassen die Kategorien des Quellmodells (OrdinalEncoder sortiert → gleiche Nummern), leere Werte (LoanGoal NaN im
  Simulator) gelten als unbekannt; danach strenger Warmstart. (Erster Versuch, Gewichte über Namen zuzuordnen, scheiterte an der
  Einbettungsbreite – verworfen.)
- **Lauf** `data/bpic17_psp_konformanz_v3.sh` (tmux psp_k3, ab 13:01): Arme warm von den T-Learner-Armen des echten Logs, nur Wahlpunkt-
  Proben, LR 1e-4, Aktion 10 / Referenz 5 Epochen, TAGs `k3_zaehler_zeit_tf1_t_{aktion,referenz}`, Hülle `t_learner_k3.pkl`; danach
  `online_konformanz.py` (Aufbau F, 12 Seeds, Arme bank / regel / orakel / konformanz, KonformanzPolicy 100 Ziehungen). Aktion Epoche 1:
  Validierung 11,89. Test der schnellen Arme (Seed 5): Abweichungen je Fall bank 0,69 / regel 0,56 / orakel 0,59, konform 65,5 / 66,5 /
  70,5 % (ein Seed).
- Nebenbei: `conformance_probe.py` und `wahrheit.py` lasen ihre Kommandozeile beim Import – conformance_probe jetzt erst in main.

## 2026-11-20 12:40: Kalibrierung mit getrennten Prefix-Hälften (svh) – online +2,15, schwächer als svk

- **Online** tlsvh_fest300 (12 × 200): +2,15 ± 0,42 gegen die Bank; Regel schneller um 0,48 ± 0,25, tlsvk um 0,45 ± 0,24 (z 1,9),
  tlsv um 0,08 ± 0,21. Abweichungen von der Regel konzept / nach V. / nachlegen / leads 1 / 6 / 21 / 2 % (svk 0 / 6 / 1 / 1).
- **Diagnose:** Vorteil der Regel-Option nachlegen 1. Besuch 0,85 (Wahrheit 2,11; svk 2,11), konzept 2,05, nach V. 2,16, leads 2,08;
  Prognose falsch 0,5 / 1,2 / 17,3 / 1,0 %, Rückstand 0,28 d. Testfälle (`je_punkt_svh_400.md`): Fehler je Gruppe −0,14 … +1,62 d (ähnlich
  svk), SD der Ziehungen 16–20 d (sv 14–16, svk 11,7–15,5) – breiter als ohne Kalibrierung; rauschfreie Streuung der Fallmittel bis 1,5 d,
  Rangkorrelation weiter ≈ 0 (−0,11 … +0,06) → die zusätzliche Streuung ist kein Fallwissen.
- **Bilanz der drei Kalibrierungen:** global (svk) am besten (+2,61 = Regel), schrumpft aber Fall- und Ziehungsstreuung; gepaart (svp)
  scheitert am Rauschen der einzelnen Ausgänge (−0,48); getrennte Hälften (svh) ohne Schrumpfdruck, aber verrauschter Gradient, breitere
  Ziehungen, zu kleiner Nachlegen-Effekt (+2,15). In BPIC17 bleibt svk die beste Wahl; ob sein Schrumpfen fallabhängige Effekte kostet,
  kann nur der Simulator mit künstlichen Effekten zeigen.

## 2026-11-20 12:00: Conformance-Abweichungen im Simulator (effekte_v3) und Conformance-Policy (policies/konformanz_policy.py)

- **Protokoll-Abweichungen** in `kuenstliche_effekte` (Regel `abweichung: {art, aktivitaet, anteil}`): auslassen (Event bzw. ganzer
  Arbeitsschritt fehlt im Protokoll), doppeln (+1 ms), vertauschen (nach dem folgenden Event); Münze beim Options-Event, eigener
  Zufallsstrom; nur engine.events / case.prefix ändern sich, der Simulator läuft intern weiter → wirkt auf Conformance, nicht auf Zeit.
- **effekte_v3.yaml** = v2 + Abweichungen: konzept Betrag ≥ 11.000 € sofort annehmen → 40 % Prüfung übersprungen; < 11.000 €
  vervollständigen → 30 % A_Complete doppelt (doppelter Versand war im angepassten Modell konform – Messung 11:40); nachlegen 1. Besuch
  ab 1,71 d senden → 50 % Nachfassen ausgelassen; unter 0,84 d weiteres Angebot → 60 % O_Created vertauscht. nach V. / leads keine.
- **Bank-Log v3** (`probe.py`, 4.000 Fälle): Durchlaufzeit +0,16 d, Conformance 79,6 → 59,9 % konform, Abweichungen 0,49 → 0,72 je Fall
  (stärker als geschätzt, vor allem das ausgelassene Nachfassen, weil die Bank zu 92 % sendet).
- **Wahrheit v3** (`wahrheit.py` mit Alignment je Fall, 3 × 4.000; Abweichungen andere − Regel-Option je Fall): konzept ≥ 11.000 €
  −0,06 / < 11.000 € +0,53; nachlegen ab 1,71 d −0,26 / 0,84–1,71 d +0,28 / unter 0,84 d +1,38; leads < 11.000 € −0,43 (Nebenwirkung des
  Zeiteffekts, keine Abweichung eingebaut) / sonst +0,08; nach V. +0,05 / +0,09. Erster Lauf mit 15 % ab 0,84 d: nachlegen kippte nicht
  (das weitere Angebot erzeugt von sich aus ≈ +0,25). Zeiteffekte unverändert wie v2.
- **KonformanzPolicy** (`prescriptive/policies/konformanz_policy.py`, Unterklasse von PSPPolicy): je Option samples PSP-Suffixe
  (option_suffixes), Spur = Prefix + Option + Suffix auf Aktivitätsebene, pm4py-Alignment (apply_trace, Zwischenspeicher je Spur) gegen
  BPIC17_ToBe_angepasst, Option mit den wenigsten erwarteten Abweichungen; Gleichstand → Bank. Test (`tests/konformanz_policy_test.py`,
  40 Fälle, v3, PSP svk vom echten Log): konforme Spur 0, Prüfung ausgelassen 1, A_Complete doppelt 1; 159 Entscheidungen in 460 s
  (≈ 2,9 s bei 50 Ziehungen, Zwischenspeicher 64 % Treffer). ~~Das PSP vom echten Log zieht selbst 0,5–1,9 Abweichungen je Suffix
  (Aktionsarm mehr)~~ – falsch, Option doppelt in der Spur (Korrektur 2026-11-21 00:20).
- Offen (braucht OK): Bank-Log v3 erzeugen, PSP darauf trainieren, Online-Auswertung mit Conformance als Zielgröße.

## 2026-11-20 11:25: Angepasstes normatives Modell BPIC17_ToBe_angepasst – 80,6 % der echten Fälle konform

Nutzer: angepasste Kopie von BPIC17_ToBe, neue Abweichungen gern auch (wie Mustroph), Zielgröße Conformance zuerst separat in eigener
Policy unter policies/. Offene Teilfrage („zwei Angebote erstellen, dann senden“) von mir entschieden: keine Abweichung (sonst wäre die
Option „weiteres Angebot“ an angebot_nachlegen immer ein Verstoß, kein fallabhängiger Effekt); umstellbar.
- `kuenstliche_effekte_20261120/modell_anpassen.py` → `process_models/BPIC17_ToBe_angepasst.bpmn` (Original unverändert, 42 neue
  Elemente ohne Layout): (1) W_Complete application vor und nach A_Concept beliebig oft; (2) mehrere Angebotsvarianten vor dem Versand,
  mehrere Versände (erste Runde und Folgeangebote); (3) Prüfung nach Nachreichung fortgesetzt statt neu gestartet (resume ist in der
  Projektion unsichtbar); (4) neues Angebot ohne vorheriges Storno; (5) Storno offener Angebote nach A_Pending; (6) erneute Prüfung
  nach Betrugsverdacht.
- `conformance_probe.py` (Standard jetzt das angepasste Modell; Konformität = keine Log- / Modellzüge, pm4py rechnet stille Übergänge mit
  Kosten 1, Abweichungen mit 10.000 – die frühere Angabe „0 % konform“ war daher auch ein Messfehler): echter Log (1.000 Fälle) Fitness
  0,981, **80,6 % konform**, 0,51 Abweichungen je Fall (Original-ToBe: 0,911 / gut 2 je Fall); simulierter RCT-Log 0,978, 72,5 %, 0,60.
- Übrige Abweichungen im echten Log (plausible echte Verstöße): O_Returned zusätzlich (0,098) bzw. fehlend (0,056), neues Angebot während
  der Prüfung (O_Create Offer / O_Created 0,061), O_Cancelled außer der Reihe (0,047; Simulator 0,133 – passt zu dessen zu vielen späten
  Angeboten), erneute Prüfung (0,025).
- `svh` (getrennte Hälften): Prognose falsch 0,5 / 1,2 / 17,3 / 1,0 %, Rückstand 0,28 d (svk 0,03); Online seit 11:13.

## 2026-11-20 10:50: Ziel Conformance – Alignments gegen BPIC17_ToBe, erste Messung

Nutzer: Sinn des Effekt-Simulators soll sein zu prüfen, ob das PSP auch Conformance-Abweichungen verhindern kann (Alignment wie
`Conformance Checking Mustroph/src/conformance_checking/conformance.py`: BPMN → Petri-Netz, pm4py-Alignments, Kosten / Fitness getrennt
für Prefix und Suffix; dort auch für die gezogenen Suffixe). Mustrophs künstlicher Repair-Simulator erzeugt Abweichungen als 4 verbotene
Ablaufvarianten (falsche Reihenfolge, doppelte Reparatur, ausgelassene Qualitätskontrolle). → Die Effektschicht muss Abweichungen
erzeugen bzw. verhindern können.
- `kuenstliche_effekte_20261120/conformance_probe.py`: Modell `process_models/BPIC17_ToBe.bpmn` (24 Aktivitäten, pm4py liest es),
  Projektion A_/O_ complete, W_ start; 400 Fälle je Log. Echter Log: Fitness 0,911, **0 % konform**; simulierter RCT-Log: 0,858, 0 %.
- Ursache überwiegend systematische Fehlpassung, keine echten Abweichungen: (1) W_Complete application beginnt im Log vor A_Concept,
  das Modell verlangt A_Concept zuerst (fast jeder Fall: „fehlt“ + „unerlaubt“); (2) das Modell verlangt O_Sent direkt nach jedem
  O_Created – zwei Angebote erstellen und dann senden (Option „weiteres Angebot“ an angebot_nachlegen) ist eine Abweichung; (3) Zahl der
  O_Cancelled. Vor einer Nutzung als Zielgröße muss Modell oder Projektion angepasst werden (offen, Entscheidung des Nutzers).

## 2026-11-20 10:35: Ablauf-Effekte im Simulator (effekte_v2) – Auftreten von Aktivitäten statt nur Zeit, gegenläufig je Teilgruppe

Nutzer: Effekte sollten eher die Durchlaufzeit oder das Auftreten anderer Aktivitäten beeinflussen (v1 wirkte nur auf Wartezeiten; der
Betrag ist nur das Merkmal der Teilgruppe). Umsetzung in `kuenstliche_effekte` (zusätzlich, v1 unverändert nutzbar):
- **ablauf** {Event: Zuschlag}: Hülle um das Next-Event-Modell addiert ab der Option bei jedem Ziehen einen Zuschlag auf die Logits
  (+1 = Chance × e), nur für Events mit ≥ 0,1 % Wahrscheinlichkeit (keine neuen Übergänge), nicht direkt nach Entscheidungs-Events.
  Die Zeit zieht das Modell passend zum Event (z. B. Frist vor A_Cancelled).
- **anteil** {Event: α}: an der ersten Stelle, an der das Event erlaubt ist, mit Wahrscheinlichkeit α erzwingen (α < 0 verbieten); eigener
  Zufallsstrom je Fall. Nötig, wo das Modell je Fall fast sicher ist.
- **Diagnose konzept** (`diagnose/`): A_Incomplete ist nur direkt nach W_Call incomplete files|start möglich (dort 99,6 %); die Verzweigung
  in die Nachfrageschleife liegt nach W_Validate application (complete / ate_abort), dort je Fall ≈ 99 % oder ≈ 0,0001 % → Logit-Zuschlag
  wirkungslos. Als Anteil erzwungen: +0,23 Schleifen je Fall, aber keine Änderung der Durchlaufzeit (Schleife kostet nur bei angenommenen
  Fällen Zeit: 21,0 statt 12,8 d; über alle +0,4 d). Abbruch kostet ≈ 12 d (30,2 statt 18,3) → konzept über Annahme / Abbruch.
- **effekte_v2.yaml**: konzept (Betrag ≥ 11.000 €) sofort annehmen → A_Cancelled +1,8 / O_Accepted −1,8, vervollständigen → −0,8 / +0,8;
  nachlegen (1. Besuch, ≥ 0,84 d) weiteres Angebot → O_Accepted +1,5 / A_Cancelled −1,5, senden → A_Cancelled +0,1; nach V. und leads
  Zeiteffekte wie v1.
- **Nähe zum Original** (Bank, 4.000 Fälle, Saaten 7 / 8 / 9): Durchlaufzeit +0,16 / −0,21 / −0,61 d; A_Cancelled je Fall +0,01–0,03,
  O_Accepted ± 0,01, A_Incomplete −0,06–0,07 (≈ −9 %, Nebenwirkung), A_Denied −0,01.
- **Wahrheit je Teilgruppe** (3 Saaten × 4.000, Vorteil der Regel-Option, Teilgruppe / Rest): konzept −1,43 / +4,35 (andere Option:
  A_Cancelled −0,54, O_Accepted +0,56 je Fall in der Teilgruppe), nach V. −3,59 / +4,69, nachlegen −1,37 / +1,40 (A_Cancelled −0,20,
  O_Accepted +0,22), leads −2,82 / +2,15. → zwei Punkte wirken über den Ablauf, zwei über die Zeit; an allen vier kippt die beste Option
  mit der Teilgruppe. Zwischenstände: konzept mit Zuschlag ±1,2 / 0,55 nur −0,10; nachlegen schwankt mit dem Zuschlag beim Senden
  (−2,29 bei +0,25, −1,37 bei +0,1).

## 2026-11-20 09:50: Simulator mit künstlichen Effekten – Effektschicht gebaut, Log bleibt nah am Original, Effekte je Teilgruppe gegenläufig

Nutzer: „originaltreu nachbauen mit allen Aktivitäten und Lifecycles, wir ändern nur Kleinigkeiten wie Effekte gewisser Punkte und
Aktionen künstlich“ (Idee aus SCOPE / SimBPIC17; SimBPIC17 selbst ist nur ein Spielzeug-BPIC17 mit 7 Aktivitätstypen).
- **Umsetzung** `process-simulation-engine/kuenstliche_effekte/` (neues Paket, keine bestehende Datei geändert): EffectEngine (Unterklasse
  der CaseEngine) – wird an einem Wahlpunkt eine Option ausgeführt (Bank oder Policy), gibt die Effektdatei Tage vor: > 0 Verzögerung des
  nächsten Wartespalts (Zeit vor dem nächsten W_* start / resume), < 0 Beschleunigung (von den folgenden Wartespalten abgezogen, je höchstens
  bis 60 s). Regeln mit Bedingungen auf Betrag, Kreditzweck, Antragsart, bisheriger Dauer, nur erster Besuch. `aktivieren()` schaltet nur im
  eigenen Prozess ein; effect_log je ausgeführter Option.
- **effekte_v1.yaml**: je Punkt eine Teilgruppe (konzept Betrag ≥ 11.000 €, nach V. Car / Existing loan takeover, nachlegen 1. Besuch ab
  0,84 d Dauer, leads Betrag < 11.000 €), dort die andere Option um D = 4,5 / 4,0 / 3,5 / 3,0 d besser gestellt, aufgeteilt in Verzögerung
  der Regel-Option und Beschleunigung der anderen Option im Verhältnis der Bank-Anteile (damit sich die Effekte im Bank-Log aufheben).
- **Nähe zum Original** (`probe.py`, 4.000 Fälle, Bank, gleiche Saat ohne / mit): eingebaute Tage je Fall netto +0,06; Durchlaufzeit
  +0,81 / +0,11 / +0,52 d (Saaten 7 / 8 / 9, Mittel ≈ +0,5 d ≈ 2 %; ohne 21,8–22,4 d), Median ähnlich, P90 +1,1–1,9 d; Events je Fall
  +0,2–0,5; keine neuen Event-Typen außer seltenen Zufallstreffern.
- **Wahre Effekte je Teilgruppe** (`wahrheit.py`, Option an einem Punkt für alle Fälle erzwungen, 3 Saaten × 4.000, Vorteil der Regel-Option
  in Tagen, ohne → mit Effekten): konzept Teilgruppe +4,78 → −0,70, Rest +4,16 → +5,20; nach V. +4,92 → −3,60, Rest +3,63 → +4,29;
  nachlegen +1,51 → −2,33, Rest +1,74 → +1,32; leads +1,81 → −3,32, Rest +1,21 → +1,69 → an allen vier Punkten kehrt sich das Vorzeichen
  in der Teilgruppe um, sonst bleibt die Regel-Option besser. Wirksame Größe teils über dem Entwurf (nach V. ≈ 8,5 statt 4,0, leads ≈ 5,1
  statt 3,0: bei erzwungener Option an jedem Besuch summieren sich die Effekte; dazu Folgewirkungen im Simulator). Abweichungen im Rest
  bis ≈ 1 d (Paarungsrauschen / Rückwirkung über Mitarbeiter). Die Wahrheit muss also weiter simuliert werden; der Entwurf gibt Richtung
  und Teilgruppe vor.
- Hinweis: die eigenen Effekte hier (konzept 4,5, nach V. 4,3) sind größer als die bisherige Wahrheit (2,79 / 2,40), weil die Option
  für alle Fälle und an jedem Besuch erzwungen wird (andere Messung als Aufbau F).
- Nächste Schritte (brauchen OK): Bank-Log mit Effekten erzeugen (≈ 32.000 Fälle, Minuten), PSP darauf trainieren (Warmstart vom ZZ-TF1,
  dann T-Learner-Arme, Suffix-Verlust, Kalibrierung), Vergleich mit Causal Forest / X-Learner / Regel / Orakel (kennt die Teilgruppen),
  Online-Auswertung im Simulator mit Effekten.

## 2026-11-20 09:20: Gepaarte Kalibrierung (svp) gescheitert – online −0,48 gegen die Bank, Regel 3,1 Tage voraus

- **Online** tlsvp_fest300 (12 × 200): −0,48 ± 0,47 gegen die Bank; Regel schneller um 3,11 ± 0,48, tlsvk um 3,08 ± 0,50, tlsv um
  2,71 ± 0,56. Abweichungen von der Regel konzept / nach V. / nachlegen / leads 0 / 4 / 84 / 83 %, Angebote je Fall 2,31 (Regel 1,26).
- **Diagnose** (`wirkung_kalibrierung_gepaart/`): Vorteil der Regel-Option nachlegen 1. Besuch −0,88 (Wahrheit 2,11), leads −0,76 (1,32),
  konzept 3,64 (2,79), nach V. 3,48 (2,40); Prognose falsch 0 / 1 / 74,5 / 80,8 %, Rückstand 1,76 d. Prüfreihe: AIPW-Vorzeichen
  97,5 / 98 / 28,5 / 24 %.
- **Testfälle** (`je_punkt_svp_400.md`): Fehler Referenz / Aktion nachlegen +1,85 / −1,26, konzept +1,19 / +0,32, nach V. +2,69 / −0,32,
  leads +1,92 / +0,03 d – der Referenzarm liegt überall 1,2–2,7 d zu hoch, obwohl das Training je Gruppe nur −0,17 … +0,35 meldete.
  SD der Ziehungen 16–19,5 d (sv 14–16, svk 11,7–15,5, vorher 12,8–13,9); Streuung der Fallmittel an nachlegen 1,4 / 1,7 d, aber
  Rangkorrelation weiter ≈ 0 (−0,10 … +0,09) → die zusätzliche Streuung ist Rauschen, kein Fallwissen. Per-Schritt-Validierung
  unauffällig (12,15 statt 12,93).
- **Deutung (teils Hypothese):** der gepaarte Term regressiert das Batch-Mittel auf die einzelnen echten Restzeiten derselben Prefixe
  (Rauschen ≈ 13 d / √8 ≈ 4,6 d je Gruppe), mit Gewicht 3 rund zehnmal stärker als der CRPS-Gradient. Folge: verrauschte Updates und
  wahrscheinlich Anpassung an die zufälligen Ausgänge der Trainingsfälle (Lücke Training ↔ Test), breitere Ziehungen, Vorzeichen an
  nachlegen / leads gekippt. Unverzerrt heißt nicht brauchbar.
- **Schrumpfen bei svk nachgemessen** (`je_punkt_sv_400.md` / `je_punkt_svk_400.md` neu mit Streuung): rauschfreie Streuung der Fallmittel
  sv → svk z. B. nach V. Referenz 1,26 → 0,75, nachlegen Aktion 0,74 → 0,46, konzept Referenz 0,42 → 0; SD der Ziehungen 14–16 → 11,7–15,5 d.
  Rangkorrelation in sv, svk, svp ≈ 0 → in BPIC17 ging dadurch kein Fallwissen verloren.
- **Besserer Schätzer (nicht umgesetzt, braucht OK):** die Prefixe einer Gruppe im Batch in zwei disjunkte Hälften teilen und
  (Mittel Hälfte 1 − globales Gruppenmittel) · (Mittel Hälfte 2 − globales Gruppenmittel) bestrafen: unverzerrt für das Quadrat der
  Abweichung des Gruppenmittels, kein Druck auf Streuung zwischen Fällen oder Ziehungen, kein Rauschen der einzelnen echten Ausgänge.
- Bestes PSP bleibt tlsvk (+2,61).

## 2026-11-20 02:15: Gepaarte Kalibrierung umgesetzt und gestartet (svp)

- Nutzer „ja“ zur Korrektur (Eintrag 01:45). `psp_model.py` SEQUENCE `calibration_mode` ("global" = Stand svk, Standard; "paired"):
  Zielwert = echtes Mittel derselben Prefixe der Gruppe im Batch, Quadrat aus zwei unabhängigen Ziehungshälften
  ((Mittel A − echt) · (Mittel B − echt)). CLI `train_psp_real.py --sequence-calibration-mode paired`.
- Simulationsprüfung (8 Prefixe × 16 Ziehungen, wahres Modell): erwartete Strafe global 1,31 / 2,44 / 5,82 bei Streuung der Fallmittel
  0 / 3 / 6 d (wächst mit der Streuung, auch mit der Streuung der Ziehungen), gepaart ≈ 21 in allen Fällen (nur Rauschen der echten Werte,
  konstant) → kein Schrumpfdruck mehr; Preis: verrauschterer Gradient der Kalibrierung.
- `je_punkt.py` misst jetzt zusätzlich die rauschfreie Streuung der Fallmittel je Zelle und die SD der Ziehungen.
- Lauf 01:59 `data/bpic17_psp_suffix_kalibrierung_gepaart.sh` (tmux psp_svp): warm von den _sv-Armen, c = 3, Aktion 10 / Referenz 5
  Epochen, TAGs `..._t_{aktion,referenz}_svp`, Hülle `t_learner_svp(_crn).pkl`; danach je_punkt svp (+ svk, sv neu mit Streuung),
  `wirkung_kalibrierung_gepaart/` + Prognose, Prüfreihe, Online `tlsvp_fest300` (Zustimmung in `online_ok_svp`). Epoche 1 Aktion:
  Fehler je Gruppe −1,27 / +1,26 / −0,70 / −0,40 d (svk Epoche 1: −1,32 / +1,36 / +0,14 / −0,03).

## 2026-11-20 01:45: Online – T-Learner mit Suffix-Verlust + Kalibrierung (tlsvk_fest300) +2,61 Tage, gleichauf mit der Regel

- Aufbau F, 12 Seeds × 200, 300 Ziehungen, nachlegen nur 1. Besuch: **tlsvk +2,61 ± 0,47 gegen die Bank, Regel +2,63; Regel minus tlsvk
  0,02 ± 0,29** (z 0,08). tlsvk minus tlsv +0,37 ± 0,25, minus tlcrn +0,32 ± 0,36. Angebote je Fall 1,30 (Regel 1,26, tlsv 1,41),
  Anteil Regel-Option 98,6 %.
- Abweichungen von der Regel konzept / nach V. / nachlegen / leads 0 / 6 / 1 / 1 % (tlsv 9 / 20 / 10 / 1, Prognose 0,0 / 0,0 / 1,5 / 0,6);
  97 % der Fälle ohne Abweichung (Nachteil −0,10 ± 0,30, Paarungsrauschen ≈ 0). Erwarteter Rückstand aus den Abweichungsraten 0,06 d
  (Prognose 0,03), beobachtet 0,02 ± 0,29. Nur nach V. weicht noch spürbar ab (6 % statt 0 % vorhergesagt).
- **Damit erreicht ein PSP mit gezogenen Suffixen (Event + Zeit je Schritt) erstmals die feste Regel** und liegt auf Höhe von Causal Forest
  (+2,62 im korrigierten Simulator, anderer Aufbau, nicht gepaart vergleichbar).
- Vorbehalt (Nachfrage des Nutzers 01:00, „lernt das PSP fallabhängige Effekte dann nicht mehr?“): die Kalibrierung vergleicht das Mittel
  von ≈ 8 Prefixen × 16 Ziehungen je Gruppe im Batch mit dem globalen Gruppenmittel; die quadrierte Abweichung bestraft dadurch im Mittel
  auch die Streuung der Fallmittel innerhalb der Gruppe und die Streuung der Ziehungen (gezogene SD 14,8 → 12,8 d, rauschfreie Streuung
  des Vorteils nach V. 1,0 → 0,4 d). In BPIC17 ohne Folgen (Rangkorrelation vorher wie nachher ≈ 0), allgemein ein Schrumpfdruck gegen
  fallabhängige Effekte. Vorgeschlagene Korrektur (nicht umgesetzt, wartet auf OK): Zielwert = echtes Mittel derselben Prefixe im Batch,
  quadrierte Abweichung als Produkt zweier unabhängiger Ziehungshälften → für das wahre Modell im Mittel null, kein Druck auf Streuung.

## 2026-11-20 00:55: Kalibrierung trainiert – Fehler je Gruppe fast null, Vorteile nahe der Wahrheit, Prognose Rückstand 0,03 d; Online läuft

- **Training** (`data/bpic17_psp_suffix_kalibrierung.sh`, c = 3, 22:42–00:04): Fehler je Gruppe (gezogen − echtes Gruppenmittel, Mittel über
  Batches, letzte Epoche) Aktion nachlegen +0,08 / konzept +0,31 / nach V. −0,47 / leads −0,20; Referenz −0,07 / −0,17 / +0,15 / +0,06 d
  (vorher bis −3,9 d). Val-Verlust 15,34 (tlsv 15,18).
- **Testfälle** (`suffix_verlust_20261119/je_punkt_svk_400.md`, 400 × 30, erster Besuch): Fehler nachher Referenz / Aktion nachlegen
  +0,17 / −0,20, konzept +0,06 / +0,75, nach V. +1,85 / −0,54, leads +0,75 / +0,96 d. Rangkorrelation je Zelle weiter ≈ 0 (−0,05 … +0,09):
  kein fallindividuelles Signal, nur die Gruppenmittel stimmen jetzt.
- **Diagnose** (`wirkung_kalibrierung/`, 300 Simulatorzustände, Bedingung event), Vorteil der Regel-Option ZZ-TF1 / tlsv / **svk** / Wahrheit:
  nachlegen 1. Besuch 0,32 / 0,98 / **2,11** / 2,11; konzept 1,57 / 1,04 / **2,84** / 2,79; nach V. 2,38 / 1,79 / **3,42** / 2,40;
  leads 1,08 / 1,86 / **1,58** / 1,32 (nachlegen ab 2. Besuch 4,15 statt 0,42, wegen first_only ohne Belang).
- **Prognose** (Φ-Formel, `wirkung_kalibrierung/prognose.md`): falsch 0,0 / 0,0 / 1,5 / 0,6 % (konzept / nach V. / nachlegen / leads),
  erwarteter Rückstand zur Regel 0,027 d (tlsv 0,19 vorhergesagt, 0,40 beobachtet; ZZ-TF1 0,54).
- **Prüfreihe** (Register-Zeile „T-Learner Suffix-Verlust + Kalibrierung (Zufallszahlen je Event)“): AIPW-Vorzeichen 97 / 98 / 88,5 / 95 %
  (tlsv 66 / 89,5 / 75 / 94), Anteil |z| > 2 0,60 / 0,79 / 0,27 / 0,28; Anteil Aktion 0,97 / 0,98 / 0,12 / 0,05 → entscheidet fast wie die
  feste Regel. Simulator-Zustände richtig (100 Ziehungen) 1,00 / 1,00 / 0,90 / 0,98.
- **Kosten:** Optionskorrelation an nachlegen 0,70 → 0,34 (Rauschen bei 300 0,62 → 0,93 d; gesamt 0,72 → 0,57) – wird vom größeren
  Vorteil überdeckt. Ungültige Übergänge je 100 Events zwischen / innerhalb 0,61 / 0,40 → 0,85 / 0,48; Suffixe mit ungültigem Übergang
  15,4 → 17,7 %. Gezogene Durchlaufzeit 23,4 (echt 22,8, SD 12,8).
- **Mechanismus an nachlegen nicht wie im Simulator:** Simulator: andere Option +6,9 Events; Modell −0,21, Angebote Regel / andere 0,27 / 0,48.
  Die Tage stimmen, der Weg dahin (Schleife mit zusätzlichen Angeboten) nicht – der Vorteil sitzt in den gezogenen Zeiten.
- **Deutung:** Die Kalibrierung zwingt das Mittel je (Punkt, Option) auf das beobachtete Mittel im echten Log. Da es kaum fallindividuelle
  Effekte gibt (Screening) und die beobachteten Gruppenunterschiede hier dieselbe Richtung wie die Simulator-Wahrheit haben, entscheidet
  das PSP jetzt praktisch wie die Regel. Vorbehalt: die Zielwerte sind beobachtete (nicht kausale) Gruppenmittel; generisch (nur
  konfigurierte Wahlpunkte und Optionen), aber bei stark verzerrter Zuweisung im Log würde die Kalibrierung die Verzerrung übernehmen.
- **Online** `tlsvk_fest300` (12 Seeds, Zustimmung des Nutzers 22:53) seit 00:30, Ende ≈ 01:50 (tlsv brauchte 77 min).

## 2026-11-19 22:45: Kalibrierung je (Punkt, Option) umgesetzt, Vorhersage günstig, Training gestartet

- **Genauere Messung** (`je_punkt_sv_400.md`, 400 Testfälle × 30 Ziehungen je Zelle; der Lauf mit 600 brach am Speicher ab, Arbeitsprozesse
  bekamen die Argumente nicht): Fehler vorher (gezogen − echt, Tage) Referenz / Aktion: nachlegen −1,85 / −3,88, konzept −2,68 / −0,81,
  nach V. −0,89 / −1,94, leads −1,96 / −2,28. Nach dem Suffix-Verlust: +0,01 / −1,54, −0,62 / +1,74, +1,36 / +0,27, +0,21 / +1,10 –
  Verschiebung je Arm fast gleich (Aktion +2,2 … +3,4, Referenz +1,9 … +2,3), bestätigt den Befund 20:45. Rangkorrelation je Zelle ≈ 0.
- **Vorhersage bei voller Kalibrierung** (Verschiebung je Gruppe = −Fehler vorher, übertragen auf die Diagnose-Zustände, Φ-Formel):
  Vorteil der Regel-Option nachlegen 0,32 → 2,34 (Wahrheit 2,11 / 1,55), konzept 1,57 → 3,44 (2,79), nach V. 2,39 → 1,33 (2,40),
  leads 1,08 → 1,40 (1,32); falsch 0,1 / 0,0 / 16 / 1,3 %; erwarteter Rückstand zur Regel ≈ 0,10 d (tlsv beobachtet 0,40). Idealisiert.
- **Umsetzung** (`psp_model.py`, opt-in): SEQUENCE `calibration` c; Gruppen = (Wahlpunkt, Option) über `PSPModel._sample_groups`
  (wie _weight_points), Gruppe als zweiter Zusatz-Tensor; die Prefixe des Suffix-Verlusts je Gruppe gleich häufig; Verlust zusätzlich
  c · Mittel über Gruppen im Batch von (Mittel gezogen − echtes Gruppenmittel über alle Trainingsproben)² / scale_days. Ausgabe je Epoche:
  Fehler je Gruppe. `train_psp_real.py --sequence-calibration c`.
- **Rauchtest** (`rauchtest_kalibrierung.log`, Aktionsarm warm vom T-Learner, c = 1, 2 Epochen): läuft, Validierung je Schritt 12,21 / 12,28
  (unverändert); echte Gruppenmittel Aktion nachlegen 22,19 / konzept 19,75 / nach V. 20,85 / leads 24,79 d; Fehler je Gruppe Ep. 1 → 2:
  −1,52 → −1,26, +1,10 → +1,27, −0,11 → +0,03, +0,13 → −0,41 → nach V. / leads kalibriert, konzept / nachlegen noch nicht getrennt.
- **Lauf** 22:42 `data/bpic17_psp_suffix_kalibrierung.sh` (tmux psp_svk): beide Arme warm von den _sv-Armen, c = 3, Aktion 10 / Referenz 5
  Epochen, TAGs `..._t_{aktion,referenz}_svk`; danach Hülle `t_learner_svk.pkl`, Kopie `..._svk_crn.pkl`, Messung je Gruppe
  (`je_punkt_svk_400.md`), Wirkung + Prognose (`wirkung_kalibrierung/`), Prüfreihe (Label „T-Learner Suffix-Verlust + Kalibrierung
  (Zufallszahlen je Event)“). Online (Arm `tlsvk_fest300`, eingetragen) läuft nur, wenn `suffix_verlust_20261119/online_ok_svk` existiert.

## 2026-11-19 20:45: Suffix-Verlust verschiebt je Arm, nicht je Punkt – Nachlegen-Gewinn ist Nebenwirkung der Arm-Verschiebung

Nutzer: „Dann mach das mal“ (Gewichtung je Wahlpunkt). Vorher gemessen: `suffix_verlust_20261119/je_punkt.py` (echte Testfälle, erster
Besuch je Punkt und Option, 100 Fälle × 100 Ziehungen, `je_punkt_sv_100.md`). Verschiebung der gezogenen Restzeit vorher → nachher:
Aktionsarm nachlegen +2,49 / konzept +2,38 / nach V. +2,34 / leads +3,27; Referenzarm +1,85 / +2,12 / +2,19 / +2,11 – fast gleich je Arm,
obwohl die Fehler vorher je Punkt verschieden waren (Aktion −4,32 / −0,99 / −1,96 / −1,99; Referenz −2,64 / −3,47 / −0,61 / −3,15 d).
Rangkorrelation gezogen ↔ echt je Zelle ≈ 0 vorher wie nachher. → Die Wirkung in der Diagnose 18:30 (nachlegen / leads +0,67 / +0,78,
konzept / nach V. −0,53 / −0,59) ist fast ganz der Unterschied der Arm-Verschiebungen (≈ 0,5 d), mit dem Vorzeichen, das sich aus der
Lage der Regel-Option (Referenz bei nachlegen / leads, Aktion bei konzept / nach V.) ergibt. Eine bloße Gewichtung je Punkt im Verlust
änderte nur die Größe der Arm-Verschiebung (ausgeglichen: beide Arme ≈ gleich → Nachlegen-Gewinn weg). Sinnvoller: Kalibrierung je
(Punkt, Option) – Mittel der gezogenen Restzeit je Gruppe gegen das echte Mittel. Vorab genauere Messung (600 Fälle × 30 Ziehungen,
`je_punkt_sv_600.md`) und Vorhersage der Wirkung, bevor trainiert wird.

## 2026-11-19 20:10: Online – T-Learner mit Suffix-Verlust (tlsv_fest300) +2,23 Tage, Fehler an nachlegen 34 → 10 %, mehr Abweichungen an konzept / nach V.

Aufbau F wie tlcrn (12 Seeds × 200, 300 Ziehungen, keyed_noise, first_only), Modell `models/psp_model_durchlaufzeit_t_learner_sv_crn.pkl`,
Lauf `data/bpic17_psp_tl_sv_online.sh` (18:48–20:05, 12 Prozesse), Bank-Prüfung reproduziert. Logs `suffix_verlust_20261119/online_tlsv.log`,
`abweichungen.log`, Auswertung `modellvergleich_20261116/f_zwoelf_auswertung.md`.
- Gewinn gegen Bank **+2,23 ± 0,50** (tlcrn +2,29, tlrct +1,98, c300 +1,97, Regel +2,63). Gepaart: Regel vor tlsv 0,40 ± 0,32 (n.s.),
  tlcrn vs tlsv 0,05 ± 0,42 (gleich), tlsv vor c300 0,26 ± 0,31.
- Abweichungen von der Regel (konzept / nach V. / nachlegen 1. / leads): **9 / 20 / 10 / 1 %** (tlcrn 1 / 3 / 34 / 8; Prognose 4,4 / 6,8 /
  5,7 / 0,2). 81 % der Fälle ganz ohne Abweichung (tlcrn 63 %). Beitrag je Fall: nachlegen 0,12 d (tlcrn 0,78), konzept 0,09, nach V. 0,02.
- Erwarteter Nachteil aus den Abweichungsraten 0,42 d (tlcrn 0,61), beobachtet 0,40; Fälle ohne Abweichung +0,11 ± 0,34 (kein
  Paarungsglück, bei tlcrn −0,77). **Bereinigt ≈ +2,2 (tlcrn ≈ +2,0)** → bestes PSP, Abstand zur Regel ≈ 0,4 d.
- Deutung: Der Suffix-Verlust löst das Nachlegen-Problem weitgehend (Prognose bestätigt: 34 → 10 %). Der neue Rest-Rückstand kommt aus
  konzept / nach V.: die Wirkung dort wurde kleiner (Diagnose 1,57 → 1,04 / 2,39 → 1,79), online mehr Abweichungen als prognostiziert
  (nach V. 20 statt 7 %). Nahe liegender nächster Hebel: Punktmischung im Suffix-Verlust (Verlust je Arm über Prefixe aller Punkte
  gemischt; konzept-Prefixe ggf. untergewichtet) – nicht untersucht.

## 2026-11-19 18:30: Suffix-Verlust, beide Arme trainiert – Nachlegen-Wirkung 0,32 → 0,98 d, erwartete Fehler an nachlegen 33 → 6 %; konzept / nach V. etwas schwächer

Lauf `data/bpic17_psp_suffix_verlust.sh` (16:10–18:17). Training: Aktion 10 Epochen (CRPS je Epoche 6,8–7,3 d ohne klaren Trend, gezogene
Restzeit im Training 20,3 → 21,3 bei echt 21,6), Referenz 5 Epochen (CRPS 7,16 → 7,13, gezogen 21,4 / echt 21,5); Validierungsverlust je
Schritt unverändert (Aktion 12,21–12,30). Modelle `models/psp_model_durchlaufzeit_t_learner_sv.pkl` (Hülle), `..._sv_crn.pkl` (keyed_noise).
- **Wirkungsdiagnose** (`suffix_verlust_20261119/wirkung/`, gleiche 300 Zustände und Saaten wie crn_schluessel, Bedingung event, gepaart je
  Zustand neu − alt): Vorteil der Regel-Option nachlegen 1. 0,32 → **0,98 d** (+0,67 ± 0,09; Simulator-Wahrheit 2,11 bzw. 1,55 mit Bank
  danach), nachlegen später 0,55 → 0,93 (+0,38 ± 0,15; Wahrheit 0,42), leads 1,08 → 1,86 (+0,78 ± 0,11; 1,32), aber konzept 1,57 → 1,04
  (−0,53 ± 0,09; 2,79) und nach V. 2,39 → 1,79 (−0,59 ± 0,14; 2,40). Mittel je Verlauf +2,2 bis +2,5 d (nachlegen 18,2 → 20,4), SD 12,4 → 14,1,
  ρ unverändert ≈ 0,7, Rauschen bei 300 Ziehungen 0,60 → 0,62.
- **Prognose online** (`prognose.py`, Φ-Formel wie 10:45, Wahlen je Fall aus tlcrn): falsch konzept / nach V. / nachlegen 1. / leads
  0,3 / 3,4 / 33,1 / 3,4 % → **4,4 / 6,8 / 5,7 / 0,2 %**; erwarteter Rückstand zur Regel 0,54 → **0,19 d** (tlcrn online: nachlegen 34 %,
  Prognose damit gut getroffen). Vorsicht: rauschfreie Streuung an nachlegen 1. auf 0 geschätzt; die Rechnung je Zustand ohne Entrauschen
  ergibt 41 → 23 %.
- **Prüfreihe** (Register „T-Learner Suffix-Verlust (Zufallszahlen je Event)“; die alte T-Learner-Zeile ist ohne keyed_noise, Unterschiede
  bei Vorzeichen-Anteilen also teils von der Kopplung): AIPW-Vorzeichen 66 / 90 / 75 / 94 % (alt 70 / 85 / 56 / 68); gegen Simulator-Wahrheit
  nachlegen1 0,28 → 0,98 (richtig 68 → 80 %), leads 0,80 → 1,86 (95 %), konzept 1,47 → 1,04, nach V. 2,47 → 1,79; **Gedächtnis: Restzeit
  nach Zusatzangebot −0,07 → 1,11 d** (Wahrheit 1,55; RCT-T-Learner 1,32), Nachfassen −0,28 → 0,41 (1,92); Lead-Gedächtnis 0,56 → 0,06
  und eingefügtes Angebot −0,07 (beide nur Referenzmodell, ohne Option gezogen); Gültigkeit der Abläufe unverändert (wie TF1);
  Durchlaufzeit gezogen 20,6 → 23,1 (echt 22,8).
- Deutung: Der Suffix-Verlust holt die späte Nachlegen-Wirkung zu einem großen Teil zurück, ohne Gültigkeit oder Schrittfehler zu
  verschlechtern, und erreicht beim Gedächtnistest fast den RCT-T-Learner – mit Daten nur aus dem echten Log. Preis: die Wirkung an konzept
  / nach V. wird kleiner (bleibt richtig gerichtet). Online-Test (`..._sv_crn.pkl`, Aufbau F, 12 Seeds, 300 Ziehungen) wartet auf Zustimmung.

## 2026-11-19 17:55: Online RCT-Lauf (Aufbau F, 12 Seeds) – T-Learner (RCT) +1,98, halbiert die Fehler an nachlegen (zirkulär)

`rct_tlearner_20261118/online_12_seeds.log`, Auswertung `modellvergleich_20261116/f_zwoelf_auswertung.md`, `rct_tlearner_20261118/abweichungen.log`.
- Gewinn gegen Bank: **tlrct_fest300 +1,98 ± 0,42**, zzrct_fest100 +0,99 ± 0,50 (zum Vergleich tlcrn +2,29, tl +1,52, c300 +1,97, Regel +2,63).
  Gepaart: tlrct vs tl (gleiche Bauart, Daten aus dem echten Log) +0,47 ± 0,21 (z 2,3); zzrct vs zztf1 +1,06 ± 0,42; Regel vor tlrct 0,65 ± 0,19.
- Abweichungen von der Regel (konzept / nach V. / nachlegen 1. / leads): tlrct 6 / 4 / **19** / 11 % (tl 8 / 10 / 39 / 20, tlcrn 1 / 3 / 34 / 8).
  Erwarteter Nachteil aus den Abweichungsraten 0,49 d (tlcrn 0,61, tl 0,93); Fälle ohne Abweichung −0,06 ± 0,34 (Paarungsrauschen klein).
- Deutung: Mit unverfälschten Daten halbiert sich der Fehleranteil an nachlegen auch ohne Zufallszahlen je Event; bereinigt liegt tlrct
  (≈ 2,63 − 0,49 = +2,14) gleichauf mit tlcrn (≈ +2,0). Zirkulär (Training im Prüf-Simulator) → nur Obergrenze dafür, was
  bessere Trainingsdaten bringen; für das echte Log nicht verfügbar. tlrct mit keyed_noise nicht getestet.

## 2026-11-19 16:15: RCT-Lauf, Prüfreihe der fertigen Modelle – mit RCT-Daten trägt das Ziehen den Nachlegen-Effekt (zirkulär); Suffix-Verlust-Training gestartet

Prüfreihe 15:11 (Register, Labels „ZZ-TF1 (RCT)“ / „T-Learner (RCT)“; Training auf 32.000 Simulator-RCT-Fällen, Prüfung an echten
Testprefixen bzw. Simulatorzuständen). Vergleich jeweils mit T-Learner (ZZ-TF1) vom echten Log:
- **Gegen die Simulator-Wahrheit** (Vorteil Regel-Option, Tage; Wahrheit konzept 2,79 / nach V. 2,40 / nachlegen1 2,11 / nachlegen2 0,42 / leads 1,32):
  ZZ-TF1 (RCT) 1,35 / 2,13 / 0,58 / 2,07 / 1,15; T-Learner (RCT) 1,64 / 1,63 / **0,94** / 1,85 / 1,23; T-Learner (ZZ-TF1) 1,47 / 2,47 / 0,28 / 0,77 / 0,80.
  Anteil richtig nachlegen1 65 / 72 % (68 %).
- **Gedächtnis:** Restzeit nach eingefügtem Zusatzangebot 1,07 / **1,32 d** (Wahrheit 1,55; T-Learner ZZ-TF1 −0,07), Nachfassen 0,49 / 0,81
  (1,92; −0,28), Lead-Gedächtnis 0,85 / 0,84 (2,83; 0,56).
- AIPW-Vorzeichen offline 82/85/66/78 bzw. 78/86/66/78 % (T-Learner ZZ-TF1 70/85/56/68). Storno-Frist weiter nicht gelernt
  (Median seit letztem O_Sent 10,7 / 11,1 d, 3,5 % in 29–32 d; echt 30,7 / 73 %).
- Deutung: Mit unverfälschten, vom Simulator erzeugten Daten tragen Architektur und freies Ziehen den späten Nachlegen-Effekt zu großen
  Teilen (1,3 von 1,55 d). Vorsicht: zirkulär (Trainingsdaten aus der Prüfumgebung, Simulator-Dynamik einfacher als der echte Log) → nur
  Obergrenze. Ob beim echten Log Confounding oder die schwerer lernbare echte Dynamik den Effekt kostet, bleibt offen. Online (zzrct_fest100,
  tlrct_fest300) läuft seit 15:13.
- 16:10 Training mit Suffix-Verlust gestartet (Nutzer: „führe das aus ja“): `data/bpic17_psp_suffix_verlust.sh`, tmux psp_sv, beide Arme
  warm vom T-Learner (ZZ-TF1), Aktion 10 / Referenz 5 Epochen, TAGs `real_alle_s60_normal_zaehler_zeit_tf1_t_{aktion,referenz}_sv`; danach
  Hülle `models/psp_model_durchlaufzeit_t_learner_sv.pkl`, Kopie mit keyed_noise `..._sv_crn.pkl`, Wirkungsdiagnose
  `suffix_verlust_20261119/wirkung.py` (gleiche 300 Zustände und Saaten wie crn_schluessel, gepaart gegen T-Learner ZZ-TF1) und Prüfreihe
  (Label „T-Learner Suffix-Verlust (Zufallszahlen je Event)“). Kein Online-Test ohne neues OK.

## 2026-11-19 15:05: Suffix-Verlust (sequence_loss) umgesetzt – Rauchtest: gezogene Restzeit nach Zusatzangebot 18,3 → 20,3 d (echt 22,7)

Nutzer: „ja“ zum Verlust auf ganzen gezogenen Suffixen. Umsetzung `prescriptive/estimators/psp_model.py` (SEQUENCE, `_sequence_draws`,
`_crps`, `_sequence_trainer`, `PSPModel._sequence_targets`; params `sequence_loss`, nur mit time_on_event; Standard aus), Optionen
`train_psp_real.py --sequence-loss --sequence-weight/-draws/-batch/-steps`. Je Batch ziehen 32 Prefixe je 16 Suffixe frei wie draw_cases
(Event aus verrauschten Logits ohne Gradient, Zeiten μ + σ·z mit Gradient, neue Dropout-Masken je Schritt, Werte ohne Gradient zurück);
CRPS der Summe der Abstände (≥ 0) gegen die echte Restzeit / 10 d, Gewicht 1 zum Verlust des Originals (Loss Attenuation, GradNorm,
Teacher Forcing 1 unverändert). Ordner `results_100k/suffix_verlust_20261119/`.
- **Gleichheit** (`gleichheit.py`, je Arm 40 echte nachlegen-Prefixe × 200 Ziehungen): Training zieht wie die Bewertung – Aktion 18,31 vs
  18,13 d (SD 12,86 / 12,86), Referenz 18,12 vs 18,17 (12,71 / 13,01). Nebenbefund: beide Arme ziehen für jeden Prefix ≈ 18 d
  (echt nach Zusatzangebot 22,5, nach Senden 20,2).
- **Rauchtest** (`rauchtest.py`, nur Aktionsarm, warm vom T-Learner-Arm, alle Fälle, 2 Epochen à ~2 min, Modell nur im Testordner):
  CRPS 7,19 → 6,84 d, gezogene Restzeit im Training 20,3 → 20,7 (echt 21,2 / 21,0); Validierungsverlust je Schritt 12,21 / 12,25 wie der
  Arm vorher (12,25) – die Schrittvorhersage leidet nicht.
- **Kurzprüfung** (`kurzpruefung.py`, 100 Testfälle mit Zusatzangebot × 100 Ziehungen): gezogene Restzeit 18,33 → **20,33 d** (echt 22,74),
  Rangkorrelation je Prefix 0,21 → 0,23, Streuung der Mittel je Prefix 1,33 → 1,55 d.
- Offen: Entscheidend ist der Unterschied zwischen den Armen am selben Prefix; dafür müssen beide Arme mit dem Verlust trainiert werden.
  Vorschlag (wartet auf Zustimmung): beide Arme warm vom T-Learner, Aktion ~10, Referenz ~5 Epochen, dann Kopplungs-/Effektdiagnose
  mit Zufallszahlen je Ereignis, dann online.

## 2026-11-19 14:10: Simulator erzeugt zu viele späte neue Angebote – nicht zu langsame Erstellung, nicht die Zuteilung

`modellvergleich_20261116/nachlegen_simulator_angebote.py` (Ausgabe `nachlegen_simulator_angebote/`: `nachlegen_simulator_angebote.md`,
`zellen.csv`, `kontraste.csv`, `uebergaenge_suspend.csv`, Rohfälle `faelle.pkl`): korrigierter Simulator, je Zuteilung (random wie Aufbau F,
learned + berechtigung erfahrung als Katalogkopie) 10 Seeds × 2.000 gewertete Fälle; erster Besuch an angebot_nachlegen 50/50 gelost,
zusätzlich die natürliche Wahl des Simulator-Modells festgehalten (EventTimeModel.draw umhüllt). Echter Log beobachtet.
- **Korrektur zu 11:15:** Das erzwungene O_Create Offer direkt nach der Entscheidung kommt im Simulator wie im Log sofort (0,002 vs 0,005 d).
  Die „0,92 vs 0,50 d je Zusatzangebot“ mischten sofortige und späte Angebote. Ursache ist die **Häufigkeit später neuer Angebote**.
- **Simulator-Fehler, für alle Fälle ohne Eingriff:** nach W_Call after offers|suspend folgt ein neues Angebot in 9,6–9,7 % (echt 5,5 %) bei
  1 Angebot, 11,8 vs 7,6 % bei 2, 12,4–13,6 vs 10,2 % bei 3; neue Angebote nach dem ersten O_Sent je Fall 0,32–0,33 vs 0,23 (+40 %);
  Wartezeit davor 6,1–6,3 vs 5,5 d. Fälle dauern 22,4–22,7 vs 21,9 d, Storno-Anteil gleich (0,33).
- **Zuteilung ist es nicht:** random und learned fast gleich (randomisierter Effekt Restzeit +1,86 ± 0,22 vs +1,74 ± 0,22 d, Übergangsanteile gleich).
  Also ein Fehler des Next-Event-Modells des Simulators (Übergang suspend → O_Create Offer zu hoch), gleichmäßig über die Angebotszahl.
- **Auswirkung auf die Wahrheit an nachlegen:** extra − senden in der Zeit vor späten Angeboten: Simulator randomisiert +0,83–0,93 d,
  Simulator beobachtet +0,96–1,18, echt beobachtet +0,51 → über diesen Kanal grob **0,3–0,5 d** zu viel. Insgesamt liegt der beobachtete
  Kontrast des Simulators (Restzeit, unerzwungen extra − senden) aber im Rauschen des echten (+2,66 ± 0,54 learned / +3,24 ± 0,59 random vs
  +2,28 ± 0,24 echt); der randomisierte Effekt (+1,7–1,9) liegt in beiden Zuteilungen deutlich unter dem beobachteten (Selektion: natürlich
  extra gewählte Fälle sind ohnehin ≈ 1,6 d langsamer).
- **Folgerung:** Die Simulator-Wahrheit an nachlegen (+1,55 d) ist eher um einige Zehntel Tage zu hoch, die Richtung (senden besser) bleibt.
  Für tlcrn würde der Rückstand zur Regel dadurch nur um grob 0,1 d kleiner. Das PSP liegt in diesem Kanal auch unter dem echten Log
  (+0,08 vs +0,51 d), der Hauptteil seines Defizits ist also PSP-seitig. Für die Arbeit als Einschränkung des Simulators festhalten;
  eine Korrektur des Simulators würde alle bisherigen Online-Ergebnisse verschieben (nur mit Zustimmung).

## 2026-11-19 12:55: Online – T-Learner mit Zufallszahlen je Ereignis (tlcrn_fest300) bestes PSP, +2,29 Tage

Aufbau F (korrigierter Simulator, 12 Seeds × 200 Fälle, gepaart, 300 Ziehungen fest, first_only angebot_nachlegen); einziger
Unterschied zu tl_fest300: params keyed_noise (Hüllen-Kopie `models/psp_model_durchlaufzeit_t_learner_zztf1_crn.pkl`).
Ergebnisse `modellvergleich_20261116/f_zwoelf_auswertung.md`, `abweichungen_regel.md`, Log `crn_schluessel/online_tlcrn.log`.

- **Gewinn gegen die Bank +2,29 ± 0,44 Tage** (T-Learner gleiche Ziehungen +1,52, (c) 300 +1,97, Regel +2,63).
- Gepaart: **gegen T-Learner +0,77 ± 0,32 (z 2,4)**, gegen (c) 300 +0,32 ± 0,30 (z 1,0), **Regel nur noch 0,35 ± 0,37 schneller (z 0,9, nicht signifikant)**.
- Abweichungen von der Regel (an echten Wahlen) konzept 8 → 1 %, nach V. 10 → 3 %, nachlegen (1.) 39 → 34 %, leads 20 → 8 %;
  vorhergesagt waren 0,3 / 0 / 30 / 3 % – an den drei klaren Punkten wirkt die Kopplung fast wie berechnet, an nachlegen bleibt der Bias.
- Vorsicht bei der Höhe: Fälle **ohne** Abweichung sind unter tlcrn 0,77 ± 0,37 Tage schneller als unter der Regel, obwohl dort
  dieselben Wahlen fallen – das ist Paarungsrauschen des Simulators (Zuteilungsstrom), etwa −2 SE. Aus den Abweichungsraten mit
  der wahren mittleren Wirkung erwartet: Nachteil 0,61 statt 0,93 Tage (tl_fest300) → **bereinigter Gewinn ≈ +2,0 Tage**, genau die
  Vorhersage aus der Kopplungsmessung (+1,9–2,0). Gegen T-Learner also sicher besser, gegen (c) 300 etwa gleich.
- Der verbleibende Rückstand liegt fast ganz an angebot_nachlegen (Beitrag +0,78 Tage je Fall, 31 % der Fälle mit Zusatzangebot);
  Ursache ist der Bias des Effekts (siehe 11:15: Angebotserstellung im Simulator langsamer als im Log, Nachfass-Wartezeiten und Storno-Anstieg zu klein).
- Folgerung: Zufallszahlen je Ereignis sind ein allgemeiner Hebel (keine datensatzspezifische Eingabe, ändert keine Invariante,
  Verteilung je Ziehung unverändert) und sollten für alle PSP-Bewertungen Standard werden; die Ziehungszahl wird dadurch weniger wichtig.

## 2026-11-19 11:15: Gezogene vs. echte Verläufe nach einem Zusatzangebot – wo die Zeit fehlt

`modellvergleich_20261116/nachlegen_verlaeufe.py` (`nachlegen_verlaeufe.md`, `_token.csv`, `_kennzahlen.csv`): ab der Entscheidung am ersten
nachlegen bis zur ersten W_Validate application|start (sonst Fallende); Simulator randomisiert mit Bank-Fortsetzung, PSP je 60 Zustände ×
100 Ziehungen, echter Log beobachtet (3.221 extra / 26.907 senden). Zeit je Token = Abstände vor dem Token.
- **extra − senden, Zeit bis Validierung/Ende:** echt +2,59, Simulator +2,28, T-Learner +0,21, Steigung −0,15, β-NLL 1 +0,40 d.
- **Events stimmen weitgehend:** O_Sent +1,06 / +1,06 / +0,90 (echt / Simulator / β-NLL), O_Created +1,27 / +1,28 / +1,18,
  Nachfass-Schleifen +0,38 / +0,18 / +0,15, O_Cancelled +0,68 / +0,65 / +0,49; Abstand der Option selbst ≈ 0 überall.
- **Zerlegung gegen den Simulator (online maßgeblich; 1,88 d fehlen bei β-NLL):** Zeit vor O_Create Offer +1,08 vs. +0,08 (≈ 1,0 d) – der
  Simulator braucht je zusätzlichem Angebot im Mittel 0,92 d, PSP 0,36–0,44, **echter Log 0,50**; Wartezeiten der Nachfass-Schritte
  (ate_abort +0,11 vs −0,23, resume +0,23 vs +0,05, withdraw −0,01 vs −0,20; ≈ 0,7 d); Storno +0,79 vs +0,63 (≈ 0,16 d).
- **Gegen den echten Log (2,19 d fehlen, beobachtet):** Storno +1,70 vs +0,63 (≈ 1,1 d; Storno-Anteil nach Zusatzangebot echt +7,8 Pp,
  Simulator +4,2 Pp, PSP +2,1–2,9 Pp), Zeit vor O_Create Offer +0,51 vs +0,08 (≈ 0,4 d), Nachfass-Wartezeiten ≈ 0,6 d.
- **Folgerung:** Kein einzelner Fehler. Ein guter Teil des Simulator-Effekts an nachlegen (≈ 0,5 d) entsteht, weil der Simulator zusätzliche
  Angebote langsamer erstellt als der echte Log (0,92 vs 0,50 d) – dort liegt das PSP näher an der Realität und wird online dafür bestraft.
  Echte Defizite des PSP: zu geringer Anstieg der Stornos nach Zusatzangebot und zu kurze Nachfass-Wartezeiten im frei gezogenen Verlauf.
- **Korrektur 14:10:** nicht langsamere Erstellung, sondern zu viele späte neue Angebote im Simulator; Anteil an der Wahrheit grob 0,3–0,5 d (Eintrag 14:10).

## 2026-11-19 11:05: Sonde „zwei Angebote“ – die Information steckt in h; der β-NLL-Kopf nutzt sie, der Fehler liegt im Ziehen

`zeitkopf_sonde_20261118/angebote_sonde.py` (Ordner `angebote_sonde/`): 3.000 Testfälle, Schnitt nach dem ersten O_Sent, echte Suffixe mit
Teacher Forcing 1, h = Eingang des Abstandskopfs; ausgewertet an 5.807 Warteschritten (W_Call after offers|ate_abort / |resume), Ridge
außerhalb der Stichprobe (5 Fallgruppen). Angebotszahlen nur in der Sonde.
- **Lesbarkeit:** Zahl erstellter / verschickter Angebote aus h R² 0,51–0,57; „≥ 2 O_Sent“ zu 93 % richtig (Grundrate 82 %). Die
  Ereigniszähler kommen in h an, abgeschwächt.
- **Wartezeit je Schritt, ≥ 2 minus 1 O_Sent (Tage):** echt ate_abort +0,76, resume −0,60 (gesamt +0,13). Ridge je Event auf h: +0,59 /
  −0,65 (gesamt +0,01–0,05); mit Angebotszahlen zusätzlich +0,15 gesamt, R² unverändert 0,284 → 0,285 → h enthält die Information.
  Trainierte Köpfe: T-Learner / Steigung ate_abort −0,36 / −0,31 (**falsches Vorzeichen**), resume −0,56 / −0,79; **β-NLL 1: +0,57 /
  −0,57 ≈ echt**.
- **Abschnitt letztes O_Sent → A_Validating** (Summe der Kopf-Mittel entlang der echten Events, 1.673 / 422 Fälle mit 1 / ≥ 2 O_Sent):
  echt 8,59 / 8,76 d; T-Learner 8,08 / 6,31, Steigung 8,75 / 7,01, ZZ-TF1 8,12 / 6,29 (je ~1,8 d zu kurz bei ≥ 2), **β-NLL 1 8,88 / 8,79**.
  Warteschritte je Fall 1,90 / 2,15.
- **Deutung:** Ergebnis A – h enthält die Information; die attenuierten Köpfe nutzen sie falsch (1/σ²: ate_abort hat 28 % Abstände < 1 h),
  der β-NLL-Kopf richtig. Trotzdem zieht die β-NLL-Variante an nachlegen keinen Effekt (Kundenantwort −0,44 statt +0,64, bis letztes O_Sent
  +0,23 statt +0,83) → der Fehler liegt nicht im Zeitkopf bei echten Events, sondern in den frei gezogenen Verläufen nach einem
  Zusatzangebot (Events, Schleifen, Zeitpunkt späterer O_Sent). Außerdem ist der Effekt populationsabhängig: über alle Fälle mit ≥ 2 O_Sent
  nur +0,17 d, bei Zusatzangebot am ersten nachlegen +0,64 / +0,83 d. Punkt 2a (nichtlinearer Kopf) und 2b (LSTM vergisst) sind damit
  nicht der nächste Schritt; nächster Prüfschritt: gezogene vs. echte Verläufe nach Zusatzangebot schrittweise vergleichen.

## 2026-11-19 10:45: Zufallszahlen je Ereignis statt je Schritt – Kopplung ρ 0,15 → 0,7, Rauschen −40 %, Verteilung unverändert

- Umsetzung (Nutzer: „teste die Zufallszahlen nach Event-Typ“): `psp_model.py` params `keyed_noise` (Standard aus, nur beim Ziehen,
  nur time_on_event-Modelle), `PSPModel._keyed_draw`, Hash `_keyed_uniform` (splitmix64). Event-Wahl: Logit-Rauschen + Gumbel-Max,
  Schlüssel (Saat, Ziehung, voriges Event, dessen Vorkommen im Suffix); Zeiten: Normal, Schlüssel (Saat, Ziehung, gezogenes Event,
  Vorkommen). Datensatzunabhängig; Dropout, Verlust, Training unverändert. Selbsttest: gleichverteilt, deterministisch, unkorreliert.
- Diagnose `modellvergleich_20261116/crn_schluessel.py` (300 Simulatorzustände × 100 Ziehungen, gleiche Saat; „schritt“ reproduziert
  punkte_alle). T-Learner schritt → event: ρ konzept 0,12 → 0,73, nach V. 0,13 → 0,69, nachlegen 1. 0,14 → 0,70, nachlegen später
  0,11 → 0,37, leads 0,17 → 0,72; Rauschen je Entscheidung bei 300 Ziehungen ≈ 1,0 → 0,57–0,60 d (später 0,82). Mittel je Verlauf
  (19,9 / 19,8 d usw.) und SD je Verlauf (12,8 / 12,8) gleich → Verteilung unverändert; Vorteil der Regel-Option im Rahmen des Rauschens
  gleich (1,47 / 1,57; 2,47 / 2,38; 0,28 / 0,32; 0,80 / 1,08). Zustände mit richtigem Vorzeichen (100 Ziehungen): konzept 75 → 95 %,
  nach V. 93 → 95 %, nachlegen 1. 68 → 67 %, leads 72 → 92 %. ZZ-TF1 (ein Netz) gleiche Kopplung (ρ 0,12–0,23 → 0,63–0,73).
- Erwartung online T-Learner, 300 Ziehungen (Φ-Formel): falsch konzept 6,9 → 0,3 %, nach V. 0,9 → 0 %, nachlegen 38,8 → 29,7 %, leads
  21,2 → 3,4 %; erwarteter Rückstand zur Regel (mit Simulator-Wahrheit bewertet) 0,88 → 0,48 d. nachlegen bleibt wegen der Verzerrung.

## 2026-11-19 10:00: MC-Dropout trägt nichts zum Entscheidungsrauschen bei (Diagnose an/aus)

`modellvergleich_20261116/dropout_anteil.py` (Ordner `dropout_anteil/`): dieselben 300 Simulatorzustände wie punkte_alle, 100 Ziehungen je
Option, gleiche Saat; „aus“ = Masken durch Einsen ersetzt (Encoder + Decoder, trainierte Rate fest 0,1 in allen 8 Zellen), Event- und
Zeitziehung unverändert. T-Learner und ZZ-TF1 (ein Netz). „an“ reproduziert punkte_alle des T-Learners exakt (1,47 / 2,47 / 0,28 / 0,77 / 0,80).
- **Dropout-Anteil an der Varianz eines Verlaufs: −10 … +2 % (≈ 0)**; SD je Verlauf an/aus 12,5–14,9 d praktisch gleich; ρ der Optionen
  an 0,11–0,17 (T-Learner) / 0,12–0,23 (ZZ-TF1), aus 0,08–0,16 / 0,12–0,23; Rauschen je Entscheidung bei 300 Ziehungen an / aus ≈ 1,0 / 1,0 d
  an allen Punkten. → Das Rauschen kommt aus der Event- und Zeitziehung je Schritt, nicht aus den Masken; ein geringerer Dropout kann es
  nicht senken. ρ ist auch mit einem Netz (ZZ-TF1) so niedrig → die zwei Netze des T-Learners sind nicht der Grund (Schritt-Ausrichtung).
- Auch die Verzerrung kommt nicht vom Dropout: Vorteil an nachlegen 1. Besuch an 0,28 / aus 0,05 d (T-Learner; ZZ-TF1 0,12 / 0,01).
- Rauschfreie Streuung der Schätzungen zwischen Zuständen (Varianz der Zustandsmittel minus SE², grob): T-Learner nachlegen 1. Besuch ≈ 0
  → das PSP schätzt dort für fast alle Fälle denselben kleinen Vorteil (0,28 d); mit unbegrenzt vielen Ziehungen wählte es meist die Regel,
  für ≤ 5 % Fehlentscheidungen bräuchte es aber rund 10.000 Ziehungen je Option (SE 0,17 d).

## 2026-11-19 09:20: Effekt je Punkt über alle Quellen (Nutzerfrage: kann AIPW sich irren?)

Vorteil der Regel-Option in Tagen, konzept / nach V. / nachlegen 1. / leads: AIPW echter Log 5,37 / 2,94 / 1,46 / 2,22 (E-Wert 2,25 / 1,75 /
1,45 / 1,60, Placebo |d| ≤ 0,07); Causal Forest (Testfälle, Mittel, SD) 4,21 (0,58) / 2,28 (0,25) / 1,18 (0,23) / 2,22 (0,63), wählt 100 %
Regel; Simulator randomisiert 2,79 / 2,40 / 2,11 (Bank-Fortsetzung 1,55) / 1,32; T-Learner (Simulatorzustände) 1,47 / 2,47 / 0,28 / 0,80;
Steigung + β-NLL 1: 2,22 / 3,09 / 0,57 / 1,07. An nachlegen zusätzlich naive Differenz im echten Log (beobachtet, unbereinigt) +2,29 d
Restzeit ab 1. O_Created. Alle Referenzen gleiches Vorzeichen; PSP an nachlegen unter jeder Referenz. AIPW und Forest teilen Daten und
Annahme (keine unbeobachteten Störgrößen), sind also keine unabhängige Bestätigung; online zählt ohnehin die randomisierte Simulator-Wahrheit.

## 2026-11-19 02:50: Zeitkopf je Event mit β-NLL 1 – Schranke verfehlt (nachlegen 0,57 d), Kundenantwort unverändert → 1/σ² ist nicht die Ursache

- Training 01:37–02:30 (Aktion 25 min, Referenz 28 min). Sonde (1.500 Fälle; Aktion / Referenz, in Klammern Steigung): Kopf an A_Cancelled
  21,2 / 20,9 d (20,6 / 21,4; echt 20,6), MAE dort 5,1 / 5,0 (6,3 / 5,8), σ 6,8 / 7,4; gezogen seit O_Sent bis Storno Median 27,8 / 27,9
  (26,7 / 27,8; echt 30,7), Anteil 29–32 d 13 %; MAE alle Schritte 0,42 / 0,43 (0,42); Antwort O_Sent → Validierung 9,4 / 8,8 bzw.
  9,8 / 9,0 d bei 1 / ≥ 2 Angeboten (Steigung 9,2 / 7,2; echt 7,0 / 6,9).
- Prüfreihe, Vorteil der Regel-Option (Simulator / Steigung / β-NLL 1): konzept 2,79 / 2,47 / 2,22; nach V. 2,40 / 2,88 / 3,09; **nachlegen
  1. Besuch 2,11 / 0,37 / 0,57**; nachlegen später 0,42 / 0,82 / 0,47; leads 1,32 / 1,27 / 1,07. MC-SE je Entscheidung 1,52 → 1,71.
- Wartezeiten nach dem Nachlegen (extra − senden): bis letztes O_Sent +0,23 (Steigung +0,24, echt +0,83), Kundenantwort **−0,44**
  (−0,43, echt +0,64), Nachfass-Start → Validierung −0,27 (−0,25, echt +1,45) – praktisch unverändert.
- **Schranke (≥ 1,0 d an nachlegen) verfehlt → kein Online-Test**; Suffixgüte läuft noch.
- **Folgerung:** Mit ungewichtetem Mittel (β = 1) lernt der Zeitkopf die längere Kundenantwort nach zwei Angeboten genauso wenig. Die
  1/σ²-Gewichtung erklärt also den Storno-Versatz (Befund 00:50), nicht den fehlenden Effekt an nachlegen. Die Vermutung von 01:35 ist
  widerlegt. Wahrscheinlicher: Der Zustand h an diesen Schritten enthält die Information „zwei Angebote offen“ nicht in einer Form, die der
  lineare Kopf je Event nutzen kann (ungeprüft).
- **Suffixgüte β-NLL 1** (1.000 Testprefixe, Aktion / Referenz, in Klammern Steigung): DLS 0,410 / 0,409 (0,407 / 0,408); Länge-MAE 10,7 /
  10,4 (10,9 / 10,5); 1. Abstand MAE 14,6 / 15,1 h (12,9 / 12,2); Restzeit MAE 8,87 / 8,94 d (8,48 / 8,45); **Bias +3,2 / +3,2 d** (+1,7 / +1,5);
  CRPS 5,54 / 5,56 (5,39 / 5,36); Abdeckung 80 % 69 / 66 % (71 / 75 %). Auf zufälligen Prefixen schlechter: die ungewichteten Mittel heben
  die kurzen Abstände an (zu lange Restzeit). Zeile in `suffix_tabelle_bild.py` ergänzt.

## 2026-11-19 01:50: Online-Ergebnis Zeitkopf je Event (Steigung), 12 Seeds – realistische Zeiten, kein Entscheidungsgewinn

- `tlks_fest300` (Aufbau F, 12 × 200 Fälle, 300 Ziehungen, first_only): Gewinn gegen Bank **+1,41 ± 0,38 d**; gepaart gegen T-Learner
  (`tl_fest300` +1,52) **−0,11 ± 0,30**, gegen (c) 300 −0,56 ± 0,42, gegen Regel −1,22 ± 0,28 (z 4,4). Regel-Option 80 % der Wahlen
  (T-Learner 76 %), 1,79 Angebote je Fall (Regel 1,26).
- Abweichungsanalyse neu mit 12 Seeds (`abweichungen_regel.py`): Abweichungen konzept / nach V. / nachlegen 1. / leads 2 / 2 / 43 / 7 %
  (T-Learner 8 / 10 / 39 / 20 %); Fälle ohne Abweichung −0,34 ± 0,40; Zusatzangebote an nachlegen kosten **3,00 d je Fall** (SE über Seeds
  0,45; Wahrheit bei Zufallsauswahl 1,55) und machen 1,19 der 1,22 d Rückstand aus. Der Zeitkopf hat drei Punkte an die Regel gebracht,
  an nachlegen bleibt der Fehler (geschätzter Effekt 0,37 statt ~1,55 d).

## 2026-11-19 01:40: Online-Test Versatz gestrichen; Zeitkopf je Event mit β-NLL β = 1 gestartet

- Auf Wunsch des Nutzers („okay dann mach das“) Online-Test `tlko_fest300` gestrichen (Pipeline psp_kopf_versatz vor dem Start beendet):
  Modell ≈ unveränderter T-Learner, Prüfreihe nachlegen 1. Besuch −0,14 d (T-Learner +0,28) → online ≈ T-Learner erwartet.
- Neu `data/bpic17_psp_zeitkopf_bnll.sh`, tmux psp_kopf_bnll, Start 01:37: wie Variante Steigung (vom T-Learner-Arm, Steigung je Event,
  nur Zeitköpfe, LR 1e-3 / 3e-2, Aktion 30 / Referenz 12 Epochen), einziger Unterschied `--beta-nll 1.0` (Mittel ohne 1/σ²-Gewicht,
  σ weiter gelernt). Hülle `t_learner_kopf_steigung_bnll1.pkl`, Label „T-Learner Kopf je Event (Steigung, β-NLL 1)“, Online-Arm
  `tlkb_fest300`. Danach Sonde, Storno, Prüfreihe, `nachlegen_wartezeiten.py` (nimmt jetzt weitere Ordner als Argument), Schranke:
  online nur bei geschätztem Vorteil an nachlegen 1. Besuch ≥ 1,0 d (Steigung 0,37); dann Suffixgüte.

## 2026-11-19 01:05: Wo die PSP-Policies von der Regel abweichen und was es kostet – der ganze Rückstand kommt aus Abweichungen ohne Signal

Skript `modellvergleich_20261116/abweichungen_regel.py` (Ausgabe `abweichungen_regel.md`, `_punkte.csv`, `_faelle.csv`), Aufbau F, 12 Seeds
(tlks bisher 6). Gepaart je Seed und Fall-ID gegen den Regel-Lauf; nachlegen nur 1. Besuch (first_only). Vergleichsmaßstab: wahrer Nachteil der
Nicht-Regel-Option im Simulator (randomisiert): konzept +2,79, nach V. +2,40, nachlegen 1. Besuch +1,55 (Bank-Fortsetzung), leads +1,32 d.
- **Abweichungsanteile** (konzept / nach V. / nachlegen 1. / leads): c_fest300 1 / 5 / 9 / 6 %; c_fest100 10 / 13 / 22 / 17 %; zztf1 24 / 12 / 52 /
  47 %; tl_fest300 8 / 10 / 39 / 20 %; tl_fest500 4 / 6 / 36 / 13 %; **tlks 2 / 1 / 45 / 7 %** (Zeitkopf: an drei Punkten näher an der Regel, an
  nachlegen häufiger Zusatzangebote als der T-Learner).
- **Fälle ohne Abweichung kosten nichts:** Differenz zur Regel −0,08 … +0,55 d (SE 0,3–0,5), d. h. die Paarung trägt, der Rückstand entsteht in
  den abweichenden Fällen.
- **Abweichende Fälle sind nicht billiger als zufällig gewählte, eher teurer:** nachlegen beobachtet 1,77 / 4,61 / 2,95 / 2,47 / 1,91 / 3,10 d
  (c100 / c300 / zztf1 / tl / tl500 / tlks; SE 0,6–1,5) gegen wahr 1,55; leads 3,32 / 6,51 / 4,57 / 0,89 / 2,30 / 4,14 gegen 1,32. Gesamt:
  beobachteter Rückstand ≥ erwarteter bei zufälliger Auswahl in jedem Arm (c300 0,66 vs 0,22; tl 1,11 vs 0,93; tl500 1,24 vs 0,73; tlks 1,45
  vs 0,76; zztf1 2,70 vs 1,54). Die PSP-Policies wählen die Fälle für Abweichungen also ohne nutzbares Signal.
- Die abweichenden Fälle sind keine besonders langen Fälle (Durchlaufzeit unter Regel 18,4–19,3 vs. 18,3–18,7 d ohne Abweichung).
- **Hauptquelle des Verlusts: Zusatzangebot an angebot_nachlegen** (tl 0,76 von 1,11 d, tlks 1,27 von 1,45 d).
- Folgerung: Im Simulator ist die Regel-Option an allen Punkten für praktisch jeden Fall besser (wahre Effekte +1,3 … +2,8 d bei
  Fall-zu-Fall-SD 0,2–0,7 d, Befund 16.11.). Eine Fall-für-Fall-Policy kann die Regel dort nicht schlagen, nur verlieren; jede Abweichung kostet
  im Mittel den vollen Effekt. Eine vorsichtige Policy (nur bei deutlichem Unterschied abweichen) kann höchstens Regel-Niveau erreichen. Deckt
  sich mit AIPW-Screening (kein Policy-Gewinn am echten Log) und den CATE-Schätzern (Causal Forest wählt 100 % Regel).
- **Warum der Causal Forest die Regel trifft und PSP nicht (01:10):** Die Abweichungsraten folgen fast genau aus mittlerer PSP-Schätzung des
  Regel-Vorteils (punkte_alle, Simulatorzustände) und Monte-Carlo-SE je Entscheidung (MC-SE bei 100 Ziehungen / √3), P(falsch) = Φ(−Schätzung/SE).
  tl_fest300 erwartet / beobachtet: konzept 6,8 / 8 %, nach V. 0,9 / 10 %, nachlegen 1. 38,8 / 39 %, leads 21,3 / 20 %; tlks: 0,3 / 2, 0,1 / 1,
  33,7 / 45, 8,3 / 7 %. Ursache an nachlegen: PSP schätzt den Vorteil auf 0,28 / 0,37 d (Wahrheit 1,55 Bank-Fortsetzung bzw. 2,11 RCT), obwohl es
  fast so viele zusätzliche Events zieht wie der Simulator (5,6 / 5,8 vs. 6,9) – die Zusatzschritte kosten in den Ziehungen kaum Zeit. Dazu
  ±0,9–1,0 d Monte-Carlo-Rauschen je Fall. Der Forest lernt die Outcome-Differenz direkt und mittelt über Tausende echte Fälle: Effekte
  1,2–4,2 d bei Rauschen 0,1–0,4 d → nie falsches Vorzeichen.
- **Wo die Zeit des Zusatzangebots fehlt (01:15, `nachlegen_wartezeiten.py`/`.md`; extra − senden, Tage):** Events stimmen inzwischen
  (O_Sent vor Validierung echt +1,06, Simulator +1,06, Steigung +0,91, T-Learner +0,92; (c) nur +0,33). Es fehlen zwei Wartezeiten:
  (1) Zeit bis zum letzten O_Sent echt +0,83, Simulator +1,46, Steigung +0,24; (2) Kundenantwort letztes O_Sent → W_Validate start echt
  +0,64, Simulator +0,50, Steigung **−0,43** (zwei Angebote → Antwort in den Ziehungen kürzer statt länger). Nachfass-Start → Validierung
  echt +1,45, Simulator +1,92, Steigung −0,25, T-Learner −0,28, (c) +0,16. Der echte Log ist beobachtet (nicht randomisiert), zeigt aber
  dasselbe Muster wie der randomisierte Simulator → das Modell könnte es aus seinen Trainingsdaten lernen, tut es aber nicht.
- **Wo die Kundenantwort im Log steckt (01:35):** Im Abschnitt letztes O_Sent → W_Validate start liegt der längste Abstand zu 53 % vor
  W_Call after offers|ate_abort und zu 39 % vor |resume (72 % der Abschnittszeit). Diese Events haben gemischte Abstände (ate_abort: 28 % < 1 h,
  67 % > 1 d; resume: 9 % / 88 %) – dasselbe Muster wie bei A_Cancelled, wo die sicheren kurzen Fälle das 1/σ²-Gewicht tragen. Vermutung
  (ungeprüft): deshalb fehlt die längere Antwort nach zwei Angeboten. Rechnung: träfe das PSP an nachlegen den Effekt (1,55 d) bei heutigem
  Rauschen (0,88 d), wäre es dort zu 4 % falsch; erwarteter Rückstand zur Regel ≈ 0,13 d (≈ Causal Forest −0,12) statt 1,45.

## 2026-11-19 00:50: Zeitkopf je Event – nur Versatz wirkt nicht (1/σ²-Gewichtung), Steigung nötig; Suffixgüte der Steigung

- **Variante nur Versatz (Sonde 1.500 Fälle, 00:25): wirkungslos.** Kopf an A_Cancelled 3,4 / 3,6 d (Aktion / Referenz; echt 20,6; vorher
  3,6), σ 23,2 / 22,3 d, gezogen seit O_Sent bis Storno Median 9,6 / 10,3 d, MAE alle Schritte 0,58 / 0,57 – praktisch der unveränderte
  T-Learner. Gelernte Versätze: A_Cancelled-Mittel −0,26 / −0,05 Einheiten (Steigungsvariante +2,1 / +2,5); große Versätze (bis 6,7) nur bei
  seltenen Events (W_Personal Loan collection), dort Adam-Rauschen.
- **Ursache (an den Sondendaten geprüft):** Loss Attenuation gewichtet den Fehler des Mittels mit 1/σ². Vor A_Cancelled sind 13 % der echten
  Abstände < 1 Tag; dort ist der Kopf sicher (Kopf 1,3 d, σ klein), und diese Schritte tragen ~100 % des 1/σ²-Gewichts. Das gewichtete Residuum
  ist 0,02 Einheiten (ungewichtet 10,3): Ein gemeinsamer Versatz je Event steht damit im Gleichgewicht bei ~0. Die Steigung auf h trennt beide
  Gruppen (kurze Stornos 11,1 d, lange 23,3 d) und kann deshalb das Mittel der langen heben. Folgerung: Ein Versatz allein reicht unter Loss
  Attenuation nicht, wenn die Abstände eines Events gemischt sind; die zustandsabhängige Steigung ist der wirksame Teil.
- **Suffixgüte Variante Steigung** (1.000 Testprefixe, 100 Ziehungen, gleiche Prefixe wie T-Learner; Aktion / Referenz, vorher T-Learner):
  DLS 0,403 → 0,407 / 0,408; Länge-MAE 11,6 / 11,1 → 10,9 / 10,5; 1. Abstand MAE 11,7 / 11,8 → 12,9 / 12,2 h; Restzeit MAE 8,32 / 8,35 →
  8,48 / 8,45 d; **Restzeit-Bias −0,3 / −0,6 → +1,7 / +1,5 d** (zufällige Prefixe, nicht nur Wahlpunkte: jetzt leicht zu lang); CRPS 5,50 / 5,56
  → 5,39 / 5,36; Abdeckung 80 % 78 / 80 → 71 / 75 % (schmaler). Ungültige Übergänge je 100 Events (Prüfreihe) 0,63 → 0,31 zwischen, 0,39 → 0,25
  innerhalb. Storno-Spalten: Median 27,6 / 26,9 d, Anteil 29–32 d 14 % (echt 30,7 d / 73 %). In `suffix_tabelle_bild.py` ergänzt.
- Online Steigung (`tlks_fest300`) läuft seit 23:38, je Seed ~63 min bei 6 Workern, 3 von 12 Seeds fertig (00:44); Ende ~02:00. Danach Online
  Versatz (`tlko_fest300`, ~2 h) – erwartbar ≈ T-Learner, da das Modell sich kaum geändert hat.
  **Zwischenstand 01:00 (Seeds 0–5, gepaart):** Gewinn gegen Bank +1,81 ± 0,43 d; gegen T-Learner (`tl_fest300`, gleiche Seeds +2,05)
  **−0,24 ± 0,41**, gegen (c) −0,62 ± 0,56, gegen Regel −1,45 ± 0,49. Realistische Zeiten bringen bisher keinen Entscheidungsgewinn.

## 2026-11-18 22:30: β-NLL nach Epoche 3 abgebrochen (6,8 d); Zeitkopf je Event auf den T-Learner-Armen gestartet

- β-NLL Epoche 3 (kleine Sonde): Kopf an A_Cancelled 6,8 d (echt 19,3), gezogen Median 12,8 d, MAE alle Schritte 0,60. Auf Wunsch des Nutzers
  („ja und abbrechen“) um 22:25 abgebrochen samt Begleiter und angehängter T-Learner-Kette. Checkpoints ep1–3 und Modelle in
  `beta_nll_20261118/checkpoints`, `modelle/`.
- **Umsetzung:** `dropout_uncertainty_decoder.add_event_time_parameters(slopes)`: je Zeitausgabe (Mittel und log-Varianz von Abstand und
  Fallzeit, getrennt) ein eigener Versatz je Event, mit `slopes` zusätzlich eine Steigung je Event auf h (Wechselwirkung); mit null begonnen.
  psp_model: TRAIN `time_event_head` ("offset"/"slope"), `head_only` (nur Zeitköpfe lernen, Rest eingefroren, `_head_only_trainer` ohne
  GradNorm, da es am eingefrorenen Zustand misst), `offset_learning_rate`; `_load_net` erkennt die neuen Gewichte; train_psp_real
  `--time-event-head`, `--head-only`, `--offset-learning-rate`. Verlust (attenuiert), Dropout-Masken und gezogene Suffixe unverändert.
- **Tests (CPU, Aktion-Arm, 120 Testfälle):** Null-Start identisch (Abweichung 0), nur 36.832 von 1,15 Mio. Gewichten lernen, NLL sinkt in
  60 Schritten, Kopf an A_Cancelled 3,5 → 4,2 d (Versatz wandert ≈ Lernrate je Schritt → Versätze mit 3e-2), Checkpoint-Rundreise exakt, Ziehen läuft.
- **Läufe** (`data/bpic17_psp_zeitkopf_je_event.sh`, Ordner `zeitkopf_je_event_20261118/`): je Arm des T-Learners (ZZ-TF1) nur die Zeitköpfe
  nachtrainiert auf den Wahlpunkt-Proben des Arms, TF 1, LR 1e-3 / Versätze 3e-2, Aktion 30 / Referenz 12 Epochen. Variante **Steigung**
  (tmux psp_kopf_steigung, Hülle `t_learner_kopf_steigung.pkl`, Online-Arm `tlks_fest300`) und Vergleich **nur Versatz** (psp_kopf_versatz,
  `t_learner_kopf_versatz.pkl`, `tlko_fest300`). Danach je Variante Sonde (1.500 Fälle), Storno-Messung, Prüfreihe, Online F 12 Seeds.
  Ausgangswerte der Arme (Sonde 16:50): Kopf an A_Cancelled 3,6 d (echt 20,6), Summe O_Sent → Storno 8,3 (echt 30,6), gezogen Median
  9,8 / 10,2 d, MAE alle Schritte 0,57 d.
- **Zwischensonde Aktion-Arm, Variante Steigung, nach 10 Epochen** (~670 Schritte, 25 s je Epoche; 300 Fälle): Kopf an A_Cancelled
  **3,6 → 9,6 d** (echt 19,3), σ dort 21,7 → 15,6 d, Summe O_Sent → Storno 8,3 → 15,5 d (echt 30,7), gezogen Median 9,8 → 14,2 d,
  MAE alle Schritte 0,57 → **0,55** (besser, β-NLL wurde schlechter). Schneller als β-NLL (6,8 d nach 3 Epochen à 1,2 h), aber noch nicht am Ziel.
- **Aktion-Arm fertig (30 Epochen, ~2.000 Schritte, 13 min):** Kopf an A_Cancelled **20,1 d** (echt 19,3), MAE dort 17,7 → 7,1 (Ridge-Sonde
  additiv 7,9 / je Event 5,4), σ 21,7 → 6,9 d; Summe O_Sent → Storno 8,3 → **27,6 d** (echt 30,7; = Ridge je Event); gezogen seit O_Sent bis
  Storno Median 9,8 → **25,7 d**, Anteil 29–32 d 4 → 13 %; MAE alle Schritte 0,57 → **0,47 d**. Offen: Antwort nach O_Sent bei ≥ 2 Angeboten
  weiter zu kurz (6,0 statt 6,8 d), bei 1 Angebot zu lang (9,2 statt 7,0). Validierungsverlust fiel bis zuletzt (12,08 → 11,24).
- **Volle Sonde beider Arme, Variante Steigung (1.500 Testfälle, 505 echte A_Cancelled-Schritte; 23:25)**, vorher → nachher:
  Kopf an A_Cancelled 3,6 → **20,6 / 21,4 d** (Aktion / Referenz; echt 20,6), MAE dort 17,7 → 6,3 / 5,8 (Ridge je Event 5,4), σ 21,7 → 6,8 / 7,3 d;
  Summe O_Sent → Storno 8,3 → 27,6 / 28,2 d (echt 30,6); Fallzeit-Kopf an A_Cancelled seit O_Sent 28,8 / 27,7 d; gezogen seit O_Sent bis
  Storno Median 9,8 / 10,2 → **26,7 / 27,8 d** (echt 30,7), Anteil 29–32 d 4 → 13 % (echt 73 %: Lage stimmt, Streuung noch zu breit);
  Antwort O_Sent → Validierung 9,2 / 7,2 d bei 1 / ≥ 2 Angeboten (echt 7,0 / 6,9; vorher 8,3 / 6,2); MAE alle Schritte 0,57 → **0,42 d**
  (Ridge je Event 0,41). Eventkopf unverändert kalibriert (P(A_Cancelled) je Uhr-Bereich = echt). Ereignisfolgen identisch zum T-Learner.
- **Storno-Messung Variante Steigung** (storno_abstand.py, 300 Prefixe × 20 Ziehungen bis EOS, Aktion / Referenz, vorher T-Learner-Arme):
  Abstand vor A_Cancelled 10,3 / 10,8 → **20,3 / 21,9 d** (echt 19,1); seit letztem O_Sent Mittel 16,0 / 16,2 → **26,5 / 26,1** (echt 25,3),
  Median 11,3 / 11,7 → 27,6 / 26,9 (echt 30,7), P10 / P90 12,4 / 38,0 (echt 5,0 / 30,9), Anteil 29–32 d 3 → 14 % (echt 73 %);
  Restzeit bis Storno 17,9 → **28,2 / 27,5** (echt 27,6); **Restzeit gesamt 17,9 / 18,1 → 20,6 / 20,7 d (echt 20,4)**. Zum Vergleich (c):
  Abstand vor Storno 3,1 d, seit O_Sent Median 8,6, Restzeit gesamt 15,4 d – (c) hat dasselbe Zeitproblem stärker.

## 2026-11-18 20:25: β-NLL Epoche 2 stockt (6,2 → 6,5 d) – Ursache: der Mittelwert des Zeitkopfs sieht das Event praktisch nicht

- Sonde ep2 (300 Fälle): Kopf-Mittel an A_Cancelled 6,5 d (ep1 6,2; echt 19,3), σ dort 18,2 d, Summe O_Sent → Storno 11,8 d, gezogen Median 12,0 d,
  Anteil 29–32 d 5 %, MAE alle Schritte 0,60 d. → β = 0,5 bringt nach der ersten Epoche kaum noch etwas.
- **Gewichte des Zeitkopfs** (Versatz je Event = Embedding(Event)·W + b, Embedding 68 × 17, geteilt von allen vier Zeitausgaben):
  ZZU-TF1 Mittel-Versatz des Abstands für A_Cancelled **+0,0 d**, für alle geprüften Events ≤ 0,4 d (‖W_mean‖ 0,11), aber log-Varianz-Versatz
  A_Cancelled +1,64, O_Sent −4,13, resume −1,84 (‖W_var‖ 1,01). β ep2: A_Cancelled Mittel-Versatz +0,7 d, log-Varianz +1,32.
  → „Zeit je Event“ wirkt im trainierten Netz fast nur auf die Varianz; der Mittelwert ist praktisch Linear(h) ohne Event. Das Event trägt
  seine lange Wartezeit über die Varianz statt über den Mittelwert. β-NLL ändert das Verhältnis von Mittelwert- zu Varianzgradient innerhalb
  derselben Probe nicht (beide mal w), daher nur kleine Bewegung.
- Das erklärt die Sonde 16:50 vollständig: die additive Ridge mit freiem Versatz je Event trifft das Mittel (20,6 d), der Kopf nicht.
- Vorschlag (braucht OK, Vorschlag 2 von 16:10 in enger Form): eigene Parameter je Event für Mittel und Varianz (freier Versatz je Event
  getrennt je Ausgabe, optional Steigung je Event = Wechselwirkung), am schnellsten als Nachtraining nur des Zeitkopfs auf den fertigen
  T-Learner-Armen (Rest eingefroren, Ereignisfolgen bleiben), dann Online-Test. β-NLL-Lauf läuft bis dahin weiter (Ende ~1–3 Uhr; Schranke
  wird voraussichtlich halten).

## 2026-11-18 19:35: β-NLL Epoche 1 – der Kopf bewegt sich in die richtige Richtung, langsam; Restzeit-Hilfsziel fertig – kleiner Gewinn, Storno unverändert

**β-NLL 0,5, Epoche 1** (kleine Sonde, 300 Fälle; Ausgangswerte ZZU-TF1 aus der vollen Sonde): Kopf-Mittel an echten A_Cancelled-Schritten
**3,6 → 6,2 d** (echt 19,3 in dieser Stichprobe), MAE 17,6 → 15,0, σ des Kopfs dort 22,7 → 18,5 d; Summe O_Sent → Storno (Kopf, echte Events)
8,4 → 11,5 d; gezogen seit O_Sent bis A_Cancelled Median 10,8 → 12,2 d, Anteil 29–32 d 3 %; MAE über alle Schritte 0,57 → 0,61 d (leicht
schlechter). → Die Varianz gibt den Fehler an den Mittelwert ab, wie von β-NLL erwartet, aber nach einer Epoche erst ~15 % des Wegs. Hält
das Tempo, liegt das Kopf-Mittel nach Epoche 5 um 15 d (Schranke für den T-Learner ≥ 10 d). Bleibt es nach Epoche 2–3 unter ~9 d: β = 1
erwägen. Epochen dauern ~2 h (drei Trainings auf der GPU).

**ZZ-TF1 + Restzeit-Hilfsziel fertig** (5 Ep., LR 3e-5, Kopf nur als Hilfsziel, Bewertung weiter aus Suffixen; Register „ZZ-TF1 + Restzeit-Hilfsziel“),
gegen ZZ-TF1 final:
- AIPW-Vorzeichen 81 / 95 / 49 / **62 %** (74 / 96 / 49 / 51); gegen Simulator Konzept +1,61 (1,36; Wahrheit 2,79), nach V. +2,83 (2,29; 2,40),
  Nachlegen 1. Besuch −0,10 (+0,12; 2,11), später +0,63 (0,54; 0,42), **Leads +0,62 (−0,03; 1,32), 70 % richtig (52 %)**.
- Lead-Gedächtnis 0,47 (0,21; echt 2,83); nach Extra-Angebot zusätzlich versendet 0,85 (0,55; Simulator 1,07), Nachfassen 0,30 (0,49; 1,92).
- Storno ab O_Sent Median 11,8 d, Anteil 29–32 d 2,8 % – unverändert. Gültig 0,46 + 0,31 je 100.
- Suffixgüte gleich bis leicht besser: DLS 0,419 (0,411), Schritt 1 / 10 87,4 / 27,7 %, Restzeit-MAE 8,32 (8,50), Bias −0,42, Abdeckung 82 %.
- → Das Hilfsziel schärft etwas das Fernsignal bei Leads und die Nahpunkte, ändert aber nichts an Nachlegen und an der Storno-Zeit (der
  Zeitkopf bleibt das Problem, siehe Sonde 16:50). Nicht online getestet. Kandidat, mit β-NLL kombiniert zu werden.

## 2026-11-18 17:10: T-Learner auf der β-NLL-Basis angehängt (Nutzer: „weiter“; `data/bpic17_psp_t_learner_bnll.sh`, tmux psp_tl_bnll)

- Wartet auf „Alles fertig“ der β-NLL-Pipeline, dann Schranke `beta_nll_20261118/schranke.py` (vorher festgelegt: Kopf-Mittel an echten
  A_Cancelled-Schritten ≥ 10 d, MAE alle Schritte ≤ 0,70 d; am Ausgangsmodell geprüft → Halt bei 3,6 d). Nur wenn erfüllt: Arme wie beim
  T-Learner (Aktion 40 / Referenz 20 Ep., LR 3e-5, nur Wahlpunkt-Proben), aber warm vom β-Modell mit Uhren und β-NLL 0,5; Hülle
  `models/psp_model_durchlaufzeit_t_learner_bnll05.pkl`; Prüfreihe „T-Learner (β-NLL)“ (t_learner_20261117/staende.json → Register), Storno und
  Sonde beider Arme, online Aufbau F 12 Seeds Arm `tlbnll_fest300` (300 Ziehungen, first_only; in f_zwoelf_seeds/-auswerten und
  online_sammeln eingetragen), Auswertung. oom_score_adj 1000. Erwartetes Ende bei „weiter“: späterer Vormittag 19.11.

## 2026-11-18 16:50: Zeitkopf-Sonde gemessen – der Kopf lernt den Mittelwert langer Wartezeiten nicht (Varianz schluckt den Fehler); β-NLL gestartet

Sonde `zeitkopf_sonde_20261118/sonde.py` (1.500 Testfälle, Schnitt nach dem ersten O_Sent, echte Events mit Teacher Forcing 1, 37.423 Schritte,
505 echte A_Cancelled-Schritte; Vorhersage außerhalb der Stichprobe über 5 Fallgruppen; `sonde.csv`, `schritte_*.csv.gz`). Alle vier Modelle
(ZZ-TF1, ZZU-TF1, T-Learner Aktion / Referenz) fast gleich; Werte für ZZ-TF1:

| A_Cancelled-Schritt (echt 20,6 d) | Mittel | MAE | Korr. | Summe O_Sent → Storno, Median (echt 30,6) |
|---|---|---|---|---|
| Zeitkopf (trainiert) | **3,6 d** | 17,7 | 0,58 | **8,3 d** |
| Ridge additiv [h, One-Hot Event] (gleiche Form wie der Kopf) | 20,6 | 7,9 | 0,64 | 25,5 |
| Ridge je Event (Wechselwirkung) | 20,7 | 5,4 | 0,67 | 27,6 |
| Regel 30,7 − Uhr | 25,1 | 5,2 | 0,43 | 30,7 |

- **Hauptursache ist das Training des Kopfs, nicht seine additive Form:** schon eine additive lineare Auslese desselben h trifft den Mittelwert
  (20,6 statt 3,6 d). Der Kopf gibt an A_Cancelled σ ≈ 22 d aus (Median aller Schritte 0,08 d) – die gelernte Varianz schluckt den Fehler, der
  Mittelwert bleibt stehen (Gauß-NLL: Gradient ∝ 1/σ²). Passt zur β-NLL-Hypothese.
- **Wechselwirkung bringt zusätzlich etwas:** A_Cancelled MAE 7,9 → 5,4 d, Storno-Median 25,5 → 27,6 d; über alle Schritte MAE Kopf 0,57 /
  additiv 0,58 / je Event **0,41 d** (−29 %). → Zeitkopf mit Wechselwirkung (Vorschlag 2) bleibt nach β-NLL sinnvoll.
- **Uhren sind überflüssig:** h kodiert „Tage seit dem letzten O_Sent“ auch ohne Uhren (Ridge R² ZZ-TF1 0,76, ZZU-TF1 0,74). Erklärt „Uhren ohne Wirkung“.
- **Eventkopf ist in Ordnung:** P(A_Cancelled) je Uhr-Klasse trifft die echte Häufigkeit (0–5 d 0,024 vs. 0,025; 29–32 d 0,002 vs. 0,002;
  > 32 d 0,023 vs. 0,027). Wann storniert wird, stimmt; nur die Zeit daran nicht.
- **Antwortzeit O_Sent → A_Validating** (gegeben die echten Events): echt 8,4 d bei 1 wie bei ≥ 2 O_Sent; Kopf 8,1 / **6,2** (verkürzt bei mehreren
  Angeboten um 1,9 d), Sonde je Event 8,6 / 8,2 → gleicher Kopffehler, passt zu „Nachfassen nach Extra-Angebot −0,28 statt +1,92“ beim T-Learner.
- Gezogen (200 Prefixe × 20): Tage vom letzten O_Sent bis A_Cancelled Median aus der Fallzeit 11,9 d, aus der Abstandssumme (so bewertet die
  Policy bei time_from gaps) 10,8 d; Anteil 29–32 d 3–4 % (echt 73 %).
- Gestartet 16:33 (Nutzer: Sonde und Vorschlag 3): **β-NLL 0,5** = fertiges ZZU-TF1 warm, `--beta-nll 0.5` (neu: `Loss(beta_nll)` in `losses.py`,
  psp_model `beta_nll`, train_psp_real `--beta-nll`; β = 0 bitgleich geprüft), LR 1e-4, 5 Epochen, TF 1, sonst wie ZZU-TF1. Skript
  `data/bpic17_psp_beta_nll.sh`, tmux psp_bnll, Begleiter psp_bnll_begleiter (je Epoche Checkpoint + kleine Sonde 300 Fälle → `beta_nll_20261118/
  fortschritt.log`), danach Prüfreihe „β-NLL 0,5“, Storno-Messung, volle Sonde, Suffixgüte; im Register angemeldet. Drei Trainings teilen die GPU.
  Erfolg: Kopf-Mittel an A_Cancelled ≈ 20 d, Storno ab O_Sent nahe 30,7 d, Antwortzeit bei ≥ 2 O_Sent nicht kürzer, Gültigkeit und Suffixgüte gehalten.

## 2026-11-18 16:10: Code-Befund – „Zeit je Event“ ist nur ein additiver Versatz je Event (Deutung, noch nicht gemessen; gemessen 16:50: nur Nebenursache)

- `dropout_uncertainty_decoder.py` `numeric_prediction` (Z. 138–147): Mittel und log-Varianz der Zeit = `Linear(h)` + `Linear(Embedding(Event))`.
  Das gezogene Event verschiebt die Zeit nur um eine Konstante; wie die Zeit vom Zustand h (Zähler, Uhren, Option) abhängt, ist für alle
  Events gleich. (Auch `conditioned_prediction` ist Linear auf [h, Embedding] → additiv.)
- Folge: Wechselwirkungen wie „Abstand vor A_Cancelled = Frist − Tage seit O_Sent“ oder „Kundenantwort nach zwei Angeboten länger“ kann der
  Kopf nicht ausdrücken. Wird A_Cancelled bei kleiner Uhr gezogen, bekommt es h-typischen Abstand + festen Storno-Versatz. Passt zu: Abstand
  vor A_Cancelled bei allen TF1-Modellen 10–14 d, Storno ab O_Sent Median ~11 d, Uhren ohne Wirkung, T-Learner mit richtigen Ereignisfolgen
  nach Extra-Angebot, aber Restzeit −0,07 statt +1,55 und Nachfassen −0,28 statt +1,92.
- Verstärkend (Hypothese): Die Gauß-NLL mit gelernter Varianz gewichtet den Mittelwert-Gradienten mit 1/σ² → lange, verrauschte Wartezeiten
  (genau die entscheidungsrelevanten) werden im Mittel schlecht gelernt; die σ-Spirale bei ZZ (8 → 45 d) ist dieses bekannte Muster
  (Seitzer et al. 2022, β-NLL).
- Vorschlag: (1) Sonde auf eingefrorenem ZZU-TF1/T-Learner mit Teacher Forcing 1 (Kopf-µ vs. eventspezifische Sonde auf h für den Abstand vor
  A_Cancelled und Antwortzeit nach O_Sent); (2) Zeitkopf mit Wechselwirkung (FiLM oder kleines MLP auf [h, Embedding]) und nur den Kopf
  nachtrainieren; (3) β-NLL nur mit OK des Nutzers (ändert die Gewichtung, nicht das Prinzip).

## 2026-11-18 15:40: ZZ-TF0,5 eingeplant – **19:45 auf Wunsch des Nutzers angehalten** (vor Ende von Epoche 1, GPU für RCT und β-NLL; Neustart: tmux + `data/bpic17_psp_tf_boden05.sh`) (Wunsch des Nutzers; `tf_boden05_20261118/`, `data/bpic17_psp_tf_boden05.sh`, tmux psp_tf05)

- Fertiges ZZ-TF1 (--init-from) 6 Epochen nachtrainiert mit Teacher-Forcing-Boden 0,5 (Quote 1,0 / 0,67 / 0,5 × 4, frei laufende Schritte
  speisen argmax ein wie bisher), LR 1e-4, sonst wie ZZ-TF1. Startet nach dem Restzeit-Hilfsziel-Training, danach Prüfreihe, Storno, Suffixgüte.
- Rauchtests: mit 300 Fällen + Warmstart Größenfehler (kleineres Vokabular, nur Testeffekt; voller Lauf hat dasselbe Vokabular wie ZZ-TF1);
  ohne Warmstart läuft der frei laufende Zweig bei 0,5 mit dem aktuellen Code (Verlust 7,61 → 6,12).
- **Vorwissen (Eintrag 11-17 11:08, bei der Planung übersehen):** ZZ-Zwischenstände TF 0,73 … 0,20 und Piloten mit fester Untergrenze
  0 / 0,5 / 0,8 / 1,0 zeigen einen monotonen Zielkonflikt ohne günstigen Zwischenwert (Pilot 0,5: Konzept-Vorzeichen 71 %, ohne Ende 40 %
  wie bei 0); das Fernsignal kam bei ZZ erst bei TF 0. Neu an diesem Lauf ist nur: volles ZZ-Modell (Zähler + Zeit je Event) und bei
  festem Boden 0,5 ausgelernt (ZZ-Zwischenstände waren Durchgangsstationen, Piloten nur 3.000 Fälle). Erwartung daher gering.

## 2026-11-18 11:55: T-Learner mit 500 Ziehungen (12 Seeds) – Wahlen regeltreuer, Ergebnis nicht besser

- Gewinn gegen Bank +1,39 ± 0,44 (300 Ziehungen: +1,52 ± 0,43); gepaart 300 − 500 = +0,13 ± 0,28 (n. s.); Regel − T-Learner 500 = 1,24 ± 0,34;
  (c) 300 − T-Learner 500 = 0,58 ± 0,29 (z 2,0). Angebote je Fall 1,72.
- Regel-Option 76 → 81 % (Konzept 92 → 96, nach V. 90 → 94, Nachlegen 61 → 64, Leads 80 → 87 %).
- → Das Ziehungsrauschen ist nicht der Engpass: an Nachlegen bleibt der T-Learner auch mit 500 Ziehungen nahe am Münzwurf (64 %), weil
  seine geschätzte Wirkung dort zu klein ist (+0,28 statt +2,11 Tage, fehlende Wartezeit nach dem Extra-Angebot). Mehr Ziehungen schätzen
  einen verzerrten Mittelwert nur genauer. Schlussprüfung meldete geänderte Quelle (f_zwoelf_seeds.py, nur RCT-Einträge ergänzt) → gültig.

## 2026-11-18 11:25: Restzeit-Kopf als Hilfsziel – Test vorbereitet; RCT-Training CPU-gebremst

- Bisherige Evidenz (a) vs. (b) (= (a) + Restzeit-Kopf; log-normale Abstände, frei trainiert): AIPW 47/58/52/44 vs. 57/54/29/60 %, Rauschen
  8,1 vs. 7,8 d, online A −0,49 vs. −0,30, Suffixgüte (200 Prefixe) gleich; Gedächtnis/Simulator nie gemessen → nicht entscheidbar.
- Neuer Test (`data/bpic17_psp_restzeit_hilfsziel.sh`, Ordner `restzeit_hilfsziel_20261118/`, tmux psp_restzeit): ZZ-TF1 fertig laden, Restzeit-
  Kopf nur als Hilfsziel (Gewicht 1,0), 5 Epochen auf allen Prefixen, LR 3e-5; Bewertung weiter nur aus Suffixen; danach Prüfreihe, Storno,
  Suffixgüte. Schnelltest (300 Fälle) ok: Kopf vorhanden, value_source = suffix, Zähler an. Start wartet auf das Ende des 500er-Laufs.
- RCT-Basistraining: GPU nur ~8 % ausgelastet, CPU-Last 23–27 (12 Simulationsprozesse) → erste Epoche nach 1,5 h noch nicht fertig;
  wird nach dem 500er-Lauf schneller.

## 2026-11-18 10:55: ZZ-TF1-WP150 Ep. 90 (≈ 6 ZZ-TF1-Epochen) – unverändert gegenüber Ep. 45

- AIPW-Vorzeichen 85 / 95 / 47 / 29 %; gegen Simulator Konzept +1,45 (82 %), nach V. +2,56 (93 %), Nachlegen 1. Besuch +0,04 (48 %),
  später −0,25 (52 %), Leads −0,82 (32 %). Extra-Angebot: versendet +0,17 (Simulator 1,07), storniert +0,02 (1,25); Lead-Gedächtnis 0,22.
- → Von Ep. 45 auf 90 keine Bewegung; die restlichen 60 Epochen lassen keinen Gewinn erwarten (Lauf teilt die GPU mit dem RCT-Training).
- Auf Wunsch des Nutzers um 11:00 bei Epoche 92 abgebrochen (GPU frei für das RCT-Training); Prüfstände Ep. 15 / 45 / 90 bleiben.

## 2026-11-18 09:25: ZZ-TF1 + T-Learner auf Simulator-RCT-Daten gestartet (Wunsch des Nutzers; `rct_tlearner_20261118/`, `data/bpic17_psp_rct_tlearner.sh`)

- RCT-Daten (`rct_erzeugen.py`): korrigierter Simulator wie Aufbau F (Katalog d_oe_zustand_20261115, zufällige Zuteilung), 8 Läufe × 4.000 Fälle
  (Saat 1000 + r, IDs R<r>_), an allen 4 Wahlpunkten Option 50/50, an Nachlegen nur beim 1. Besuch (danach Bank, sonst endlose Schleifen).
- RCT-Daten fertig (09:24, 8 × ~2–3 min): 32.000 Fälle, 1.409.262 Events (44 je Fall, echter Log 38), Durchlaufzeit im Mittel 22,05 d (echt 21,90). Wahlen: Konzept annehmen 5.478 / bearbeiten 5.500; nach V. 13.824 / 13.635; Nachlegen (1. Besuch) weiteres Angebot 15.227 / senden 15.546, spätere Besuche Bank 23.661; Leads bearbeiten 10.346 / zurückziehen 10.224 → ausgeglichene Gruppen (echter Log z. B. Nachlegen 3.887 / 35.604, Leads 3.644 / 16.802). Preprocessing wie beim echten Log ohne Verluste durchlaufen.
- Pipeline unter gleichen Umständen wie auf dem echten Log: Basis ZZ-TF1 (15 Ep., LR 1e-4, TF 1, Zähler, Zeit je Event), T-Learner-Arme warm
  von dieser Basis (40 / 20 Ep., LR 3e-5, nur Wahlpunkt-Proben), Hülle, Prüfreihe (Testprefixe aus dem echten Log), Storno-Messung, online
  12 Seeds (Basis 100, T-Learner 300 Ziehungen, first_only). Neue Trainingsoptionen `--log-pickle`, `--test-cases-from`.
- Einordnung: Training auf Daten der Bewertungsumgebung (Zirkelschluss) → nur getrennt berichten, als Obergrenze „ohne Verzerrung durch
  die Wahl im Log“. Erwartete Dauer: Basis ~12–15 h (GPU geteilt mit WP150 bis ~14:30), dann ~6 h für Arme, Prüfungen, Online.

## 2026-11-18 09:40: Suffixgüte der T-Learner-Modelle (1.000 Testprefixe, 100 Ziehungen; `t_learner_20261117/suffixguete_*`)

- Aktion: Schritt 1 85,3 %, Schritt 10 26,9 %, DLS 0,403, Länge-MAE 11,6, Restzeit-MAE 8,32 d, Bias −0,31, Abdeckung 78 %.
  Referenz: 85,3 % / 25,9 % / 0,403 / 11,1 / 8,35 d / −0,60 / 80 %. (ZZ-TF1 fertig: 87,5 % / 28,7 % / 0,411 / 11,4 / 8,50 d / +0,35 / 80 %.)
- → Das kurze Feintraining auf Wahlpunkt-Proben kostet kaum allgemeine Suffixgüte (anders als WP von Grund auf: Schritt 1 nur 40 %).
  Der T-Learner hat damit gültige Abläufe, Suffixgüte auf ZZ-TF1-Niveau und ein Entscheidungssignal.

## 2026-11-18 09:25: ZZ-TF1 hat bei Epoche 12–15 ausgelernt (`verlust_diagnose_20261117/verlust_je_ausgabe_lernkurve.csv`)

- Testverlust (Teacher Forcing 1, 300 Testprefixe, nächste 10 Events), Ep. 3 / 6 / 9 / 12 / 15: Kreuzentropie 0,397 / 0,355 / 0,344 / 0,336 / 0,341,
  Treffer 86,4 / 87,6 / 87,9 / 88,2 / 88,2 %, Abstand-MAE 0,532 / 0,527 / 0,522 / 0,522 / 0,523 d, Fallzeit attenuiert −2,312 → −2,326 (12 → 15).
- → Plateau: von Ep. 12 auf 15 keine Verbesserung der Ereignisse, Zeiten minimal. Suffixgüte ebenfalls flach (DLS 0,40 / 0,41 / 0,41),
  Fernpunkte ohne Trend, Storno-Zeitpunkt über alle Epochen gleich. Mehr Epochen mit LR 1e-4 lohnen nicht; höchstens eine kurze Phase
  mit gesenkter Lernrate (LR blieb konstant, Plateau-Scheduler griff nie) – erwarteter Gewinn klein, danach müssten die T-Learner-Arme neu.

## 2026-11-18 09:10: Ist der T-Learner austrainiert? (`t_learner_20261117/arm_verlust.py/.csv`)

- Trainingslogs nicht aussagekräftig: protokollierter Trainingsverlust steigt (GradNorm-Gewichtung, wie bei ZZ-TF1), Validierung im freien
  Lauf (für TF1 irreführend; Referenz 14,84 → 15,21), keine Zwischen-Checkpoints (kein Begleiter).
- Direkt gemessen (Teacher Forcing 1, 400 Testprefixe je Gruppe am Options-Event, nächste 10 echte Events), Basis ZZ-TF1 → Arm:
  Aktion Kreuzentropie 0,232 → 0,213, Treffer 92,7 → 93,3 %, Abstand-MAE 0,127 → 0,125 d; Referenz 0,331 → 0,313, Treffer 89,2 → 89,5 %,
  1. Schritt 81,7 → 84,0 % (Nachlegen 91 → 95 %, nach V. 69 → 72 %), Abstand-MAE 0,483 → 0,480 d.
- → Das Feintraining hat auf zurückgehaltenen Fällen leicht verbessert, kein Überanpassen sichtbar; der Spielraum in der Likelihood ist
  klein. Ob mehr Epochen die Entscheidungen verbessern, ist ohne Lernkurve offen; die Hauptschwäche (Wartezeiten/Storno-Frist) hing bei
  allen TF1-Modellen nicht an der Trainingsdauer.

## 2026-11-18 08:45: Bilanz „nur Wahlpunkte“ (WP, WP150) – allein kein Vorteil, als Feintraining im T-Learner nützlich

| | ZZ-TF1 Ep. 3 | ZZ-TF1 fertig | WP150 Ep. 15 | WP150 Ep. 45 | T-Learner |
|---|---|---|---|---|---|
| AIPW-Vorzeichen K / nV / N / L | 82/96/50/29 % | 74/96/49/50 % | 80/93/56/34 % | 86/95/46/32 % | 70/85/56/68 % |
| richtig gegen Simulator N1 / Leads | 48 / 27 % | 52 / 52 % | 53 / 40 % | 47 / 27 % | 68 / 72 % |
| Rauschen / ρ | 1,79 / 0,25 | 1,85 / 0,16 | 1,36 / 0,43 | 1,53 / 0,31 | 1,76 / 0,12 |
| Suffix Schritt 1 / DLS (zufällige Testprefixe) | 83 % / 0,36 | 88 % / 0,41 | 40 % / 0,24 | 40 % / 0,26 | – |
| ungültig je 100 (zwischen + innerhalb) | 1,2 + 0,8 | 0,7 + 0,5 | 8,3 + 5,9 | 3,1 + 2,5 | 0,6 + 0,4 |

- Vom Grund auf nur auf Wahlpunkten: Fernpunkte nicht besser (Leads weiter falsch), Nahpunkte etwa wie ZZ-TF1, etwas weniger Rauschen (höhere
  Kopplung), aber deutlich schlechtere allgemeine Suffixgüte (Schritt 1 40 % statt 88 %) und mehr ungültige Übergänge.
- Nützlich nur als Feintraining auf einem voll trainierten Modell und getrennt nach Option (T-Learner): Gültigkeit bleibt (0,6 + 0,4),
  Fernpunkte drehen sich in die richtige Richtung. Der Hebel ist die Trennung nach Option, nicht die Beschränkung auf Wahlpunkte.

## 2026-11-18 08:35: T-Learner mit 500 Ziehungen auf 12 Seeds gestartet (tmux psp_tl500, Arm tl_fest500, 12 Prozesse)

- Grund: die Optionen des T-Learners sind kaum gekoppelt (ρ 0,10–0,16 → Rauschen 1,7–1,8 d je Entscheidung bei 100 Ziehungen); mit 500
  statt 300 Ziehungen sinkt es um √(300/500) ≈ 0,77 (bei 300 ≈ 1,0 d, bei 500 ≈ 0,8 d). Vergleich gepaart mit tl_fest300, (c) 300, Regel.

## 2026-11-18 08:30: Nachtergebnisse – T-Learner löst das S-Learner-Problem der TF1-Modelle; WP150 Ep. 45 und ZZU-TF1 fertig ohne Fortschritt

**T-Learner (ZZ-TF1)** (Aktion 40 Ep. auf 17.010 Proben, Referenz 20 Ep., warm vom ZZ-TF1; Prüfreihe `t_learner_20261117/`, online `f_zwoelf_seeds/tl_fest300_*`):
- Offline: AIPW-Vorzeichen 70 / 85 / 56 / **68 %**, Diff +1,16 / +2,20 / −0,28 / **−1,02** (AIPW +5,4 / +2,9 / −1,5 / −2,2) – erstmals bei einem
  TF1-Modell richtiges Vorzeichen an Nachlegen und Leads im Mittel. Rauschen 1,7–1,8, ρ 0,10–0,16.
- Gegen Simulator: Konzept +1,47 (Wahrheit 2,79, 75 % richtig), nach V. +2,47 (2,40, 93 %), **Nachlegen 1. Besuch +0,28 (2,11, 68 %)**,
  später +0,77 (0,42, 53 %), **Leads +0,80 (1,32, 72 %)**.
- Extra-Angebot: zusätzlich versendet **+0,94 (Simulator 1,07)**, storniert **+1,07 (1,25)** – die Ereignisfolgen stimmen jetzt; die Restzeit
  aber nicht (−0,07 statt +1,55; Nachfassen bis Validierung −0,28 statt +1,92) → die fehlende Wartezeit (Storno-Frist) bleibt.
- Storno-Zeitpunkt beider Arme: Median 11,3 / 11,7 d, 3 % im Fenster 29–32 d – unverändert.
- **Online, 12 Seeds × 200, 300 Ziehungen + first_only: +1,52 ± 0,43** (Regel-Option 76 %: 92 / 90 / 61 / 80 %; 1,76 Angebote/Fall).
  Gepaart: T-Learner − ZZ-TF1 = **+1,59 ± 0,24 (z 6,7)**; Regel − T-Learner = 1,11 ± 0,25; (c) 300 − T-Learner = 0,45 ± 0,29 (n. s.);
  ≈ (c) 100 (+1,57).
- → Mit gültigen Abläufen (TF1-Basis) erreicht der T-Learner online praktisch das Niveau von (c); der S-Learner auf derselben Basis brachte nichts.

**ZZ-TF1-WP150 Ep. 45** (≈ 3 ZZ-TF1-Epochen): AIPW 85 / 95 / 47 / 32 %; gegen Simulator Konzept +1,52 (87 %), nach V. +2,63 (92 %), Nachlegen
1. Besuch +0,02 (47 %), später −0,31 (55 %), Leads −0,87 (27 %) → Fernpunkte weiter falsch; nur Wahlpunkt-Training hilft nicht (bisher).

**ZZU-TF1 Ep. 12 / fertig** (Uhren): AIPW 78 / 93 / 50 / 58 bzw. 77 / 93 / 45 / 54 %; fertig gegen Simulator Konzept +2,06 (87 %), nach V. +2,06
(83 %), Nachlegen −0,11 (43 %) / später −1,19 (32 %), Leads +0,41 (57 %); Storno Median 11,5 d, 3,2 % im Fenster → Uhren ohne Wirkung.

## 2026-11-18 01:10: ZZ-TF1-WP150 Ep. 15 (≈ 1 ZZ-TF1-Epoche an Schritten) – Nahpunkte ja, Fernpunkte weiter nein

- AIPW-Vorzeichen 81 / 93 / 56 / 34 %, Diff +1,32 / +2,32 / −0,06 / +0,31 d; Rauschen 0,9–1,6 d, ρ 0,26–0,78 (geringer als ZZ-TF1).
- Gegen Simulator: Konzept +1,12 (Wahrheit 2,79, 70 % richtig), nach V. +1,81 (2,40, 90 %), Nachlegen 1. Besuch +0,01 (2,11, 53 %),
  später −1,76 (0,42, 27 %), Leads −0,24 (1,32, 40 %).
- Extra-Angebot ändert praktisch nichts (versendet +0,01 statt 1,07, storniert +0,02 statt 1,25); Lead-Gedächtnis 0,15 (echt 2,83).
- → Bisher dasselbe Bild wie bei ZZ-TF1: das Training nur auf Wahlpunkten macht die Option an den Fernpunkten nicht wirksamer.
  Nächste Prüfung Ep. 45 (~5 Uhr).

## 2026-11-17 23:40: Ideenliste (Stand, auf Wunsch des Nutzers festgehalten)

Neu (Nutzer): **PSP als Suffix-Generator + Baum als Outcome-Modell („Variante 2“)** – an jeder historischen Entscheidung im echten Log zieht
das PSP Suffixe für beide Optionen; Kennzahlen je Option (mittlere Restzeit, Quantile, Anteil Storno/Annahme, Angebote, Zeit bis zum
nächsten Event); ein Baum (Gradient Boosting) lernt Prefix-Merkmale + Kennzahlen der gewählten Option → echte Durchlaufzeit; Entscheidung
= Vergleich der Vorhersagen mit den Kennzahlen von Option A bzw. B. Korrigiert systematische Suffix-Fehler (z. B. zu frühes Storno);
Effekt nur, wo die Ziehungen der Optionen sich unterscheiden (passt zum T-Learner); optional doppelt robuster Score statt roher
Durchlaufzeit (Policy Tree). NICHT mit Simulator-Outcomes trainieren (Zirkelschluss: Training = Bewertungsumgebung).
Entscheidung käme aus dem Baum (Abweichung von „direkt aus Suffixen“, vom Nutzer selbst vorgeschlagen). Aufwand ~½ Tag.

Neu (Nutzer, 23:45): **bestes Modell zusätzlich auf RCT-Daten des Simulators trainieren** (Optionen an den 4 Punkten zufällig 50/50 → keine Verzerrung durch die Wahl, ausgeglichene Gruppen, z. B. weiteres Angebot 50 % statt ~10 %). Vorhandene RCT-Daten (rct_bpic17_channel_delta50_*) stammen vom alten Simulator (Ende Okt.) mit anderen Punkten → neue RCT-Erzeugung mit dem korrigierten Simulator nötig. Achtung Zirkelschluss: Training auf Daten der Bewertungsumgebung → online optimistisch; nur als Obergrenze bzw. getrennt berichtet. Varianten: nur RCT oder echter Log + RCT (Vortraining echt, Feintraining RCT).

Weitere offene Ideen: X-Learner bzw. DR-Learner mit PSP als Gegenwelt-Generator (Entscheidung aus zweitem Modell → OK des Nutzers);
Decision-aware decoding (Paper, catboost-Installation → OK); Zeitregel für Storno beim Ziehen (Invariante → OK); Zufallszahlen je
Event-Vorkommen koppeln; eigener Zuteilungsstrom je Fall im Simulator; Abweichungen (c) 300 ↔ Regel analysieren; X-Learner/Causal Forest auf
12 Seeds; Arbeitskosten Seed 2; Simulator-Wahrheit mit aktuellem Stand neu messen; mehr Ziehungen (500–1000) für (c).

## 2026-11-17 23:25: T-Learner-PSP gebaut und gestartet (`policies/ev_policy/t_learner_policy.py`, `data/bpic17_psp_t_learner.sh`)

- Anlass: die PSP-Policy ist ein S-Learner (Option = letztes Prefix-Event); die TF1-Modelle ignorieren die Option an Nachlegen/Leads.
- Zwei PSPs, warm vom fertigen ZZ-TF1, nachtrainiert nur auf den Wahlpunkt-Proben ihrer Gruppe (Aktion 40 Ep., Referenz 20 Ep., LR 3e-5;
  Training und Validierung gefiltert). Gruppengrößen im echten Log (Aktion / Referenz): Konzept 9.231 / 15.065, nach V. 9.457 / 22.347,
  Nachlegen 3.887 / 35.604, Leads 3.644 / 16.802. Hülle `TLearnerPSP` mit PSPModel-Schnittstelle; Prüfreihe `t_learner_20261117/`,
  danach online Aufbau F, 12 Seeds, 300 Ziehungen + first_only (Arm tl_fest300). Neue Trainingsoptionen: only_options (--arm), init_from (--init-from).

## 2026-11-17 23:00: ZZU-TF1 Ep. 9 – die Uhren lernen die Storno-Frist nicht; ZZ-TF1-WP Ep. 10; Neustart als ZZ-TF1-WP150

**ZZU-TF1 Ep. 9** (`zaehler_uhren_tf1_20261117/pruefung_ep9.md`): AIPW-Vorzeichen 77 / 93 / 49 / 41 %. Gegen Simulator: Konzept +1,24
(75 % richtig), nach V. +2,04 (87 %), Nachlegen 1. Besuch +0,01 (48 %), später −1,64 (28 %), Leads −0,12 (50 %). Extra-Angebot: versendet
+0,32 (Simulator 1,07), storniert +0,78 (1,25). **Storno ab letztem O_Sent: Median 11,3 d (echt 30,7), Anteil 29–32 d 3,3 % (echt 73 %)** –
wie Ep. 3 / 6 und wie ohne Uhren. → Nach 9 Epochen kein Effekt der Uhren; die 30-Tage-Frist ist mit normalverteilten Abständen offenbar
nicht erreichbar. Verbleibende Option: gelernte Zeitregel beim Ziehen (berührt die Invariante → nur mit Zustimmung des Nutzers).

**ZZ-TF1-WP Ep. 10** (30-Epochen-Lauf, nur Wahlpunkt-Proben): AIPW-Vorzeichen 74 / 80 / 47 / 43 % (Ep. 5: 50 / 59 / 36 / 63) – an den Nahpunkten
kommt das Signal. Gegen Simulator: Konzept +0,98 (70 %), nach V. +1,18 (83 %), Nachlegen 1. Besuch −0,08 (47 %), später −3,95 (10 %), Leads −0,17
(45 %). Kopplung ρ 0,34–0,80, Rauschen 0,8–1,4 d (kleiner als bei allen anderen TF1-Modellen). Extra-Angebot: versendet −0,22. Storno:
Median 10,8 d, 3 % im Fenster.
- Lauf auf Wunsch des Nutzers bei Ep. 18 abgebrochen (untertrainiert: 30 WP-Epochen ≈ 2 ZZ-TF1-Epochen an Schritten) und als **ZZ-TF1-WP150**
  neu gestartet (22:55): 150 Epochen, Training und Validierung nur am Options-Event (neue Option `only_points_validation` /
  `--only-points-validation`, Schnelltest 259 von 2.544 Validierungsproben), Begleiter `nur_wahlpunkte150_20261117/` mit Prüfreihe nach
  Ep. 15 / 45 / 90 / Ende, tmux psp_wp150 / psp_wp150_begleiter / psp_wp150_sg.

## 2026-11-17 21:15: ZZ-TF1-WP Ep. 5 (nur Wahlpunkt-Proben) – noch kein Optionssignal; Epochen nicht vergleichbar

- Training: 75.381 von 782.630 Proben (nur Options-Events), ~8–10 min je Epoche (Validierung weiter auf allen ~180.000 Prefixen).
- Optionssignal Ep. 5: Diff 0,01 / 0,20 / 0,36 / −0,27 Tage, AIPW-Vorzeichen 50 / 59 / 36 / 63 %, Kopplung ρ 0,66–0,85 (Rauschen nur
  0,76–1,12 d) – die Optionen werden fast gleich gezogen, die Option ändert den Verlauf kaum („Events mehr“ ≈ 0). Gegen Simulator:
  Konzept +0,19 (2,79), nach V. +0,46 (2,40), Nachlegen −0,49 / später −3,58 (8 % richtig), Leads +0,10 (1,32).
- Wichtig für den Vergleich: eine WP-Epoche hat ~75.000 Proben, eine ZZ-TF1-Epoche ~1,08 Mio. → WP Ep. 5 ≈ 0,35 ZZ-TF1-Epochen an
  Gradientenschritten; WP Ep. 30 ≈ 2,1 ZZ-TF1-Epochen. ZZ-TF1 zeigte das Konzept-Signal ab Ep. 3 (≈ 43 WP-Epochen). Der jetzige Lauf
  ist also vermutlich zu kurz; fair wäre ein Vergleich bei gleich vielen Schritten.

## 2026-11-17 21:00: ZZ-TF1 online auf 12 Seeds – kein Gewinn gegenüber der Bank (Teil 2, `f_zwoelf_auswertung.md`)

| Arm (12 Seeds × 200, first_only) | Gewinn gegen Bank | Regel-Option (Konzept / nach V. / Nachlegen / Leads) | Angebote/Fall |
|---|---|---|---|
| Regel | +2,63 ± 0,37 | 100 % | 1,26 |
| (c) 300 fest | +1,97 ± 0,51 | 94 % (99 / 95 / 91 / 94) | 1,39 |
| (c) 100 fest | +1,57 ± 0,43 | 83 % (90 / 87 / 78 / 83) | 1,56 |
| **ZZ-TF1 100 fest** | **−0,08 ± 0,48** | **62 % (75 / 88 / 48 / 53)** | **1,92** |

- Gepaart: (c) 100 − ZZ-TF1 = **+1,65 ± 0,37 (z 4,4)**, Regel − ZZ-TF1 = 2,71 ± 0,35. ZZ-TF1 entscheidet an Nachlegen und Leads praktisch per
  Münzwurf (48 / 53 %) und legt deutlich öfter nach (1,92 Angebote/Fall) → genau wie offline vorhergesagt (Fernpunkte ≈ 50 %).
- **Kernbefund:** gültige Abläufe und beste Suffix-/Zeitgüte (ZZ-TF1: DLS 0,41, 0,6 + 0,4 ungültig je 100) bringen online nichts, wenn die
  Option die späteren Verläufe nicht beeinflusst. (c) mit vielen ungültigen Übergängen entscheidet besser, weil sein Decoder die Option
  direkt in die Verteilung je Position trägt (Kopplung ρ 0,64). Suffixgüte ist kein Maß für Entscheidungsgüte.
- Bestes PSP online bleibt (c) + first_only (+ 300 Ziehungen); Rückstand zur Regel 0,66 ± 0,32.

## 2026-11-17 20:15: ZZ-TF1 fertig (Ep. 15) – der Leads-Fortschritt von Ep. 12 hält nicht; keine Verlängerung auf 20 Epochen

- AIPW-Vorzeichen 74 / 96 / 49 / 50 % (Ep. 12: 77 / 97 / 50 / 60). Gegen Simulator: Konzept +1,36 (Wahrheit 2,79, 80 % richtig),
  nach V. +2,29 (2,40, 77 %), Nachlegen 1. Besuch +0,12 (2,11, 52 %), später +0,54 (0,42, 68 %), Leads −0,03 (1,32, 52 %).
  Ep. 12 → 15: Leads +0,32 → −0,03, Nachlegen später +1,44 → +0,54 (je SE ~0,25–0,38) → der Sprung bei Ep. 12 war wohl Messrauschen
  (200 Entscheidungen bzw. 60 Zustände), kein stetiger Trend.
- Gedächtnis: Lead 0,21 (echt 2,83), Nachfassen nach Extra-Angebot 0,49 (Simulator 1,92). Rauschen 1,8–1,9, ρ 0,14–0,21.
- Storno-Zeitpunkt: ab letztem O_Sent Median 12,3 d (echt 30,7), Anteil 29–32 d 3,4 % (echt 73 %) – unverändert.
- Entscheidung nach dem Kriterium von 17:45: Fernpunkte legen von Ep. 12 auf 15 nicht zu → ZZ-TF1 nicht auf 20 Epochen fortsetzen.
- Online-Test ZZ-TF1 auf 12 Seeds läuft (Teil 2), Prüfreihe nach Suffixgüte folgt (psp_zztf1_sg).

## 2026-11-17 19:45: Aufbau F auf 12 Seeds × 200 Fälle (Teil 1 fertig; `f_zwoelf_auswertung.md`, Aufbau „L“ in online.csv)

| Arm | Gewinn gegen Bank (± SE, 12 Seeds) | Regel-Option | Angebote/Fall |
|---|---|---|---|
| feste Regel | **+2,63 ± 0,37** | 100 % | 1,26 |
| (c) 300 fest + first_only | +1,97 ± 0,51 | 94 % | 1,39 |
| (c) 100 fest + first_only | +1,57 ± 0,43 | 83 % | 1,56 |
| Bank | 0 (21,25 d) | – | 1,37 |

- Gepaart je Seed: **Regel − (c) 100 = 1,06 ± 0,26 (z 4,1)**, **Regel − (c) 300 = 0,66 ± 0,32 (z 2,1)**, (c) 300 − (c) 100 = +0,40 ± 0,41 (z 1,0).
- → (c) erreicht die Regel nicht; mit 300 Ziehungen schrumpft der Rückstand auf ~0,7 Tage (knapp signifikant). Der Vorteil von 300 gegenüber
  100 Ziehungen zeigt in die erwartete Richtung (Wahlen 83 → 94 % regeltreu), ist aber auch mit 12 Seeds nicht gesichert (SD je Seed ~1,4 d,
  größer als aus 3 Seeds geschätzt). Restrückstand passt zu den verbleibenden 6 % Wahlen gegen die Regel (u. a. Nachlegen 91 %).
- Gewinn je Seed (Regel / c100 / c300): sehr verschieden, z. B. Seed 8 −0,25 / −0,39 / −2,04, Seed 6 2,72 / 0,91 / 4,23.
- Teil 2 (ZZ-TF1, 12 Seeds) startet nach dem fertigen ZZ-TF1.

## 2026-11-17 18:40: ZZU-TF1 Ep. 6 (Uhren) – weiterhin kein Vorteil, Storno-Frist nicht getroffen

- Optionssignal (`zaehler_uhren_tf1_20261117/pruefung_ep6.md`): AIPW-Vorzeichen 79 / 97 / 47 / 29 % (ZZ-TF1 Ep. 6 ohne Uhren 88 / 95 / 47 / 32).
- Gegen Simulator: Konzept +1,36 (Wahrheit 2,79, 82 % richtig), nach V. +2,71 (2,40, 95 %), Nachlegen 1. Besuch −0,15 (2,11, 47 %),
  später −1,66 (0,42, 27 %), Leads −0,86 (1,32, 35 %) – Fernpunkte falsche Richtung. Extra-Angebot: versendet +0,28 (Simulator 1,07),
  storniert +0,53 (1,25); Lead-Gedächtnis 0,59 (echt 2,83). Rauschen 1,6–1,8, ρ 0,14–0,30.
- Storno-Zeitpunkt (`storno_abstand.py`, gleiche 300 Prefixe, Saat 0): ab letztem O_Sent Median 11,7 d (echt 30,7), Anteil 29–32 d 3,7 %
  (echt 73 %), Abstand vor A_Cancelled 10,2 d (19,1) – praktisch wie Ep. 3 (10,9 d / 3,4 %) und wie ZZ-TF1 Ep. 12 (12,1 / 3,2 %).
  → Die Uhren lernen die Frist bis Ep. 6 nicht. Nächste Messung Ep. 9 (~22 Uhr); bleibt es so, ist die Frist mit normalverteilten
  Abständen offenbar nicht erreichbar (Zeitregel beim Ziehen nur mit Zustimmung des Nutzers).

## 2026-11-17 18:25: Aufbau F auf 12 Seeds (läuft) – `modellvergleich_20261116/f_zwoelf_seeds.py`, Auswertung `f_zwoelf_auswerten.py`

- Teil 1 (tmux psp_f12, ab 18:22, 6 Prozesse): Seeds 3–11 für Bank, Regel, (c) 100 fest + first_only, (c) 300 fest + first_only. Bankprüfung
  Seed 0 reproduziert. Seeds 0–2 werden aus den bisherigen Läufen übernommen (gleicher Aufbau).
- Teil 2 (tmux psp_f12_zz, wartet auf das fertige ZZ-TF1 und das Ende von Teil 1; 4 Prozesse): ZZ-TF1 100 fest + first_only, Seeds 0–11.
- Fragen: (1) erreicht (c) die Regel (Ziel SE ≈ 0,2)? (2) bringen 300 Ziehungen online etwas? (3) ZZ-TF1 (gültige Abläufe) gegen (c)?

## 2026-11-17 18:35: Warum gepaarte Online-Vergleiche in F um ~1 Tag je Seed schwanken (Simulator-Zufallsströme)

- Zufall im Simulator: je Fall ein eigener Strom für nächstes Event und Zeiten (`np.random.default_rng([seed, Fallnummer])`), Ankünfte aus
  einem eigenen Strom – beides von Entscheidungen unberührt. Die zufällige Mitarbeiterzuteilung (Aufbau F: `zuteilung: random`) zieht
  dagegen aus Pythons globalem `random` (`resources/selection_strategies.py`, `random.choice`) – ein Strom für alle Fälle.
- Gleiche Wahlen → identisches Ergebnis: Causal Forest vs. Regel (in jedem Fall dieselben Wahlen) in allen 600 Fällen Differenz 0,0.
- (c) 300 + first_only vs. Regel, Fall für Fall: in 365 Fällen mit genau denselben Wahlen unterscheidet sich die Durchlaufzeit trotzdem
  in 94 % der Fälle (Mittel +0,22 d, SD 10,3 d je Fall); in 235 Fällen mit abweichender Wahl Mittel +1,79 d, SD 20,4 d. Sobald ein Fall
  anders läuft, fragt er zu anderen Zeiten Mitarbeiter an, der globale Zuteilungsstrom verschiebt sich, alle folgenden Fälle bekommen
  andere Mitarbeiter/Wartezeiten → die Paarung geht verloren.
- Folge: Rückstand (c) − Regel je Seed 1,38 / 1,07 / 0,05 (SD ≈ 0,7) → mit 3 Seeds SE ≈ 0,4; für SE ≈ 0,2 (Unterschiede ≥ ~0,5 d
  trennbar) etwa 10–12 Seeds. Alternative: Zuteilung aus einem Strom je Fall (Simulator-Änderung; alle Bank-/Regel-Referenzen in F
  müssten neu gerechnet werden).

## 2026-11-17 18:20: (c) first_only mit 300 statt 100 Ziehungen in Aufbau F (`c_first_only_300_im_korrigierten_simulator.py`, Plan 1a)

- Gewinn gegen Bank +0,68 / +0,89 / +3,77 → **+1,78 ± 1,00** (100 Ziehungen: +1,32 / +0,16 / +3,71 → +1,73 ± 1,04); gepaart 300 − 100:
  −0,64 / +0,73 / +0,06 → **+0,05 ± 0,40** → kein messbarer Gewinn. Rückstand auf die Regel 0,83 ± 0,40 (100: 0,89 ± 0,49).
- Die Wahlen werden aber deutlich regeltreuer: Anteil Regel-Option 82 → 94 % (Nachlegen 76 → 91, Konzept 90 → 98, nach V. 83 → 93,
  Leads 83 → 95 %). Dass das Ergebnis trotzdem kaum steigt, passt zum Rauschen des Aufbaus: sobald eine Wahl abweicht, laufen die
  Zufallsströme des Simulators auseinander, die gepaarten Unterschiede je Seed schwanken um ~1 Tag. 3 × 200 Fälle reichen nicht, um
  Unterschiede < ~1 Tag zwischen (c) und Regel zu trennen.
- Angebote je Fall 1,40 (Bank 1,31, Regel 1,27). Auffällig erneut Seed 2: Arbeitskosten 130 € (Bank 54 €), wie „mit Rückfall“ Seed 2 (121 €) –
  nicht untersucht, betrifft nur Kosten, nicht die Durchlaufzeit.
- Schlussprüfung des Skripts meldete geänderte Quellen (psp_model.py um 17:56 um only_points erweitert); Jobs hatten den Code um 17:26
  geladen, Änderung betrifft nur das Training → Ergebnisse gültig (HINWEIS.txt im Ergebnisordner).

## 2026-11-17 18:05: Neu – Training nur auf Wahlpunkt-Proben (ZZ-TF1-WP), vorgemerkt

- Bisher: alle Modelle auf allen Prefixen, Wahlpunkt-Proben 3-fach (75.381 von 933.392 Proben ≈ 8 %, ~20 % des Gewichts). Nie nur Wahlpunkte.
- Neue Option `only_points` (psp_model) / `--only-points` (train_psp_real.py), Standard aus; Schnelltest 300 Fälle: 1.065 von 11.744 Proben.
- Lauf ZZ-TF1-WP: Einstellungen wie ZZ-TF1 + nur Wahlpunkt-Proben, 30 Epochen; startet automatisch, sobald ZZ-TF1 fertig ist (tmux psp_wp);
  Prüfreihe nach Ep. 5 / 10 / 20 / Ende (`nur_wahlpunkte_20261117/begleiter.py`), Suffixgüte je Stand. Vergleich mit ZZ-TF1 bei gleicher Epoche.
- Erwartung: Fernpunkte vielleicht besser (Option endet jedes Beispiel, Training genau auf dem Abgefragten); allgemeine Suffixgüte auf
  zufälligen Prefixen eher schlechter (weniger Startzustände); Storno-Frist und Kopplung dürften unverändert bleiben.

## 2026-11-17 17:35: Gumbel-Kopplung auf ZZ-TF1 Ep. 12 hilft nicht (`beschraenkt_20261117/rauschen_zztf1_12_gumbel`, Plan Punkt 1b)

- Kopie `modelle/zztf1_12_gumbel.pkl` (nur Gumbel-Kopplung der Eventwahl, keine Masken, gleiche Verteilung). psp_rauschen, 200 Entscheidungen
  je Punkt, 100 Ziehungen: Korrelation der Optionen 0,11 / 0,15 / 0,12 / 0,22 (ohne Gumbel 0,14 / 0,13 / 0,11 / 0,20), gepaarte SE
  1,80 / 1,86 / 1,75 / 1,81 (ohne 1,80 / 1,87 / 1,78 / 1,79), AIPW-Vorzeichen 74 / 93 / 45 / 59 % (ohne 77 / 97 / 50 / 60). → keine Wirkung.
- Wie bei (c) (Schnelltest 16.11.: ρ 0,62 → 0,52). Deutung (nicht bewiesen): die gemeinsamen Zufallszahlen (Dropout-Masken, Rauschen) sind je
  Schritt-Index gekoppelt; sobald sich die Verläufe beider Optionen um ein Event verschieben (z. B. Extra-Angebot), passt das Rauschen nicht
  mehr zusammen. Ein TF1-Netz nutzt seine Vorgeschichte und läuft deshalb schnell auseinander; (c) ignoriert sie weitgehend (ρ 0,64).
  Mögliche Alternative: Rauschen je Token-Vorkommen statt je Schritt (z. B. Abstand des k-ten O_Sent in beiden Optionen mit derselben Zahl).
- Restliche Prüfreihe (punkte_alle usw.) läuft weiter, Ergebnis erscheint im Register als „ZZ-TF1 ep12 Gumbel“.

## 2026-11-17 18:00: Aktualisierter Plan (nach first_only, Rauschformel, Storno-Messung, Paper Mustroph & Rinderle-Ma 2026)

Stand: bestes PSP online (c) fest 100 + first_only +1,73 (Regel/Forest +2,62); ZZ-TF1 gültige Abläufe + beste Suffixe, Fernpunkte wachsen
mit Epochen, aber ρ 0,15 → Rauschen 1,8; kein Modell trifft die 30-Tage-Storno-Frist; Screening: keine fallweise Heterogenität → Ziel ist,
die Regel zu erreichen (Richtung und Größe der Effekte richtig), nicht sie zu schlagen.
1. Sofort, CPU: (c) first_only 300 Ziehungen fest in F (Rauschkosten online); Gumbel-Kopplung ohne Masken auf ZZ-TF1 Ep. 12 offline
   (steigt ρ von 0,15? `psp_beschraenkt.beschraenken(..., gumbel=True, masken=False)` + psp_rauschen/punkte_alle).
2. Nach ZZ-TF1 final (~20 Uhr): Prüfreihe + Storno-Messung; online in F mit first_only (fest 100; mit Gumbel, falls 1. hilft) gepaart zu (c).
   Fortsetzen auf 20 Epochen mit LR 3e-5 nur, wenn Leads/Nachlegen von Ep. 12 auf 15 weiter zulegen.
3. Decision-aware decoding (Paper) als Prototyp für die Fernpunkte: Entscheidungspunkte per Inductive Miner, Entscheidungsmodell je Punkt
   mit Datenzustand inkl. Zähler/Uhren/Beträge, Umgewichten der Eventwahl beim Ziehen (ε 1e-3, β_max 2, α 0,1); Prüfung: DLS, ungültig,
   Wirkung eines Extra-Angebots/Leads auf die Folgeverzweigungen gegen Simulator-Wahrheit. Semantic loss im Training laut Paper wirkungslos → weglassen.
   Braucht catboost (nicht installiert) oder sklearn-Bäume.
4. Storno-Frist: erst ZZU-TF1 Ep. 6/9 messen; lernen die Uhren sie nicht, eine gelernte Zeitregel beim Ziehen (A_Cancelled ≈ letzter O_Sent
   + empirische Frist) – berührt die Invariante „Zeit je Schritt vom Decoder“ → nur mit Zustimmung des Nutzers.

## 2026-11-17 17:45: Lernkurve ZZ-TF1 auf Testfällen – lohnen mehr Epochen? (`verlust_diagnose_20261117/verlust_je_ausgabe_lernkurve.csv`)

- Roher Verlust mit Teacher Forcing 1 auf 300 Testprefixen (nächste 10 echte Events), Ep. 3 → 6 → 9 → 12:
  Aktivität Kreuzentropie 0,397 → 0,355 → 0,344 → **0,336** (Treffer 86,4 → 87,6 → 87,9 → 88,2 %); Fallzeit attenuiert −2,235 → −2,281 →
  −2,297 → −2,312; Abstand-MAE 0,532 → 0,527 → 0,522 → 0,522 Tage. Zum Vergleich (c)-TF1 nach 15 Ep.: 0,339 / 87,7 % / 0,525 d.
- Kein Überanpassen sichtbar (Testverlust fällt weiter), aber die Gewinne werden klein (Kreuzentropie −0,042 / −0,011 / −0,008 je 3 Ep.;
  Abstände schon flach). Die Entscheidungssignale an den Fernpunkten wachsen dagegen noch deutlich (Leads 29 → 32 → 41 → 60 % AIPW-Richtung,
  Nachlegen später gegen Simulator 0,35 → 1,44). → Mehr Epochen sind plausibel nützlich für die Fernpunkte, kaum für Suffix-/Zeitgüte.
- Fortsetzen ist ohne Neustart möglich, aber nicht eingebaut: last.pt enthält nur Gewichte + kwargs (kein Optimierer-/GradNorm-Zustand);
  Fortsetzen = Gewichte laden, AdamW neu. Bei TF 1 und konstanter LR 1e-4 entspricht das (bis auf die Optimierer-Momente) längerem Training.

## 2026-11-17 17:20: ZZ-TF1 Ep. 12 und ZZU-TF1 Ep. 3 (Uhren) + Stornierung in gezogenen Suffixen

**ZZ-TF1 Ep. 12** (`zaehler_zeit_tf1_20261117/pruefung_ep12.md`): AIPW-Vorzeichen 77 / 97 / 50 / 60 % (Ep. 9: 80 / 97 / 46 / 41).
Gegen Simulator: Konzept +1,61 (Wahrheit 2,79, 83 % richtig), nach V. +3,04 (2,40, 93 %), Nachlegen 1. Besuch −0,13 (2,11, 47 %),
**Nachlegen später +1,44 (0,42, 77 % richtig)**, **Leads +0,32 (1,32, 62 %)** – Leads erstmals in richtiger Richtung (Ep. 3–9 falsch).
Suffixgüte beste bisher: DLS 0,41, Schritt 10 27 %, ungültig 0,6 + 0,4 je 100, Restzeit-MAE 8,28, Bias −0,20, Abdeckung 82 %.
Rauschen 1,81, ρ 0,15. Gedächtnis weiter schwach: Lead 0,45 (echt 2,83), Nachfassen nach Extra-Angebot 0,53 (Simulator 1,92).

**ZZU-TF1 Ep. 3** (Uhren; `zaehler_uhren_tf1_20261117/pruefung_ep3.md`): AIPW-Vorzeichen 85 / 99 / 42 / 35 % ≈ ZZ-TF1 Ep. 3 (82 / 96 / 50 / 29).
Gegen Simulator: Konzept +1,53, nach V. +3,91 (überschätzt), Nachlegen −0,33 / −0,88, Leads −0,97 (alle drei Fernpunkte falsche Richtung).
Extra-Angebot ändert kaum etwas: zusätzlich versendet +0,19 (Simulator 1,07), O_Cancelled +0,05 (1,25). Suffix: DLS 0,36, ungültig 1,2 + 0,9,
Restzeit-MAE 8,54, Bias +0,03. Rauschen 1,72, ρ 0,25. → nach 3 Epochen kein Vorteil der Uhren erkennbar.

**Stornierung in gezogenen Suffixen** (`zaehler_uhren_tf1_20261117/storno_abstand.py/.csv`; 300 Testprefixe, Schnitt nach dem 1. O_Sent,
20 Ziehungen bis EOS; Tage):

| Quelle | storniert | Abstand vor A_Cancelled | ab letztem O_Sent: Median (P10–P90) | Anteil 29–32 d | Restzeit bis Storno |
|---|---|---|---|---|---|
| echter Log | 35 % | 19,1 | **30,7** (5,0–30,9) | **73 %** | 27,6 |
| (c) | 33 % | 3,1 | 8,6 (0,2–22,1) | 1 % | 13,4 |
| (c)-TF1 | 34 % | 6,4 | 15,8 (4,0–32,6) | 4 % | 19,5 |
| ZZ-TF1 Ep. 12 | 35 % | 10,8 | 12,1 (3,4–36,2) | 3 % | 17,8 |
| ZZU-TF1 Ep. 3 | 32 % | 10,4 | 10,9 (3,0–36,6) | 3 % | 17,6 |

- Der Anteil Stornierungen stimmt bei allen, der Zeitpunkt bei keinem: die 30-Tage-Frist (73 % der echten Stornos im Fenster 29–32 d)
  trifft kein Modell (1–4 %); Stornos kommen im Mittel 10–17 Tage zu früh. Die Uhren haben das nach 3 Epochen nicht behoben.
- Vermutete Grenze (nicht bewiesen): der Zeitkopf zieht normalverteilte Abstände; eine scharfe Frist braucht eine kleine Varianz
  genau dann, wenn „Tage seit O_Sent“ hoch ist – möglich mit Uhren, aber offen, ob das Netz es lernt. Nächste Prüfungen Ep. 6 / 9.

## 2026-11-17 13:30: (c) in Aufbau F mit first_only = angebot_nachlegen (`modellvergleich_20261116/c_first_only_im_korrigierten_simulator.py`, Auswertung `c_first_only_auswertung.md/.csv`)

- Gleiche Bedingungen wie 16.11. (Bankreferenz bitgleich reproduziert), einzige Änderung: an Nachlegen entscheidet (c) nur beim 1. Besuch.
  Gewinn gegen Bank (Seeds 0/1/2, gepaart) und Rückstand auf die feste Regel (+2,62):

| Arm | ohne first_only | mit first_only | Rückstand Regel (mit) | Angebote/Fall (mit; ohne) |
|---|---|---|---|---|
| fest 100 | +0,03 | **+1,73 ± 1,04** (+1,32 / +0,16 / +3,71) | 0,89 ± 0,49 | 1,53; 2,44 |
| adaptiv ohne Rückfall | +0,88 | +1,30 ± 0,72 (+0,19 / +1,06 / +2,64) | 1,32 ± 0,29 | 1,50; 2,31 |
| adaptiv mit Rückfall | −0,33 | −0,28 ± 0,76 (−0,31 / −1,58 / +1,05) | 2,90 ± 0,35 | 1,34; 1,87 |

- Gepaart mit minus ohne first_only: fest 100 +2,00 / +0,91 / +2,20 → **+1,70 d**; adaptiv ohne +0,42; mit Rückfall +0,05. Die Schleife
  kostete also bei fest 100 fast den ganzen Gewinn; Angebote je Fall jetzt 1,5 (Bank 1,31, Regel 1,27), ≥ 4 Angebote 2,7 % statt 17 %.
- Rückfall auf die Bank schadet: auch „unsichere“ Entscheidungen (|z| ≤ 2) wählen ohne Rückfall zu 78–88 % die Regel-Option (sichere 88–97 %),
  der Rückfall gibt 56–86 % der Wahlen an die Bank → Arm ≈ Bank. Die 2-SE-Schwelle je Fall verwirft ein über die Fälle konsistentes Vorzeichen.
- Bester (c)-Stand online in F: fest 100 + first_only, +1,73 d, noch 0,9 d hinter Regel/Causal Forest (+2,62). Auffällig: Arbeitskosten
  „mit Rückfall“ Seed 2 121 € (Bank 54 €) – nicht untersucht.
- Folgerung: first_only an Nachlegen bei allen PSP-Online-Läufen setzen; kein Rückfall auf die Bank; nächster Schritt 300 Ziehungen fest
  + first_only (Erwartung: weniger Rauschen → näher an der Regel).

## 2026-11-17 13:40: Trainingsproben Original vs. (c) – wie viel vom Suffix sieht das Netz?

- Proben (`new_event_log_loader_v2.py`, encode_*_column): je Fall jede Prefixlänge 1…n, Ziel = die nächsten S Events (S = min_suffix_size
  = Decoderlänge, an jeden Fall werden S EOS-Zeilen gehängt). Original-Repo (BPIC17-Notebook) und unser 10k-Original: S = 5; (c): S = 60.
- BPIC17 (31.509 Fälle, 38,2 Events/Fall, 1,20 Mio. Prefixe, Rest im Median 19 Events): ganzes Suffix inkl. Fallende im Ziel bei S = 5
  nur für 13,1 % der Prefixe, S = 40 85,0 %, S = 60 96,2 %. Beim Ziehen läuft der Decoder trotzdem bis EOS → beim Original sind alle
  Schritte ab 6 ungeübte Extrapolation; das Fallende langer Fälle lernt es nie (passt zu psp-diagnosis: kein fallspezifisches Signal).

## 2026-11-17 13:30: Woher das Entscheidungsrauschen kommt (register.csv, alle Modelle)

- Das Rauschen einer Einzelentscheidung (gepaarte SE, 100 Ziehungen) folgt für alle Modelle fast exakt
  **SE ≈ √(2·(1 − ρ)) · SD / √n** (SD = Streuung der gezogenen Durchlaufzeit, ρ = Korrelation der Optionen, n = Ziehungen):
  (c) 1,29 gemessen / 1,30 Formel, (c)-TF1 1,97 / 1,98, ZZ final 4,91 / 4,94, (a) 8,13 / 8,12, Piloten 1,05 / 1,06.
- Über die Epochen sinkt es nicht: ZZ 1,5 (Ep. 3) → 2,7 (Ep. 5) → 4,1 (Ep. 9) → 4,9 (Ep. 15), weil die SD 14 → 46 d aufbläht;
  ZZ-TF1 1,8 / 1,7 / 1,8 (Ep. 3/6/9) flach; OE-TF1 0,97 (Ep. 1) → 1,54 (Ep. 26), weil ρ 0,57 → 0,12 fällt (das Netz nutzt seine eigene
  Vorgeschichte, die Optionen laufen auseinander). Piloten (6 Ep.) rauschen wenig, weil sie beide Optionen fast gleich ziehen (kein Signal).
- Folgerung: eine kalibrierte SD von ~15 d ist echte Unsicherheit und bleibt; senkbar ist das Rauschen nur über mehr Ziehungen
  (∝ 1/√n; (c) bräuchte im Median 160–390 für |z| = 2), höhere Kopplung ρ (bei TF1 0,12–0,25 der größte Hebel: ρ 0,64 → SE × 0,65)
  und das Beheben künstlicher Streuung (ZZ-Storno-Wartezeiten), nicht über längeres Training.

## 2026-11-17 13:05: Wann fällt (c) auf die Bank zurück? (`modellvergleich_20261116/c_rueckfall_analyse.py/.md/.csv`, Aufbau F, adaptiv mit Rückfall)

- Regel der Policy (`policies/ev_policy/psp_policy.py`): je Option 30 Ziehungen (gemeinsame Zufallszahlen); ist |Mittelwertdifferenz| ≤ 2 gepaarte SE,
  100 Ziehungen; dann immer noch ≤ 2 SE → None = Bankverhalten (der Simulator zieht das nächste Event). Unsicherheit = Monte-Carlo-Rauschen
  der Suffixe, keine Modell-/Epistemik-Unsicherheit. Gepaarte SE mit 100 Ziehungen ≈ 1,8 Tage → Schwelle ≈ 3,6 Tage, die echten Effekte
  (Simulator-Wahrheit 0,4–2,8 Tage) liegen fast alle darunter.
- Rückfall bei 65 % der 2.115 Wahlen mit 2 Optionen: handle_leads 79 %, konzept 72 %, nach Vervollst. 57 %, nachlegen 62 %. Kaum abhängig
  von Antragsart/Betrag; stark vom Fallfortschritt: nachlegen 1. Besuch 82 % Rückfall (Fallalter < 1 d 82 %), 2. Besuch 46 %, 3. 34 %, ab 4. 27 %
  (Fallalter > 30 d: 7 %). Grund: späte Prefixe haben kürzere Restsuffixe → kleinere Streuung → Differenz passiert die 2-SE-Schwelle.
- Was (c) wählt, wenn es „sicher“ ist: konzept/nach V. fast immer Annehmen (83/2, 176/5), leads fast immer zurückziehen (70/7) = wie die Regel.
  Nachlegen 1. Besuch senden 98 / weiteres Angebot 9 (= Regel), **ab 2. Besuch fast nur weiteres Angebot: 66/7, 53/0, 154/1** – die Regel
  sendet immer, Simulator-Wahrheit ab 2. Besuch +0,42 d für senden. → selbstverstärkende Angebotsschleife: jedes neue Angebot führt
  wieder an den Punkt, wo (c) erneut nachlegt (bis 31 Besuche in einem Fall). Bekannt seit 10.11. (first_only in PSPPolicy), in F aber
  ohne first_only gelaufen.
- Folgen (Angebote je Fall / Anteil ≥ 4 / max): Bank 1,31 / 1,3 % / 6; Regel 1,27 / 0,2 % / 4; (c) mit Rückfall 1,86 / 11 % / 32; ohne 2,31 / 15 % / 48;
  fest 100 2,44 / 17 % / 30. Durchlaufzeit der (c)-Fälle (mit Rückfall) nach Besuchen am Punkt: 1 → 19,3 d, 2–3 → 23,9 d, ≥ 4 → 38,8 d (52 von 600 Fällen).
  Damit erklärt die Schleife einen großen Teil des Rückstands von (c) auf die Regel in F (1,7 d gepaart).
- Folgerung: (c) online in F mit first_only = angebot_nachlegen erneut prüfen (kostet nichts am Modell); allgemein: das Signal ab dem 2. Angebot
  ist ein Beobachtungszusammenhang, kein Effekt.

## 2026-11-17 12:45: Causal Forest (Ziel Durchlaufzeit), X-Learner und Regel in Aufbau F (`modellvergleich_20261116/cf_im_korrigierten_simulator.py`)

- Bankreferenz vom 15.11. exakt reproduziert. Gewinn gegen Bank (3 Seeds × 200, ± SE): **Causal Forest +2,62 ± 0,61 = feste Regel +2,62 ± 0,61**
  (fallweise identisch: der Forest wählt immer die Regel-Option; ebenso der Forest mit Kostenziel), X-Learner +2,29 ± 0,59 (bei
  niedrigeren Gesamtkosten 173 € statt 184 €), XGB +1,18, (c) adaptiv ohne Rückfall +0,88, D (zustandsbewusst) +0,82, (c) fest 100 +0,02.
- Gepaart je Seed: Regel minus (c) adaptiv ohne Rückfall +1,75 / +1,58 / +1,88 → **+1,74 ± 0,09 Tage**; Regel minus (c) fest 100
  +2,74 / +2,71 / +2,31 → +2,59 ± 0,14. Im älteren Aufbau A lag (c) nur 1,14 ± 0,11 hinter der Regel → der Rückstand von (c) ist in F
  wirklich größer (nicht nur Rauschen), Ursache offen (Angebotsschleife 2,44 Angebote/Fall, Optionsprüfung bei Konzept).

## 2026-11-17 12:30: Korrektur – Aufbau F (korrigierter Simulator) lief mit ZUFÄLLIGER Zuteilung

- `d_oe_zustand_20261115/catalog.yaml` und `c_im_korrigierten_simulator` (manifest: allocation random) nutzen die eingefrorene Katalogkopie
  mit `zuteilung: random`. F = zufällige Zuteilung + Optionsprüfung (15.11.), nicht gelernte Zuteilung. Frühere Aussagen („F mit gelernter
  Zuteilung“, „gelernte Zuteilung als mögliche Ursache für den Einbruch von (c)“) sind falsch. Der Unterschied zum älteren Aufbau A ist damit
  nur die Optionsprüfung (betrifft nur Konzept) plus Laufumfang (3 × 200 statt 10 × 1.000, andere Fälle) → Einbruch von (c) in F am ehesten
  Zufall/kleiner Lauf oder Konzept-Effekt der Optionsprüfung; offen. Passender AIPW-Vergleich: `screening_validierung_20261117_zufall`.
- Ziele in F: (c) und XGB Durchlaufzeit; Causal Forest Gesamtkosten (!) – trotzdem +2,62 Tage; D/OE je Durchlaufzeit und Gesamtkosten.

## 2026-11-17 10:08: AIPW auf simulierten Daten – liegt die Schwäche bei Nachlegen/Leads am Simulator? Nein.

`scripts/validate_screening.py`, 20.000 Fälle je Lauf (je ~5 min); Ergebnisse `screening_validierung_20261117_gelernt` (aktueller
Simulator: gelernte Zuteilung + Optionsprüfung) und `..._zufall` (zufällige Zuteilung wie die Online-Vergleiche). Effekt der Aktion in
Tagen (positiv = Aktion kürzer), Reihenfolge Konzept / nach V. / Nachlegen / Leads:
- echter Log (AIPW): +5,23 / +2,96 / −1,42 / −2,16 (SE 0,27–0,32)
- Simulator wahr (randomisiert), gelernt: +3,80 / +2,10 / **−1,32** / **−1,22** (SE 0,20–0,27); zufall: +3,56 / +1,91 / −1,64 / −1,38
- Anteil der echten Größe (gelernt): 73 / 71 / **93** / **56** %; echt gegen Simulator wahr (z): 3,9 / 2,3 / **−0,3** / −2,4
- AIPW auf beobachtet simulierten Daten trifft die Simulator-Wahrheit überall (|z| ≤ 1,8) → das Screening ist verlässlich.
Deutung: **Nachlegen bildet der Simulator in Richtung und Größe richtig ab** (93 %, z −0,3) → die schwachen Entscheidungen dort sind ein
Modellproblem. **Leads: Richtung klar richtig** (−1,22 ± 0,25), Größe nur 56 % → online kleinere Gewinne möglich, aber Modelle mit falschem
Vorzeichen (ZZ-TF1, OE, OE-TF1) liegen auch gegen den Simulator falsch → ebenfalls Modellproblem. Konzept/nach V. im Simulator ~70 %
der echten Größe. Gelernte vs. zufällige Zuteilung ändert wenig.

## 2026-11-17 09:50: ZZ final + ZZ-TF1 Ep. 3 / 6 – TF1 erreicht Gültigkeit, Suffix- und Zeitgüte, aber keine Fernwirkung

**ZZ final (freier Lauf, 15 Ep.)**: Vorzeichen 76 / 86 / 78 / 67 %, Simulator richtig 78 / 88 / 72 / 22 / 87 %, Lead-Gedächtnis 2,19,
eingefügtes Angebot +1,12, Nachfassen +2,63 – Gedächtnis da; aber Effekte 1,5–2,5-fach zu groß, Durchlaufzeit 35 Tage (SD 46), Rauschen
4,9, ungültig 47 + 26 je 100, Restzeit-MAE 10,6 / Bias +7,9. Insgesamt nicht besser als (c).

**ZZ-TF1 Ep. 3 / 6 (Teacher Forcing 1):**
- Gültig: 1,2 + 0,8 / 0,7 + 0,4 Verstöße je 100 Events; ohne Ende 2 / 1 %; Enden Ep. 6 55 / 33 / 11 % (echt 55 / 33 / 12).
- Suffixgüte bestes Niveau: DLS 0,36 / 0,40, Schritt 10 24 / 26 %, erster Abstand 12,9 / 11,7 h (bester Wert aller Modelle),
  Restzeit-MAE 8,56 / 8,25, Bias +0,56 / −0,74, CRPS 5,5. Restzeit ab erstem O_Created 19,5 (SD 13,6) – echter Log 20,3 (SD 12,7),
  Simulator 22,2 (14,8) → näher am echten Log als der Simulator.
- Entscheidungen an den Nahpunkten beste bisher: Konzept 82 / 88 % AIPW-Vorzeichen, gegen Simulator 87 % richtig; nach V. 96 / 97 %.
  Rauschen 1,67–1,79 (fast wie (c)), Korrelation der Optionen 0,25 / 0,19.
- Fernpunkte fehlen weiter: Nachlegen 50 / 46 % (Effekt ≈ 0), Leads 29 / 32 % (falsche Richtung: Start −1,0 statt +1,3 Tage gegen
  Simulator – wie bei OE: frühe Phase richtig, späte Wartezeit fehlt). Lead-Gedächtnis 0,44 / 0,65 (echt 2,83), Nachfassen nach
  Extra-Angebot 0,37 / 0,29 (Simulator 1,92) → die Zähler werden unter TF1 kaum genutzt (das befürchtete Anreizproblem).
- Stornierung zu früh: Abstand vor A_Cancelled 14,0 Tage (echt 20,7, Simulator 20,9); Restzeit storniert 23,9 statt 28,2 (echt).

**Neuer Befund zur Stornierung (echter Log):** A_Cancelled kommt **30,7 Tage nach dem (letzten) O_Sent** – IQR 30,5–30,8 Tage, 77 % im
Fenster 29–32 Tage. Eine harte Frist. Ohne „Zeit seit dem letzten O_Sent“ als Eingabe kann ein Netz unter TF1 sie nur schätzen
(normalverteilter Abstand, Mittel 14 statt 21); im freien Lauf führt dieselbe Unkenntnis zur riesigen Varianz (ZZ).
→ **Vorschlag**: Zeit seit dem letzten Auftreten je Token („Uhren“) als Eingabe in Encoder und Decoder, wie die Zähler (beim Ziehen aus
den gezogenen Abständen fortgeschrieben). Trifft die Frist direkt (Stornierungskanal) und liefert dem Netz eine bequeme Größe für
Fernwirkungen (Zeit seit Lead-Bearbeitung, seit Angebotsversand). Training ZZ-TF1 + Uhren ≈ 17 h GPU.

## 2026-11-17 03:20: Woher die Streuung von ZZ kommt (Nachlegen-Verläufe, Ep. 11 gegen Simulator)

- Fast alles aus einem einzigen Zeitkopf: dem Abstand vor A_Cancelled. Simulator: Mittel 27 Tage, P90 30 (scharfe 30-Tage-Frist);
  ZZ Ep. 3: 28 / 38; ZZ Ep. 11: **46 / 103**; größter Einzelabstand je Verlauf P99 136 Tage (Simulator 31), Maximum 491 Tage.
  Restzeit ohne die langen Abstände (≥ 10 Tage): SD 7,7 (Simulator 8,4, (c) 12,0) – der Rest stimmt also.
- Mechanismus: Abstände sind normalverteilt um den vorhergesagten Mittelwert (linearer StandardScaler, negative Werte auf 0). Im
  freien Lauf kennt der Zustand die eigene (falsche) Vorgeschichte, nicht wie lange der Kunde schon schweigt → der Fehler des
  Mittelwerts wächst, die Loss Attenuation antwortet mit größerer gelernter Varianz (σ grob 8 → 45 Tage) → gezogene Stornierungs-
  abstände von 100+ Tagen. Häufigkeit der Stornierung ist dabei normal (0,18 lange Storno-Abstände je Verlauf, Simulator 0,26).
- Folge: jede Option, die die Stornierungswahrscheinlichkeit leicht ändert, bewegt die Restzeit um ein Vielfaches → überschätzte
  Effekte (nach V. 8,7 statt 2,4) und Rauschen 4,3. Bei TF 0,73 (Ep. 3) war der Kopf scharf (28 / 38) → gutes Vorzeichen für ZZ-TF1.
  Bei ZZ-TF1 zu prüfen: Abstand vor A_Cancelled Mittel ≈ 27, P90 ≈ 30–40.

## 2026-11-17 01:10: Prüfreihe beschränktes Ziehen + Gesamtbild + Plan (Ziel des Nutzers: Outcome gut geschätzt, ungültige Sequenzen selten, Suffixe und Zeit gut)

**Ergebnis Masken beim Ziehen (volle Prüfreihe, Register „(c) Masken“, „ZZ ep9 Masken+Gumbel“): keine Lösung.**
- (c) Masken: AIPW-Vorzeichen 40 / 86 / 62 / 67 % (ohne Masken 84 / 96 / 80 / 80); Konzept kippt (offline −1,65, gegen Simulator −4,30,
  13 % richtig), Rauschen 2,78 statt 1,29. Ursache sichtbar: Angebotsschleifen – 5,8 / 4,3 Angebote je Verlauf bei Konzept (echt
  ~1,3–1,6), Stornierung 61–66 % (echt ~33 %). Die Masken erlauben O_Create Offer-Schleifen (erster Ordnung gültig), und weil 62 %
  der Netzwahrscheinlichkeit gesperrt sind, landet der Rest dort. Einzig Nachlegen wird besser: 1. Besuch 2,10 (Wahrheit 2,11),
  später −0,06 (Wahrheit +0,42; ohne Masken −1,91), zusätzlich verschickt +1,15 (Wahrheit 1,07).
- ZZ ep9 Masken+Gumbel: Vorzeichen 75 / 82 / 70 / 68 % (ohne 82 / 96 / 87 / 77), gegen Simulator 3,79 / 3,86 / 1,47 / −1,61 / 2,13,
  Rauschen 3,9; ebenfalls zu viele Angebote (3,5–4,4) und Stornierungen (~50 %).

- (c) Masken+Gumbel (01:08): wie (c) Masken – Vorzeichen 37 / 85 / 68 / 66 %, Konzept −3,93, Rauschen 2,75, Korrelation 0,36 (Gumbel
  bringt hier nichts), Lead-Gedächtnis −0,57. Bestätigt: Masken auf frei trainierten Netzen nicht weiterverfolgen.

**Gesamtbild (Deutung; passt zu allen Messungen, nicht einzeln bewiesen):**
- Freier Lauf mit Ziel an fester Position macht den Decoder faktisch zu einem Vorhersager der Randverteilung je Position
  („was steht an Position t, gegeben Prefix und Option“), kaum abhängig von der eigenen Vorgeschichte. Summen von Erwartungswerten
  stimmen auch ohne stimmige Abläufe (Restzeit-Bias (c) −0,05), die Option wirkt direkt, beide Optionen bleiben gekoppelt (ρ 0,64)
  → gutes Entscheidungssignal. Aber aus Randverteilungen gezogene Abläufe sind nicht stimmig (48 ungültige Übergänge je 100, 47 %
  ohne Ende). Bekannt aus der Literatur: Scheduled Sampling ist ein inkonsistentes Trainingsziel (Bengio et al. 2015; Kritik Huszár 2015).
- Teacher Forcing 1 ist das konsistente Ziel (Maximum Likelihood) → stimmige Abläufe (0,6 je 100, DLS 0,40, Enden 53 / 34 % ≈ echt).
  Schwächen dort: (a) Fernwirkung der Option fehlt im Mittel, nicht nur im Rauschen ((c)-TF1 Nachlegen +0,14 ± 0,14 statt −1,46);
  (b) Wartezeiten nicht am richtigen Event (Storno-Abstand 16 statt 21 Tage); (c) Ziehungen der Optionen ungekoppelt (ρ ≈ 0,1).
- ZZ zeigt, dass Zähler (a) und Zeit je Event (b) liefern – bisher nur im freien Lauf gemessen.
- Alle vier Ziele zusammen verlangen daher: konsistentes Training (TF 1) + explizite Verlaufsinformation (Zähler) + Zeit je Event,
  und das Rauschen über die Schätzung senken (Gumbel-Kopplung, mehr Ziehungen auf der GPU), nicht über Artefakte.

**Plan:**
1. Training ZZ-TF1 = (c)-Einstellungen + Zähler + Zeit je Event + Teacher Forcing 1 (**gestartet 2026-11-17 01:07 parallel zu ZZ auf
   der GPU**, tmux psp_zz_tf1 / psp_zztf1_begleiter / psp_zztf1_sg (Suffixgüte je Prüfstand); Log `data/PSP_real_alle_s60_normal_zaehler_zeit_tf1.log`; Skript: `data/bpic17_psp_real_zaehler_zeit_tf1.sh`,
   Begleiter `zaehler_zeit_tf1_20261117/begleiter.py`, Prüfreihe nach Ep. 3 / 6 / 9 / 12 / final; Register erkennt die Stände).
   Auswahl über die Prüfreihe, nicht über den Validierungsverlust (der wird im freien Lauf gemessen und ist für TF1 irreführend).
   Erfolgskriterien: ungültig < 2 je 100; Enden ≈ 55 / 33 %; Lead-Gedächtnis und Nachfassen nach Extra-Angebot > 0 in richtiger
   Größe; Storno-Abstand ~21 Tage; AIPW-Vorzeichen Nachlegen/Leads deutlich > 50 %; Effektgrößen näher an der Simulator-Wahrheit.
   Risiko: Fernwirkung bleibt trotz Zählern aus (Pilot decoder_context unter TF1 half nicht) → dann Zähler um gezielte Zustandsgrößen
   ergänzen (offene Angebote, Zeit seit letztem O_Sent) oder Option je Decoderschritt einspeisen.
2. Ziehen/Schätzen auf diesem Modell: Gumbel-Kopplung (vorhanden), Masken nur als Sicherheitsnetz (bei TF1 kaum gesperrte Masse →
   keine Verzerrung), Ziehungszahl bis |z| > 2 bzw. auf der GPU; offline Prüfreihe, dann online 10 Seeds gepaart.
   **Poolen je Punkt entfällt (Entscheidung des Nutzers 2026-11-17 01:05).**
3. Simulator: Konzept-Wahrheit und KPIs mit dem korrigierten Stand neu messen.
4. ZZ (freier Lauf) zu Ende führen (Ep. 11 / 13 / final) als Vergleich; Masken auf freilaufenden Netzen nicht weiterverfolgen.

## 2026-11-17 01:15: Beschränktes Ziehen umgesetzt (`prescriptive/estimators/psp_beschraenkt.py`), Schnelltest

- Umsetzung als eigene Klasse `BeschraenktesPSP` (PSPModel unverändert; Kopien in `beschraenkt_20261117/modelle/`): gelernte
  Übergänge zwischen Events (Fallende nur nach beobachteten Endtokens) + Lifecycle-Folgen je Aktivität + Fallende erst nach
  A_Pending/Denied/Cancelled (99,7 % der Trainingsfälle haben eines); Netzwahrscheinlichkeiten renormiert, Zeit weiter vom Decoder,
  kein Neuziehen (gleiche Zufallszahlen je Schritt). Optional Gumbel-Kopplung (argmax(Logits + Gumbel je Token) statt multinomial,
  gleiche Verteilung).
- Schnelltest (`schnelltest.py`, 40 Testfälle, Nachlegen 1. Besuch, 30 Ziehungen; `schnelltest.csv`): ungültige Schritte je 100
  Events (c) 74 → 0,3, ZZ ep9 66 → 0,2; ohne Endzustand 48 % → 0 %. Aber: das Netz legt 62 % ((c)) bzw. 54 % (ZZ) seiner
  Wahrscheinlichkeit auf gesperrte Tokens → gezogen wird aus dem Rest, die Endverteilung kippt zu Stornierungen ((c): Pending 22 %,
  Cancelled 57 %; echt ~55 / 33 %), Restzeit 26 statt ~22 Tage, SD 25 statt 15, Optionskorrelation 0,62 → 0,27 (mit Gumbel 0,34).
  Optionsunterschied extra − senden (Simulator +1,55, AIPW +1,42; SE 0,4–1,2): (c) 0,63, (c) Masken 1,35, ZZ 5,83 → Masken 3,77 →
  Masken+Gumbel 2,52. Gumbel ohne Masken: Korrelation 0,62 → 0,52 (keine Verbesserung).
- Nachtrag gleicher Schnelltest (gleiche 40 Fälle, gleicher Seed; (c) bitgleich reproduziert): OE ungültig 65 je 100, ohne Ende
  36 %, Pending/Cancelled 26 / 31 %, Restzeit 21,3 (SD 15,9), Korrelation 0,34, Effekt +1,10 ± 0,63; OE-TF1 Ep. 26 ungültig 1,2,
  ohne Ende 1,6 %, Pending/Cancelled 53 / 34 % (≈ echt!), Restzeit 19,1 (SD 11,0), Korrelation 0,09, Effekt −0,48 ± 0,46 (falsche
  Richtung). → TF1 zieht von sich aus realistische Enden, aber ohne Kopplung und ohne Fernsignal; Masken auf (c)/ZZ erzwingen Enden,
  aber mit falscher Verteilung.
- Warum OE-TF1 trotz gültiger Abläufe nicht entscheidet (Nachlegen-Verläufe aus Simulatorzuständen): Stornierungsweg zu kurz –
  Restzeit bei A_Cancelled 23,5 statt 30,4 Tage (Abstand vor A_Cancelled 16,3 statt 20,9), Pending 16,8 statt 17,9 → Abstand
  Storno − Pending nur 6,7 statt 12,5 Tage, Restzeit 18,8 (SD 11,4) statt 22,2 (14,8); Extra-Angebot: zweites Angebot verschickt
  (75 %, Simulator 89 %), aber Nachfassen danach kaum länger (+0,33 statt +1,92) → Effekt +0,5 statt +1,6; Optionskorrelation 0,09
  → Rauschen bleibt, Signal fast weg (AIPW-Vorzeichen 50 %).
- Volle Prüfreihe läuft (tmux psp_beschraenkt, `pruefreihe.py`): (c) Masken, ZZ ep9 Masken+Gumbel, (c) Masken+Gumbel → Register.
- Deutung bisher: Masken reparieren Gültigkeit und Endzustände, aber die Netze haben ihre Wahrscheinlichkeit an falscher Stelle; die
  Ursache (Versatz im freien Training) bleibt → wahrscheinlich nötig: Training so ändern, dass das Netz gültige Abläufe selbst
  bevorzugt (z. B. freier Lauf je Sequenz statt je Schritt, oder Masken schon im Training).

## 2026-11-17 00:30: Warum „Zeit je Event“ die Zeiten von ZZ aufbläht (`zeit_ungueltig.py`, `zeit_ungueltig.csv`)

- Die Zeit je Event funktioniert, wofür sie gebaut ist: Abstände hängen jetzt am richtigen Event. A_Cancelled (30-Tage-Frist):
  Simulator 20,9 Tage, (c) 3,0, ZZ Ep. 3 / 7 / 9 11,1 / 22,3 / 29,2. W_Validate application|start, A_Validating, O_Sent: Simulator
  ≈ 0, ZZ ≈ 0,01–0,05, (c) 0,5–1,2 Tage – (c) verschmiert die Wartezeit über alle Events.
- Aufgebläht wird die Zeit über die ungültigen Schritte: Anteil der gezogenen Restzeit auf ungültigen Schritten ZZ Ep. 3 / 7 / 9
  9 / 30 / 43 % (1,4 / 5,0 / 10,7 Tage), Restzeit 14,5 / 16,9 / 24,6 Tage, SD 14 / 26 / 38. Jedes falsche Event (A_Cancelled,
  Wiederholung von W_Call after offers resume/ate_abort/withdraw) bringt bei ZZ seine volle typische Wartezeit mit. (c) hat genauso
  viele ungültige Schritte (53 % der Zeit), aber dort hängt die Zeit nicht am Event, die Restzeit wird eher über die Schritte
  verteilt → falsche Events kosten wenig, dafür kann (c) „Option ändert das Ende → 30-Tage-Wartezeit“ nicht ausdrücken (Endzustand-
  Kanal ≈ 0).
- Beitrag des Trainings: Der Zeitkopf wird auf das echte Event konditioniert, die Vorgeschichte ist im freien Lauf aber die eigene
  → er lernt eher kontextfreie, eventtypische Wartezeiten und „weiß“ nicht, ob die Wartezeit im gezogenen Verlauf schon vergangen ist.
- → Nicht „Zeit je Event“ ist das Problem, sondern „Zeit je Event + ungültige Abläufe“. Revidierte Empfehlung: statt der Ablation
  „nur Zähler“ zuerst ZZ-Checkpoints mit beschränktem Ziehen (beide Übergangsmasken) offline prüfen – kein Training nötig; wenn die
  Abläufe gültig sind, könnte ZZ realistische Zeiten je Event + Gedächtnis + Endzustand-Kanal verbinden.

## 2026-11-17 00:45: Warum mehr Teacher Forcing das Entscheidungssignal schwächt (Erklärung, teils Hypothese)

Vergleich je Zwilling (normaler TF-Plan → TF immer 1): (c) → (c)-TF1, D → D-TF1, OE → OE-TF1 Ep. 26.
- **Rauschen**: Streuung je Option gleich ((c) 15,4 vs. 15,2 Tage SD), aber die Korrelation der beiden Optionen fällt von
  0,64 / 0,30 / 0,34 auf 0,15 / 0,13 / 0,12 → gepaarter SE = SD·√(2(1−ρ))/√n steigt (1,29 → 1,97 bei (c)). Der ganze
  Rauschanstieg kommt aus der verlorenen Kopplung der gemeinsamen Zufallszahlen.
- **Signal**: an den „Fernpunkten“ verschwindet der Effekt fast ganz – Nachlegen 83 / 10 / 46 % → −10 / −2 / −4 % des AIPW-Effekts,
  AIPW-Vorzeichen bei allen TF1-Modellen exakt 50 % (Münzwurf); Leads 56 / 80 / −9 % → 20 / 1 / −3 %. An den „Nahpunkten“
  Konzept und nach Vervollständigung bleibt ein Teil (Konzept 27 → 17 %, nach V. 71 → 54 % bei (c)).
- Erklärung 1 (Signal, Anreiz statt Kapazität): Mit TF 1 kennt der Decoder bei jedem Schritt das echte vorige Event und kann das
  nächste fast nur daraus vorhersagen; Information von weit vorn (die Option am Prefix-Ende) senkt den Verlust kaum → das Netz
  lernt sie zu ignorieren. Die Nachlegen- und Leads-Effekte entstehen erst 10+ Schritte später (Nachfasszeit nach zwei Angeboten,
  Wartezeit O_Sent → Validierung), Konzept/nach V. wirken sofort auf die nächsten Events. Stützend: Der Pilot mit Encoder-Zustand
  an jedem Decoderschritt (decoder_context, TF 1) half nicht – die Information war da, wurde aber nicht genutzt. Im freien Lauf
  ist die eigene Eingabe oft falsch, das Netz muss sich stärker auf seinen Zustand (inkl. Option) stützen.
- Erklärung 2 (Rauschen, Exposure Bias): Ein TF1-Netz hat nie auf eigenen Ausgaben weitergerechnet; beim Ziehen reagiert es
  empfindlich auf jede Abweichung, die Verläufe beider Optionen trennen sich trotz gleicher Zufallszahlen früh. Ein im freien Lauf
  trainiertes Netz hat stabilere Dynamik (Verläufe laufen zurück in typische Bahnen) → Kopplung bleibt.
- Kehrseite: Der freie Lauf erzeugt dabei die ungültigen Übergänge (siehe nächster Eintrag), und ein Teil des Fernsignals von (c)
  kommt über Artefakte (Leads: übertriebene Vervollständigungsschleifen).
- Gestützt durch den ZZ-Trainingsverlauf (Frage des Nutzers, ob das Signal erst spät im Training kommt): Lead-Gedächtnis Ep. 3 / 7 / 9
  (TF 0,73 / 0,20 / 0) +0,51 / +1,49 / +3,70; Nachfassen nach Extra-Angebot +0,24 / −1,18 / +2,37; AIPW-Vorzeichen Nachlegen 55 / 56 /
  87 %, Leads 40 / 77 / 77 % – das Fernsignal kommt mit dem freien Lauf, die ungültigen Übergänge (7 / 23 / 42 je 100) auch.
  Zusatzprüfung Ep. 11 und 13 (tmux psp_zz_ep11_13) zeigt, ob es bis Ep. 15 weiter wächst. Idee: längere Phase im freien Lauf für
  mehr Fernsignal, Gültigkeit getrennt sichern (Masken beim Ziehen oder Versatz im Training beheben).
- Prüfbar (offline, Minuten bis 1 h): Effekt über die Schritte verfolgen (Optionsunterschied nach k Schritten, k = 1…30) für (c)
  vs. (c)-TF1; Schritt bestimmen, ab dem sich die gekoppelten Verläufe beider Optionen trennen.

## 2026-11-17 00:10: Gültigkeit der gezogenen Abläufe (Anregung des Nutzers: Übergänge zwischen UND innerhalb von Events prüfen)

- Messung (`uebergaenge_pruefen.py`, Übergangsmodelle aus 80 % der echten Fälle, Prüfung an den gespeicherten Nachlegen-Verläufen):
  (c) verletzt in 99 % der Suffixe einen nie beobachteten Übergang zwischen Events (48 je 100 Events!) und in 94 % die Lifecycle-
  Folge innerhalb einer Aktivität (26 je 100). D und OE genauso, ZZ Ep. 9 genauso. Simulator und echte Prüffälle ≈ 0 (Fehlalarm-Rate
  0,003 je 100 Events → die gelernten Übergangsmodelle sind verlässlich). Typisch bei (c): W_Validate application start → schedule,
  schedule → schedule, W_Call after offers ate_abort → suspend (Pause nach Abbruch), A_Pending zweimal.
- Teacher Forcing 1 erzeugt gültige Abläufe (OE-TF1 Ep. 26: 0,6 je 100), der frühe ZZ-Stand mit TF 0,73 auch fast (7 je 100); ab
  freiem Lauf (Ep. 7/9) wieder ungültig. Vermutung (ungeprüft): im freien Training wird das argmax-Token zurückgefüttert, beim
  Ziehen aber gesampelt → gezogene Verläufe liegen außerhalb dessen, was das Netz im Training gesehen hat.
- **Mechanismus (im Code bestätigt, 00:30):** Im freien Teil des Trainings (Teacher Forcing je Schritt per Zufall, Anteil sinkt auf 0)
  bekommt der Decoder als Eingabe sein eigenes argmax-Token, das Ziel bleibt aber das echte Token an derselben Position
  (`dropout_uncertainty_model.py` forward, `trainer.py` Ziele = echte Suffixe). Liegt die Vorhersage einen Schritt vorn (z. B.
  „start“ statt „schedule“), lernt das Netz „nach start kommt schedule“ bzw. Wiederholungen. Die typischen Verstöße sind genau
  solche Wiederholungen und Umkehrungen (start → schedule, schedule → schedule, A_Pending/A_Complete zweimal, resume → resume), und
  ihre Zahl steigt mit sinkendem Teacher Forcing (ZZ: TF 0,73 → 7, TF 0,2 → 23, TF 0 → 42 je 100 Events; TF1-Modelle ≈ 0,6).
  → Die Ursache liegt im Trainingsziel des freien Laufs, nicht in den Daten; Masken beim Ziehen behandeln nur das Symptom.
- Ungültige Schritte nach Position bei (c) (Nachlegen-Verläufe): Schritt 2–3 12–13 %, Schritt 5 23 %, Schritt 10 56 %, ab Schritt 20
  62–67 % (Simulator überall 0). Warum (c) trotzdem oft richtig entscheidet: (1) die unmittelbaren Folgen der Option (erste Schritte)
  sind meist gültig – dort wirken Konzept und nach Vervollständigung, die Punkte mit 92–97 % Treffern; (2) gemeinsame Zufallszahlen →
  Artefakte treffen beide Optionen ähnlich und heben sich in der Differenz teilweise auf; (3) die Aufgabe ist grob: binär, große und
  fast einheitliche Effekte – die feste Regel trifft ohne Modell immer. Was die Ungültigkeit kostet: die Fernwirkungen (Nachlegen
  spätere Besuche 25 %, Leads teils über Artefakte), die Effektgrößen (Konzept 27 %) und online 1,1 Tage hinter der Regel.
- Das erklärt vermutlich auch die 47 % Ziehungen ohne Endzustand und die Zeitverzerrungen (Wartezeiten nach geschlossener Arbeit,
  doppelte Endzustände).
- Vorhandenes im Code: `transition_rules` (nur Übergänge zwischen Events + Fallenden; bisher nur bei OE online getestet, 1 Seed,
  schlechter), `work_state_constraints` (handgeschriebene start/resume-Regel; nur in den zustandsbewussten Kopien zusammen mit der
  empirischen Abstandstabelle, die das Signal zerstörte). Ein gelerntes Lifecycle-Modell innerhalb von Aktivitäten fehlt; für (c)
  wurde keine Beschränkung je geprüft.
- → Neuer Kandidat (Priorität 2, vor/mit der Endzustand-Reparatur): beschränktes Ziehen in PSP mit beiden gelernten Masken
  (zwischen Events inkl. Fallende nur nach beobachteten Endtokens; innerhalb Aktivität: letzter Lifecycle → erlaubter nächster),
  Netzwahrscheinlichkeiten auf die erlaubten Tokens renormiert, Zeit weiter vom Decoder je Schritt (Invarianten bleiben). Erst
  offline an (c) prüfen (Gültigkeit, Endzustände, Suffixgüte, Optionssignal/SE/Korrelation, Zerlegung, Simulator-Wahrheit), dann
  online gepaart.

## 2026-11-16 23:47: ZZ Prüfstand Epoche 9 (erster Stand im freien Lauf, TF 0)

- **Entscheidungsrichtung wie (c)**: AIPW-Vorzeichen 82 / 96 / 87 / 77 % ((c) 84 / 97 / 80 / 81), gegen die Simulator-Wahrheit
  richtig in 80 / 95 / 75 / 28 / 80 % ((c) 82 / 92 / 78 / 25 / 85) – Nachlegen 1. Besuch besser, Leads etwas schlechter.
- **Gedächtnis jetzt da**: Lead-Gedächtnis +3,70 Tage (echt +2,83 ± 1,11; (c) +0,15); nach Extra-Angebot zusätzlich +0,82 Angebote
  verschickt (Simulator +1,07), Nachfassen bis Validierung +2,37 (Simulator +1,92, Ep. 7 noch −1,18) → **Vorzeichen gedreht,
  die Wartezeit-Verknüpfung nach zwei Angeboten stimmt jetzt**. Eingefügtes zweites Angebot weiterhin −0,51 (Richtung falsch).
- **Aber Zeitskala aufgebläht**: gezogene Durchlaufzeit 26–34 Tage (echt 22–24; Ep. 3 noch 15–18 zu kurz), SD ~40 Tage (echt ~15)
  → Effekte 1,3–3,5-fach zu groß (Konzept 3,61 vs. 2,79; nach V. 6,71 vs. 2,40; Nachlegen 3,76 vs. 2,11; Leads 3,52 vs. 1,32) und
  gepaarter SE 3,9–4,3 Tage ((c) 1,3). Signal/Rauschen daher ähnlich wie (c) (z ≈ 1), nicht besser. 45–50 % Ziehungen ohne Endzustand.
- Verluste Ep. 9: Standard 8,99 ((c) Ep. 9: 8,85), attenuiert 277 ((c) 22). Suffixgüte Ep. 3 / 7: DLS 0,34 / 0,31 ((c) 0,25),
  Schritt 10 17 / 13 % ((c) 9 %), Restzeit-MAE 8,23 / 7,87 ((c) 8,13), aber Bias −3,7 / −2,0 und CRPS 6,1 / 6,3 ((c) 5,4).
- Deutung: Die Zähler liefern das fehlende Langzeitgedächtnis (Leads, Nachlegen); die Zeitköpfe sind instabil (Skala springt
  zwischen Epochen, zu breite Verteilungen) – wahrscheinlich durch „Zeit je Event“. → Ablation „nur Zähler“ (ohne Zeit je Event)
  ist der nächste Trainingskandidat; Endstand (Ep. 15) abwarten.

## 2026-11-16: (c) + Zähler + Zeit je Event (ZZ), Prüfstände Epoche 3 und 7

- Training `real_alle_s60_normal_zaehler_zeit` läuft (15 Epochen à ~67 min, Ende ~06:10 am 17.11.); Prüfreihe nach Ep. 3,
  Ep. 7 (von Hand), Ep. 9 und final.
- **Ep. 7: Gedächtnis besser als bei allen bisherigen Modellen**: Lead-Gedächtnis +1,49 Tage (echt +2,83 ± 1,11; (c) +0,15),
  beim Nachlegen werden zusätzlich +0,68 Angebote verschickt ((c) +0,34, Simulator +1,07). Leads gegen Simulator 1,22 (Wahrheit 1,32).
- **Aber die Wartezeit-Verknüpfung ist falsch herum**: eingefügtes zweites Angebot −0,55 Tage (Ep. 3 −1,12; (c) +0,51; OE +1,18);
  Nachfassen bis Validierung nach Extra-Angebot −1,18 (Simulator +1,92). Vermutung (ungeprüft): Zähler werden als
  „Fall weit fortgeschritten“ gelesen.
- **Rauschen verdoppelt**: SD der gezogenen Durchlaufzeit ~26 Tage (echt ~15, (c) 15,4, Ep. 3 14,1) → MC-SE je Entscheidung
  2,3–3,3 Tage ((c) 1,3); Konzept-Signal verschwunden (0,05 statt Wahrheit 2,79), AIPW-Vorzeichen 55/70/56/76 %.
- Attenuierter Validierungsverlust bleibt hoch (Ep. 7: 192, Ep. 8: 243; (c) bei Ep. 7: 27), Standardverlust wie (c) (9,3 / 9,1).
  Erste Vermutung „Varianzen zu selbstsicher“ ist durch die zu breiten Ziehungen widerlegt; Ursache offen.
- Offen: ob die Streuung im freien Lauf (ab Ep. 8) wieder sinkt; ob das Vorzeichen beim eingefügten Angebot dreht.

## 2026-11-16: OE mit Teacher Forcing 1 (OE-TF1, 100-Epochen-Lauf, bei Ep. ~27 gestoppt)

- TF1 verbessert die Abläufe (DLS 0,40 statt 0,27; zweites Angebot verschickt 69 % statt 20 %), nicht die Entscheidungen:
  Konzept 1,17 statt 1,54, Nachlegen 1. Besuch 0,19 statt 0,48, Leads −0,44 statt −0,14; eingefügtes Angebot −0,17 statt +1,18.
- Ep. 1 (bester Val-Verlust) ist überall deutlich schlechter → **Val-Verlust unter TF1 taugt nicht zur Modellauswahl**.
- Gleiches Muster wie (c)-TF1 und D-TF1: beste Suffixgüte, schwächeres Optionssignal (Korrelation der Optionen 0,12–0,15
  statt 0,6 → gemeinsame Zufallszahlen wirken kaum).

## 2026-11-16: Modellvergleich und Diagnose (`modellvergleich_20261116/`)

- Simulator-Wahrheit (RCT): Konzept annehmen +2,79, nach Vervollständigung +2,40, Nachlegen senden +2,11 (1. Besuch; mit Bank
  danach +1,55) bzw. +0,42 (später), Leads zurückziehen +1,32 Tage.
- Kein Modell erinnert sich an den Lead beim ersten O_Sent (außer ZZ teilweise, s. o.); D/(c) haben das richtige Vorzeichen bei
  Leads nur über übertriebene Vervollständigungsschleifen. OE bildet die frühe Phase realistisch ab und verpasst die späte Wartezeit.
- Nachlegen: Modelle verschicken nach einem zusätzlichen Angebot selten beide Angebote (OE 20 %, (c) 25 %, D 31 %, echt 80 %);
  OE kennt die längere Nachfasszeit bei zwei Angeboten, sie wird nur nicht ausgelöst.
- Stateful D (empirische Abstandstabelle statt Decoder-Zeiten) verliert das Leads-Signal; gemeinsame Zufallszahlen sind dort kaputt.
- 3 × 200-Online-Läufe können Modelle mit < ~1,5 Tagen Unterschied nicht ordnen.

## Ältere Erkenntnisse (vor dem 16.11.)

- Log-Abstände ((a), (b)) explodieren beim Ziehen (gezogene Durchlaufzeit ~40 Tage, SD ~72) → kein Signal.
- Kurzer Decoder (10k-Modelle, Decoder 5) → Restverläufe zu kurz, kein Signal.
- s40 hat gutes Konzept-Signal (90 %), aber Restzeit-Bias +6,4 Tage.
- Zeitklassen ((d), D, OE, (e)): bessere Suffixe ab Schritt 5, aber höheres Rauschen und schwächere Optionskorrelation als (c).
- Durchlaufzeit an den Wahlpunkten ist kaum vorhersagbar (alle Modelle ≈ naive MAE 10–11 Tage; Abbruch AUC ~0,55).

## Offene Hebel (noch nicht umgesetzt)

- ~~PSP-Optionsdifferenz je Punkt poolen~~ – vom Nutzer am 2026-11-17 gestrichen, nicht weiterverfolgen.
- `first_only` beim Nachlegen (spätere Besuche folgen der Bank).
- Online-Vergleich mit X-Learner im korrigierten Simulator ergänzen.

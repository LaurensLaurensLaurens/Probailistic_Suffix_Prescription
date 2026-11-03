# PMSP Processing Time Model Analysis

## Problemstellung

PMSP benötigt **Kosten pro (Task, Resource)** für die Optimierung. Die aktuellen Processing-Time-Modelle sagen jedoch **inter-event time** voraus (Zeit zwischen zwei Log-Events, inkl. Wartezeit), nicht die **execution time** (wie lange die Ressource tatsächlich arbeitet).

## Verfügbare Modelle

### 1. **"ml"** (Random Forest) - **AKTUELL VERWENDET**

**Vorhersage**: Inter-event time (Zeit zwischen zwei Events)

**Vorteile:**
- ✅ Schnell (~91ms pro Vorhersage)
- ✅ Deterministisch (gut für Optimierung)
- ✅ Kann Resource-spezifische Features nutzen
- ✅ Robust gegenüber Outliers

**Nachteile:**
- ❌ Sagt **inter-event time** voraus, nicht execution time
- ❌ Problem mit unbekannten Aktivitäten (z.B. `A_Create Application` → Fallback auf falsche Defaults)
- ❌ Enthält Wartezeiten, die nicht zur Ressourcenkosten gehören sollten

**Beispiel-Problem:**
```
START → A_Create Application: 129.686s (36h) 
→ Modell kennt A_Create Application nicht
→ Fallback auf W_Validate application
→ Vorhersage enthält Wartezeit zwischen Events
```

---

### 2. **"distribution"** (Log-Normal per Transition)

**Vorhersage**: Inter-event time (Zeit zwischen zwei Events)

**Vorteile:**
- ✅ Sehr schnell
- ✅ Deterministisch (nach Sampling)
- ✅ Einfach und interpretierbar

**Nachteile:**
- ❌ Sagt **inter-event time** voraus, nicht execution time
- ❌ Keine Resource-spezifischen Features
- ❌ Braucht viele Beobachtungen pro Transition-Paar
- ❌ Problem mit seltenen/neuen Transitionen

---

### 3. **"probabilistic_ml"** (LSTM)

**Vorhersage**: Inter-event time (Zeit zwischen zwei Events)

**Vorteile:**
- ✅ Kann Sequenzen lernen
- ✅ Berücksichtigt Event-Historie

**Nachteile:**
- ❌ **Langsam** (LSTM-Inferenz)
- ❌ **Stochastisch** (nicht ideal für deterministische Optimierung)
- ❌ Sagt **inter-event time** voraus, nicht execution time
- ❌ Braucht TensorFlow (schwerer Dependency)

**Nicht empfohlen für PMSP** ❌

---

### 4. **"xgboost"** (Activity-Specific XGBoost)

**Vorhersage**: Inter-event time in log10(hours) → konvertiert zu seconds

**Vorteile:**
- ✅ Activity-spezifische Modelle (bessere Genauigkeit)
- ✅ Kann Quantile-Regression

**Nachteile:**
- ❌ Sagt **inter-event time** voraus, nicht execution time
- ❌ Komplexere Architektur
- ❌ Möglicherweise langsamer als RF

---

## 🔴 KRITISCHES PROBLEM: Falsche Einheit

**Alle Modelle sagen inter-event time voraus**, aber PMSP braucht **execution time** (resource hold time).

### Inter-Event Time vs. Execution Time

```
Inter-event time = Wartezeit + Execution time + System-Delay
                  ↑              ↑
                  |              └─ Das braucht PMSP!
                  └─ Sollte nicht in PMSP-Kosten
```

**Beispiel:**
- Inter-event time: 36h (inkl. Wartezeit über Nacht)
- Execution time: 2h (tatsächliche Arbeit)
- PMSP sollte 2h verwenden, nicht 36h!

---

## ✅ EMPFEHLUNG: `predict_resource_hold_time()`

Es gibt bereits eine Methode, die **execution time** vorhersagt:

```python
predictor.predict_resource_hold_time(activity, resource)
```

**Funktionsweise:**
- Für `W_` Aktivitäten: Lognormal-Verteilung basierend auf **work bursts** (start→suspend, resume→complete)
- Für `A_`/`O_` Aktivitäten: 60s fix (System-State-Changes)

**Vorteile:**
- ✅ Sagt **execution time** voraus (korrekte Einheit für PMSP)
- ✅ Schnell (einfache Distribution-Sampling)
- ✅ Deterministisch (nach Sampling)
- ✅ Resource-spezifisch
- ✅ Korrekte Behandlung von A_/O_ Aktivitäten

**Nachteile:**
- ⚠️ Braucht `resource_hold` Modell (wird automatisch geladen)
- ⚠️ Keine Context-Features (hour, weekday, etc.)

---

## 🎯 FINALE EMPFEHLUNG

### Option 1: **`predict_resource_hold_time()` verwenden** ⭐ **BESTE WAHL**

**Warum:**
1. Sagt die **korrekte Einheit** voraus (execution time)
2. Schnell genug für PMSP
3. Resource-spezifisch
4. Behandelt A_/O_ Aktivitäten korrekt (60s)

**Implementierung:**
```python
# In resource_optimization.py, predict_processing_seconds():
if hasattr(opt_predictor, 'predict_resource_hold_time'):
    seconds = opt_predictor.predict_resource_hold_time(
        curr_activity=allocation_activity,
        resource=candidate_resource
    )
else:
    # Fallback auf altes Verhalten
    seconds = opt_predictor.predict(...)
```

### Option 2: **"ml" mit Fixes** (wenn Context-Features wichtig sind)

**Fixes nötig:**
1. Unbekannte Aktivitäten explizit behandeln (z.B. A_/O_ → 60s)
2. Modell auf **execution time** statt inter-event time trainieren
3. Oder: `predict()` Ergebnis mit `predict_resource_hold_time()` kombinieren

**Nachteile:**
- Komplexer
- Braucht Modell-Retraining
- Immer noch Problem mit unbekannten Aktivitäten

### Option 3: **"distribution" mit Fixes** (wenn sehr schnell sein muss)

**Fixes nötig:**
- Gleiche wie bei "ml"
- Zusätzlich: Resource-spezifische Distributions

---

## Performance-Vergleich

| Methode | Vorhersage-Zeit | Einheit | Resource-spezifisch | Context-Features |
|---------|----------------|---------|---------------------|------------------|
| `predict_resource_hold_time()` | ~0.1ms | ✅ Execution time | ✅ Ja | ❌ Nein |
| "ml" | ~91ms | ❌ Inter-event | ✅ Ja | ✅ Ja |
| "distribution" | ~0.5ms | ❌ Inter-event | ❌ Nein | ❌ Nein |
| "probabilistic_ml" | ~200ms+ | ❌ Inter-event | ✅ Ja | ✅ Ja |
| "xgboost" | ~50ms | ❌ Inter-event | ✅ Ja | ✅ Ja |

---

## Zusammenfassung

**Für PMSP sollte `predict_resource_hold_time()` verwendet werden**, weil:

1. ✅ **Korrekte Einheit**: Execution time, nicht inter-event time
2. ✅ **Schnell genug**: ~0.1ms vs. ~91ms für "ml"
3. ✅ **Resource-spezifisch**: Berücksichtigt (activity, resource) Kombinationen
4. ✅ **Robust**: Korrekte Behandlung von A_/O_ Aktivitäten
5. ✅ **Bereits implementiert**: Keine neuen Modelle nötig

**Nachteil:** Keine Context-Features (hour, weekday, etc.), aber für PMSP-Kosten ist das akzeptabel, da die Optimierung hauptsächlich relative Kosten zwischen Ressourcen vergleicht.

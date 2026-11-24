"""
Künstliche Effekte für den BPIC17-Simulator (2026-11-20, Wunsch des Nutzers: „originaltreu nachbauen mit allen Aktivitäten und Lifecycles,
wir ändern nur Kleinigkeiten wie Effekte gewisser Punkte und Aktionen künstlich“; Idee aus SCOPE / SimBPIC17: bekannte Effektfunktion
je Fall statt nur gelernter Dynamik).

Grundlage bleibt der gelernte Simulator (simulation/case_engine_lifecycle_aware.CaseEngine) unverändert: alle Aktivitäten und Lifecycles,
Mitarbeiter, Zuteilung, Ankünfte, Fallattribute. Dazu kommt eine Effektschicht (EffectEngine, Unterklasse):

- Wird an einem Control-Flow-Wahlpunkt eine Option ausgeführt (gleich ob die Policy sie gewählt hat oder die Bank), rechnet delay_days
  aus der Effekt-Konfiguration die künstliche Verzögerung in Tagen für diesen Fall (Regeln je Wahlpunkt und Option, Bedingungen auf
  Fallattributen und der bisherigen Falldauer, siehe effekte_v1.yaml).
- Eine Verzögerung (tage > 0) kommt ganz auf den nächsten Wartespalt danach: die Zeit vor dem nächsten Beginn oder Fortsetzen eines
  Arbeitsschritts (W_* start / resume; die Engine behandelt genau diese Zeit als Warten auf Kunde oder Warteschlange). Alle späteren
  Events des Falls verschieben sich mit.
- Eine Beschleunigung (tage < 0) wird von den folgenden Wartespalten abgezogen (jeder höchstens bis auf MIN_SPALT Sekunden), bis sie
  verbraucht ist. Damit lassen sich Effekte bauen, die sich unter dem Bankverhalten im Mittel aufheben (Log bleibt nah am Original).
- Endet der Fall vorher, entfällt der Rest (effect_log: angewandt_tage).
- Ablauf-Effekt (Regel mit ablauf: {Aktivität|lifecycle: Zuschlag}, 2026-11-20 Nutzer: „Auftreten anderer Aktivitäten“): ab der Option
  bekommt der Fall bei jedem folgenden Ziehen des nächsten Events einen Zuschlag auf die Logits dieser Event-Typen (+1: Chance × e,
  −1: Chance / e), für den Rest des Falls oder höchstens `schritte` Schritte. Nur Event-Typen, die das Modell an der Stelle ohnehin für
  möglich hält (Wahrscheinlichkeit ≥ MIN_P, wie min_probability des Simulators), bekommen ihn – keine neuen Übergänge. Nicht direkt nach
  dem Entscheidungs-Event eines Wahlpunkts (die Wahl der Bank an anderen Punkten bleibt). Die Zeit bis zum Event zieht das Modell wie
  immer passend zum gezogenen Event (z. B. die lange Frist vor A_Cancelled).
- Ablauf-Anteil (Regel mit anteil: {Aktivität|lifecycle: α}, 2026-11-20): wo das Modell fast sicher ist (z. B. Verzweigung nach
  W_Validate application: je Fall ≈ 99 % oder ≈ 0 % Nachfrageschleife), wirkt ein Logit-Zuschlag nicht. Dann: an der ersten Stelle nach der
  Option, an der das Event als Nachfolger erlaubt ist (im Training beobachteter Übergang), wird es mit Wahrscheinlichkeit α gezogen (α > 0)
  bzw. mit Wahrscheinlichkeit |α| verboten (α < 0, nur wenn ein anderes Event möglich ist). Mit einmal: false bei jeder solchen Stelle.
  Zufallszahlen aus einem eigenen Strom je Fall (Saat aus Fall-ID und Engine-Saat), der übrige Ablauf bleibt ungestört.
- Protokoll-Abweichung (Regel mit abweichung: {art, aktivitaet, lifecycle?, anteil}, 2026-11-20 Nutzer: das PSP soll Conformance-
  Abweichungen verhindern können, neue Arten wie Mustroph gern): mit Wahrscheinlichkeit anteil (Münze beim Options-Event, eigener
  Zufallsstrom) wird beim nächsten Auftreten der Aktivität im Log
    auslassen   – das Event weggelassen (Arbeitsschritt W_*: alle seine Events bis zum Abschluss complete / ate_abort / withdraw;
                  z. B. W_Validate application = Prüfung übersprungen),
    doppeln     – das Event (bzw. aktivitaet|lifecycle, Standard complete, bei W_* start) ein zweites Mal geschrieben (+1 ms),
    vertauschen – das Event erst nach dem folgenden Event des Falls geschrieben (+1 ms; endet der Fall vorher, am Ende).
  Nur das Protokoll (engine.events und case.prefix, also auch, was eine Policy sieht) ändert sich; der Simulator läuft intern weiter
  wie ohne Abweichung (Zustand des LSTM, Mitarbeiter, Zeiten). Die Abweichung wirkt also auf die Conformance, nicht auf die Dauer.
- effect_log: je ausgeführter Option Fall, Punkt, Option, Tage, Ablauf-Zuschläge, Abweichung (gezogen / angewandt) – damit ist der
  eingebaute Effekt je Fall bekannt.

Einschalten nur im eigenen Prozess (aktivieren): ersetzt dort simulation.case_engine_lifecycle_aware.build_engine durch eine Hülle, die
jede neu gebaute Engine zur EffectEngine macht. Keine Datei des Simulators wird geändert; andere Prozesse (z. B. laufende Online-Tests)
sehen davon nichts. Die letzte gebaute Engine steht in LETZTE_ENGINE (für effect_log nach Simulator.run).
"""
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional
import zlib

import numpy as np
import yaml

from simulation import case_engine_lifecycle_aware as engine_module
from simulation.case_engine_lifecycle_aware import EVENT, OPENING, CaseEngine, to_seconds

DAY = 86400.0
HERE = Path(__file__).resolve().parent
STANDARD = HERE/"effekte_v1.yaml"
BEDINGUNGEN = ("betrag_ab", "betrag_unter", "ziel_in", "typ_in", "dauer_ab_tage", "dauer_unter_tage", "nur_erster_besuch")
MIN_SPALT = 60.0  # Sekunden, kürzester Wartespalt nach einer Beschleunigung
ARTEN = ("auslassen", "doppeln", "vertauschen")
SCHLIESSEND = ("complete", "ate_abort", "withdraw")
MIN_P = 1e-3  # Ablauf-Zuschlag nur für Event-Typen mit mindestens dieser Wahrscheinlichkeit (= min_probability des Simulators)
LETZTE_ENGINE: Optional["EffectEngine"] = None


def laden(pfad=None) -> Dict:
    """Effekt-Konfiguration (yaml) laden und prüfen."""
    pfad = Path(pfad) if pfad else STANDARD
    effekte = yaml.safe_load(pfad.read_text()) or {}
    for punkt, optionen in (effekte.get("punkte") or {}).items():
        for option, regeln in (optionen or {}).items():
            if "|" not in option:
                raise ValueError(f"Effekte {punkt}: Option {option!r} ohne Lifecycle (Aktivität|lifecycle)")
            for regel in regeln or []:
                unbekannt = set(regel) - set(BEDINGUNGEN) - {"tage", "ablauf", "anteil", "einmal", "schritte", "abweichung"}
                if unbekannt or not ({"tage", "ablauf", "anteil", "abweichung"} & set(regel)):
                    raise ValueError(f"Effekte {punkt} / {option}: Regel {regel} (erlaubt {BEDINGUNGEN}, tage, ablauf, anteil, einmal, schritte, abweichung)")
                ab = regel.get("abweichung")
                if ab is not None and (ab.get("art") not in ARTEN or "aktivitaet" not in ab or not 0 <= float(ab.get("anteil", 1.0)) <= 1):
                    raise ValueError(f"Effekte {punkt} / {option}: abweichung {ab} (art aus {ARTEN}, aktivitaet, anteil 0–1)")
                if any(not -1.0 <= float(a) <= 1.0 for a in (regel.get("anteil") or {}).values()):
                    raise ValueError(f"Effekte {punkt} / {option}: anteil zwischen −1 und 1, {regel['anteil']}")
                if any("|" not in token for token in list(regel.get("ablauf") or {}) + list(regel.get("anteil") or {})):
                    raise ValueError(f"Effekte {punkt} / {option}: ablauf braucht Aktivität|lifecycle, {regel['ablauf']}")
    effekte["pfad"] = str(pfad)
    return effekte


def trifft_zu(regel: Dict, attribute: Dict, dauer_tage: float, besuch: int = 1) -> bool:
    """Gilt die Regel für einen Fall mit diesen Attributen, dieser bisherigen Dauer (Tage) und diesem Besuch des Punkts (1 = erster)?"""
    betrag = float(attribute.get("case:RequestedAmount") or 0.0)
    if "betrag_ab" in regel and not betrag >= float(regel["betrag_ab"]):
        return False
    if "betrag_unter" in regel and not betrag < float(regel["betrag_unter"]):
        return False
    if "ziel_in" in regel and attribute.get("case:LoanGoal") not in regel["ziel_in"]:
        return False
    if "typ_in" in regel and attribute.get("case:ApplicationType") not in regel["typ_in"]:
        return False
    if "dauer_ab_tage" in regel and not dauer_tage >= float(regel["dauer_ab_tage"]):
        return False
    if "dauer_unter_tage" in regel and not dauer_tage < float(regel["dauer_unter_tage"]):
        return False
    if regel.get("nur_erster_besuch") and besuch > 1:
        return False
    return True


def delay_days(effekte: Dict, punkt: str, option: str, attribute: Dict, dauer_tage: float, besuch: int = 1) -> float:
    """Künstlicher Effekt (Tage, > 0 Verzögerung, < 0 Beschleunigung), wenn an punkt die Option option (Aktivität|lifecycle) ausgeführt
    wird: Summe der zutreffenden Regeln."""
    regeln = ((effekte.get("punkte") or {}).get(punkt) or {}).get(option) or []
    return float(sum(float(regel.get("tage", 0.0)) for regel in regeln if trifft_zu(regel, attribute, dauer_tage, besuch)))


def ablauf_regeln(effekte: Dict, punkt: str, option: str, attribute: Dict, dauer_tage: float, besuch: int = 1) -> List[Dict]:
    """Die zutreffenden Regeln mit Ablauf-Effekt (ablauf, schritte) für diese Option an diesem Punkt."""
    regeln = ((effekte.get("punkte") or {}).get(punkt) or {}).get(option) or []
    return [regel for regel in regeln if (regel.get("ablauf") or regel.get("anteil")) and trifft_zu(regel, attribute, dauer_tage, besuch)]


def abweichung_regeln(effekte: Dict, punkt: str, option: str, attribute: Dict, dauer_tage: float, besuch: int = 1) -> List[Dict]:
    """Die zutreffenden Regeln mit Protokoll-Abweichung für diese Option an diesem Punkt."""
    regeln = ((effekte.get("punkte") or {}).get(punkt) or {}).get(option) or []
    return [regel for regel in regeln if regel.get("abweichung") and trifft_zu(regel, attribute, dauer_tage, besuch)]


class _ModellMitZuschlag:
    """Hülle um das Next-Event-Modell der Engine: step liefert die Logits plus die Ablauf-Zuschläge des Falls, den die Engine gerade
    weiterzieht (engine._advancing); alles andere geht unverändert an das Modell."""

    def __init__(self, inner, engine):
        self._inner, self._engine = inner, engine

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def step(self, state, tokens, seconds, *args, **kwargs):
        logits = self._inner.step(state, tokens, seconds, *args, **kwargs)
        case = self._engine._advancing
        aktiv = getattr(case, "ablauf_effekte", None) if case is not None else None
        if not aktiv or int(np.asarray(tokens).reshape(-1)[0]) in self._engine._control_flow:
            return logits
        zuschlag = np.zeros(logits.shape[-1])
        for effekt in aktiv:
            zuschlag += effekt["zuschlag"]
            if effekt["rest"] is not None:
                effekt["rest"] -= 1
        wahrsch = np.exp(logits - logits.max(axis=-1, keepdims=True))
        wahrsch /= wahrsch.sum(axis=-1, keepdims=True)
        logits = logits + np.where(wahrsch >= MIN_P, zuschlag, 0.0)
        # Anteile: an der ersten erlaubten Stelle (bzw. jeder, einmal: false) das Event erzwingen oder verbieten
        for effekt in aktiv:
            for j, alpha in list(effekt["anteil"].items()):
                if not np.isfinite(logits[0, j]):
                    continue  # hier kein erlaubter Nachfolger
                andere = (wahrsch[0] >= MIN_P) & (np.arange(logits.shape[-1]) != j)
                if alpha < 0 and not andere.any():
                    continue  # nichts anderes möglich, nicht verbieten
                if effekt["einmal"]:
                    del effekt["anteil"][j]
                if case.effekt_rng.random() < abs(alpha):
                    if alpha > 0:
                        erzwungen = np.full_like(logits, -np.inf)
                        erzwungen[0, j] = 0.0
                        logits = erzwungen
                    else:
                        logits = logits.copy()
                        logits[0, j] = -np.inf
                    effekt["umgelenkt"] = effekt.get("umgelenkt", 0) + 1
                    effekt["satz"]["umgelenkt"] = effekt["satz"].get("umgelenkt", 0) + 1
        case.ablauf_effekte = [effekt for effekt in aktiv if (effekt["rest"] is None or effekt["rest"] > 0)
                               and (effekt["zuschlag"].any() or effekt["anteil"])]
        return logits


class EffectEngine(CaseEngine):
    """CaseEngine mit künstlichen Effekten (siehe Modul). Wird über aktivieren / installieren erzeugt, nicht direkt."""

    effekte: Dict = {}

    def _reset(self) -> None:
        super()._reset()
        self.effect_log: List[Dict] = []
        self._advancing = None

    def _happen(self, case, token: int, time: float) -> None:
        # Ausgeführte Option an einem Control-Flow-Wahlpunkt: das letzte Event des Falls ist das Entscheidungs-Event, token eine Option
        eintrag = self._control_flow.get(case.last_token)
        if eintrag is not None and token in eintrag[1]:
            punkt = eintrag[0]
            option = "|".join(self._tokens[token])
            beginn = to_seconds(case.prefix[0]["time:timestamp"]) if case.prefix else time
            dauer = (time - beginn) / DAY
            besuche = getattr(case, "effekt_besuche", None)
            if besuche is None:
                besuche = case.effekt_besuche = {}
            besuch = besuche[punkt.name] = besuche.get(punkt.name, 0) + 1
            tage = delay_days(self.effekte, punkt.name, option, case.attributes, dauer, besuch)
            ablauf = ablauf_regeln(self.effekte, punkt.name, option, case.attributes, dauer, besuch)
            abweichungen = abweichung_regeln(self.effekte, punkt.name, option, case.attributes, dauer, besuch)
            if (ablauf or abweichungen) and getattr(case, "effekt_rng", None) is None:
                case.effekt_rng = np.random.default_rng([zlib.crc32(case.case_id.encode()), int(self.seed or 0)])
            satz = {"case": case.case_id, "point": punkt.name, "option": option, "besuch": besuch, "tage": tage, "dauer_tage": dauer,
                    "time": time, "angewandt_tage": 0.0, "rest_tage": tage, "umgelenkt": 0,
                    "ablauf": "; ".join([f"{k} {v:+g}" for regel in ablauf for k, v in (regel.get("ablauf") or {}).items()]
                                        + [f"{k} Anteil {v:+g}" for regel in ablauf for k, v in (regel.get("anteil") or {}).items()])}
            satz["abweichung"], satz["abweichung_angewandt"] = "", 0
            for regel in abweichungen:
                ab = regel["abweichung"]
                gezogen = case.effekt_rng.random() < float(ab.get("anteil", 1.0))
                satz["abweichung"] += f"{ab['art']} {ab['aktivitaet']} {'gezogen' if gezogen else 'nicht gezogen'}; "
                if gezogen:
                    offen = getattr(case, "abweichungen_offen", None)
                    if offen is None:
                        offen = case.abweichungen_offen = []
                    lc = ab.get("lifecycle") or ("start" if ab["aktivitaet"].startswith("W_") and ab["art"] == "doppeln" else None)
                    offen.append({"art": ab["art"], "aktivitaet": ab["aktivitaet"], "lifecycle": lc, "satz": satz, "laeuft": False})
            self.effect_log.append(satz)
            if ablauf:
                ids = self.model.encoder.token_to_id
                aktiv = getattr(case, "ablauf_effekte", None)
                if aktiv is None:
                    aktiv = case.ablauf_effekte = []
                for regel in ablauf:
                    zuschlag = np.zeros(len(self._tokens))
                    for name, wert in (regel.get("ablauf") or {}).items():
                        zuschlag[ids[name]] += float(wert)
                    anteil = {ids[name]: float(wert) for name, wert in (regel.get("anteil") or {}).items()}
                    aktiv.append({"zuschlag": zuschlag, "anteil": anteil, "einmal": bool(regel.get("einmal", True)), "satz": satz,
                                  "rest": int(regel["schritte"]) if regel.get("schritte") else None})
            if tage != 0:
                offen = getattr(case, "effekte_offen", None)
                if offen is None:
                    offen = case.effekte_offen = []
                offen.append(satz)
        super()._happen(case, token, time)

    # ---- Protokoll-Abweichungen: _happen hat das Event gerade geschrieben, _advance zieht danach das nächste (und fragt ggf. die Policy)
    def _event_index(self, event) -> int:
        for i in range(len(self.events) - 1, -1, -1):
            if self.events[i] is event:
                return i
        raise RuntimeError("Event nicht im Protokoll")

    def _schreiben_nach(self, case, nach, neu) -> None:
        neu["time:timestamp"] = nach["time:timestamp"] + timedelta(milliseconds=1)
        self.events.insert(self._event_index(nach) + 1, neu)
        case.prefix.append(neu)

    def _protokoll(self, case) -> None:
        if len(case.prefix) == getattr(case, "effekt_n", 0):
            return  # kein neues Event (z. B. Ankunft)
        event = case.prefix[-1]
        alte = getattr(case, "gehalten", None) or []
        case.gehalten = []
        offen = getattr(case, "abweichungen_offen", None)
        if offen:
            act, lc = event["concept:name"], event["lifecycle:transition"]
            for ab in offen:
                if ab["aktivitaet"] != act:
                    continue
                if ab["art"] == "auslassen":
                    del self.events[self._event_index(event)]
                    case.prefix.pop()
                    ab["satz"]["abweichung_angewandt"] += 1
                    ab["fertig"] = not (act.startswith("W_") and lc not in SCHLIESSEND)  # Arbeitsschritt: bis zum Abschluss weglassen
                elif ab["lifecycle"] is not None and ab["lifecycle"] != lc:
                    continue
                elif ab["art"] == "doppeln":
                    self._schreiben_nach(case, event, dict(event))
                    ab["satz"]["abweichung_angewandt"] += 1
                    ab["fertig"] = True
                else:  # vertauschen: zurückhalten, nach dem nächsten Event des Falls schreiben
                    del self.events[self._event_index(event)]
                    case.prefix.pop()
                    case.gehalten.append(event)
                    ab["satz"]["abweichung_angewandt"] += 1
                    ab["fertig"] = True
                break
            case.abweichungen_offen = [ab for ab in offen if not ab.get("fertig")]
        if alte and case.prefix:  # früher zurückgehaltene Events nach dem jetzt letzten Event schreiben
            vorher = case.prefix[-1]
            for alt in alte:
                self._schreiben_nach(case, vorher, alt)
                vorher = alt
        elif alte:
            case.gehalten = alte + case.gehalten
        case.effekt_n = len(case.prefix)

    def _finish(self, case, time: float) -> None:
        gehalten = getattr(case, "gehalten", None)
        if gehalten and case.prefix:
            vorher = case.prefix[-1]
            for alt in gehalten:
                self._schreiben_nach(case, vorher, alt)
                vorher = alt
            case.gehalten = []
        super()._finish(case, time)

    def _advance(self, case, now: float) -> None:
        self._protokoll(case)
        self._advancing = case
        try:
            super()._advance(case, now)
        finally:
            self._advancing = None

    def _schedule(self, time: float, kind: int, case_id) -> None:
        # Nur der Termin des nächsten Events, den _advance gerade setzt (case.pending), und nur vor einem Wartespalt
        case = self._advancing
        offen = getattr(case, "effekte_offen", None) if case is not None else None
        if kind == EVENT and offen and case.case_id == case_id and case.pending is not None:
            token = case.pending[0]
            activity, lifecycle = self._tokens[token]
            if self._is_work_item(activity) and lifecycle in OPENING:
                for satz in offen:
                    if satz["rest_tage"] > 0:  # Verzögerung: ganz auf diesen Spalt
                        time += satz["rest_tage"] * DAY
                        satz["angewandt_tage"] += satz["rest_tage"]
                        satz["rest_tage"] = 0.0
                    else:  # Beschleunigung: höchstens bis auf MIN_SPALT, Rest auf die folgenden Wartespalten
                        kuerzung = min(-satz["rest_tage"] * DAY, max(0.0, time - case.last_time - MIN_SPALT))
                        time -= kuerzung
                        satz["angewandt_tage"] -= kuerzung / DAY
                        satz["rest_tage"] += kuerzung / DAY
                        if satz["rest_tage"] > -1e-9:
                            satz["rest_tage"] = 0.0
                case.effekte_offen = [satz for satz in offen if satz["rest_tage"] != 0.0]
                case.pending = (token, time)
        super()._schedule(time, kind, case_id)


def installieren(engine: CaseEngine, effekte: Dict) -> EffectEngine:
    """Eine gebaute CaseEngine zur EffectEngine machen (gleiche Daten, zusätzliche Effektschicht)."""
    if not isinstance(engine, CaseEngine) or type(engine) not in (CaseEngine, EffectEngine):
        raise TypeError(f"Effekte nur für CaseEngine (lifecycle-aware), nicht {type(engine).__name__}")
    ids = engine.model.encoder.token_to_id
    unbekannt = sorted({name for optionen in (effekte.get("punkte") or {}).values() for regeln in (optionen or {}).values()
                        for regel in regeln or [] for name in list(regel.get("ablauf") or {}) + list(regel.get("anteil") or {})
                        if name not in ids})
    aktivitaeten = {tok[0] for tok in engine._tokens}  # Sondertokens haben keinen Lifecycle
    unbekannt += sorted({regel["abweichung"]["aktivitaet"] for optionen in (effekte.get("punkte") or {}).values()
                         for regeln in (optionen or {}).values() for regel in regeln or []
                         if regel.get("abweichung") and regel["abweichung"]["aktivitaet"] not in aktivitaeten})
    if unbekannt:
        raise ValueError(f"Effekte: Event-Typen kennt das Modell nicht: {unbekannt}")
    engine.__class__ = EffectEngine
    engine.effekte = effekte
    engine.effect_log, engine._advancing = [], None
    if not isinstance(engine.model, _ModellMitZuschlag):
        engine.model = _ModellMitZuschlag(engine.model, engine)
    return engine


def aktivieren(effekte: Optional[Dict] = None) -> Dict:
    """
    Im eigenen Prozess: jede danach gebaute Engine (Simulator.run importiert build_engine bei jedem Lauf) bekommt die Effekte.
    effekte: Ergebnis von laden (Standard: effekte_v1.yaml). Liefert die Konfiguration.
    """
    effekte = effekte if effekte is not None else laden()
    original = getattr(engine_module.build_engine, "_original", engine_module.build_engine)

    def build_engine(*args, **kwargs):
        global LETZTE_ENGINE
        LETZTE_ENGINE = installieren(original(*args, **kwargs), effekte)
        return LETZTE_ENGINE

    build_engine._original = original
    engine_module.build_engine = build_engine
    return effekte


def deaktivieren() -> None:
    """Den ursprünglichen build_engine wiederherstellen."""
    engine_module.build_engine = getattr(engine_module.build_engine, "_original", engine_module.build_engine)

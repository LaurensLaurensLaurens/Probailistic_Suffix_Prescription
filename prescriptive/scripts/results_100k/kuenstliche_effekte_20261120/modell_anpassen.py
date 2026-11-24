"""Angepasste Kopie des normativen BPIC17-Modells für die Conformance-Zielgröße (2026-11-20, Nutzer: „angepasste Kopie von BPIC17_ToBe,
die die Reihenfolge A_Concept / W_Complete application wie im Log erlaubt; Ziel: Großteil der echten Fälle konform, nur echte Verstöße
übrig“). Liest process_models/BPIC17_ToBe.bpmn (unverändert) und schreibt process_models/BPIC17_ToBe_angepasst.bpmn.

Änderungen (jede als Funktion unten, Begründung aus conformance_probe.py am echten Log):
1. vervollstaendigen: W_Complete application darf vor A_Concept (beliebig oft, auch gar nicht) und danach (beliebig oft) vorkommen –
   im Log beginnt die erste Sitzung vor A_Concept; an annehmen_bei_konzept sind beide Optionen konform.
2. angebote: nach O_Created darf eine weitere Angebotsvariante erstellt werden, bevor gesendet wird, und mehrere Angebote dürfen
   nacheinander verschickt werden (zwei Angebote erstellen, dann senden ist im Log üblich; Entscheidung 2026-11-20: keine Abweichung,
   sonst wäre die Option „weiteres Angebot“ an angebot_nachlegen immer ein Verstoß). Gilt für die erste Angebotsrunde und für
   Folgeangebote nach O_Cancelled.
3.–6. nach der zweiten Messung (61 % der echten Fälle konform): Prüfung nach Nachreichung fortgesetzt statt neu gestartet; neues
   Angebot ohne vorheriges Storno; Storno offener Angebote nach A_Pending; erneute Prüfung nach Betrugsverdacht.
Weitere Änderungen nur nach Messung, hier ergänzen. Neue Elemente ohne Layout (BPMNDI); pm4py braucht es nicht.
Aufruf: python3 <dieses Skript>
"""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
QUELLE = ROOT/"process_models/BPIC17_ToBe.bpmn"
ZIEL = ROOT/"process_models/BPIC17_ToBe_angepasst.bpmn"
NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
for prefix, uri in (("", NS), ("bpmndi", "http://www.omg.org/spec/BPMN/20100524/DI"),
                    ("dc", "http://www.omg.org/spec/DD/20100524/DC"), ("di", "http://www.omg.org/spec/DD/20100524/DI")):
    ET.register_namespace(prefix, uri)
q = lambda tag: f"{{{NS}}}{tag}"


class Modell:
    def __init__(self, pfad):
        self.baum = ET.parse(pfad)
        self.proc = self.baum.getroot().find(q("process"))
        self.n = 0

    def element(self, ident):
        return next(e for e in self.proc if e.get("id") == ident)

    def finde(self, endung):
        treffer = [e.get("id") for e in self.proc if (e.get("id") or "").endswith(endung) and e.tag != q("sequenceFlow")]
        assert len(treffer) == 1, (endung, treffer)
        return treffer[0]

    def task_von(self, quelle_id, name):
        """id des Tasks name, der von quelle_id aus über eine Kante erreicht wird (Namen sind im Modell mehrfach vergeben)."""
        for f in self.proc.findall(q("sequenceFlow")):
            if f.get("sourceRef") == quelle_id and self.element(f.get("targetRef")).get("name") == name:
                return f.get("targetRef")
        raise KeyError((quelle_id, name))

    def kante(self, quelle, ziel):
        treffer = [f for f in self.proc.findall(q("sequenceFlow")) if f.get("sourceRef") == quelle and f.get("targetRef") == ziel]
        assert len(treffer) == 1, (quelle, ziel, len(treffer))
        return treffer[0]

    def _neu(self, art):
        self.n += 1
        return f"anp_{art}_{self.n}"

    def gateway(self, richtung):
        ident = self._neu("xor")
        ET.SubElement(self.proc, q("exclusiveGateway"), {"id": ident, "gatewayDirection": richtung})
        return ident

    def task(self, name):
        ident = self._neu("task")
        ET.SubElement(self.proc, q("task"), {"id": ident, "name": name})
        return ident

    def verbinde(self, quelle, ziel, name=None):
        ident = self._neu("flow")
        attrs = {"id": ident, "sourceRef": quelle, "targetRef": ziel}
        if name:
            attrs["name"] = name
        ET.SubElement(self.proc, q("sequenceFlow"), attrs)
        ET.SubElement(self.element(quelle), q("outgoing")).text = ident
        ET.SubElement(self.element(ziel), q("incoming")).text = ident
        return ident

    def umlenken(self, kante, quelle=None, ziel=None):
        """Quelle und/oder Ziel einer vorhandenen Kante ändern (incoming / outgoing der Knoten mitziehen)."""
        ident = kante.get("id")
        for seite, neu, kind in (("sourceRef", quelle, "outgoing"), ("targetRef", ziel, "incoming")):
            if neu is None:
                continue
            alt = self.element(kante.get(seite))
            for c in list(alt.findall(q(kind))):
                if c.text == ident:
                    alt.remove(c)
            kante.set(seite, neu)
            ET.SubElement(self.element(neu), q(kind)).text = ident

    def schreiben(self, pfad):
        self.baum.write(pfad, encoding="UTF-8", xml_declaration=True)


def vervollstaendigen(m: Modell):
    j_src, concept, compl = m.finde("j_src"), "concept", m.finde("_compl")
    complete_app = m.task_von(concept, "W_Complete application")
    # vor A_Concept: Schleife über eine zweite W_Complete application (auch gar nicht)
    j1, g1, vor = m.gateway("Converging"), m.gateway("Diverging"), m.task("W_Complete application")
    m.umlenken(m.kante(j_src, concept), ziel=j1)
    m.verbinde(j1, g1)
    m.verbinde(g1, vor, "bearbeiten")
    m.verbinde(vor, j1)
    m.verbinde(g1, concept, "Konzept")
    # nach A_Concept: W_Complete application beliebig oft, auch gar nicht
    j2, g2 = m.gateway("Converging"), m.gateway("Diverging")
    m.umlenken(m.kante(concept, complete_app), ziel=j2)
    m.verbinde(j2, g2)
    m.umlenken(m.kante(complete_app, compl), ziel=j2)
    m.verbinde(g2, complete_app, "weiter bearbeiten")
    m.verbinde(g2, compl, "fertig")


def _angebotsrunde(m: Modell, created: str, kanal_split: str, kanal_join: str, danach: str, variante_ziel: str):
    """created → [weitere Variante zurück zu variante_ziel | senden] ; nach dem Senden weitere Angebote senden oder weiter zu danach."""
    gv, jk = m.gateway("Diverging"), m.gateway("Converging")
    m.umlenken(m.kante(created, kanal_split), ziel=gv)
    m.verbinde(gv, variante_ziel, "weitere Variante vor Versand")
    m.verbinde(gv, jk, "senden")
    m.verbinde(jk, kanal_split)
    gs = m.gateway("Diverging")
    m.umlenken(m.kante(kanal_join, danach), ziel=gs)
    m.verbinde(gs, jk, "weiteres Angebot senden")
    m.verbinde(gs, danach, "alle gesendet")


def angebote(m: Modell):
    offer_join, g_chan, j_chan, g_more = m.finde("_offer"), m.finde("g_chan"), m.finde("j_chan"), m.finde("g_more")
    offer = m.task_von(offer_join, "O_Create Offer")
    created = m.task_von(offer, "O_Created")
    _angebotsrunde(m, created, g_chan, j_chan, g_more, offer_join)
    # Folgeangebote nach O_Cancelled: gleicher Aufbau (created2 → g_chan2 → mail2 / online2 → j_chan2 → j_wait)
    created2 = "created2"
    offer2 = next(f.get("sourceRef") for f in m.proc.findall(q("sequenceFlow")) if f.get("targetRef") == created2)
    jv = m.gateway("Converging")
    for f in [f for f in m.proc.findall(q("sequenceFlow")) if f.get("targetRef") == offer2]:
        m.umlenken(f, ziel=jv)
    m.verbinde(jv, offer2)
    m.folgeangebot = jv
    _angebotsrunde(m, created2, "g_chan2", "j_chan2", "j_wait", jv)


def nachgereicht_fortgesetzt(m: Modell):
    """3. Nach A_Incomplete und nachgereichten Unterlagen wird die Prüfung oft fortgesetzt (W_Validate application resume, in der
    Projektion unsichtbar) statt neu gestartet: von g_inc direkt zur Prüfentscheidung (j_val)."""
    m.verbinde("g_inc", "j_val", "nachgeliefert, Prüfung fortgesetzt")


def neues_angebot_ohne_storno(m: Modell):
    """4. Neues Angebot nach A_Complete, ohne das alte vorher zu stornieren (Storno später oder nie): von g_cust direkt zur
    Folgeangebotsrunde."""
    m.verbinde("g_cust", m.folgeangebot, "neues Angebot, altes bleibt offen")


def storno_nach_pending(m: Modell):
    """5. Offene Angebote werden auch nach A_Pending storniert: A_Pending → [O_Cancelled]* → Ende."""
    j, g, storno = m.gateway("Converging"), m.gateway("Diverging"), m.task("O_Cancelled")
    m.umlenken(m.kante("a_pending", "end_ok"), ziel=j)
    m.verbinde(j, g)
    m.verbinde(g, storno, "offenes Angebot storniert")
    m.verbinde(storno, j)
    m.verbinde(g, "end_ok", "Ende")


def pruefung_nach_betrug(m: Modell):
    """6. Nach W_Assess potential fraud wird erneut geprüft (W_Validate application) und dann entschieden."""
    erneut = m.task("W_Validate application")
    m.verbinde("g_fraud", erneut, "erneut prüfen")
    m.verbinde(erneut, "j_val")


def main():
    m = Modell(QUELLE)
    m.proc.set("id", "bpic17_tobe_angepasst")
    m.proc.set("name", "BPIC17 To-Be (angepasst für Conformance)")
    doku = m.proc.find(q("documentation"))
    doku.text = (doku.text or "") + (" Angepasste Kopie 2026-11-20 (modell_anpassen.py): W_Complete application vor und nach A_Concept "
                                     "beliebig oft; mehrere Angebotsvarianten vor dem Versand und mehrere Versände erlaubt; Prüfung nach Nachreichung fortgesetzt; "
                                     "neues Angebot ohne Storno; Storno nach A_Pending; erneute Prüfung nach Betrugsverdacht.")
    vervollstaendigen(m)
    angebote(m)
    nachgereicht_fortgesetzt(m)
    neues_angebot_ohne_storno(m)
    storno_nach_pending(m)
    pruefung_nach_betrug(m)
    m.schreiben(ZIEL)
    print(f"geschrieben: {ZIEL} ({m.n} neue Elemente)")


if __name__ == "__main__":
    main()

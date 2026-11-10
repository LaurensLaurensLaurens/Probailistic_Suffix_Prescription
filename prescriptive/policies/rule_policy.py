"""
regel: feste Regel je Wahlpunkt, z. B. die laut Screening beste feste Option (screening/, Spalte "beste feste Regel"). An
einem Control-Flow-Punkt in RULES (bzw. rules) wählt sie immer diese Option, sofern sie angeboten ist; sonst, und an allen
anderen Punkten, None (wie die Bank). Ein Vergleichsmaßstab für die gelernten Policies: so viel bringt das, was das
Screening auf dem echten Log schon weiß, ohne Personalisierung.

Standard (BPIC17, Outcome Durchlaufzeit; screening_alle vom 2026-11-06): annehmen bei A_Concept und nach dem Vervollständigen,
kein weiteres Angebot nach O_Created, kein Rückruf bei Leads. Andere Regeln über policy_params, z. B.
{"regel": {"rules": {"handle_leads": "W_Handle leads|start"}}}.
"""
RULES = {
    "annehmen_bei_konzept": "A_Accepted",
    "annehmen_nach_vervollstaendigung": "A_Accepted",
    "angebot_nachlegen": "O_Sent (mail and online)",
    "handle_leads": "W_Handle leads|withdraw",
}


class RulePolicy:
    def __init__(self, rules=None, seed=None):
        self.rules = dict(RULES if rules is None else rules)

    # Wird aufgerufen von case_engine._ask -> choose (je Wahlpunkt); zusammengelegte Punkte (a+b): die Regel des ersten Teils
    def choose(self, point, case, options, time, engine):
        for part in point.name.split("+"):
            option = self.rules.get(part)
            if option is not None and option in options:
                return option
        return None

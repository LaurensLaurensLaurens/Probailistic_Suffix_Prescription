"""BPIC17: Kennzahlen am Entscheidungspunkt A_Validating (Anruf wegen fehlender Unterlagen) für train.py."""
from evaluation.BPIC2017.kpis import decision_kpis


def kpis(generated, real) -> dict:
    return {"decision": decision_kpis(generated)}

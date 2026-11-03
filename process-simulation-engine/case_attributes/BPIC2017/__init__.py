"""
BPIC17: der Generator der Fall- und Angebotsattribute (CASE_ATTRIBUTES "generator", ../datasets.py).

setup.py              setup_case_attributes() baut den Generator
simulator.py          AttributeSimulationEngine: LoanGoal, ApplicationType, RequestedAmount (nach Monat); die Engine ruft
                      bei jeder Ankunft start_new_case() und beim ersten O_Create Offer populate_offer_attributes() auf
registry.py           die Modelle der Angebotsattribute, je eine Datei: offered_amount, number_of_terms, monthly_cost,
                      first_withdrawal_amount, credit_score, selected, accepted (gemeinsam: base.py, utils.py)
attribute_models.pkl, attribute_distributions.pkl   gespeicherte Modelle und Verteilungen
"""
from .setup import setup_case_attributes
from .simulator import AttributeSimulationEngine, CaseState

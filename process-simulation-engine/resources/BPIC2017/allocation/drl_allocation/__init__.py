"""
Zustand für die Zuteilung drl (resources/case_allocation.py, DRLAllocation): state.py baut aus dem Zustand der Engine die
Beobachtung und die Aktionsmaske für die trainierte Strategie (MaskablePPO; Modell: resources/datasets.py, drl_model).
"""
from resources.BPIC2017.allocation.drl_allocation.state import DRLStateBuilder

__all__ = ["DRLStateBuilder"]

"""
Fall- und Angebotsattribute der simulierten Fälle.

datasets.py   je Datensatz (CASE_ATTRIBUTES), Einstieg dataset_case_attributes(): der Generator (BPIC17) oder die
              Attribute echter Fälle (RealCaseAttributes, BPIC13, BPIC19)
BPIC2017/     der Generator für BPIC17 (AttributeSimulationEngine) mit den Modellen der Angebotsattribute
"""
from .BPIC2017 import AttributeSimulationEngine, CaseState, setup_case_attributes
from .datasets import CASE_ATTRIBUTES, CaseAttributes, RealCaseAttributes, dataset_case_attributes

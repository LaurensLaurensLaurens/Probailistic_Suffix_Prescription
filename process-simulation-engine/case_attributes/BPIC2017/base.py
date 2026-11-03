from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import pandas as pd


@dataclass
class FitContext:
    """Optional: Trägt Meta-Infos aus fit() (z.B. Spaltennamen)."""
    case_id_col: str = "case:concept:name"


class AttributePredictorBase:
    """
    Minimaler Base-Contract für Attribute:
      - fit(df): lernt/initialisiert Artefakte aus Originaldaten
      - predict(...): erzeugt Wert(e)
      - validate(df, sim_df): bewertet Simulationsdaten vs Original (Metriken/Report)
    """
    name: str = "attribute"
    seed: int

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        self.model: Optional[dict[str, Any]] = None
        self.fit_ctx: FitContext = FitContext()

    def fit(self, df: pd.DataFrame) -> "AttributePredictorBase":
        raise NotImplementedError

    def is_fitted(self) -> bool:
        return self.model is not None

    def _require_fitted(self):
        if not self.is_fitted():
            raise RuntimeError(f"{self.__class__.__name__} ist nicht gefittet. Bitte zuerst .fit(df) aufrufen.")

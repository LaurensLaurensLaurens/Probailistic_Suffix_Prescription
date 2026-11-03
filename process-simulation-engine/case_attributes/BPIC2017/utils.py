from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def resolve_col(frame: pd.DataFrame, col: str) -> str:
    """
    Unterstützt case:-Prefix-Varianten:
      - "CreditScore" vs "case:CreditScore"
    """
    if col in frame.columns:
        return col
    if f"case:{col}" in frame.columns:
        return f"case:{col}"
    if col.startswith("case:"):
        plain = col.replace("case:", "", 1)
        if plain in frame.columns:
            return plain
    raise KeyError(f"Spalte '{col}' nicht gefunden. Verfügbare Spalten: {list(frame.columns)[:30]} ...")


def to_case_level(frame: pd.DataFrame, cols: Sequence[str], case_id_col: str = "case:concept:name") -> pd.DataFrame:
    """
    Case-Level wie im Notebook: groupby(case).first()
    """
    if case_id_col in frame.columns:
        return frame.groupby(case_id_col)[list(cols)].first()
    return frame[list(cols)].copy()


def detect_rounding_step(values: np.ndarray) -> float | None:
    """
    Heuristik aus dem Notebook (robust): typischste Schrittweite bei vielen Unique-Werten.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 200:
        return None

    u = np.unique(v)
    if u.size < 200:
        return None

    u = np.sort(u)
    diffs = np.diff(u)
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.size < 200:
        return None

    med = float(np.median(diffs))
    nice = np.array([0.01, 0.05, 0.1, 0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000], dtype=float)
    step = float(nice[np.argmin(np.abs(nice - med))])
    return step if step > 0 else None

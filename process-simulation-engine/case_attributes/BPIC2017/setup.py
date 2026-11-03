"""
Aufbau des Attribut-Generators für die Simulation (BPIC17).
"""
import logging
from typing import Optional

import pandas as pd

from .simulator import AttributeSimulationEngine

logger = logging.getLogger(__name__)


# Set up case attribute predictor.
# Always returns an AttributeSimulationEngine instance.
# By default loads from cached artifacts. Only retrains if explicitly requested.
# Wird aufgerufen von case_attributes.dataset_case_attributes <- simulation/case_engine.py build_engine
def setup_case_attributes(
    df: Optional[pd.DataFrame],
    seed: int = 42,
    monthly_artifact_path: Optional[str] = None,
    retrain: bool = False,
) -> AttributeSimulationEngine:
    logger.info("Setting up case attribute predictor (AttributeSimulationEngine)...")

    # Load monthly artifact if specified
    monthly_artifact = None
    if monthly_artifact_path:
        try:
            import joblib
            monthly_artifact = joblib.load(monthly_artifact_path)
        except Exception as e:
            logger.warning(f"Could not load monthly artifact: {e}")

    # By default, load from cached artifacts (df=None, retrain_models=False).
    # If cache is missing/corrupt, auto-fallback to retraining when df is available.
    used_retrain = retrain

    try:
        predictor = AttributeSimulationEngine(
            # Mit df baut der Generator die Fall-Sampler (nach Jahreszeit) aus dem Log; die Modelle kommen
            # trotzdem aus dem Cache, solange retrain nicht gesetzt ist
            df=df,
            seed=seed,
            monthly_artifact=monthly_artifact,
            retrain_models=retrain,
        )
    except ValueError:
        if (not retrain) and (df is not None):
            logger.warning(
                "Case attribute artifacts missing or invalid. Falling back to retraining from event log."
            )
            predictor = AttributeSimulationEngine(
                df=df,
                seed=seed,
                monthly_artifact=monthly_artifact,
                retrain_models=True,
            )
            used_retrain = True
        else:
            raise

    mode_desc = "retrained from event log" if used_retrain else "from cached artifacts"
    logger.info(f"Loaded AttributeSimulationEngine ({mode_desc})")
    return predictor

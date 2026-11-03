"""
Processing time estimator for batch allocation policies.

Provides a fast (resource, activity) -> seconds lookup table mined from
the event log.  Used by the MILP-based 1-Batch-1 policy to estimate
p_{ij} (processing time of task j on worker i).

Unlike the full ProcessingTimePredictionClass, this is a simple lookup of
historical means — fast enough to call inside the optimizer for every
(worker, task) pair.
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Sensible default when no data is available at all (1 hour)
_DEFAULT_GLOBAL_MEAN = 3600.0


class ProcessingTimeEstimator:
    """
    Fast lookup table: (resource, activity) -> mean processing time in seconds.

    Falls through three levels:
        1. Per (resource, activity) mean
        2. Per activity mean
        3. Global mean (default 3600s)
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self._pair_means: Dict[Tuple[str, str], float] = {}
        self._activity_means: Dict[str, float] = {}
        self._global_mean: float = _DEFAULT_GLOBAL_MEAN

        if df is not None and len(df) > 0:
            self._fit(df)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame) -> None:
        """Mine mean processing times from the event log."""
        required = {"concept:name", "org:resource", "time:timestamp"}
        missing = required - set(df.columns)
        if missing:
            logger.warning(
                "ProcessingTimeEstimator: missing columns %s — using defaults",
                missing,
            )
            return

        work = df.copy()
        work["time:timestamp"] = pd.to_datetime(work["time:timestamp"])
        work = work.sort_values(["case:concept:name", "time:timestamp"])

        # Compute inter-event duration per case (seconds between consecutive events)
        work["_duration_s"] = (
            work.groupby("case:concept:name")["time:timestamp"]
            .diff()
            .dt.total_seconds()
        )

        # Drop NaN (first event of each case) and non-positive durations
        valid = work.dropna(subset=["_duration_s"])
        valid = valid[valid["_duration_s"] > 0]

        if valid.empty:
            logger.warning("ProcessingTimeEstimator: no valid durations found")
            return

        # 1. Per (resource, activity) means
        pair_group = valid.groupby(["org:resource", "concept:name"])["_duration_s"]
        for (resource, activity), series in pair_group:
            self._pair_means[(resource, activity)] = series.mean()

        # 2. Per activity means
        act_group = valid.groupby("concept:name")["_duration_s"]
        for activity, series in act_group:
            self._activity_means[activity] = series.mean()

        # 3. Global mean
        self._global_mean = valid["_duration_s"].mean()

        logger.info(
            "ProcessingTimeEstimator fitted: %d (resource, activity) pairs, "
            "%d activities, global_mean=%.0fs",
            len(self._pair_means),
            len(self._activity_means),
            self._global_mean,
        )

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate(self, resource_id: str, activity: str) -> float:
        """
        Estimate processing time in seconds for a (resource, activity) pair.

        Falls through: (resource, activity) -> activity -> global_mean.
        """
        # Level 1: exact (resource, activity) pair
        val = self._pair_means.get((resource_id, activity))
        if val is not None:
            return val

        # Level 2: activity-level mean
        val = self._activity_means.get(activity)
        if val is not None:
            return val

        # Level 3: global mean
        return self._global_mean

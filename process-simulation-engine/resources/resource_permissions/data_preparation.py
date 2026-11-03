"""
Data Preparation Module for Resource Permission Mining.

Preprocesses BPIC17 EL for organizational model discovery.
Key steps:
1. Filter to completed activities only (lifecycle:transition = 'complete') (as was also done in the OrdinoR paper)
2. Remove system resources (e.g., User_1)
3. Handle missing values
"""
import os
import logging
from typing import List, Optional, Set
import pandas as pd
import pm4py

logger = logging.getLogger(__name__)


class ResourceDataPreparation:
    """
    Prepares event log data for resource permission mining.
    
    Applies standard preprocessing per process mining best practices:
    - Filters to completed events only
    - Removes automated/system resources
    - Validates required columns
    """
    
    REQUIRED_COLUMNS = ["concept:name", "org:resource", "case:concept:name"]
    LIFECYCLE_COL = "lifecycle:transition"
    COMPLETED_STATE = "complete"
    
    def __init__(
        self,
        log_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None
    ):
        """
        Initialize with event log.
        
        Args:
            log_path: Path to XES/CSV event log.
            df: Pre-loaded DataFrame.
        """
        if df is not None:
            self.df_raw = df.copy()
        elif log_path is not None:
            self.df_raw = self._load_log(log_path)
        else:
            raise ValueError("Either log_path or df must be provided")
        
        self.df_prepared: Optional[pd.DataFrame] = None
        self._stats = {}
    
    def _load_log(self, path: str) -> pd.DataFrame:
        """Load event log from file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Event log not found: {path}")
        
        logger.info(f"Loading event log from {path}")
        
        if path.endswith('.csv') or path.endswith('.csv.gz'):
            return pd.read_csv(path)
        else:
            # XES format
            log = pm4py.read_xes(path, return_legacy_log_object=True)
            return pm4py.convert_to_dataframe(log)
    
    def prepare(
        self,
        filter_completed: bool = True,
        exclude_resources: Optional[List[str]] = None,
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Apply all preprocessing steps.
        
        Args:
            filter_completed: If True, keep only 'complete' lifecycle events.
            exclude_resources: Resources to remove (e.g., ['User_1']).
            drop_na: If True, drop rows with missing required columns.
        
        Returns:
            Prepared DataFrame.
        """
        df = self.df_raw.copy()
        self._stats["raw_events"] = len(df)
        
        # Validate required columns
        self._validate_columns(df)
        
        # Step 1: Filter to completed events
        if filter_completed and self.LIFECYCLE_COL in df.columns:
            before = len(df)
            df = df[df[self.LIFECYCLE_COL] == self.COMPLETED_STATE]
            self._stats["after_lifecycle_filter"] = len(df)
            logger.info(f"Filtered to completed events: {before:,} -> {len(df):,} "
                       f"({len(df)/before:.1%})")
        
        # Step 2: Exclude system resources
        if exclude_resources:
            before = len(df)
            df = df[~df["org:resource"].isin(exclude_resources)]
            self._stats["after_resource_filter"] = len(df)
            logger.info(f"Excluded resources {exclude_resources}: {before:,} -> {len(df):,}")
        
        # Step 3: Drop missing values
        if drop_na:
            before = len(df)
            df = df.dropna(subset=self.REQUIRED_COLUMNS)
            self._stats["after_dropna"] = len(df)
            if before != len(df):
                logger.info(f"Dropped NA rows: {before:,} -> {len(df):,}")
        
        self.df_prepared = df
        self._stats["final_events"] = len(df)
        
        logger.info(f"Preparation complete: {self._stats['raw_events']:,} -> {len(df):,} events "
                   f"({len(df)/max(self._stats['raw_events'], 1):.1%})")
        
        return df
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def get_stats(self) -> dict:
        """Get preprocessing statistics."""
        return self._stats.copy()
    
    def get_unique_resources(self) -> Set[str]:
        """Get unique resources in prepared data."""
        if self.df_prepared is None:
            raise ValueError("Call prepare() first")
        return set(self.df_prepared["org:resource"].unique())
    
    def get_unique_activities(self) -> Set[str]:
        """Get unique activities in prepared data."""
        if self.df_prepared is None:
            raise ValueError("Call prepare() first")
        return set(self.df_prepared["concept:name"].unique())


def prepare_bpic2017(
    log_path: str,
    exclude_resources: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function for BPIC2017 preparation.
    
    Args:
        log_path: Path to BPIC2017 XES file.
        exclude_resources: Resources to exclude (default: ['User_1']).
    
    Returns:
        Prepared DataFrame with only completed activities.
    """
    exclude = exclude_resources or ['User_1']
    
    prep = ResourceDataPreparation(log_path=log_path)
    return prep.prepare(
        filter_completed=True,
        exclude_resources=exclude,
        drop_na=True
    )

"""
Berechtigungen, die mehrere Datensätze nutzen: BasicResourcePermissions (method performed, resources/datasets.py: wer
die Tätigkeit im Log ausgeführt hat; BPIC19, dazu als Vergleich in den BPIC17-Analysen). Die übrigen Methoden liegen im
Ordner des Datensatzes, der sie nutzt: OrdinoR in BPIC2017/resource_permissions/, die Teams in BPIC2013/.
"""
import logging
from typing import Dict, List, Optional, Set

import pandas as pd

from resources.resource_permissions.data_preparation import ResourceDataPreparation

logger = logging.getLogger(__name__)


class BasicResourcePermissions:
    """
    Manages resource permissions based on historical event log data.
    Maps activities to the set of resources that have performed them.
    
    By default, preprocesses data to include only completed activities
    following OrdinoR (2022) methodology.
    """

    def __init__(
        self,
        log_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        filter_completed: bool = True,
        exclude_resources: Optional[List[str]] = None
    ):
        """
        Initialize the ResourcePermissions model.

        Args:
            log_path: Path to the XES event log file.
            df: Existing pandas DataFrame containing the event log.
            filter_completed: If True, only include 'complete' lifecycle events.
            exclude_resources: Resources to exclude (e.g., ['User_1'] for system users).
        
        Raises:
            ValueError: If neither log_path nor df is provided.
        """
        self.activity_resource_map: Dict[str, Set[str]] = {}
        self.filter_completed = filter_completed
        self.exclude_resources = exclude_resources or []

        if df is not None:
            self.df = df
            # Apply preprocessing if filtering is enabled and df hasn't been preprocessed
            if filter_completed or exclude_resources:
                self.df = self._preprocess(df)
        elif log_path is not None:
            self.df = self._load_and_preprocess(log_path)
        else:
            raise ValueError("Either 'log_path' or 'df' must be provided.")

        self._build_mapping()

    def _load_and_preprocess(self, path: str) -> pd.DataFrame:
        """Load and preprocess event log."""
        prep = ResourceDataPreparation(log_path=path)
        return prep.prepare(
            filter_completed=self.filter_completed,
            exclude_resources=self.exclude_resources if self.exclude_resources else None,
            drop_na=True
        )
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to existing DataFrame."""
        prep = ResourceDataPreparation(df=df)
        return prep.prepare(
            filter_completed=self.filter_completed,
            exclude_resources=self.exclude_resources if self.exclude_resources else None,
            drop_na=True
        )

    def _build_mapping(self):
        """
        Parses the event log to build a map of Activity -> Set[Resource].
        Assumes standard column names: 'concept:name' for activity, 'org:resource' for resource.
        """
        if self.df is None or self.df.empty:
            return

        activity_col = "concept:name"
        resource_col = "org:resource"

        if activity_col not in self.df.columns or resource_col not in self.df.columns:
             raise ValueError(f"DataFrame missing required columns: {activity_col}, {resource_col}")
        grouped = self.df.dropna(subset=[activity_col, resource_col])
        
        mapping_series = grouped.groupby(activity_col)[resource_col].apply(set)
        
        self.activity_resource_map = mapping_series.to_dict()

    def get_eligible_resources(self, activity_name: str, timestamp: pd.Timestamp = None, case_type: str = None) -> List[str]:
        """
        Returns a list of all resources that have historically performed the given activity.
        
        Args:
            activity_name: The name of the activity.
            timestamp: Ignored (Basic model is not context-aware).
            case_type: Ignored (Basic model is not context-aware).

        Returns:
            List[str]: A list of resource IDs. Returns empty list if activity is unknown.
        """
        resources = self.activity_resource_map.get(activity_name, set())
        return list(resources)

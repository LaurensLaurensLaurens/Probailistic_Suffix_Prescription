"""BPIC13: Berechtigungen über die Support-Teams (method groups, resources/datasets.py)."""
from typing import Dict, List, Optional, Set

import pandas as pd


class GroupResourcePermissions:
    """
    Berechtigt sind die Mitglieder der Teams (group_column), die die Tätigkeit im Log ausgeführt haben: Mitglied eines
    Teams ist, wer im Log Events für das Team ausgeführt hat (BPIC13: org:group, die Support-Teams).
    """

    def __init__(self, df: pd.DataFrame, group_column: str, exclude_resources: Optional[List[str]] = None):
        work = df[df['org:resource'].notna() & df[group_column].notna() & ~df['org:resource'].isin(exclude_resources or [])]
        members = work.groupby(group_column)['org:resource'].agg(set)
        self.activity_resource_map: Dict[str, Set[str]] = {
            activity: set().union(*(members[group] for group in groups))
            for activity, groups in work.groupby('concept:name')[group_column].agg(set).items()
        }

    def get_eligible_resources(self, activity_name: str, timestamp: pd.Timestamp = None, case_type: str = None) -> List[str]:
        return list(self.activity_resource_map.get(activity_name, set()))

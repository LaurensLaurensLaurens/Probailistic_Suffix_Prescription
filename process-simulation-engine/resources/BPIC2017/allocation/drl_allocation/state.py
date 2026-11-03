"""
DRL State Vector Builder for resource allocation.

Builds observation vectors and action masks for the PPO agent.
State space: role-aggregated features (3K + 3A + 5 = 113 with K=10, A=26).
Action space: activity-level (A + 1 = 27: serve activity queue or postpone).
"""

import math
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


def _strip_tz(dt: datetime) -> datetime:
    """Remove tzinfo from a datetime to avoid naive/aware comparison errors."""
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


class DRLStateBuilder:
    """
    Builds observation vectors and action masks for the DRL allocation agent.

    State vector layout (3K + 3A + 5):
        [0..K)           role_idle_fraction       (K)
        [K..2K)          role_avg_remaining_busy   (K)
        [2K..2K+A)       queue_lengths             (A)
        [2K+A..2K+2A)    queue_avg_wait            (A)
        [2K+2A..3K+2A)   freed_role_onehot         (K)
        [3K+2A..3K+3A)   freed_eligible            (A)
        [3K+3A..+4)      hour_sin/cos, dow_sin/cos (4)
        [3K+3A+4]        active_cases              (1)

    All features clamped to [0, 1].
    """

    def __init__(
        self,
        activity_list: List[str],
        role_groups: List[Set[str]],
        resource_to_role: Dict[str, int],
        activity_to_roles: Dict[str, Set[int]],
        max_queue_length: float = 200.0,
        max_wait_hours: float = 48.0,
        max_remaining_busy_hours: float = 24.0,
        max_active_cases: float = 5000.0,
    ):
        """
        Args:
            activity_list: Ordered list of allocation activities (defines action indices).
            role_groups: List of sets, each set contains resource IDs in that role.
            resource_to_role: resource_id -> role group index.
            activity_to_roles: activity -> set of role group indices that can perform it.
            max_queue_length: Normalization constant for queue lengths.
            max_wait_hours: Normalization constant for average wait time (hours).
            max_remaining_busy_hours: Normalization constant for remaining busy time (hours).
            max_active_cases: Normalization constant for active case count.
        """
        self.activity_list = list(activity_list)
        self._activity_to_idx = {a: i for i, a in enumerate(self.activity_list)}
        self.role_groups = role_groups
        self.resource_to_role = resource_to_role
        self.activity_to_roles = activity_to_roles

        self.num_roles = len(role_groups)
        self.num_activities = len(activity_list)

        self._max_queue = max_queue_length
        self._max_wait = max_wait_hours
        self._max_busy = max_remaining_busy_hours
        self._max_cases = max_active_cases

        # Pre-compute role sizes for idle fraction calculation
        self._role_sizes = np.array(
            [len(g) for g in role_groups], dtype=np.float32
        )

    @property
    def observation_size(self) -> int:
        """Total observation vector size: 3K + 3A + 5."""
        return 3 * self.num_roles + 3 * self.num_activities + 5

    def build(
        self,
        freed_resource: str,
        current_time: datetime,
        resource_pool,
        active_case_count: int,
    ) -> np.ndarray:
        """
        Build observation from live ResourcePool (for inference).

        Args:
            freed_resource: The resource that just became free.
            current_time: Current simulation time.
            resource_pool: Live ResourcePool instance.
            active_case_count: Number of active cases.

        Returns:
            Observation vector of shape (observation_size,), values in [0, 1].
        """
        # Normalize timezone: strip tz info to avoid naive/aware comparison errors
        current_time = _strip_tz(current_time)

        obs = np.zeros(self.observation_size, dtype=np.float32)
        K = self.num_roles
        A = self.num_activities

        # --- Role idle fraction & avg remaining busy ---
        role_busy_count = np.zeros(K, dtype=np.float32)
        role_remaining_sum = np.zeros(K, dtype=np.float32)

        for res_id, (busy_until, _, _) in resource_pool._busy_resources.items():
            role_idx = self.resource_to_role.get(res_id)
            if role_idx is None:
                continue
            busy_until_naive = _strip_tz(busy_until)
            if busy_until_naive > current_time:
                role_busy_count[role_idx] += 1
                remaining_h = (busy_until_naive - current_time).total_seconds() / 3600.0
                role_remaining_sum[role_idx] += remaining_h

        # role_idle_fraction = 1 - busy/total (clamped)
        safe_sizes = np.maximum(self._role_sizes, 1.0)
        obs[0:K] = np.clip(1.0 - role_busy_count / safe_sizes, 0.0, 1.0)

        # role_avg_remaining_busy (normalized)
        busy_counts_safe = np.maximum(role_busy_count, 1.0)
        avg_remaining = role_remaining_sum / busy_counts_safe
        obs[K:2*K] = np.clip(avg_remaining / self._max_busy, 0.0, 1.0)

        # --- Queue lengths & avg wait ---
        for act, queue in resource_pool._waiting_queues.items():
            idx = self._activity_to_idx.get(act)
            if idx is None or not queue:
                continue
            obs[2*K + idx] = min(len(queue) / self._max_queue, 1.0)
            total_wait_h = sum(
                (current_time - _strip_tz(w.arrival_time)).total_seconds() / 3600.0
                for w in queue
            )
            obs[2*K + A + idx] = min(total_wait_h / len(queue) / self._max_wait, 1.0)

        # --- Freed role one-hot ---
        freed_role = self.resource_to_role.get(freed_resource)
        if freed_role is not None and freed_role < K:
            obs[2*K + 2*A + freed_role] = 1.0

        # --- Freed eligible activities ---
        if freed_role is not None:
            for act, roles in self.activity_to_roles.items():
                if freed_role in roles:
                    idx = self._activity_to_idx.get(act)
                    if idx is not None:
                        obs[3*K + 2*A + idx] = 1.0

        # --- Temporal encoding ---
        hour_frac = (current_time.hour + current_time.minute / 60.0) / 24.0
        dow_frac = current_time.weekday() / 7.0
        base = 3*K + 3*A
        obs[base] = (math.sin(2 * math.pi * hour_frac) + 1) / 2
        obs[base + 1] = (math.cos(2 * math.pi * hour_frac) + 1) / 2
        obs[base + 2] = (math.sin(2 * math.pi * dow_frac) + 1) / 2
        obs[base + 3] = (math.cos(2 * math.pi * dow_frac) + 1) / 2

        # --- Active cases ---
        obs[base + 4] = min(active_case_count / self._max_cases, 1.0)

        return obs

    def build_from_snapshot(
        self,
        freed_resource: str,
        current_time_dt: datetime,
        pool_snapshot: dict,
        active_case_count: int,
    ) -> np.ndarray:
        """
        Build observation from a serialized pool snapshot (for training, thread-safe).

        Args:
            freed_resource: The resource that just became free.
            current_time_dt: Current simulation time as datetime.
            pool_snapshot: Dict with 'busy_resources' and 'waiting_queues' (plain dicts).
            active_case_count: Number of active cases.

        Returns:
            Observation vector of shape (observation_size,), values in [0, 1].
        """
        obs = np.zeros(self.observation_size, dtype=np.float32)
        K = self.num_roles
        A = self.num_activities

        busy_resources = pool_snapshot.get("busy_resources", {})
        waiting_queues = pool_snapshot.get("waiting_queues", {})

        # --- Role idle fraction & avg remaining busy ---
        role_busy_count = np.zeros(K, dtype=np.float32)
        role_remaining_sum = np.zeros(K, dtype=np.float32)

        for res_id, info in busy_resources.items():
            role_idx = self.resource_to_role.get(res_id)
            if role_idx is None:
                continue
            busy_until = info["busy_until"]
            if busy_until > current_time_dt:
                role_busy_count[role_idx] += 1
                remaining_h = (busy_until - current_time_dt).total_seconds() / 3600.0
                role_remaining_sum[role_idx] += remaining_h

        safe_sizes = np.maximum(self._role_sizes, 1.0)
        obs[0:K] = np.clip(1.0 - role_busy_count / safe_sizes, 0.0, 1.0)

        busy_counts_safe = np.maximum(role_busy_count, 1.0)
        avg_remaining = role_remaining_sum / busy_counts_safe
        obs[K:2*K] = np.clip(avg_remaining / self._max_busy, 0.0, 1.0)

        # --- Queue lengths & avg wait ---
        for act, queue_items in waiting_queues.items():
            idx = self._activity_to_idx.get(act)
            if idx is None or not queue_items:
                continue
            obs[2*K + idx] = min(len(queue_items) / self._max_queue, 1.0)
            total_wait_h = sum(
                (current_time_dt - item["arrival_time"]).total_seconds() / 3600.0
                for item in queue_items
            )
            obs[2*K + A + idx] = min(total_wait_h / len(queue_items) / self._max_wait, 1.0)

        # --- Freed role one-hot ---
        freed_role = self.resource_to_role.get(freed_resource)
        if freed_role is not None and freed_role < K:
            obs[2*K + 2*A + freed_role] = 1.0

        # --- Freed eligible activities ---
        if freed_role is not None:
            for act, roles in self.activity_to_roles.items():
                if freed_role in roles:
                    idx = self._activity_to_idx.get(act)
                    if idx is not None:
                        obs[3*K + 2*A + idx] = 1.0

        # --- Temporal encoding ---
        hour_frac = (current_time_dt.hour + current_time_dt.minute / 60.0) / 24.0
        dow_frac = current_time_dt.weekday() / 7.0
        base = 3*K + 3*A
        obs[base] = (math.sin(2 * math.pi * hour_frac) + 1) / 2
        obs[base + 1] = (math.cos(2 * math.pi * hour_frac) + 1) / 2
        obs[base + 2] = (math.sin(2 * math.pi * dow_frac) + 1) / 2
        obs[base + 3] = (math.cos(2 * math.pi * dow_frac) + 1) / 2

        # --- Active cases ---
        obs[base + 4] = min(active_case_count / self._max_cases, 1.0)

        return obs

    def build_action_mask(
        self,
        freed_resource: str,
        eligible_map: Dict[str, Set[str]],
        waiting_activities: Set[str],
    ) -> np.ndarray:
        """
        Build action mask for MaskablePPO.

        Action i (0..A-1) feasible iff freed_resource is eligible for activity_list[i]
        AND there's waiting work for that activity.
        Action A (postpone) is always feasible.

        Args:
            freed_resource: The resource that just became free.
            eligible_map: activity -> set of eligible resource IDs.
            waiting_activities: Set of activities that have waiting work.

        Returns:
            Boolean mask of shape (num_activities + 1,).
        """
        mask = np.zeros(self.num_activities + 1, dtype=np.bool_)

        for i, act in enumerate(self.activity_list):
            if act in waiting_activities and freed_resource in eligible_map.get(act, set()):
                mask[i] = True

        # Postpone is always feasible
        mask[self.num_activities] = True

        return mask

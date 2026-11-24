"""Work-item state shared by intervention validation and constrained PSP draws.

Closing an unobserved task is allowed: withdraw/abort often cancel planned work.
A second opening of active work is never a new waiting task.
"""


class WorkState:
    def __init__(self, states=None, business_end=False, previous=None):
        self.states = dict(states or {})
        self.business_end = business_end
        self.previous = previous

    def copy(self):
        return WorkState(self.states, self.business_end, self.previous)

    @classmethod
    def from_events(cls, events):
        result = cls()
        for event in events:
            result.push(f"{event['concept:name']}|{event.get('lifecycle:transition', 'complete')}")
        return result

    @property
    def active(self):
        return frozenset(a for a, status in self.states.items() if status == "active")

    @property
    def complete(self):
        return self.business_end and not self.active

    def permits(self, token):
        activity, _, lifecycle = str(token).rpartition("|")
        if not activity.startswith("W_"):
            return True
        status = self.states.get(activity)
        if lifecycle == "start":
            return status not in ("active", "paused")
        if lifecycle == "resume":
            return status in (None, "paused")
        return True

    def push(self, token):
        activity, _, lifecycle = str(token).rpartition("|")
        if activity.startswith("W_"):
            if lifecycle in ("start", "resume"):
                self.states[activity] = "active"
            elif lifecycle == "suspend":
                self.states[activity] = "paused"
            elif lifecycle in ("complete", "withdraw", "ate_abort"):
                self.states[activity] = "closed"
            elif lifecycle == "schedule" and activity not in self.active:
                self.states[activity] = "scheduled"
        self.business_end |= activity in ("A_Pending", "A_Denied", "A_Cancelled")
        self.previous = str(token)

    def context(self, token):
        activity = str(token).rpartition("|")[0]
        return self.states.get(activity, "unknown"), bool(self.active)

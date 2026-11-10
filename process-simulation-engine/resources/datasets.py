"""
Mitarbeiter je Datensatz: Berechtigungen, Dienstplan und Zuteilung für einen Datensatz. dataset_resources ist der
Einstieg (die Engine simulation/case_engine.py, Auswertungen); die Logik selbst ist für alle Datensätze gleich.

Je Datensatz (RESOURCES):
- permissions: wer eine Tätigkeit darf, method
    ordinor     OrdinoR, full recall (BPIC2017/resource_permissions/ordinor.py): Gruppen ähnlicher Mitarbeiter, jedes
                Mitglied darf, was die Gruppe getan hat (Regeln aus dem Log dazu, log_rules). file: gespeichertes Modell,
                relativ zu resources/; fehlt die Datei, wird es aus dem Log entdeckt und dort gespeichert
    groups      die Mitglieder der Teams (group_column), die die Tätigkeit im Log ausgeführt haben (BPIC2013/)
    performed   wer die Tätigkeit im Log ausgeführt hat (resource_permissions/resource_permissions.py)
- roster: always, alle rund um die Uhr im Dienst (resources/daily_roster.py AlwaysOnDuty; die Werkstatt); Standard: DailyRoster, im Dienst von
  der ersten bis zur letzten Stunde mit eigenen Events am Tag
- learned_allocation, drl_model: gespeicherte Modelle der Zuteilungen learned und drl (nur wo die Engine mit
  Mitarbeitern läuft; fehlt learned, wird es gelernt)
Die Systemnutzer (Systemschritte, rund um die Uhr im Dienst, nehmen keinem Mitarbeiter Arbeit ab) sind die Mitarbeiter,
auf die das Muster screening.system_users im Katalog (prescriptive/catalog.yaml) passt, wie im Screening. Welche
Tätigkeiten Arbeitsschritte mit Mitarbeitern sind, sagt work_item_prefix in next_event_prediction/datasets.py; ob die
Engine mit Mitarbeitern läuft, DATASETS in simulation/dataset_simulator.py.
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Set

import pandas as pd

RESOURCES = {
    # 144 Mitarbeiter: OrdinoR mit 10 Gruppen grenzt sinnvoll ab
    "bpic17": {"permissions": {"method": "ordinor", "file": "BPIC2017/resource_permissions/ordinor_fullrecall.pkl"},
               # learned: gefittet mit dem Pool der Engine (Erfahrung, Dienstplan, frei); learned_allocation.pkl ist die alte Fassung (±2 h)
               "learned_allocation": "models/allocation/learned_allocation_dienstplan.pkl", "drl_model": "models/drl_allocation/drl_allocation_model"},
    # 1.440 Mitarbeiter in 649 Support-Teams (org:group). OrdinoR mit 10 Gruppen erlaubte fast jedem alles (1.439 statt
    # 569 für Accepted/In Progress [1st]); mit den Teams 1.011. Nur complete-Events, die Fälle laufen in der Simulation
    # (noch) ohne Mitarbeiter
    "bpic13": {"permissions": {"method": "groups", "group_column": "org:group"}},
    # 628 Mitarbeiter ohne Team-Spalte; OrdinoR erlaubte z. B. 544 statt 29 Record Invoice Receipt. Nur complete-Events
    "bpic19": {"permissions": {"method": "performed"}},
    # Die Werkstatt: jeder darf, was er im Log getan hat (Sachbearbeiter, Prüfer, Mechaniker und Expertinnen getrennt)
    "repair": {"permissions": {"method": "performed"}, "roster": "always"},
}
RESOURCES_FOLDER = os.path.dirname(os.path.abspath(__file__))


def resource_spec(dataset):
    if dataset not in RESOURCES:
        raise ValueError(f"Keine Mitarbeiter für '{dataset}' festgelegt (RESOURCES), möglich: {', '.join(RESOURCES)}")
    return RESOURCES[dataset]


def system_users(dataset, log) -> Set[str]:
    """Die Systemnutzer des Datensatzes im Log: Muster screening.system_users im Katalog (wie das Screening, str.contains)."""
    import interventions
    pattern = interventions.catalog()[dataset].get("screening", {}).get("system_users")
    if not pattern:
        return set()
    resources = pd.Series(log["org:resource"].dropna().unique()).astype(str)
    return set(resources[resources.str.contains(pattern)])


def main_system_user(log, users: Set[str]) -> Optional[str]:
    """Der Systemnutzer mit den meisten Events: ihm gibt die Engine Systemschritte, die kein Mitarbeiter darf."""
    counts = log.loc[log["org:resource"].isin(users), "org:resource"].value_counts()
    return str(counts.index[0]) if len(counts) else None


@dataclass
class Resources:
    """Was die Engine über die Mitarbeiter wissen muss: wer eine Tätigkeit darf (permissions), wer wann im Dienst ist (availability)."""
    permissions: object  # get_eligible_resources(activity)
    availability: object  # resources, is_available(resource, time)
    system_users: Set[str]
    system_user: Optional[str]  # der wichtigste davon (main_system_user)

    def candidates(self, activity: str, time) -> List[str]:
        """Der Pool für eine Tätigkeit zu einem Zeitpunkt: berechtigt (permissions) und im Dienst (availability), ohne Systemnutzer."""
        return sorted(r for r in self.permissions.get_eligible_resources(activity)
                      if r not in self.system_users and self.availability.is_available(r, time))

    def select(self, activity: str, time, strategy) -> Optional[str]:
        """
        Aus dem Pool (candidates) wählt die Auswahlstrategie (resources/selection_strategies.py, create_strategy); None,
        wenn niemand berechtigt und im Dienst ist. Wer gerade belegt ist, weiß nur der Aufrufer: er gibt einen
        Mitarbeiter mit strategy.notify_release frei (zählt bei shortest_queue).
        """
        pool = self.candidates(activity, time)
        if not pool:
            return None
        chosen = strategy.select(pool, activity)
        strategy.notify_assignment(chosen, activity)
        return chosen


def system_resources(dataset, log):
    """Für Logs ohne Arbeitsschritte (work_item_prefix None): nur die Systemnutzer, die Engine teilt dort niemanden zu."""
    users = system_users(dataset, log)
    return Resources(None, None, users, main_system_user(log, users))


def dataset_permissions(spec, log, users, main, n_resource_clusters=10):
    """
    Das Berechtigungsmodell nach spec (RESOURCES[datensatz]["permissions"]), ohne die Systemnutzer. Die Methode liegt
    oben, wenn mehrere Datensätze sie nutzen, sonst im Ordner ihres Datensatzes.
    """
    if spec["method"] == "ordinor":
        from .BPIC2017.resource_permissions.ordinor import load_permissions
        return load_permissions(os.path.join(RESOURCES_FOLDER, spec["file"]), log, users, main, n_resource_clusters)
    if spec["method"] == "groups":
        from .BPIC2013.resource_permissions import GroupResourcePermissions
        return GroupResourcePermissions(log, spec["group_column"], exclude_resources=sorted(users))
    if spec["method"] == "performed":
        from .resource_permissions.resource_permissions import BasicResourcePermissions
        return BasicResourcePermissions(df=log, exclude_resources=sorted(users))
    raise ValueError(f"Unbekannte Methode der Berechtigungen {spec['method']}, möglich: ordinor, groups, performed")


def dataset_resources(dataset, log, n_resource_clusters=10):
    """Berechtigungen und Dienstplan des Datensatzes aus dem vorverarbeiteten Log."""
    from .daily_roster import AlwaysOnDuty, DailyRoster
    spec = resource_spec(dataset)
    users = system_users(dataset, log)
    main = main_system_user(log, users)
    permissions = dataset_permissions(spec["permissions"], log, users, main, n_resource_clusters)
    roster = AlwaysOnDuty(log, users) if spec.get("roster") == "always" else DailyRoster(log, users)
    return Resources(permissions, roster, users, main)

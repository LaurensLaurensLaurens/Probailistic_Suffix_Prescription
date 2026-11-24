"""Test der Option-Learner-Hülle (2026-11-22) mit Platzhaltern: die beiden Arme des echten T-Learners stehen für die 8 Netze. Prüft bauen,
speichern, laden, Weiterleitung je (Wahlpunkt, Option) und das Ziehen an jedem Wahlpunkt."""
import os, sys, tempfile, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
os.environ["PSP_CATALOG"] = str(ROOT/"prescriptive/data/catalog_neue_punkte_zuteilung_random.yaml")
from prescriptive.helpers.datasets import add_paths
add_paths()
from prescriptive.policies.ev_policy import option_learner_policy as ol
from prescriptive.estimators.psp_model import PSPModel
M = ROOT/"prescriptive/models"
akt, ref = M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_aktion.pkl", M/"psp_model_durchlaufzeit_real_alle_s60_normal_zaehler_zeit_tf1_t_referenz.pkl"
paare = {("annehmen_bei_konzept", "A_Accepted|complete"): akt, ("annehmen_bei_konzept", "W_Complete application|start"): ref,
         ("annehmen_nach_vervollstaendigung", "A_Accepted|complete"): akt, ("annehmen_nach_vervollstaendigung", "W_Complete application|resume"): ref,
         ("angebot_nachlegen", "O_Create Offer|complete"): akt, ("angebot_nachlegen", "O_Sent (mail and online)|complete"): ref,
         ("handle_leads", "W_Handle leads|start"): akt, ("handle_leads", "W_Handle leads|withdraw"): ref}
ziel = Path(tempfile.mkdtemp(dir=str(Path(__file__).parent)))/"huelle.pkl"
ol.bauen(paare, ziel)
h = PSPModel.load(str(ziel))
print(type(h).__name__, len(h.netze), "Netze; Punkte", sorted(h.points))
sys.path.insert(0, str(ROOT/"prescriptive/scripts/results_100k/kuenstliche_effekte_20261120"))
import online_konformanz as ok  # nur simulator() (ohne künstliche Effekte)
gesehen = {}
class Spion:
    def choose(self, point, case, options, time, engine):
        if len(options) == 2 and point.name not in gesehen:
            drawn, _ = h.option_suffixes(point, case.prefix, options, time, 2)
            gesehen[point.name] = {o: [s[0] for s in d[0][:2]] for o, d in zip(options, drawn)}
            print(point.name, {o: type(h.model_for(point, o)).__name__ for o in options}, gesehen[point.name])
        return None
ok.simulator().run(500, policy=Spion(), window=(400, 500), seed=7)
assert len(gesehen) == 4, gesehen
import shutil; shutil.rmtree(ziel.parent)
print("OK")

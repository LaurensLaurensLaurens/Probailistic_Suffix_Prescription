# Funktionstest der neuen psp_lookahead in der echten Werkstatt (600 Fälle, Fenster 100-399, Outcome kosten), gegen psp
import sys, os, time, numpy as np, pandas as pd
sys.path[:0]=[os.getcwd(), "..", "../prescriptive"]
from prescriptive.helpers.datasets import add_paths; add_paths()
from prescriptive import profiles; profiles.set_outcome("repair", sys.argv[1] if len(sys.argv) > 1 else "kosten")
from preprocessing import chosen_outcome_spec
from simulation.repair_workshop import RepairWorkshop
from prescriptive.policies.psp_policy import PSPPolicy
from prescriptive.policies.psp_lookahead_policy import PSPLookaheadPolicy
from interventions import WAIT
M="../prescriptive/models/psp_model_repair.pkl"; spec=chosen_outcome_spec("repair")
for name, policy in [("psp", PSPPolicy(M, seed=0)), ("psp_lookahead", PSPLookaheadPolicy(M, seed=0))]:
    t=time.time(); e=RepairWorkshop(seed=0); e.set_policy(policy, [e.point], window=(100, 400))
    ev=pd.DataFrame(e.run(600)); tab=e.case_table(); part=tab[tab["number"].between(100,399)]
    extra=""
    if spec["type"]=="negative_repair_cost":
        from prescriptive.costs.repair.outcome import case_costs
        extra=f", Kosten je Fall {case_costs(ev[ev['case:concept:name'].isin(part['case'])], spec)['Kosten (€)'].mean():,.0f} €"
    waits=sum(c["choice"]==WAIT for c in e.choices)
    print(f"{name}: {time.time()-t:.0f} s, Fragen {len(e.choices)}, Warten {waits}{extra}, Durchlaufzeit {part['days'].mean():.2f} Tage, "
          f"Warten auf Reparatur {part['waited'].mean():.3f} Tage, {getattr(policy,'stats',None)}", flush=True)

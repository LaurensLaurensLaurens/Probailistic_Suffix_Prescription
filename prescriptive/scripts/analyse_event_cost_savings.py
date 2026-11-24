"""Decompose observed costs and add a paired, cost-trained forest reference."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import warnings
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115"
FOREST = ROOT/"prescriptive/policies/ev_policy_event_costs/results/vergleich_200_faelle_30_ziehungen_seed0_20261113/models/cate_model_cf_event_costs.pkl"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PSP_CATALOG"] = str(SOURCE/"catalog.yaml")
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()
import numpy as np
import pandas as pd


def now():
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat(timespec="seconds")


def write(path, value):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+"\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forest_job(task):
    out, seed = task
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    from prescriptive.policies.ev_policy_event_costs.cate_policy import CATEEventCostPolicy
    from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
    folder = out/f"forest_cost_seed{seed}"
    folder.mkdir(exist_ok=False)
    started = time.time()
    write(folder/"status.json", {"status": "running", "seed": seed, "started_at": now()})
    raw, _, _ = load_dataset("bpic17")
    sim = Simulator("bpic17", preprocess("bpic17", raw))
    costs = EventCosts(SOURCE/"event_costs.yaml")
    policy = CATEEventCostPolicy(str(FOREST), seed=seed, cost_config=costs)
    assert policy.model.learner == "cf"
    events, choices = sim.run(2200, policy=policy, window=(700,900), seed=seed)
    numbers = events["case:concept:name"].astype(str).str.rsplit("_", n=1).str[-1].astype(int)
    events = events.loc[numbers.ge(700)&numbers.lt(900)]
    bills = costs.case_costs(events)
    dates = events.groupby("case:concept:name")["time:timestamp"]
    bills["cycle_days"] = (dates.max()-dates.min()).dt.total_seconds()/86400
    bank_bills = pd.read_csv(SOURCE/f"bank_reference_seed{seed}/case_costs.csv",index_col=0)
    assert len(bills)==200 and set(bills.index)==set(bank_bills.index)
    bank = json.loads((SOURCE/f"bank_reference_seed{seed}/result.json").read_text())
    result = {"seed": seed, "cases": len(bills), "cost_eur": float(bills.total_eur.mean()),
        "cycle_days": float(bills.cycle_days.mean()), "bank_cost_eur": bank["cost_eur"],
        "saving_eur": bank["cost_eur"]-float(bills.total_eur.mean()),
        "saving_percent": 100*(1-float(bills.total_eur.mean())/bank["cost_eur"]), "seconds": time.time()-started}
    bills.to_csv(folder/"case_costs.csv")
    events.to_pickle(folder/"events.pkl")
    choices.to_csv(folder/"choices.csv",index=False)
    write(folder/"result.json",result)
    write(folder/"status.json",{"status":"complete","updated_at":now()})
    print("COST FOREST",json.dumps(result),flush=True)
    return result


def decompose(out, with_forest=False):
    from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
    from interventions import choice_points, option_token
    costs = EventCosts(SOURCE/"event_costs.yaml")
    folders = {"Bank":SOURCE/"bank_reference_seed0", "D-Kosten":SOURCE/"D_cost_seed0",
        "OE-Kosten":SOURCE/"OE_cost_seed0", "D-Zeit":SOURCE/"D_time_seed0", "OE-Zeit":SOURCE/"OE_time_seed0",
        "XGB-Zeit":SOURCE/"xgb_reference/xgb_time_seed0"}
    if with_forest:
        folders["Forest-Kosten"] = out/"forest_cost_seed0"
    summaries, activities, choice_rates, distribution = [], [], [], []
    bank = pd.read_csv(folders["Bank"]/"case_costs.csv",index_col=0)
    bank_fees = bank.base_eur.mean()+bank.lifecycle_eur.mean()
    points = {f"{point.activity}|{point.lifecycle}":point for point in choice_points("bpic17") if point.kind=="control_flow"}
    bank_first = []
    for name,folder in folders.items():
        bills = pd.read_csv(folder/"case_costs.csv",index_col=0)
        events = pd.read_pickle(folder/"events.pkl")
        ledger = costs.event_costs(events)
        ledger["activity"] = events["concept:name"].astype(str).str.strip().to_numpy()
        by_activity = ledger.groupby("activity")[["base_eur","lifecycle_eur","time_eur","active_hours","total_eur"]].sum()/len(bills)
        by_activity["policy"] = name
        activities.append(by_activity.reset_index())
        row={"policy":name,"cases":len(bills),"events_per_case":len(events)/len(bills),
            **bills[["total_eur","base_eur","lifecycle_eur","time_eur","active_hours"]].mean().to_dict()}
        row["fees"] = row["base_eur"]+row["lifecycle_eur"]
        row["extra_fees_vs_bank"] = row["fees"]-bank_fees
        row["work_saving_vs_bank"] = bank.time_eur.mean()-row["time_eur"]
        row["total_saving_vs_bank"] = bank.total_eur.mean()-row["total_eur"]
        summaries.append(row)
        delta = bills.total_eur-bank.total_eur
        distribution.append({"policy":name,"cheaper_cases":int(delta.lt(-1e-6).sum()),"more_expensive_cases":int(delta.gt(1e-6).sum()),
            "equal_cases":int(delta.abs().le(1e-6).sum()),"median_cost_delta":float(delta.median()),
            "mean_cost_delta":float(delta.mean()),"largest_extra_cost":float(delta.max()),"largest_saving":float(-delta.min())})
        counts = {}
        for case_id,part in events.sort_values(["case:concept:name","time:timestamp"],kind="stable").groupby("case:concept:name",sort=False):
            records=part.to_dict("records")
            first_seen=False
            for i,(event,nxt) in enumerate(zip(records,records[1:])):
                token=f"{str(event['concept:name']).strip()}|{event['lifecycle:transition']}"
                next_token=f"{str(nxt['concept:name']).strip()}|{nxt['lifecycle:transition']}"
                point=points.get(token)
                if point is not None and next_token in [option_token(o) for o in point.options]:
                    counts[(point.name,next_token)]=counts.get((point.name,next_token),0)+1
                    if name=="Bank" and not first_seen:
                        at=pd.Timestamp(event['time:timestamp']).timestamp()
                        bank_first.append({"case":case_id,"sunk_before_first_choice":costs.state(records[:i+1]).total(at),
                            "total_cost":float(bills.loc[case_id,"total_eur"]),"first_point":point.name})
                        first_seen=True
        for (point,option),count in counts.items():
            choice_rates.append({"policy":name,"point":point,"next_event":option,"count":count})
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/"cost_components_seed0.csv",index=False)
    pd.concat(activities,ignore_index=True).to_csv(out/"activity_costs_seed0.csv",index=False)
    pd.DataFrame(choice_rates).to_csv(out/"actual_options_seed0.csv",index=False)
    pd.DataFrame(distribution).to_csv(out/"paired_case_distribution_seed0.csv",index=False)
    pd.DataFrame(bank_first).to_csv(out/"bank_first_decision_costs.csv",index=False)
    stats={"bank_fee_share":float(bank_fees/bank.total_eur.mean()),"bank_active_minutes":float(bank.active_hours.mean()*60),
        "bank_work_cost_share":float(bank.time_eur.mean()/bank.total_eur.mean()),
        "bank_cases_with_observed_choice":len(bank_first),
        "mean_sunk_cost_before_first_choice":float(pd.DataFrame(bank_first).sunk_before_first_choice.mean()),
        "sunk_share_across_bank_cases_with_choice":float(sum(r['sunk_before_first_choice'] for r in bank_first)/sum(r['total_cost'] for r in bank_first))}
    write(out/"decomposition_summary.json",stats)
    print(summary.to_string(index=False),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib')
    files=[Path(__file__),FOREST,SOURCE/'catalog.yaml',SOURCE/'event_costs.yaml',ROOT/'process-simulation-engine/simulation/case_engine_lifecycle_aware.py',
        ROOT/'process-simulation-engine/simulation/dataset_simulator.py',ROOT/'prescriptive/policies/ev_policy_event_costs/costs.py']
    hashes={str(path):sha(path) for path in files}
    write(out/'manifest.json',{'prepared_at':now(),'seeds':[0,1,2],'scored_cases':200,'warmup':700,'cooldown':1300,
        'forest_model':str(FOREST),'source_sha256':hashes,'purpose':'separate cost composition, intervention behaviour and a cost-trained reference; no optimal-cost bound'})
    write(out/'status.json',{'status':'running','updated_at':now(),'completed_forest_seeds':[]})
    decompose(out)
    rows=[]
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(forest_job,(out,seed)) for seed in [0,1,2]]
            for future in as_completed(futures):
                rows.append(future.result())
                pd.DataFrame(rows).sort_values('seed').to_csv(out/'forest_runs.csv',index=False)
                write(out/'status.json',{'status':'running','updated_at':now(),'completed_forest_seeds':sorted(r['seed'] for r in rows)})
        assert all(sha(path)==expected for path,expected in hashes.items())
        summary=decompose(out,True)
        indexed=summary.set_index('policy')
        lines=['**Warum die Kostenersparnis gegenüber der Bank klein bleibt**','',
            'Alle nachfolgenden Komponenten beziehen sich auf dieselben 200 Fälle, Seed 0, im aktuellen korrigierten Simulator. '
            'Es handelt sich um die konfigurierten Beispieltarife. Kalenderwartezeit hat keinen eigenen Kostensatz; aktive Arbeitszeit und jedes Event werden berechnet.','',
            '| Verfahren | Basis + Lifecycle (€) | Aktive Zeit (€) | Gesamt (€) |','|---|---:|---:|---:|']
        for name,row in indexed.iterrows():
            lines.append(f'| {name} | {row.fees:.2f} | {row.time_eur:.2f} | {row.total_eur:.2f} |')
        lines += ['', 'Die Basis-/Lifecycle-Gebühren sind keine bewiesenen unvermeidbaren Fixkosten: Sie können durch weniger Events sinken. '
            'Eine Verkürzung reiner Wartezeit spart in dieser Kostenfunktion jedoch direkt nichts. Bei unveränderter Eventfolge bringen '
            '10 % niedrigere Arbeitskosten nur rund 2,33 % weniger Gesamtkosten.','',
            f"D-Kosten: {indexed.loc['D-Kosten','work_saving_vs_bank']:.2f} € weniger Arbeitskosten und "
            f"{-indexed.loc['D-Kosten','extra_fees_vs_bank']:.2f} € weniger Eventgebühren, zusammen "
            f"{indexed.loc['D-Kosten','total_saving_vs_bank']:.2f} € Ersparnis pro Fall.", '',
            f"XGB-Zeit: {indexed.loc['XGB-Zeit','work_saving_vs_bank']:.2f} € weniger Arbeitskosten bei "
            f"{indexed.loc['XGB-Zeit','extra_fees_vs_bank']:.2f} € zusätzlichen Eventgebühren. "
            'XGBoost optimiert hier die Durchlaufzeit, nicht Eventkosten.','',
            'Die PSP-Kostenpolicies fallen im abgeschlossenen Seed 0 bei etwa 60 % ihrer adaptiv bewerteten Zweierwahlen '
            'wegen nicht ausreichend getrennter Monte-Carlo-Schätzungen auf den Bankvorschlag zurück. Das begrenzt die Änderungen '
            'gegenüber der Bank; der tatsächlich andere Aktionsanteil kann daraus allein nicht abgeleitet werden. '
            'Die Netzgewichte lernen beobachtete Fortsetzungen und identifizieren damit nicht automatisch kausale Kostenunterschiede.','',
            '**Kostenoptimierte Referenz, dieselben drei Seeds:**','',
            '| Seed | Bank €/Fall | Causal Forest €/Fall | Ersparnis (%) |','|---|---:|---:|---:|']
        for r in sorted(rows,key=lambda r:r['seed']):
            lines.append(f"| {r['seed']} | {r['bank_cost_eur']:.2f} | {r['cost_eur']:.2f} | {r['saving_percent']:+.2f} |")
        frame=pd.DataFrame(rows)
        lines += ['',f"Mittel über Seeds: Bank {frame.bank_cost_eur.mean():.2f} €, Forest {frame.cost_eur.mean():.2f} €. "
            f"Mittlere seedweise Ersparnis: {frame.saving_percent.mean():.2f} %. "
            'Das ist eine Referenz mit vorhandenen Gewichten und kein Nachweis des optimal erreichbaren Kostenniveaus.','',
            'Dateien: cost_components_seed0.csv, activity_costs_seed0.csv, actual_options_seed0.csv, '
            'paired_case_distribution_seed0.csv, bank_first_decision_costs.csv, decomposition_summary.json und forest_runs.csv.']
        (out/'bericht.md').write_text('\n'.join(lines)+'\n')
        write(out/'status.json',{'status':'complete','updated_at':now(),'completed_forest_seeds':[0,1,2]})
    except BaseException as error:
        write(out/'status.json',{'status':'failed','updated_at':now(),'error':repr(error)})
        raise


if __name__=='__main__':
    main()

"""Read-only diagnosis of current PSP/XGB labor outcomes and fixed-state rankings."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115'
OUT = SOURCE / 'psp_vs_xgb_labor_diagnosis'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PSP_CATALOG'] = str(SOURCE / 'catalog.yaml')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()
import numpy as np
import pandas as pd
import torch
from interventions import choice_points, option_token
from interventions.work_state import WorkState
from simulation.case_engine_lifecycle_aware import CaseEngine
from prescriptive.estimators.psp_model import PSPModel
from prescriptive.estimators.xgb_model import XGBModel
from prescriptive.estimators.conditional_gaps import SuffixSamplingError
from prescriptive.policies.ev_policy_event_costs.costs import EventCosts


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')


def main():
    OUT.mkdir(exist_ok=False)
    benchmark = json.loads((SOURCE / 'manifest.json').read_text())
    protected = {**benchmark['source_sha256'], **benchmark['prepared_sha256']}
    def validate():
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in protected.items())
    validate()
    torch.set_num_threads(1)
    costs = EventCosts(SOURCE / 'event_costs.yaml')
    summaries, comparisons, decision_rows = [], [], []
    for seed in range(3):
        bank = pd.read_csv(SOURCE / f'bank_reference_seed{seed}/case_costs.csv', index_col=0)
        xgb = pd.read_csv(SOURCE / f'xgb_reference/xgb_time_seed{seed}/case_costs.csv', index_col=0)
        folders = [('Bank', SOURCE / f'bank_reference_seed{seed}'),
                   ('D-Kosten', SOURCE / f'D_cost_seed{seed}'), ('D-Zeit', SOURCE / f'D_time_seed{seed}'),
                   ('XGB-Zeit', SOURCE / f'xgb_reference/xgb_time_seed{seed}')]
        for label, folder in folders:
            if not (folder / 'result.json').exists():
                continue
            bills = pd.read_csv(folder / 'case_costs.csv', index_col=0).reindex(bank.index)
            assert len(bills) == 200 and bills.time_eur.notna().all()
            summaries.append({'seed': seed, 'policy': label, 'labor_eur': bills.time_eur.mean(),
                              'total_eur': bills.total_eur.mean(), 'cycle_days': bills.cycle_days.mean(),
                              'labor_saving_percent': 100 * (1 - bills.time_eur.mean() / bank.time_eur.mean())})
            if label.startswith('D-'):
                delta = bills.time_eur - xgb.time_eur
                rng = np.random.default_rng(0)
                means = delta.to_numpy()[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(axis=1)
                comparisons.append({'seed': seed, 'policy': label, 'labor_delta_D_minus_XGB': delta.mean(),
                                    'median_case_delta': delta.median(),
                                    'case_resampling_95_low_exploratory': np.quantile(means, .025),
                                    'case_resampling_95_high_exploratory': np.quantile(means, .975),
                                    'D_dearer_cases': int(delta.gt(1e-6).sum()),
                                    'XGB_dearer_cases': int(delta.lt(-1e-6).sum())})
            if seed == 0 and label != 'Bank':
                choices = pd.read_csv(folder / 'choices.csv')
                for point, group in choices[choices.options.eq(2)].groupby('point'):
                    decision_rows.append({'policy': label, 'point': point, 'two_option_calls': len(group),
                                          'bank_fallbacks': int(group.choice.isna().sum()),
                                          'explicit_choices': group.choice.value_counts().to_dict()})
    pd.DataFrame(summaries).to_csv(OUT / 'completed_labor_runs.csv', index=False)
    pd.DataFrame(comparisons).to_csv(OUT / 'paired_labor_differences.csv', index=False)
    write('two_option_decisions_seed0.json', decision_rows)

    # Prefixes come from the completed current bank run, before any intervention.
    events = pd.read_pickle(SOURCE / 'bank_reference_seed0/events.pkl')
    points = {f'{p.activity}|{p.lifecycle}': p for p in choice_points('bpic17') if p.kind == 'control_flow'}
    candidates = {p.name: [] for p in points.values()}
    for case_id, part in events.groupby('case:concept:name', sort=True):
        records = part.to_dict('records')
        seen = set()
        for index, (event, following) in enumerate(zip(records, records[1:])):
            token = f"{event['concept:name']}|{event['lifecycle:transition']}"
            point = points.get(token)
            if point is None or point.name in seen:
                continue
            context = records[:index + 1]
            options = CaseEngine.control_options(SimpleNamespace(prefix=context, sessions={}), point.options)
            next_token = f"{following['concept:name']}|{following['lifecycle:transition']}"
            if len(options) != 2 or next_token not in [option_token(option) for option in options]:
                continue
            seen.add(point.name)
            candidates[point.name].append({'case': case_id, 'point': point, 'events': context,
                                           'options': options, 'now': pd.Timestamp(event['time:timestamp']).timestamp()})
    rng = np.random.default_rng(20261115)
    selected = []
    for name in sorted(candidates):
        values = candidates[name]
        selected += [values[index] for index in rng.choice(len(values), min(3, len(values)), replace=False)]
    write('selected_bank_states.json', [{'case': row['case'], 'point': row['point'].name,
          'prefix_events': len(row['events']), 'options': row['options']} for row in selected])
    model = PSPModel.load(SOURCE / 'models/D_stateful.pkl')
    model.set_threads(1)
    xgb = XGBModel.load(ROOT / 'prescriptive/models/xgb_model_durchlaufzeit_xgb_real_10k_neuepunkte.pkl')
    xgb.set_threads(1)
    write('model_configuration.json', {'window_size': model.encoder_decoder.window_size,
          'min_suffix_size': model.encoder_decoder.min_suffix_size, 'params': model.params,
          'xgb_trained_on': xgb.trained_on, 'xgb_summary': xgb.summary})
    original_draw = model.draw_cases
    def extended_draw(*args, **kwargs):
        if kwargs.get('max_steps') is None:
            kwargs.setdefault('completion_steps', 1000)
        return original_draw(*args, **kwargs)

    rankings, failures = [], []
    for number, row in enumerate(selected):
        point, context, options, now = row['point'], row['events'], row['options'], row['now']
        xgb_values = xgb.values(xgb.points[point.name], context, options, now)
        for horizon in ('policy_window', 'up_to_1000'):
            model.draw_cases = original_draw if horizon == 'policy_window' else extended_draw
            model.sampling_audit = {}
            torch.manual_seed(400000 + number)
            try:
                drawn, elapsed = model.option_suffixes(model.points[point.name], context, options, now, 100)
            except SuffixSamplingError as error:
                failures.append({'case': row['case'], 'point': point.name, 'horizon': horizon, 'error': str(error)})
                continue
            days, labor, lengths = [], [], []
            observed = costs.state(context)
            sunk_labor = observed.summary(now)['time_eur']
            for suffixes in drawn:
                days.append(np.array([sum(event[2] for event in suffix) / 86400 for suffix in suffixes]))
                lengths.append(np.array([len(suffix) for suffix in suffixes]))
                work_costs = []
                for suffix in suffixes:
                    state, at = observed.copy(), now
                    work = WorkState.from_events(context)
                    for token, _, gap in suffix:
                        assert work.permits(token)
                        work.push(token)
                        activity, lifecycle = token.rsplit('|', 1)
                        at += gap
                        state.push(activity, lifecycle, at)
                    assert work.complete
                    work_costs.append(state.summary(at)['time_eur'] - sunk_labor)
                labor.append(np.asarray(work_costs))
            difference = days[0] - days[1]
            se = difference.std(ddof=1) / 10
            mean_difference = difference.mean()
            rankings.append({'case': row['case'], 'point': point.name, 'prefix_events': len(context),
                'horizon': horizon, 'options': options, 'D_mean_remaining_days': [array.mean() for array in days],
                'D_mean_remaining_labor_eur': [array.mean() for array in labor],
                'D_mean_suffix_events': [array.mean() for array in lengths],
                'D_raw_time_choice': options[int(np.argmin([array.mean() for array in days]))],
                'D_100_draw_uncertain': bool(abs(mean_difference) <= 2 * se),
                'D_paired_time_delta_days': mean_difference, 'D_paired_time_mcse_days': se,
                'XGB_time_choice': options[int(xgb_values.argmax())], 'XGB_values': xgb_values.tolist(),
                'sampling_audit': model.sampling_audit})
            write('fixed_state_rankings.json', rankings)
        print(f'FIXED STATE {number + 1}/{len(selected)} {point.name} {row["case"]}', flush=True)
    model.draw_cases = original_draw
    write('sampling_failures.json', failures)
    validate()
    write('validation.json', {'benchmark_inputs_unchanged': True, 'fixed_bank_states': len(selected),
          'draws_per_option': 100, 'horizons': ['policy_window', 'up_to_1000'],
          'completed_rankings': len(rankings), 'failures': len(failures),
          'caveat': 'Descriptive path differences and exploratory fixed-bank-state predictions; no causal attribution or architecture-only comparison. Case bootstrap does not capture resource coupling or between-seed uncertainty.'})
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

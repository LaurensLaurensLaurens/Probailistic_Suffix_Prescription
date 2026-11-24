"""Paired online D-time experiment changing only the final uncertainty fallback.

Existing guarded runs are reused after an exact simulator replay and input-hash
validation. New runs retain the same adaptive 30→100 draws per decision and all
sampling-error/feasibility protections; uncertain estimates choose their mean.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import warnings
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'prescriptive/policies/ev_policy_event_costs/results/d_oe_zustand_20261115'
MODEL = SOURCE / 'models/D_stateful.pkl'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PSP_CATALOG'] = str(SOURCE / 'catalog.yaml')
warnings.filterwarnings('ignore', category=FutureWarning)
sys.path.insert(0, str(ROOT))
from prescriptive.helpers.datasets import add_paths
add_paths()
import numpy as np
import pandas as pd


def now():
    return datetime.now(ZoneInfo('Europe/Vienna')).isoformat(timespec='seconds')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def check(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    for path, expected in manifest['source_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError(f'Experiment input changed: {path}')
    return manifest


def prepare(out):
    original = json.loads((SOURCE / 'manifest.json').read_text())
    protected = {**original['source_sha256'], **original['prepared_sha256']}
    for path, expected in protected.items():
        if sha(path) != expected:
            raise RuntimeError(f'Original benchmark input changed: {path}')
    extra = [Path(__file__), ROOT / 'prescriptive/policies/ev_policy/uncertainty_ablation.py',
             SOURCE / 'manifest.json', ROOT / 'prescriptive/policies/xgb_policy.py',
             ROOT / 'prescriptive/estimators/xgb_model.py']
    protected.update({str(path): sha(path) for path in extra})
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'prepared_at': now(), 'model': str(MODEL), 'model_sha256': sha(MODEL),
                'objective': 'negative_cycle_days', 'seeds': [0, 1, 2],
                'scored_cases': 200, 'warmup': 700, 'cooldown': 1300, 'workers': 2,
                'initial_samples_per_option': 30, 'max_samples_per_option': 100, 'z': 2.,
                'arms': {'with_guard': 'existing D-time run: bank on final uncertain estimate',
                         'without_guard': 'identical adaptive draw rule; choose best mean on final uncertain estimate'},
                'retained': ['frozen event weights', 'conditional gap model', 'work-state rules',
                             'transition rules', 'sampling-error bank fallback', 'random employee allocation',
                             'event tariffs', 'case IDs and simulator seeds'],
                'budget_note': 'Same adaptive budget per reached decision. Different choices can change the number of later decisions and total samples.',
                'guarded_references': str(SOURCE / 'D_time_seed{seed}'),
                'source_sha256': protected}
    write(out / 'manifest.json', manifest)
    write(out / 'status.json', {'status': 'prepared', 'updated_at': now(), 'completed_new_runs': 0,
                              'total_new_runs': 3, 'completed_pairs': 0, 'total_pairs': 3})
    summarize(out)


def simulation():
    from prescriptive.helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    raw, _, _ = load_dataset('bpic17')
    return Simulator('bpic17', preprocess('bpic17', raw))


class RecordedChoices:
    def __init__(self, rows):
        self.rows, self.index = rows, 0

    def choose(self, point, case, options, at, engine):
        row = self.rows[self.index]
        assert row['case'] == case.case_id and row['point'] == point.name
        assert abs(float(row['time']) - at) < 1e-5
        assert int(row['options']) == len(options)
        self.index += 1
        return None if pd.isna(row['choice']) else row['choice']


def replay(out):
    """Verify reuse of the guarded reference in the unchanged simulator."""
    check(out)
    source = SOURCE / 'D_time_seed0'
    policy = RecordedChoices(pd.read_csv(source / 'choices.csv').to_dict('records'))
    events, _ = simulation().run(2200, policy=policy, window=(700, 900), seed=0)
    numbers = events['case:concept:name'].astype(str).str.rsplit('_', n=1).str[-1].astype(int)
    actual = events.loc[numbers.ge(700) & numbers.lt(900)].reset_index(drop=True)
    expected = pd.read_pickle(source / 'events.pkl').reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False,
                                  rtol=0, atol=1e-8)
    assert policy.index == len(policy.rows)
    write(out / 'reference_replay.json', {'status': 'passed', 'seed': 0, 'cases': 200,
                                         'decisions': policy.index, 'events': len(actual),
                                         'same_complete_event_frame': True, 'completed_at': now()})
    print('GUARDED REFERENCE REPLAY PASSED', flush=True)


def probe(out):
    import torch
    from types import SimpleNamespace
    from prescriptive.policies.ev_policy.uncertainty_ablation import UncertaintyAblationPolicy
    check(out)
    bank = pd.read_pickle(SOURCE / 'bank_reference_seed0/events.pkl')
    part = bank.loc[bank['case:concept:name'].eq('Application_772')].reset_index(drop=True)
    cut = int(np.flatnonzero(part['concept:name'].eq('A_Concept') & part['lifecycle:transition'].eq('complete'))[0])
    context = part.iloc[:cut + 1].to_dict('records')
    at = pd.Timestamp(context[-1]['time:timestamp']).timestamp()
    case = SimpleNamespace(case_id='Application_772', prefix=context)
    arms = {bank_on_uncertain: UncertaintyAblationPolicy(model_path=str(MODEL), samples=30,
            seed=0, bank_on_uncertain=bank_on_uncertain) for bank_on_uncertain in (True, False)}
    attempts = []
    for seed in range(5):
        draws, choices, random_states = [], [], []
        for bank_on_uncertain, policy in arms.items():
            torch.manual_seed(seed)
            policy.rng = np.random.default_rng(seed)
            point = policy.model.points['annehmen_bei_konzept']
            assert policy.model._outcome_kind() == 'negative_cycle_days'
            assert policy.model.params['sampling_control'] == {'max_samples': 100, 'z': 2.}
            choices.append(policy.choose(point, case, list(point.options), at, None))
            draws.append(policy.decision_rows[-1])
            random_states.append(torch.get_rng_state())
        assert draws[0]['estimates'] == draws[1]['estimates']
        assert torch.equal(*random_states)
        uncertain = draws[0]['uncertain_estimate']
        assert uncertain == draws[1]['uncertain_estimate']
        if uncertain:
            assert choices[0] is None and choices[1] in point.options
            assert [entry['samples_per_option'] for entry in draws[0]['estimates']] == [30, 100]
        else:
            assert choices[0] == choices[1]
        attempts.append({'torch_seed': seed, 'guarded': draws[0], 'unguarded': draws[1]})
        if uncertain:
            break
    assert any(attempt['guarded']['uncertain_estimate'] for attempt in attempts)
    check(out)
    write(out / 'policy_probe.json', {'status': 'passed', 'same_sample_values_and_torch_rng': True,
                                     'attempts': attempts, 'completed_at': now()})
    print('REAL CHECKPOINT FALLBACK ABLATION PROBE PASSED', flush=True)


def reference(out, seed):
    source, target = SOURCE / f'D_time_seed{seed}', out / f'with_guard_seed{seed}'
    if not (source / 'result.json').exists():
        return
    files = ['result.json', 'case_costs.csv', 'choices.csv', 'events.pkl']
    fingerprints = {name: sha(source / name) for name in files}
    if target.exists():
        saved = json.loads((target / 'reference.json').read_text())
        assert saved['source_sha256'] == fingerprints, 'Completed guarded reference changed'
        assert all(sha(target / name) == value for name, value in fingerprints.items())
        return
    target.mkdir()
    for name in files:
        shutil.copy2(source / name, target / name)
    write(target / 'reference.json', {'reused_from': str(source), 'source_sha256': fingerprints,
                                      'imported_at': now()})


def summarize(out):
    manifest = check(out)
    rows, pairs = [], []
    for seed in manifest['seeds']:
        reference(out, seed)
        locations = {'Bank': SOURCE / f'bank_reference_seed{seed}',
                     'XGBoost-Zeit': SOURCE / f'xgb_reference/xgb_time_seed{seed}',
                     'D-mit-Rueckfall': out / f'with_guard_seed{seed}',
                     'D-ohne-Unsicherheitsrueckfall': out / f'without_guard_seed{seed}'}
        tables = {}
        for label, folder in locations.items():
            if not (folder / 'result.json').exists():
                continue
            bills = pd.read_csv(folder / 'case_costs.csv', index_col=0)
            assert len(bills) == manifest['scored_cases']
            tables[label] = bills
            result = json.loads((folder / 'result.json').read_text())
            assert np.isclose(bills.total_eur.mean(), result['cost_eur'], rtol=0, atol=1e-8)
            rows.append({'seed': seed, 'policy': label, 'cases': len(bills),
                         'labor_eur': float(bills.time_eur.mean()), 'total_eur': float(bills.total_eur.mean()),
                         'cycle_days': float(bills.cycle_days.mean()),
                         'active_minutes': float(bills.active_hours.mean() * 60)})
        labels = ['D-mit-Rueckfall', 'D-ohne-Unsicherheitsrueckfall']
        if all(label in tables for label in labels):
            guarded, unguarded = [tables[label] for label in labels]
            assert set(guarded.index) == set(unguarded.index) == set(tables['Bank'].index)
            delta = unguarded.reindex(guarded.index) - guarded
            pairs.append({'seed': seed, 'cases': len(guarded),
                          'labor_delta_without_minus_with_eur': float(delta.time_eur.mean()),
                          'total_delta_without_minus_with_eur': float(delta.total_eur.mean()),
                          'cycle_delta_without_minus_with_days': float(delta.cycle_days.mean()),
                          'active_minutes_delta_without_minus_with': float(delta.active_hours.mean() * 60)})
            delta.to_csv(out / f'paired_cases_seed{seed}.csv')
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'runs.csv', index=False)
    pd.DataFrame(pairs).to_csv(out / 'paired_seed_results.csv', index=False)
    lines = ['**D mit und ohne Unsicherheitsrückfall: Durchlaufzeitziel**', '',
             'Identischer aktueller D-Checkpoint, unveränderte Gewichte und Simulatorquellen. '
             'Seeds 0–2, je 200 bewertete Fälle, 700 Warmup und 1.300 Cooldown. '
             'Je erreichter Entscheidung dieselbe adaptive Regel: zunächst 30, bei unsicherer Differenz 100 Ziehungen je Option. '
             'Nur nach einer weiterhin unsicheren Schätzung wird die Bankfortsetzung durch die Option mit dem besten Mittelwert ersetzt. '
             'Bei fehlgeschlagenen vollständigen Suffixziehungen bleibt der Bank-Rückfall erhalten.', '',
             f'Abgeschlossen: {len(pairs)} von 3 gepaarten Seeds. Die Referenzen mit Rückfall sind die vorhandenen D-Zeitläufe; '
             'sie werden mit Prüfsummen gesichert und durch eine exakte Seed-0-Wiedergabe des Simulators geprüft.', '',
             '| Seed | Policy | Arbeitskosten €/Fall | Gesamtkosten €/Fall | Durchlaufzeit Tage |',
             '|---|---|---:|---:|---:|']
    for row in frame.itertuples():
        lines.append(f'| {row.seed} | {row.policy} | {row.labor_eur:.2f} | {row.total_eur:.2f} | {row.cycle_days:.3f} |')
    if pairs:
        lines += ['', '**Gepaarte Änderung: ohne minus mit Unsicherheitsrückfall. Negative Werte bedeuten eine Verbesserung.**', '',
                  '| Seed | Arbeitskosten Δ € | Gesamtkosten Δ € | Durchlaufzeit Δ Tage |', '|---|---:|---:|---:|']
        for row in pairs:
            lines.append(f"| {row['seed']} | {row['labor_delta_without_minus_with_eur']:+.2f} | "
                         f"{row['total_delta_without_minus_with_eur']:+.2f} | {row['cycle_delta_without_minus_with_days']:+.3f} |")
        if len(pairs) == 3:
            mean = pd.DataFrame(pairs).mean(numeric_only=True)
            lines.append(f"| Mittel | {mean.labor_delta_without_minus_with_eur:+.2f} | "
                         f"{mean.total_delta_without_minus_with_eur:+.2f} | {mean.cycle_delta_without_minus_with_days:+.3f} |")
    lines += ['', 'Das Trainingsziel bleibt die Durchlaufzeit. Arbeitskosten und Gesamtkosten sind zusätzlich gemessene Ergebnisse. '
              'Die Policies können unterschiedliche spätere Zustände und Entscheidungshäufigkeiten erzeugen; deshalb sind '
              'die Gesamtsummen ihrer Ziehungen trotz gleicher Regel je Entscheidung nicht identisch. '
              'Der Vergleich isoliert eine Policyregel bei diesem D-Modell und beweist keinen allgemeinen Architektureffekt.']
    (out / 'bericht.md').write_text('\n'.join(lines) + '\n')
    return len(pairs)


def job(task):
    out, seed = task
    import torch
    from prescriptive.policies.ev_policy.uncertainty_ablation import UncertaintyAblationPolicy
    from prescriptive.policies.ev_policy_event_costs.costs import EventCosts
    check(out)
    folder = out / f'without_guard_seed{seed}'
    folder.mkdir(exist_ok=False)
    started = time.time()
    policy = UncertaintyAblationPolicy(model_path=str(MODEL), seed=seed, samples=30, bank_on_uncertain=False)
    torch.set_num_threads(1)
    assert policy.model._outcome_kind() == 'negative_cycle_days'
    progress = {'status': 'running', 'seed': seed, 'started_at': now(), 'decisions': 0}
    write(folder / 'progress.json', progress)
    original = policy.choose
    def choose(point, case, options, at, engine):
        selected = original(point, case, options, at, engine)
        progress['decisions'] += 1
        if progress['decisions'] == 1 or progress['decisions'] % 10 == 0:
            progress.update(updated_at=now(), elapsed_seconds=time.time() - started,
                            uncertainty_overrides=policy.uncertainty_overrides,
                            sampling_fallbacks=getattr(policy, 'sampling_fallbacks', 0),
                            original_policy_audit=getattr(policy, 'decision_audit', {}))
            write(folder / 'progress.json', progress)
            write(folder / 'decision_estimates.json', policy.decision_rows)
        return selected
    policy.choose = choose
    try:
        events, choices = simulation().run(2200, policy=policy, window=(700, 900), seed=seed)
        numbers = events['case:concept:name'].astype(str).str.rsplit('_', n=1).str[-1].astype(int)
        events = events.loc[numbers.ge(700) & numbers.lt(900)]
        bills = EventCosts(SOURCE / 'event_costs.yaml').case_costs(events)
        dates = events.groupby('case:concept:name')['time:timestamp']
        bills['cycle_days'] = (dates.max() - dates.min()).dt.total_seconds() / 86400
        bank = pd.read_csv(SOURCE / f'bank_reference_seed{seed}/case_costs.csv', index_col=0)
        assert len(bills) == 200 and set(bills.index) == set(bank.index)
        bills.to_csv(folder / 'case_costs.csv')
        choices.to_csv(folder / 'choices.csv', index=False)
        events.to_pickle(folder / 'events.pkl')
        write(folder / 'decision_estimates.json', policy.decision_rows)
        base_audit = getattr(policy, 'decision_audit', {})
        uncertain = base_audit.get('uncertain_fallbacks', 0)
        assert uncertain == policy.uncertainty_overrides
        result = {'model': 'D', 'objective': 'time', 'arm': 'without_uncertainty_bank_fallback',
                  'seed': seed, 'cases': len(bills), 'model_sha256': sha(MODEL),
                  'labor_eur': float(bills.time_eur.mean()), 'cost_eur': float(bills.total_eur.mean()),
                  'cycle_days': float(bills.cycle_days.mean()), 'active_hours': float(bills.active_hours.mean()),
                  'uncertain_estimates': uncertain, 'uncertainty_overrides': policy.uncertainty_overrides,
                  'actual_uncertainty_bank_fallbacks': 0,
                  'sampling_failure_bank_fallbacks': getattr(policy, 'sampling_fallbacks', 0),
                  'adaptive_calls': base_audit.get('adaptive_calls', 0),
                  'expanded_calls': base_audit.get('expanded_calls', 0),
                  'samples_per_option_total': base_audit.get('samples_per_option_total', 0),
                  'seconds': time.time() - started}
        check(out)
        write(folder / 'result.json', result)
        write(folder / 'progress.json', {**progress, 'status': 'complete', 'updated_at': now()})
        print('UNGUARDED COMPLETE', json.dumps(result), flush=True)
        return result
    except BaseException as error:
        write(folder / 'progress.json', {**progress, 'status': 'failed', 'updated_at': now(), 'error': repr(error)})
        raise


def run(out):
    manifest = check(out)
    assert json.loads((out / 'policy_probe.json').read_text())['status'] == 'passed'
    assert json.loads((out / 'reference_replay.json').read_text())['status'] == 'passed'
    pending = [seed for seed in manifest['seeds'] if not (out / f'without_guard_seed{seed}/result.json').exists()]
    completed = len(manifest['seeds']) - len(pending)
    def status(state, **extra):
        pairs = summarize(out)
        write(out / 'status.json', {'status': state, 'updated_at': now(), 'completed_new_runs': completed,
              'total_new_runs': 3, 'completed_pairs': pairs, 'total_pairs': 3, **extra})
        return pairs
    status('running', started_at=now())
    try:
        with ProcessPoolExecutor(max_workers=manifest['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(job, (out, seed)) for seed in pending]
            for future in as_completed(futures):
                future.result()
                completed += 1
                status('running')
        while status('waiting_for_guarded_reference') < 3:
            parent_status = json.loads((SOURCE / 'status.json').read_text())['status']
            if parent_status in ('failed', 'complete'):
                raise RuntimeError('Guarded reference is missing; original pipeline has ended')
            time.sleep(30)
        check(out)
        status('complete')
    except BaseException as error:
        status('failed', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'probe', 'replay', 'run', 'launch', 'summarize'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    os.environ['MPLCONFIGDIR'] = str(out / 'matplotlib')
    if args.phase == 'prepare':
        prepare(out)
    elif args.phase == 'probe':
        probe(out)
    elif args.phase == 'replay':
        replay(out)
    elif args.phase == 'run':
        run(out)
    elif args.phase == 'summarize':
        summarize(out)
    else:
        check(out)
        assert json.loads((out / 'policy_probe.json').read_text())['status'] == 'passed'
        assert json.loads((out / 'reference_replay.json').read_text())['status'] == 'passed'
        with (out / 'pipeline.log').open('x') as stream:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'run', '--out', str(out)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (out / 'pipeline.pid').write_text(str(process.pid) + '\n')
        print(json.dumps({'pid': process.pid, 'out': str(out)}), flush=True)


if __name__ == '__main__':
    main()

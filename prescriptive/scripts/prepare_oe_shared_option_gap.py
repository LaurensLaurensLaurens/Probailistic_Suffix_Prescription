"""Copy OE and verify one model-drawn first gap shared by all suffixes of an option."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import warnings

P = Path(__file__).resolve().parents[1]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PSP_CATALOG"] = str(P / "data/catalog_neue_punkte_zuteilung_random.yaml")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(P))
from helpers.datasets import add_paths
add_paths()

import numpy as np
import pandas as pd
import torch
from interventions import option_token
from prescriptive.estimators.psp_model import ACTIVITY, GAP_CLASS, TIME_COLUMNS, PSPModel, gap_class_names
from prepare_d_option_encoder import weights_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    source = P / "models/psp_model_durchlaufzeit_real_alle_s60_token_oe.pkl"
    target = P / "models/psp_model_durchlaufzeit_real_alle_s60_token_oe_shared_gap.pkl"
    if target.exists() or (out / "run_manifest.json").exists():
        raise FileExistsError("Model or completed preparation already exists")
    out.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    reference = P / "scripts/results_100k/token_oe_suffixe30_vergleich_c_seed0_20261113"
    reference_manifest = json.loads((reference / "run_manifest.json").read_text())
    changed_sources = {str(P / "estimators/psp_model.py"), str(P / "scripts/run_token_oe_ue_seed.py")}
    unchanged_checks = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() == value
                        for name, value in reference_manifest["sha256"].items() if name not in changed_sources}
    assert all(unchanged_checks.values()), unchanged_checks
    torch.set_num_threads(1)
    model = PSPModel.load(source)
    model.set_threads(1)
    assert model.gap_tokens() and not model.gap_classes() and model.option_in_encoder()
    assert not model.params.get("shared_option_gap", False)
    assert getattr(model, "transition_rules", None) is None
    assert getattr(model, "option_gap_distributions", None) is None
    original_params = dict(model.params)
    original_weights = weights_hash(model)
    model.params = {**model.params, "shared_option_gap": True}
    fixture = P / "scripts/results_100k/token_oe_suffixe30_paired200_seed0/failed_prefix.pkl"
    probe = pickle.loads(fixture.read_bytes())
    point = model.points[probe["point"].name]
    events, options, now = probe["events"], probe["options"], probe["now"]
    original_events = pickle.dumps(events)
    base = model._encode(events)
    base_cats, base_nums, _, _ = model._batch([base[:4]], 30)
    ev = model._evaluator()
    cat_at = {name: i for i, name in enumerate(ev.prefix_cat_attributes)}
    num_at = {name: i for i, name in enumerate(ev.prefix_num_attributes)}
    scalers = model.encoder_decoder.continuous_encoders
    calls = []
    original_draw = model.draw_cases

    def observe(encoded, n, **kwargs):
        entry = {"n": n, "first": kwargs.get("first")}
        if kwargs.get("last") is not None:
            cats, nums, _, _ = model._batch(encoded, n)
            for at, value in kwargs["last"]["cats"].items():
                cats[at][:, -1] = torch.as_tensor(value, dtype=cats[at].dtype)
            for at, value in kwargs["last"]["nums"].items():
                nums[at][:, -1] = value.reshape(-1)
            for actual, expected in zip(cats, base_cats):
                assert torch.equal(actual[:, :-1], expected[:, 1:])
            for actual, expected in zip(nums, base_nums):
                assert torch.allclose(actual[:, :-1], expected[:, 1:], rtol=1e-6, atol=1e-6)
            assert all(torch.all(value == value[0:1]) for value in cats + nums)
            entry["cats"], entry["nums"] = cats, nums
        entry["result"] = original_draw(encoded, n, **kwargs)
        calls.append(entry)
        return entry["result"]

    model.draw_cases = observe
    torch.manual_seed(123)
    started = time.time()
    try:
        drawn, elapsed = model.option_suffixes_many(point, [events], options, now, 30)
    finally:
        del model.draw_cases
    probe_seconds = time.time() - started
    assert [call["n"] for call in calls] == [1, 30, 1, 30]
    assert elapsed == [base[4]] and pickle.dumps(events) == original_events
    details = []
    for i, option in enumerate(options):
        first, rest = calls[2*i:2*i+2]
        assert first["first"] == [option_token(option)]
        gap = first["result"][0][0][0][2]
        assert np.isfinite(gap) and gap >= 0
        suffixes = drawn[i][0]
        assert len(suffixes) == 30
        for k, suffix in enumerate(suffixes):
            assert suffix[0] == (option_token(option), base[4] + gap, gap)
            assert suffix[1:] == rest["result"][0][k]
        actual_gap = model._decode(scalers[TIME_COLUMNS[1]], rest["nums"][num_at[TIME_COLUMNS[1]]][:, -1])
        actual_elapsed = model._decode(scalers[TIME_COLUMNS[0]], rest["nums"][num_at[TIME_COLUMNS[0]]][:, -1])
        assert np.allclose(actual_gap, gap, rtol=1e-5, atol=0.1)
        assert np.allclose(actual_elapsed, base[4] + gap, rtol=1e-5, atol=0.1)
        gap_name = str(gap_class_names([gap])[0])
        for column, name in [(GAP_CLASS, gap_name), (ACTIVITY, option_token(option) + "|" + gap_name)]:
            encoder = model.encoder_decoder.categorical_encoders[column]
            expected_id = int(encoder.transform(np.array([[name]], dtype=object)).reshape(-1)[0]) + 1
            assert torch.all(rest["cats"][cat_at[column]][:, -1] == expected_id)
        stamp = pd.Timestamp(now, unit="s") + pd.to_timedelta(gap, unit="s")
        for column, expected in [(TIME_COLUMNS[2], stamp.weekday()), (TIME_COLUMNS[3], stamp.hour*3600 + stamp.minute*60 + stamp.second)]:
            actual = model._decode(scalers[column], rest["nums"][num_at[column]][:, -1])
            assert np.allclose(actual, expected, atol=0.01)
        details.append({"option": str(option), "gap_seconds": gap, "gap_class": gap_name,
                        "suffixes": 30, "identical_extended_prefix": True, "gap_counted_once": True})
    torch.manual_seed(123)
    shared_one = model.option_suffixes_many(point, [events], options, now, 1)
    assert all(shared_one[0][i][0][0][0][2] == details[i]["gap_seconds"] for i in range(len(options)))
    model.params = original_params
    torch.manual_seed(123)
    default_one = model.option_suffixes_many(point, [events], options, now, 1)
    assert default_one == shared_one
    model.params = {**original_params, "shared_option_gap": True}
    assert weights_hash(model) == original_weights
    model.save(target)
    saved = PSPModel.load(target)
    assert saved.params == model.params and weights_hash(saved) == original_weights
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    validation = {"status": "passed", "case": probe["case"], "point": point.name, "probe_seconds": probe_seconds,
                  "sampling_call_sizes": [call["n"] for call in calls], "options": details,
                  "original_model_unchanged": True, "weights_unchanged": True, "observed_history_unchanged": True,
                  "shared_gap_independent_of_suffix_count": True, "one_suffix_equals_default_oe_exactly": True}
    (out / "sampling_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    paths = [source, target, Path(__file__), P/"estimators/psp_model.py", P/"scripts/run_token_oe_ue_seed.py",
             P/"data/catalog_neue_punkte_zuteilung_random.yaml"]
    manifest = {"prepared_at": datetime.now().astimezone().isoformat(timespec="seconds"), "seed": 0,
                "samples_per_option": 30, "scored_cases": 200, "warmup": 700, "cooldown": 1300, "arrivals": 2200,
                "allocation": "random", "transition_rules": False, "source_model": str(source), "model": str(target),
                "source_model_params": original_params, "model_params": saved.params,
                "only_config_change": {"shared_option_gap": [False, True]},
                "first_gap_source": "OE decoder, option-constrained joint event/time-class token and truncated Gaussian time",
                "draws_per_case_and_option": 1, "shared_across_all_30_suffixes": True, "common_random_numbers": True,
                "reference_model_and_catalog_unchanged": unchanged_checks,
                "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"model": str(target), "validation": validation}, ensure_ascii=False))


if __name__ == "__main__":
    main()

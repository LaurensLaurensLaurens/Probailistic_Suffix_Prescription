"""Copy (d) with one model-drawn option gap per suffix; validate real inputs."""
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


def weights_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.net.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = P / "models/psp_model_durchlaufzeit_real_alle_s60_klassen.pkl"
    target = P / "models/psp_model_durchlaufzeit_real_alle_s60_klassen_option_encoder.pkl"
    if target.exists():
        raise FileExistsError(target)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    reference = P / "scripts/results_100k/token_oe_suffixe30_vergleich_c_seed0_20261113"
    reference_manifest = json.loads((reference / "run_manifest.json").read_text())
    reference_checks = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
                        for name, digest in reference_manifest["sha256"].items()}
    assert all(reference_checks.values()), reference_checks

    torch.set_num_threads(1)
    model = PSPModel.load(source)
    model.set_threads(1)
    assert model.gap_classes() and not model.gap_tokens()
    assert model.params.get("option_as") is None and model.params.get("time_from") == "gaps"
    assert getattr(model, "transition_rules", None) is None
    assert getattr(model, "option_gap_distributions", None) is None
    original_params = dict(model.params)
    weight_hash = weights_hash(model)
    model.params = {**model.params, "option_as": "encoder"}

    fixture = P / "scripts/results_100k/token_oe_suffixe30_paired200_seed0/failed_prefix.pkl"
    probe = pickle.loads(fixture.read_bytes())
    original_events = pickle.dumps(probe["events"])
    events, options, now = probe["events"], probe["options"], probe["now"]
    point = model.points[probe["point"].name]
    base = model._encode(events)
    base_cats, base_nums, _, _ = model._batch([base[:4]], 30)
    ev = model._evaluator()
    cat_at = {name: i for i, name in enumerate(ev.prefix_cat_attributes)}
    num_at = {name: i for i, name in enumerate(ev.prefix_num_attributes)}
    scalers = model.encoder_decoder.continuous_encoders
    calls = []
    original_draw = model.draw_cases

    def observe(encoded, n, **kwargs):
        entry = {"n": n, "first": kwargs.get("first"), "last": kwargs.get("last")}
        if kwargs.get("last") is not None:
            cats, nums, _, _ = model._batch(encoded, n)
            for position, value in kwargs["last"].get("cats", {}).items():
                cats[position][:, -1] = torch.as_tensor(value, dtype=cats[position].dtype)
            for position, value in kwargs["last"].get("nums", {}).items():
                nums[position][:, -1] = torch.as_tensor(value, dtype=nums[position].dtype).reshape(-1)
            # Fixed-width tensors are left-padded. Appending the option shifts
            # the same observed history left by one position in that window.
            for actual, expected in zip(cats, base_cats):
                assert torch.equal(actual[:, :-1], expected[:, 1:]), "Observed categorical history changed"
            for actual, expected in zip(nums, base_nums):
                assert torch.allclose(actual[:, :-1], expected[:, 1:], rtol=1e-6, atol=1e-6), "Observed time history changed"
            entry["actual_cats"], entry["actual_nums"] = cats, nums
        result = original_draw(encoded, n, **kwargs)
        entry["result"] = result
        calls.append(entry)
        return result

    model.draw_cases = observe
    torch.manual_seed(123)
    started = time.time()
    try:
        drawn, elapsed = model.option_suffixes_many(point, [events], options, now, 30)
    finally:
        del model.draw_cases
    probe_seconds = time.time() - started
    assert pickle.dumps(events) == original_events
    assert elapsed == [base[4]] and len(calls) == 4
    details = []
    for index, option in enumerate(options):
        first, rest = calls[2 * index:2 * index + 2]
        assert first["first"] == [option_token(option)]
        assert rest["last"] is not None and len(drawn[index][0]) == 30
        gaps = np.array([suffix[0][2] for suffix in first["result"][0]])
        assert np.isfinite(gaps).all() and (gaps >= 0).all()
        assert len(np.unique(gaps)) > 1, "Expected varying first gaps across suffix draws"
        actual = rest["actual_nums"]
        actual_gaps = model._decode(scalers[TIME_COLUMNS[1]], actual[num_at[TIME_COLUMNS[1]]][:, -1])
        actual_elapsed = model._decode(scalers[TIME_COLUMNS[0]], actual[num_at[TIME_COLUMNS[0]]][:, -1])
        assert np.allclose(actual_gaps, gaps, rtol=1e-5, atol=0.1)
        assert np.allclose(actual_elapsed, base[4] + gaps, rtol=1e-5, atol=0.1)
        names = gap_class_names(gaps)
        encoder = model.encoder_decoder.categorical_encoders[GAP_CLASS]
        expected_ids = encoder.transform(names.reshape(-1, 1)).reshape(-1) + 1
        assert np.array_equal(rest["actual_cats"][cat_at[GAP_CLASS]][:, -1].numpy(), expected_ids)
        stamps = pd.Timestamp(now, unit="s") + pd.to_timedelta(gaps, unit="s")
        weekdays = model._decode(scalers[TIME_COLUMNS[2]], actual[num_at[TIME_COLUMNS[2]]][:, -1])
        daytime = model._decode(scalers[TIME_COLUMNS[3]], actual[num_at[TIME_COLUMNS[3]]][:, -1])
        assert np.allclose(weekdays, np.asarray(stamps.weekday), atol=1e-5)
        expected_daytime = np.asarray(stamps.hour * 3600 + stamps.minute * 60 + stamps.second)
        assert np.allclose(daytime, expected_daytime, atol=0.01)
        for k, suffix in enumerate(drawn[index][0]):
            assert suffix[0][0] == option_token(option)
            assert suffix[0][2] == gaps[k] and suffix[0][1] == base[4] + gaps[k]
            assert suffix[1:] == rest["result"][0][k], "Option gap or future suffix duplicated"
        details.append({"option": str(option), "gaps_seconds": gaps.tolist(), "unique_gaps": len(np.unique(gaps)),
                        "observed_history_identical": True, "encoder_times_and_class_match_draw": True,
                        "option_gap_counted_once": True})
    assert weights_hash(model) == weight_hash
    model.save(target)
    saved = PSPModel.load(target)
    assert saved.params == {**original_params, "option_as": "encoder"}
    assert weights_hash(saved) == weight_hash
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    validation = {"status": "passed", "case": probe["case"], "point": point.name, "samples_per_option": 30,
                  "probe_seconds": probe_seconds, "weights_unchanged": True, "original_model_unchanged": True,
                  "separate_event_and_gap_class_outputs": True, "drawn_option_gap_per_suffix": True,
                  "common_observed_history": True, "options": details,
                  "paired_option_gaps_identical_on_probe": details[0]["gaps_seconds"] == details[1]["gaps_seconds"]}
    (out / "sampling_validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n")
    paths = [source, target, Path(__file__), P / "estimators/psp_model.py", P / "scripts/run_token_oe_ue_seed.py",
             P / "data/catalog_neue_punkte_zuteilung_random.yaml"]
    manifest = {"started_at": datetime.now().astimezone().isoformat(timespec="seconds"), "seed": 0,
                "samples_per_option": 30, "scored_cases": 200, "warmup": 700, "cooldown": 1300,
                "arrivals": 2200, "allocation": "random", "transition_rules": False,
                "source_model": str(source), "model": str(target), "original_model_params": original_params,
                "model_params": saved.params, "only_config_change": {"option_as": [None, "encoder"]},
                "reference": str(reference), "reference_source_hashes_unchanged": reference_checks,
                "time_logic": "same observed prefix; one model-drawn option gap per suffix; encode option with its gap and class, then draw continuation",
                "paired_random_numbers": True,
                "note": "With separate event/class heads, pre-option class and Gaussian predictions do not depend on the forced candidate. Common random numbers may therefore give matching first gaps across candidates.",
                "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({"model": str(target), "out": str(out), "validation": validation["status"],
                      "probe_seconds": probe_seconds, "unique_gaps_per_option": [row["unique_gaps"] for row in details],
                      "paired_option_gaps_identical": validation["paired_option_gaps_identical_on_probe"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

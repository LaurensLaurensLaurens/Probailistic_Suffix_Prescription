"""Copy (c) and estimate option-specific Gaussian gap parameters from outer training cases."""
import json
import os
from pathlib import Path
import sys

P = Path(__file__).resolve().parents[1]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PSP_CATALOG"] = str(P / "data/catalog_neue_punkte_zuteilung_random.yaml")
sys.path.insert(0, str(P))
import numpy as np
from helpers.datasets import ACTIVITY, CASE, LIFECYCLE, TIMESTAMP, add_paths, load_dataset


def main():
    add_paths()
    from preprocessing import preprocess
    from prescriptive.estimators.psp_model import PSPModel
    from interventions import option_token
    source = P / "models/psp_model_durchlaufzeit_real_alle_s60_normal.pkl"
    target = P / "models/psp_model_durchlaufzeit_real_alle_s60_normal_shared_gap.pkl"
    if target.exists():
        raise FileExistsError(target)
    model = PSPModel.load(str(source))
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw)
    del raw
    train = log.loc[~log[CASE].astype(str).isin(set(map(str, model.test_cases)))].sort_values([CASE, TIMESTAMP], kind="stable")
    tokens = train[ACTIVITY].astype(str) + "|" + train[LIFECYCLE].astype(str)
    following = tokens.groupby(train[CASE]).shift(-1)
    gaps = (train[TIMESTAMP].groupby(train[CASE]).shift(-1) - train[TIMESTAMP]).dt.total_seconds()
    distributions = {}
    for name, point in model.points.items():
        if point.kind != "control_flow":
            continue
        at = tokens == point.activity + "|" + point.lifecycle
        distributions[name] = {}
        for option in point.options:
            observed = gaps.loc[at & (following == option_token(option))].dropna().to_numpy(dtype=float)
            if not len(observed) or not np.isfinite(observed).all() or (observed < 0).any():
                raise ValueError(f"Keine gültigen Trainingsabstände für {name}: {option}")
            distributions[name][option] = {"mean_seconds": float(observed.mean()),
                                            "std_seconds": float(observed.std(ddof=0)),
                                            "n": len(observed), "median_seconds": float(np.median(observed)),
                                            "zero_share": float(np.mean(observed == 0))}
    model.option_gap_distributions = distributions
    model.option_gap_info = {"source": "observed next-event gaps, all outer training cases excluding model.test_cases",
                              "source_cases": int(train[CASE].nunique()), "excluded_test_cases": len(model.test_cases),
                              "parameters": "empirical mean and population standard deviation (Gaussian moment fit)",
                              "sampling": "independent draw per option and decision; lower-truncated normal at 0 seconds",
                              "draws_per_case_and_option": 1,
                              "note": "truncation changes the distribution mean relative to the raw Gaussian location"}
    model.save(str(target))
    report = {"source_model": str(source), "model_copy": str(target),
              "distributions": distributions, "metadata": model.option_gap_info}
    path = P / "data/PSP_c_shared_option_gap_distributions.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

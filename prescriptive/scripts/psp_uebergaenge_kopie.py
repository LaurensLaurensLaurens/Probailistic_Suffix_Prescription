"""
Kopie eines PSP-Modells mit Übergangsregeln wie im Simulator (2026-11-13; estimators/psp_model.py, Kommentar
"Übergangsregeln"): aus den Trainingsfällen des echten Logs (alle Fälle außer model.test_cases) je Paar direkt
aufeinanderfolgender Events (Aktivität|Lifecycle) "erlaubt" und je Event, ob ein Fall danach endet. Die Kopie heißt
models/psp_model_durchlaufzeit_<tag>_ue.pkl; das Original bleibt unverändert.

Aufruf (im Ordner process-simulation-engine):
    .venv/bin/python -u ../prescriptive/scripts/psp_uebergaenge_kopie.py --tag real_alle_s60_normal real_alle_s60_token_oe
"""
import argparse
import os
import sys

PRESCRIPTIVE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PRESCRIPTIVE)

import numpy as np

from helpers.datasets import ACTIVITY, CASE, LIFECYCLE, TIMESTAMP, add_paths


def rules_from(log, lifecycle_tokens=True):
    """Erlaubte Paare (a, b) und Events, nach denen ein Fall endet, aus einem Log (sortiert nach Fall und Zeit)."""
    tokens = (log[ACTIVITY].astype(str) + ("|" + log[LIFECYCLE].astype(str) if lifecycle_tokens else "")).to_numpy()
    case = log[CASE].to_numpy()
    same = case[1:] == case[:-1]
    pairs = sorted(set(zip(tokens[:-1][same], tokens[1:][same])))
    ends = sorted(set(tokens[np.r_[~same, True]]))
    return {"pairs": pairs, "ends": ends}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", nargs="+", required=True, help="Modelle models/psp_model_durchlaufzeit_<tag>.pkl")
    args = parser.parse_args()
    add_paths()
    from helpers.datasets import load_dataset
    from helpers.training import MODELS
    from preprocessing import preprocess
    from prescriptive.estimators.psp_model import PSPModel
    raw, _, _ = load_dataset("bpic17")
    log = preprocess("bpic17", raw).sort_values([CASE, TIMESTAMP], kind="stable")
    del raw
    for tag in args.tag:
        path = os.path.join(MODELS, f"psp_model_durchlaufzeit_{tag}_ue.pkl")
        if os.path.exists(path):
            raise FileExistsError(f"Modellkopie existiert bereits: {path}")
        model = PSPModel.load(os.path.join(MODELS, f"psp_model_durchlaufzeit_{tag}.pkl"))
        train = log[~log[CASE].astype(str).isin(set(map(str, model.test_cases)))]
        model.transition_rules = rules_from(train, getattr(model, "lifecycle_tokens", True))
        model.transition_rules["quelle"] = f"{train[CASE].nunique():,} Trainingsfälle des echten Logs (ohne model.test_cases)"
        model.save(path)
        print(f"{tag}: {len(model.transition_rules['pairs'])} erlaubte Übergänge, Ende nach {len(model.transition_rules['ends'])} Events -> {path}", flush=True)


if __name__ == "__main__":
    main()

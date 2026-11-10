"""
Validierung des Screenings im Simulator (screening/validation.py): trifft das Screening auf einem beobachteten Log die
wahre Wirkung, und gibt der Simulator wieder, was das Screening im echten Log findet?

Aufruf (im Ordner process-simulation-engine; Wahlpunkte aus dem Katalog bzw. PSP_CATALOG):
    .venv/bin/python -u ../prescriptive/scripts/validate_screening.py --out ../prescriptive/scripts/results_100k/screening_validierung \
        --n-cases 20000 --workers 4
Ergebnis: <out>/validierung.csv (je Kandidat), <out>/validierung_mitarbeiter.csv (je Mitarbeiter).
"""
import argparse
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
PRESCRIPTIVE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PRESCRIPTIVE)

from helpers.datasets import add_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="bpic17")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-cases", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidates", nargs="*", default=None, help="nur diese Kandidaten (Namen aus dem Katalog)")
    args = parser.parse_args()
    add_paths()
    import pandas as pd
    from helpers.datasets import load_dataset
    from preprocessing import preprocess
    from simulation.dataset_simulator import Simulator
    from screening.validation import validate

    started = time.time()
    os.makedirs(args.out, exist_ok=True)
    raw, _, _ = load_dataset(args.dataset)
    log = preprocess(args.dataset, raw)
    del raw
    simulator = Simulator(args.dataset, log, None)
    print(simulator, flush=True)
    table, details = validate(simulator, log, n_cases=args.n_cases, seed=args.seed, workers=args.workers, candidates=args.candidates)
    table.to_csv(os.path.join(args.out, "validierung.csv"))
    details.to_csv(os.path.join(args.out, "validierung_mitarbeiter.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 100):
        print(table.round(3).T.to_string())
    print(f"######## Fertig in {time.time() - started:.0f} s; Ergebnisse in {args.out}", flush=True)


if __name__ == "__main__":
    main()

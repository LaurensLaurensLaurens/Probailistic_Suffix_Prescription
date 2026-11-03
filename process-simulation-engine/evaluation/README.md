# Evaluation and log comparison

## Single comparison (two logs)

Use **log_comparison.ipynb** or **log_comparison_only_with_start_complete.ipynb** to compare two event logs side-by-side (e.g. with vs without terminated resources). Set `LOG_A_PATH` / `LOG_B_PATH` (or the dirs) in the first cells and run.

## Distribution of changes (many runs)

To get a **distribution** of metric changes instead of a single point:

1. **Generate many paired runs** (same arrivals, with vs without excluded resources):

   ```bash
   python -m integration.run_termination_comparison --n-runs 10 --exclude-resources User_128,User_129 --output-csv evaluation/termination_comparison_runs.csv
   ```

   Options:
   - `--n-runs` — number of run pairs (default: 10)
   - `--exclude-resources` — comma-separated resource names to exclude (required)
   - `--num-cases` — cases per run (default: full log)
   - `--output-csv` — path for the comparison CSV (default: `evaluation/termination_comparison_runs.csv`)
   - `--output-dir` — base directory for run subdirs (`run_0/with_termination`, `run_0/without_termination`, …)
   - `--event-log` — path to source event log
   - `--seed` — base random seed (run `i` uses `seed + i` so arrivals match within each pair)

2. **Open a log_comparison notebook** and run the **“5 — Distribution of changes (many runs)”** section. It loads the CSV and shows:
   - Summary table (mean, std, min, max, median of each metric’s difference)
   - Histograms of the difference (without − with termination) per metric

## Shared metrics

**evaluation/log_metrics.py** provides `load_log()` and `compute_all_metrics()` used by the notebooks and by `integration/run_termination_comparison.py`.

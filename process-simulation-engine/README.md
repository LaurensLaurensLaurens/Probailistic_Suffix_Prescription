# Process Simulation Engine

Data-driven discrete-event simulation (DES) engine for the BPIC 2017 loan application process.
Built for TUM Business Process Prediction, Simulation and Optimization Practical Course. Python 3.12+.

## Reproduction

### 1. Clone and set up environment

```bash
git clone https://github.com/BPPSO-Simulation-engine-Optimization/process-simulation-engine.git && cd process-simulation-engine
conda env create -f environment.yml
conda activate pse_env
```

### 2. Install PM4Py

The patched PM4Py build **must** be installed separately (not included in `environment.yml`):

```bash
pip install pm4py-release/
```

### 3. Download event log

Download the BPIC 2017 event log from [4TU.ResearchData](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884/1) and place it as:

```
eventlog/eventlog.xes.gz
```

### 4. Download trained models

Models (~4.4 GB) are not tracked in git. Download from [Google Drive](https://drive.google.com/drive/folders/1NpMEYyLnDi6gp3OdN7sf_f6kCCRPOPog?usp=sharing) and extract into the `models/` directory.

Resource models (`resources/resource_permissions/`, `resources/resource_availabilities/`) are tracked in git.

### 5. Run simulation

```bash
conda activate pse_env
python -m integration.test_integration \
 	--next-activity lifecycle_dual_start_complete_baseline \
	--arrivals advanced \
	--processing advanced \
	--attributes advanced \
	--resource-strategy random \
	--resource-allocation-mode greedy \
	--num-cases 31509 \
	--event-log eventlog/eventlog.xes.gz
```

Output: `integration/output/simulated_log.csv`

## CLI Reference

**Prediction:**

| Flag | Choices | Default |
|------|---------|---------|
| `--next-activity` | `lstm`, `process_transformer`, `lifecycle_dual_full_baseline`, `lifecycle_dual_start_complete_baseline` | `lstm` |
| `--arrivals` | `basic`, `advanced` | `basic` |
| `--processing` | `basic`, `advanced` | `basic` |
| `--attributes` | `basic`, `advanced` | `basic` |

**Resources:**

| Flag | Choices | Default |
|------|---------|---------|
| `--resource-strategy` | `random`, `round_robin`, `shortest_queue` | `random` |
| `--resource-allocation-mode` | `greedy`, `batch`, `drl`, `pmsp` | `greedy` |

**Process Transformer:**

| Flag | Description | Default |
|------|-------------|---------|
| `--pt-lifecycle-mode` | `native` or `gt_activity_gated` | `native` |
| `--pt-max-duration-days` | Max duration cap (days) | `30` |
| `--temperature` | Sampling temperature (PT only) | `1.5` |

**General:**

| Flag | Description | Default |
|------|-------------|---------|
| `--num-cases` | Cases to simulate | same as original log |
| `--event-log` | Path to event log | `Dataset/BPI Challenge 2017.xes` |
| `--output-dir` | Output directory | `integration/output` |
| `--verbose` | Enable verbose logging | off |
| `--profile` | Performance profiling | off |

**PMSP/DRL (advanced allocation):**

| Flag | Description | Default |
|------|-------------|---------|
| `--pmsp-dummy-delta` | PMSP dummy cost multiplier | `1.0` |
| `--pmsp-solver-time-limit` | CP-SAT solver time limit (s) | `2.0` |
| `--pmsp-prediction-batch-size` | Max predictions per task (0=unlimited) | `0` |
| `--drl-model-path` | Path to trained DRL model | `models/drl_allocation/drl_allocation_model` |
| `--processing-model-path` | Processing time model base path | auto |

## Architecture

| Directory | Purpose |
|-----------|---------|
| `simulation/` | Core DES engine (event queue, case manager, clock, log export) |
| `next_activity_prediction/` | LSTM, Process Transformer v2, DFG fallback |
| `processing_time_prediction/` | Distribution, RandomForest, LSTM methods |
| `Instance Spawn Rate/` | Case arrival prediction & case attribute prediction |
| `resources/` | Three-tier allocation: permissions, availability, resource pool |
| `integration/` | CLI entry point (`test_integration.py`), config, benchmarking |
| `evaluation/` | Batch evaluation scripts |
| `process_model/` | BPMN process model |
| `pm4py-release/` | Vendored/patched PM4Py build |

## Evaluation and Testing

**Benchmarking** compares simulated vs. ground truth logs (statistics, DFG, variants, throughput, resources, next-activity quality):

```bash
# Runs automatically after simulation; results in integration/output/
```

**Batch evaluation:**

```bash
python evaluation/batch_evaluate.py
```

**Tests** (no pytest framework, run individually):

```bash
python resources/tests/test_resource_permissions.py
python resources/tests/test_advanced_permissions.py
python resources/tests/test_allocation_integration.py
python resources/tests/test_bpic17_integration.py
python resources/tests/test_benchmark.py
python resources/tests/test_caching.py
```

## Evaluation Data

Simulated event logs for all allocation strategies (5 replications each, 10,000 cases per run) and the full-scale 31,509-case validation run are available on Google Drive:

https://drive.google.com/file/d/1LgNv8BvfSBVb9VrfepryuIHXilVG9dTb/view?usp=sharing

## Dataset

**BPI Challenge 2017** - Loan application process from a Dutch financial institution (2016-2017).
31,509 cases, 1.2M events, 26 activities.
Source: [4TU.ResearchData](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884/1)

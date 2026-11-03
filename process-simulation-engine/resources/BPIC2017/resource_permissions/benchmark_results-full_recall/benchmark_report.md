# OrdinoR Benchmark Report

Generated: 2025-12-13 10:45
Log: 475306 events, 24 activities
Configuration: OrdinoR (FullRecall profiling)
Params: n_resource_clusters=10, mode=full_recall

## Coverage

| Approach | Covered | Total | Coverage |
|----------|---------|-------|----------|
| Basic | 24 | 24 | 100.0% |
| OrdinoR | 24 | 24 | 100.0% |

✓ OrdinoR coverage meets 90% threshold.

## Precision & Recall (Activity-Based)

| Metric | Value |
|--------|-------|
| Avg Precision | 70.6% |
| Avg Recall | 100.0% |
| F1 Score | 82.8% |

✓ Precision ≥60% - most eligible resources have actually performed the activity.

## Group Quality

- **Groups discovered:** 10
- **Avg group size:** 14.4 (σ=20.8)

| Group | Size |
|-------|------|
| 0 | 1 |
| 1 | 9 |
| 2 | 75 |
| 3 | 8 |
| 4 | 21 |
| 5 | 7 |
| 6 | 9 |
| 7 | 6 |
| 8 | 4 |
| 9 | 4 |


## Generalization (Holdout Test)

| System | Test Recall |
|--------|-------------|
| Basic | 97.0% |
| OrdinoR | 100.0% |

✓ OrdinoR outperforms basic on unseen cases.

## Recommendation

**Use OrdinoRResourcePermissions** for simulation. Higher recall and acceptable precision.

# OrdinoR Benchmark Report

Generated: 2025-12-13 09:33
Log: 475306 events, 24 activities
Configuration: OrdinoR (Case Type + AHC + OverallScore)
Params: n_resource_clusters=10

## Coverage

| Approach | Covered | Total | Coverage |
|----------|---------|-------|----------|
| Basic | 24 | 24 | 100.0% |
| OrdinoR | 23 | 24 | 95.8% |

✓ OrdinoR coverage meets 90% threshold.

## Precision & Recall (Activity-Based)

| Metric | Value |
|--------|-------|
| Avg Precision | 93.4% |
| Avg Recall | 49.3% |
| F1 Score | 64.6% |

✓ Precision ≥60% - most eligible resources have actually performed the activity.

## Group Quality

- **Groups discovered:** 10
- **Avg group size:** 2.0 (σ=0.0)

| Group | Size |
|-------|------|
| 0 | 2 |
| 1 | 2 |
| 2 | 2 |
| 3 | 2 |
| 4 | 2 |
| 5 | 2 |
| 6 | 2 |
| 7 | 2 |
| 8 | 2 |
| 9 | 2 |


## Generalization (Holdout Test)

| System | Test Recall |
|--------|-------------|
| Basic | 97.0% |
| OrdinoR | 56.6% |

✗ OrdinoR underperforms basic on unseen cases.

## Recommendation

**Consider hybrid approach.** Use basic for high-frequency activities, OrdinoR for rare ones.

## Problem Summary
-> one cluster of specialists gets the capability assigned, and the large "everyone else" cluster is too noisy to pass the threshold | not suitable for simulation; but might be interesting when looking at optimization

"We used OrdinoR for organizational model discovery with FullRecall profiling to ensure simulation completeness"
"OverallScore profiling is designed for precision optimization in conformance checking scenarios, but creates artificial constraints for simulation; FullRecall is the appropriate choice when the goal is capturing all valid resource-activity relationships"

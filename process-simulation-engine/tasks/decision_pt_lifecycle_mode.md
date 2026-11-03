# Decision: `pt_lifecycle_mode` (PT-Scoped Lifecycle Logging)

## Decision
Introduce a PT-only lifecycle logging mode:
- `native` (default): keep existing lifecycle behavior.
- `gt_activity_gated`: for Process Transformer only, synthesize `start` for GT start-capable activities and force completion logs to `complete`.

## Why
- Process Transformer currently predicts activity + duration, not lifecycle.
- Ground-truth logs include `start` only for specific activities, not globally.
- We need GT-aligned `start/complete` output without redesigning the full lifecycle engine.

## Scope
- Strictly Process Transformer path.
- Non-PT predictors (including lifecycle-dual) remain predictor-native and unchanged.

## Guardrail
- If `pt_lifecycle_mode=gt_activity_gated` is used with a non-PT predictor, raise `ValueError` immediately.

## Timestamp Semantics
- Synthetic `start` is emitted at actual execution start (dispatch time), not at prediction or queue-entry time.

## Non-Goals
- No simulation of additional lifecycle states (`schedule`, `suspend`, `resume`, `withdraw`, `ate_abort`) in this change.
- No changes to model artifacts or predictor API contracts.

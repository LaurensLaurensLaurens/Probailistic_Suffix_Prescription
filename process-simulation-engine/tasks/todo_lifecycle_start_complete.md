# PT-Scoped Lifecycle Logging Implementation Plan

## Context
- The Process Transformer path predicts next activity and duration, but not lifecycle labels.
- Ground-truth data contains `start` only for a subset of activities, so synthetic starts must be activity-gated.
- Other predictors (especially lifecycle-dual) already handle lifecycle explicitly and must not be changed.

## Decisions Locked
- Scope is PT-only via `pt_lifecycle_mode`.
- Supported values:
  - `native` (default): current behavior.
  - `gt_activity_gated`: emit synthetic `start` for GT start-capable activities and log completions as `complete`.
- Misconfiguration guard:
  - if `pt_lifecycle_mode=gt_activity_gated` and predictor is not Process Transformer, fail fast with `ValueError`.

## GT Start-Capable Activities
- `W_Assess potential fraud`
- `W_Call after offers`
- `W_Call incomplete files`
- `W_Complete application`
- `W_Handle leads`
- `W_Validate application`

## File-Level Changes

### 1) `integration/config.py`
- [x] Add `pt_lifecycle_mode: Literal["native", "gt_activity_gated"] = "native"`.

### 2) `integration/test_integration.py`
- [x] Add CLI arg `--pt-lifecycle-mode {native,gt_activity_gated}`.
- [x] Add early guard in `main()`:
  - non-PT predictor + `gt_activity_gated` -> raise `ValueError`.
- [x] Map CLI value to `config.pt_lifecycle_mode`.
- [x] Pass `pt_lifecycle_mode` into `DESEngine(...)`.
- [x] Print PT lifecycle mode in configuration summary when PT is selected.

### 3) `simulation/engine.py`
- [x] Extend `DESEngine.__init__` with `pt_lifecycle_mode`.
- [x] Add PT detection helper for explicit type and direct instance wiring.
- [x] Validate `pt_lifecycle_mode` against allowed values.
- [x] Enforce PT-only guard for `gt_activity_gated`.
- [x] Add GT start-capable activity set.
- [x] Add label normalization helper for loop suffixes (`"X 2"` -> `"X"`).
- [x] Add `_should_emit_pt_gt_start(...)`.
- [x] Add `_append_synthetic_start_record(...)` and append synthetic starts:
  - in `_schedule_activity_with_resource(...)` at dispatch time.
  - in `_schedule_activity_without_resource(...)` at immediate start time.
- [x] In `_on_activity_complete(...)`, force logged completion lifecycle to `complete` for PT gated mode.
- [x] Keep all non-PT predictor behavior unchanged.

### 4) `README.md`
- [x] Add concise note documenting:
  - PT-only nature of `--pt-lifecycle-mode`.
  - default `native`.
  - hard-error behavior for non-PT + `gt_activity_gated`.

## Validation Plan
- [x] Static validation: `python -m py_compile` for edited Python files.
- [x] Guardrail check:
  - run integration CLI with non-PT + `--pt-lifecycle-mode gt_activity_gated` and confirm fast failure.
- [ ] Regression check:
  - run with default `--pt-lifecycle-mode native` to confirm unchanged execution path.

## Review Checklist
- [x] No lifecycle changes for lifecycle-dual or other non-PT predictors.
- [x] Synthetic `start` is emitted at execution start, not at prediction time.
- [ ] Queue waits do not produce premature `start`.
- [x] Completion lifecycle is `complete` in PT gated mode.

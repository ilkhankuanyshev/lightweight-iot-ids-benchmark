# Task 3 change record

Base repository: https://github.com/ilkhankuanyshev/lightweight-iot-ids-benchmark

Base commit: `51759722daee772e40310fccc0ce72e9b03dea8c`.
No upstream branch or tag was modified.

## Changed inherited modules

- `latency_worker.py`: buffer actual per-call timestamps, durations and status;
  distinguish warm-up from benchmark; serialize after timing; Windows peak-working-set
  fallback; retain benchmark errors before raising to the parent.
- `measure.py`: pass raw-output destination and provenance context into the worker.
- `benchmark.py`: assign contextual identities and raw destinations to every
  measured model; fail guardrails when required values are missing/non-finite.
- `analyze.py`: suppress hypothesis tests on synthetic smoke inputs.
- `.gitignore`: allowlist compact Task 3 data artifacts while continuing to exclude
  original datasets, result directories, weights, caches and virtual environments.
- `data/README.md`: distinguish the new audit sample from original Task 2 data policy.

## Added modules and artifacts

- `process_data.py`: schema validation, error denominators, non-destructive outlier
  flags, per-unit quantiles and deterministic bootstrap aggregation.
- `visualize.py`: labelled feature-workload and distribution plots; unit-level
  uncertainty; accessible styling; vector PDF and 300-DPI PNG exports.
- `task3.py`: actual smoke collection and saved-log analysis entry point.
- `scripts/execute_notebook.py`: fresh-kernel sequential execution using current Python.
- `scripts/build_report.py`: generate report tables from the saved measurements.
- `tests/test_task3.py`: invalid data, duplicate identities, mixed provenance,
  warm-up exclusion, error accounting, retained spikes and missing guardrails.
- Domain model, raw provenance, executed notebook, reports and tested package lock.

## Deliberate scope decisions

1. Use the existing synthetic fixture; do not claim real CICIoT2023 observations.
2. Measure execution time instead of generating artificial latency distributions.
3. Use 100 RF trees for the baseline, and 100 warm-up / 1,000 timed calls.
4. Preserve Task 2's 15-split/full-data configuration separately from the two-fold
   smoke run. Full training and edge-platform measurement are not performed here.
5. Use feature count as the x-axis. The workload plot is not a load-generator or
   queueing experiment, even though its rubric-compatible filename contains "load".
6. Retain the original statistical routines for future audit, but do not treat the
   overlapping CV folds as unquestionably independent units.

# Reproduction and submission

## Reproduce the saved analysis (no training or data download)

From the repository root, using Python 3.13 on Windows as tested:

```bash
python -m pip install -r requirements-task3-lock.txt
python -m src.task3
python scripts/execute_notebook.py
```

The notebook uses the same modules as the command line. Saved raw data are
immutable inputs; new collection deliberately replaces the packaged sample:

```bash
python -m src.task3 --collect --config configs/task3_smoke.yaml
```

Random seeds reproduce the split and bootstrap resampling. Wall-clock timings
are hardware- and load-dependent and will not be bitwise identical.

The inherited requirements.txt and Dockerfile describe Task 2's Python 3.12
environment. The delivered smoke run was performed on local Python 3.13/Windows,
not that Docker image. requirements-task3-lock.txt records the tested environment;
it contains Windows-only packages and is not a cross-platform lock.

## Full real-data experiment

1. Obtain the CICIoT2023 CSV dataset from its official page, following its terms:
   https://www.unb.ca/cic/datasets/iotdataset-2023.html
2. Run the inherited preparation command:
   `python -m src.prepare_data --raw-dir data/raw --out data/processed/ciciot2023_capped.csv --cap 50000 --seed 42`.
3. Keep configs/config.yaml's full 5x3 CV design, all models and feature counts.
   Explicitly add `data_origin: real_ciciot2023` under measurement only after
   verifying the input dataset provenance. Do not reuse smoke_augment.
4. Train unrestricted with `python -m src.benchmark --config configs/config.yaml --phase train`.
5. Measure the saved run in the documented 1 CPU / 1 GiB container or Raspberry Pi,
   using `python -m src.benchmark --phase measure --run-dir results/<run>`.
   Replace `<run>` with the actual directory. Never merge platforms in one summary.
6. Preserve `metrics.csv`, `env_measure.json`, `config_snapshot.yaml` and `raw/*.jsonl`.
   Concatenate only one experiment's raw files; retain a manifest of all expected
   configurations, including those which crashed before emitting JSONL.
7. Run process_data on that raw log into a separate output directory. Review
   per-class recall/FPR, missing splits and failures before ranking configurations.
   Replace the smoke figures, tables and notebook input only after this audit.
8. The inherited analyze.py implements the Task 2 tests but the repeated-CV folds
   overlap. Review the independence and multiple-endpoint assumptions before
   interpreting p-values as confirmatory; a small p-value is not a substitute for
   independent experimental units. No hypothesis conclusion is made in Task 3.

The 50,000-per-class cap changes prevalence; add an untouched, natural-prevalence
holdout and acquisition/session-disjoint sensitivity analysis before deployment claims.
CPU quotas on x86 do not emulate Raspberry Pi instruction set, caches or memory.

## Git workflow (commands provided; no remote write performed)

Copy these files over a working checkout of the existing project, then review:

```bash
git switch -c task3-results
git status --short
git add README.md docs notebooks src tests scripts configs data reports .gitignore requirements-task3-lock.txt
git diff --cached --stat
git commit -m "Add Task 3 reproducible smoke analysis and domain model"
git tag -a task3-smoke-v1 -m "Measured local smoke timings; synthetic input; no scientific claims"
git push origin task3-results
git push origin task3-smoke-v1
```

Publish the final scientific release only when real results are available. The
archive is not evidence that a remote commit, tag or public release has occurred.

## Five-minute audit

- Confirm data_origin in provenance.json and every JSONL record.
- Restart and run the notebook; inspect sequential execution counts and outputs.
- Check warm-up/error counts, flagged outliers and configuration failures.
- Check both plot formats, units, markers, confidence bands and captions.
- Inspect domain equations and the requirement-to-artifact map in the report.
- Review git status for caches, virtual environments, model weights or large dumps.

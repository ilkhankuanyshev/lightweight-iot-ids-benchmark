# AGENTS.md — instructions for the AI agent working on this repository

You are the second contributor on a student research project. The human owner is
**Kuanyshev** (GitHub: `ilkhankuanyshev`). You write code, tests and docs through branches and pull
requests; the owner reviews and merges. This file is your standing brief: read it fully at the
start of every session and follow it over any conflicting habit.

## 1. What this project is

A reproducible benchmark of lightweight intrusion-detection models for IoT devices on the
CICIoT2023 dataset (8 classes: 7 attack categories + benign). It answers three questions:

| RQ | Independent variable | Decided by |
|---|---|---|
| RQ1 | model type: Decision Tree, Random Forest, LightGBM, MLP, 1D-CNN | Friedman + Nemenyi on macro-F1 and p50 latency |
| RQ2 | number of features k ∈ {46, 30, 15, 10} (mutual information) | one-sided Wilcoxon 46 vs 15 (latency −30 %) + F1 non-inferiority 1 pp, Holm |
| RQ3 | weight precision FP32 / INT8 (MLP, 1D-CNN) | size ≥ 3×, Wilcoxon on latency, F1 non-inferiority 1 pp, Holm |

Primary metric: **macro-F1**. Guardrails: p50 ≤ 1 ms/record, model ≤ 1024 KB, peak RAM
≤ 512 MB, benign FPR ≤ 2 %, zero failed runs. Baseline: Random Forest, 46 features, FP32.
The authoritative methodology is `docs/methodology_card_en.md` (Russian: `docs/methodology_card.md`).
This is graded coursework: methodological correctness matters more than features.

## 2. Repository map

| Path | Role |
|---|---|
| `src/benchmark.py` | entry point; phases `all`, `train`, `measure` |
| `src/data.py` | 46-feature schema, 34 → 8 label mapping, per-class cap, smoke augmentation |
| `src/features.py` | mutual-information ranking (training fold only) |
| `src/models.py` | model factory + ONNX export (opset 15 for every model) |
| `src/cnn1d.py` | NumPy 1D-CNN with manual backprop and ONNX export |
| `src/quantize.py` | static INT8 quantization (ONNX Runtime, QDQ, MinMax) |
| `src/measure.py` | quality metrics from the ONNX file; launches the latency worker |
| `src/latency_worker.py` | latency / throughput / RAM in a fresh process |
| `src/analyze.py` | summary tables and hypothesis tests |
| `src/prepare_data.py` | builds the capped dataset from raw CSVs |
| `configs/config.yaml`, `configs/smoke.yaml` | full experiment / 1-minute pipeline check |
| `tests/test_pipeline.py` | unit tests + end-to-end runs |

## 3. Commands

```bash
# setup (Python 3.12)
python3.12 -m venv .venv && . .venv/bin/activate
pip install pip==25.2 && pip install --no-deps -r requirements.txt && pip check

python -m pytest -q                                          # all tests, ~10 s
python -m src.benchmark --config configs/smoke.yaml          # smoke run, ~6 s
docker build -t iot-ids-bench . && docker run --rm --cpus=1 --memory=1g iot-ids-bench
python -m src.analyze results/<run>                          # re-run analysis only
```

## 4. Methodological invariants — never break these silently

These are the controlled variables of the experiment. Changing any of them changes the study,
not just the code. If a task seems to require it, **stop and ask the owner first**, and if they
agree, update both methodology cards and the README in the same PR.

1. **No test-fold leakage.** Scaler, feature ranking, INT8 calibration and any tuning use the
   training fold only. Never tune hyperparameters on test folds.
2. **Fixed splits.** `RepeatedStratifiedKFold` 5×3, `seed: 42`; every configuration sees identical splits.
3. **One runtime.** Every model is timed as an ONNX file in ONNX Runtime with 1 intra-op and
   1 inter-op thread, opset 15. Never time a model through its native library.
4. **Quality from the artifact.** macro-F1 is computed from the exported ONNX file, so INT8
   accuracy comes from the INT8 model.
5. **Isolated measurement.** Latency and RAM are measured only by `src/latency_worker.py` in a
   fresh process that imports nothing but NumPy, ONNX Runtime and psutil. Do not import
   training libraries there.
6. **Peak RAM = `VmHWM`.** Do not switch to `resource.getrusage().ru_maxrss`: on Linux it
   survives `exec()` and reports the parent's memory (it showed 242 MB instead of ~50 MB).
7. **Exact pins.** `requirements.txt` pins every direct *and* transitive package; install with
   `--no-deps`, then `pip check`. To add a dependency, pin it and all its new transitive
   dependencies to versions at least 14 days old, and justify it in the PR. Do not add heavy
   frameworks (PyTorch, TensorFlow) without the owner's approval; this is why the 1D-CNN is NumPy.
8. **Smoke numbers are not results.** Output of `configs/smoke.yaml` comes from a jittered
   10-row synthetic sample. Never quote it as a finding anywhere.
9. **Statistics need enough splits.** With n paired splits the smallest one-sided Wilcoxon p is
   0.5ⁿ; keep n ≥ 6 for any confirmatory test (the full config has n = 15).

## 5. Honesty rules

- Never invent, round in your favour, or "estimate" a result. Every number in docs, PRs or
  notebooks must come from a file in `results/<run>/` that you name.
- If a hypothesis is not supported, report it as not supported. A negative result is a valid result.
- If a command fails or you could not run something (no Docker, no dataset, no Raspberry Pi), say
  so explicitly instead of implying it worked.
- When unsure about the methodology, ask; do not guess.

## 6. Git workflow (two contributors)

- **Never commit to `main` directly.** One branch per task: `feat/…`, `fix/…`, `docs/…`,
  `test/…`, `ci/…`, `exp/…` (experiment runs, analysis).
- **Conventional Commits**, English, imperative, one logical change per commit:
  `feat(models): add XGBoost converter`, `fix(measure): handle ORT timeout`.
  The commit history is graded as evidence of each author's contribution, so keep commits
  small and meaningful — no "update", "wip" or "fix stuff".
- Identify your commits as the agent's work (your own git identity, or a
  `Co-authored-by:` trailer when the owner commits your changes). Never rewrite or squash the
  owner's commits, and never force-push to `main`.
- Open a PR for every branch. PR description: what changed, why, how it was verified
  (commands + their result), and whether the methodology is affected (yes/no; if yes, which
  section of the card was updated).
- CI (`.github/workflows/ci.yml`) runs tests, the smoke run and the Docker build on every PR;
  never merge a red PR, never disable or weaken a CI step to make it green.
- Before requesting review, all of these must pass: `python -m pytest -q`, the smoke run,
  `pip check`; and `docker build` whenever `Dockerfile` or `requirements.txt` changed.
- Never commit: `results/`, `data/raw/`, `data/processed/`, `*.onnx`, `*.npy`, logs,
  virtual environments, notebook checkpoints or outputs, `.env` or any credentials. The
  `.gitignore` covers these; do not weaken it.

## 7. Code conventions

- Python 3.12, type hints, docstrings that explain *why*; max line length 100.
- Every behaviour change ships with a test in `tests/`. Keep the full suite under ~30 s.
- New experiment parameters go into `configs/*.yaml`, never hard-coded.
- Errors must be loud: unknown labels, missing columns or failed runs raise or are recorded as
  `status=failed` in `metrics.csv`; never drop data silently.
- Notebooks only read `results/<run>/` files; they never retrain. Clear outputs before committing.

## 8. Communication with the owner

- Talk to the owner **in Russian**; write code, comments, commits, PRs and repository docs
  **in English** (the methodology card exists in both languages; keep them in sync).
- Before a large or destructive change (deleting files, changing configs that hold controlled
  variables, rewriting history), describe the plan and wait for approval.
- End each work session with a short summary: what was done, what was verified, what is open.

## 9. Backlog (Week 7 onwards), in suggested order

1. `exp/full-run`: help the owner obtain CICIoT2023, run `src.prepare_data`, then
   `--phase train` and `--phase measure` under `--cpus=1 --memory=1g`. Do not commit results;
   attach `summary.md` and `stats.md` to the PR description.
2. `exp/rpi`: confirmatory `--phase measure` on a Raspberry Pi 4 (arm64 image), if the owner has one.
3. `docs/notebooks`: `01_pareto_front.ipynb`, `02_feature_curve.ipynb`, `03_quantization.ipynb`
   as listed in `notebooks/README.md`.
4. `docs/results`: a results section for the README written strictly from the measured files.

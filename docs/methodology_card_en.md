# Methodology Passport Card

Version of 2026-10-12 · Author: Kuanyshev · Russian version: [methodology_card.md](methodology_card.md)

## 1. Project information

Individual project; continuation of Task 1 (Research Proposal) on the CICIoT2023 dataset.

| Field | Value |
| --- | --- |
| Title | Accuracy–Cost Trade-off of Lightweight ML Models for IoT Intrusion Detection on Resource-Constrained Devices |
| Author | Kuanyshev (individual) |
| GitHub repository | https://github.com/ilkhankuanyshev/lightweight-iot-ids-benchmark |
| Context | CICIoT2023, 46 features, 8 classes (7 attack categories + benign); target platform: Raspberry Pi 4 or Docker limited to 1 vCPU / 1 GB RAM |

## 2. Study type

Quantitative experimental study (controlled comparative benchmark).

Justification:

- Every dependent variable is numeric and measured instrumentally: macro-F1, inference latency (ms), model size (KB), peak RAM (MB).
- The independent variables (model type, number of features, weight precision) are manipulated by the researcher while all other conditions stay fixed — a classic factorial experiment.
- Hypotheses are decided by statistical tests (Friedman, Wilcoxon) at α = 0.05, not by interpretation.
- No qualitative component (interviews, expert ratings) is needed: model-selection recommendations come from the macro-F1 vs latency Pareto front, i.e. also from numbers. A mixed-methods design would therefore not add validity.

## 3. Hypotheses

The three hypothesis pairs from Task 1 keep their meaning and are refined so that each is decided by one pre-specified test. Unit of observation: one split of 5×3 repeated stratified cross-validation (n = 15 paired observations).

**RQ1 — model type (DT, RF, LightGBM, MLP, 1D-CNN)**

- **H0.1:** the five models have equal median macro-F1 and equal median p50 latency.
- **H1.1:** at least one model differs significantly in macro-F1 or in p50 latency (Friedman test, α = 0.05). The directional claim from Task 1 ("tree ensembles give a better F1/latency ratio") is checked with the Nemenyi post-hoc test and the Pareto front: no neural network may dominate RF or LightGBM on both metrics at once.

**RQ2 — number of features (46 → 15)**

- **H0.2:** at k = 15, p50 latency drops by less than 30 % (or not significantly), or macro-F1 drops by more than 1 pp relative to k = 46.
- **H1.2:** at k = 15, p50 latency drops by at least 30 % (one-sided Wilcoxon signed-rank test) and macro-F1 is non-inferior to k = 46 with a 1 pp margin.

**RQ3 — weight precision (FP32 → INT8), MLP and 1D-CNN only**

- **H0.3:** INT8 quantization shrinks the model by less than 3×, or does not reduce latency significantly, or lowers macro-F1 by more than 1 pp.
- **H1.3:** the INT8 model is at least 3× smaller than FP32, its p50 latency is significantly lower (one-sided Wilcoxon), and its macro-F1 is non-inferior to FP32 with a 1 pp margin.

**Week 5 refinements relative to Task 1:**

1. 5-fold CV is replaced by 5×3 repeated stratified CV. With n = 5 the exact two-sided Wilcoxon test cannot reach p < 0.05 (minimum p = 0.0625), so H1.2 and H1.3 could never be supported.
2. H0.2 is rewritten as the exact complement of H1.2: in Task 1 it covered latency only, while H1.2 also contains a macro-F1 condition.
3. "F1 drop of at most 1 pp" is tested as non-inferiority, not as the absence of a significant difference.
4. The confirmatory comparison in RQ2 is 46 vs 15 features only; levels 30 and 10 build the curve and are analysed descriptively, to avoid multiplying tests.
5. Holm correction is applied within each test family (across models).

## 4. Variable matrix

Three factors are varied; everything else is fixed in `configs/config.yaml`, `requirements.txt` and `Dockerfile`.

**Independent variables (IV)**

| Variable | Levels | config.yaml key |
| --- | --- | --- |
| Model type | 5: Decision Tree, Random Forest, LightGBM, MLP, 1D-CNN | `models.*` |
| Number of input features k | 46 / 30 / 15 / 10 (top-k by mutual information) | `features.counts` |
| Weight precision | FP32 / INT8 (MLP and 1D-CNN only; static post-training quantization in ONNX Runtime) | `quantization.*` |

**Dependent variables (DV)**

| Metric | Unit | How it is measured |
| --- | --- | --- |
| macro-F1 (8 classes) | 0–1 | on the test fold |
| Inference latency p50 / p95 | ms per record | batch = 1, 1000 timed predictions |
| Throughput | records/s | batch = 256 |
| Model size | KB | size of the ONNX file on disk |
| Peak RAM of the inference process | MB | peak RSS of a separate process |
| Per-class recall, benign FPR | fraction | on the test fold |

**Controlled variables (CV)**

| Condition | Fixed value |
| --- | --- |
| Hardware platform | Docker `--cpus=1 --memory=1g` on x86-64 (main run); Raspberry Pi 4, Cortex-A72 (confirmatory run). All configurations of one comparison are measured on the same platform |
| OS / base image | `python:3.12.11-slim-bookworm` (Debian 12) |
| Inference runtime | ONNX Runtime 1.23.2 for all five models, 1 thread (intra-op = inter-op = 1) |
| Library versions | 31 packages pinned exactly in `requirements.txt`: numpy 2.3.3, scikit-learn 1.7.2, lightgbm 4.6.0, onnx 1.19.1, skl2onnx 1.19.1, etc. |
| Dataset | CICIoT2023; at most 50,000 records per class (stratified sample that also balances classes); 34 labels → 8 classes via a fixed mapping in code |
| Data split | RepeatedStratifiedKFold 5×3, seed = 42; identical folds for every configuration |
| Preprocessing | NaN/inf rows removed; StandardScaler fitted on the training fold only and embedded in the ONNX graph (MLP, 1D-CNN) |
| Feature selection | mutual-information ranking on the training fold only |
| Hyperparameters | fixed in config.yaml, never tuned on test folds |
| Measurement protocol | 100 warm-up + 1000 timed predictions; every model measured in a fresh separate process |
| INT8 calibration | 1000 records from the training fold |

## 5. Metrics and baseline

**Primary metric: macro-F1 over 8 classes.** CICIoT2023 classes are highly imbalanced (DDoS and DoS dominate, Web and Brute Force are fractions of a percent), so accuracy overstates quality, while macro-F1 weighs every class equally. The RQ3 objective is the highest macro-F1 among configurations that pass all guardrails.

**Secondary metrics:** p50 / p95 latency, throughput, model size, peak RAM, per-class recall. p50 latency is the key cost metric: in H1.2 and H1.3 it decides the outcome together with macro-F1.

| Hypothesis | Deciding metric |
| --- | --- |
| H1.1 | macro-F1 and p50 latency (Friedman test on each) |
| H1.2 | p50 latency (reduction ≥ 30 %) + macro-F1 (non-inferiority, 1 pp margin) |
| H1.3 | model size (≥ 3×) + p50 latency + macro-F1 (non-inferiority, 1 pp margin) |

**Guardrail metrics.** A configuration that violates any threshold is excluded from the RQ3 optimum but stays in the statistical comparison.

| Metric | Threshold | Rationale |
| --- | --- | --- |
| p50 latency | ≤ 1 ms per record | RQ3 constraint from Task 1 |
| Model size | ≤ 1024 KB | RQ3 constraint from Task 1 |
| Peak RAM of the inference process | ≤ 512 MB | half of the 1 GB container limit; the rest is needed by the OS and the device's network stack |
| Benign-traffic FPR | ≤ 2 % | false alarms on normal traffic make an IDS unusable in practice |
| Run stability | 0 crashes, OOM kills or timeouts | results without the full set of splits are not comparable |

No CPU-utilisation guardrail is used: inference runs single-threaded by design, so one core is near 100 % for every model and the metric would not discriminate between them.

**Baseline (reference point).**

- **Main baseline:** Random Forest, all 46 features, FP32, default hyperparameters (100 trees). It is the standard strong model in CICIoT2023 studies (Neto et al., 2023; García-Merino et al., 2026) and the "no optimisation" configuration. It is expected to violate the size guardrail, which shows what compression buys.
- **Within-hypothesis reference points:** for RQ2 the same model at k = 46; for RQ3 the FP32 version of the same network.
- **Lower bound:** DummyClassifier (most frequent class), a check that models learn more than class frequencies (macro-F1 only).

## 6. Measurement protocol and statistical testing

The protocol is implemented in `src/benchmark.py`; the same code runs on the test sample and in the full experiment (only the config changes).

1. Data: CICIoT2023 → at most 50,000 records per class (`src/prepare_data.py`), 34 labels → 8 classes.
2. Split: RepeatedStratifiedKFold 5×3, seed = 42; the same 15 splits for every configuration.
3. On each training fold, features are ranked by mutual information; the top-k are kept for k ∈ {46, 30, 15, 10}.
4. Each model is trained with fixed hyperparameters and exported to ONNX (opset 15); MLP and 1D-CNN are additionally quantized to INT8 (static quantization, calibrated on 1000 training-fold records).
5. Quality (macro-F1, per-class recall, benign FPR) is computed from the exported ONNX file on the test fold; for INT8 this is the quantized model itself.
6. Cost is measured in a fresh separate process that imports only NumPy, ONNX Runtime and psutil: 100 warm-up + 1000 timed single-record predictions, throughput at batch = 256, peak RAM from `VmHWM`.
7. Every row is checked against the guardrails; outputs are `metrics.csv`, `summary.md`, `stats.md`, `env.json`.

**Two-phase mode.** Training Random Forest on the full dataset may need more than 1 GB of RAM, so training (`--phase train`) runs without limits, while measurement (`--phase measure`) runs in a `--cpus=1 --memory=1g` container or on a Raspberry Pi that receives the folder of ONNX models. Quality does not depend on the platform; cost does.

**Statistics (α = 0.05).**

| Hypothesis | Test | Correction |
| --- | --- | --- |
| H1.1 | Friedman across 5 models (macro-F1 and p50 separately) + Nemenyi critical difference | — |
| H1.2 | one-sided Wilcoxon: p50(k=15) < 0.7 · p50(k=46); non-inferiority: ΔF1 + 0.01 > 0 | Holm across 5 models |
| H1.3 | FP32/INT8 size ratio ≥ 3 on every split; one-sided Wilcoxon on p50; non-inferiority on macro-F1 | Holm across 2 models |

**Readiness check (Week 6).** The smoke run `configs/smoke.yaml` on `data/sample/` finishes in about 6 s without errors (28 measurements), and all 7 automated tests pass: 1D-CNN gradient check, ONNX vs NumPy equivalence, mapping of all 34 labels, and end-to-end runs in both modes.

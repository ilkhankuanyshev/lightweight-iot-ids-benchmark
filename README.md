# Accuracy–Cost Trade-off of Lightweight ML Models for IoT Intrusion Detection on Resource-Constrained Devices

[![CI](https://github.com/ilkhankuanyshev/lightweight-iot-ids-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/ilkhankuanyshev/lightweight-iot-ids-benchmark/actions/workflows/ci.yml)

**Authors:** Kuanyshev (individual project)
**Course task:** Task 2 — Methodology Passport Card & Research Repository (Weeks 5–6)
**License:** MIT · **Dataset:** CICIoT2023 (Neto et al., 2023)

## Overview

Most lightweight intrusion-detection (IDS) studies report accuracy measured on powerful hardware and compare
either tree ensembles *or* compressed neural networks, never both under one protocol. This repository is a
reproducible benchmark that trains five lightweight model families on CICIoT2023 (8 classes: 7 attack
categories + benign), exports **every model to ONNX**, and measures detection quality and inference cost
in **one runtime (ONNX Runtime, 1 thread)** on a resource-constrained platform (Docker limited to
1 vCPU / 1 GB RAM, or a Raspberry Pi 4).

## Research Questions

| ID | Research question | Hypothesis (H1) |
|---|---|---|
| RQ1 | How does the model type (Decision Tree, Random Forest, LightGBM, MLP, 1D-CNN) affect macro-F1 and inference latency on a constrained device? | At least one model differs significantly (Friedman, α = 0.05); no neural network Pareto-dominates the tree ensembles. |
| RQ2 | How does reducing the number of input features (mutual-information ranking) affect latency and macro-F1? | 46 → 15 features cuts p50 latency by ≥ 30 % with macro-F1 non-inferior within 1 pp (one-sided Wilcoxon, Holm). |
| RQ3 | Which configuration (model + k + precision) maximises macro-F1 under latency ≤ 1 ms/record and size ≤ 1 MB? | Post-training INT8 makes MLP/1D-CNN ≥ 3× smaller and faster with macro-F1 non-inferior within 1 pp. |

The full methodology (variables, metrics, guardrails, baseline) is in the Methodology Passport Card:
[`docs/methodology_card_en.md`](docs/methodology_card_en.md) (English) ·
[`docs/methodology_card.md`](docs/methodology_card.md) (Russian).

## Variables at a glance

* **Independent:** model type (5 levels) · number of features k ∈ {46, 30, 15, 10} · weight precision FP32 / INT8 (neural models).
* **Dependent:** macro-F1 (**primary**) · latency p50/p95 (ms/record, batch = 1) · throughput (records/s, batch = 256) · ONNX model size (KB) · peak RSS of the inference process (MB) · per-class recall · benign FPR.
* **Controlled:** hardware limits · base image `python:3.12.11-slim-bookworm` · all 31 Python packages pinned · ONNX opset 15 · ORT 1.23.2 with 1 intra-op / 1 inter-op thread · seed 42 · identical CV splits for every configuration · fixed hyperparameters.
* **Baseline:** Random Forest (100 trees), all 46 features, FP32. Floor: majority-class `DummyClassifier`.
* **Guardrails:** p50 ≤ 1 ms · size ≤ 1024 KB · peak RAM ≤ 512 MB · benign FPR ≤ 2 % · zero failed runs.

## Planned Benchmarks

Unit of observation: one split of 5×3 repeated stratified CV (n = 15 paired values per configuration).

| ID | Purpose | Varied | Fixed | Configurations | Measurements (×15 splits) | Statistical test |
|---|---|---|---|---|---|---|
| B0 | Sanity floor | — | k = 46 | Dummy (most frequent) | 15 (quality only) | — |
| B1 | RQ1 / H1.1 | model type (5) | k = 46, FP32 | 5 | 75 | Friedman + Nemenyi CD |
| B2 | RQ2 / H1.2 | k ∈ {46, 30, 15, 10} | FP32 | 5 × 4 = 20 (incl. B1) | 300 | Wilcoxon (46 vs 15) + Holm |
| B3 | RQ3 / H1.3 | FP32 vs INT8 | MLP, 1D-CNN; each k | 2 × 4 INT8 = 8 | 120 | Wilcoxon + non-inferiority + Holm |
| B4 | RQ3 optimum | all of the above | guardrails | 28 | 420 (= B1–B3) | Pareto front macro-F1 vs latency |

## Repository Structure

```
lightweight-iot-ids-benchmark/
├── configs/
│   ├── config.yaml          # full experiment: IVs, data paths, load and timeouts
│   └── smoke.yaml           # 1-minute pipeline check on data/sample/
├── data/
│   ├── README.md            # how to obtain CICIoT2023
│   ├── sample/              # 10-row schema-conformant sample (versioned)
│   ├── raw/                 # original CSVs (git-ignored)
│   └── processed/           # capped subset built by src.prepare_data (git-ignored)
├── docs/
│   ├── methodology_card_en.md  # Methodology Passport Card (English)
│   └── methodology_card.md     # Methodology Passport Card (Russian)
├── notebooks/               # result analysis (Week 7+)
├── results/                 # one folder per run (git-ignored)
├── scripts/
│   └── run_local.sh         # smoke test without Docker
├── src/
│   ├── benchmark.py         # entry point: train -> ONNX -> evaluate -> measure -> report
│   ├── data.py              # schema, 34 -> 8 class mapping, per-class cap
│   ├── features.py          # mutual-information ranking (training fold only)
│   ├── models.py            # model factory + ONNX export
│   ├── cnn1d.py             # NumPy 1D-CNN with ONNX export
│   ├── quantize.py          # static INT8 post-training quantization (ORT, QDQ)
│   ├── measure.py           # quality metrics + launches the latency worker
│   ├── latency_worker.py    # latency / throughput / RAM in a fresh process
│   ├── analyze.py           # summary + Friedman / Wilcoxon / Holm
│   └── prepare_data.py      # builds the capped dataset from raw CSVs
├── tests/test_pipeline.py   # unit tests + end-to-end smoke run
├── Dockerfile
├── requirements.txt         # exact pins, direct + transitive
├── .github/workflows/ci.yml # CI: tests, smoke run, Docker build on every push / PR
├── CITATION.cff
└── LICENSE
```

## System Requirements

* **Docker** ≥ 24 on Linux/macOS/Windows (x86-64 or arm64), ~1.5 GB free disk for the image; **or** Python 3.12 on Linux/macOS.
* Smoke test: 1 CPU core, < 300 MB RAM, about 1 minute.
* Full experiment: the CICIoT2023 CSV release (several GB, see [`data/README.md`](data/README.md)); the capped subset is ≈ 0.2 GB. Run time is several hours under the 1-CPU limit.
* Optional target hardware: Raspberry Pi 4 (Cortex-A72, 64-bit OS) with Docker.

## Quickstart Guide

Three commands build the environment and run the smoke test on `data/sample/`:

```bash
git clone https://github.com/ilkhankuanyshev/lightweight-iot-ids-benchmark.git && cd lightweight-iot-ids-benchmark
docker build -t iot-ids-bench .
docker run --rm --cpus=1 --memory=1g -v "$(pwd)/results:/app/results" iot-ids-bench
```

The run prints one line per model and writes `results/<timestamp>_smoke/`:

```
[split 1/2 r0 f0] k=46 random_forest fp32 | F1=1.0000 p50=0.0058ms size=27.9KB RAM=50MB | guardrails OK
[split 1/2 r0 f0] k=46 mlp           int8 | F1=0.7852 p50=0.0151ms size=10.8KB RAM=51MB | guardrails OK
...
Reports: results/<run>/summary.md, results/<run>/stats.md
```

| File | Content |
|---|---|
| `metrics.csv` | one row per split × model × k × precision (all DVs, guardrail verdicts) |
| `summary.md` / `summary.csv` | mean ± std per configuration |
| `stats.md` | Friedman / Wilcoxon / Holm results for H1.1–H1.3 |
| `env.json` | Python, OS, CPU, cgroup limits, package versions |
| `run.log`, `config_snapshot.yaml` | full log and the exact config used |

The same checks run automatically on GitHub Actions for every push and pull request (badge at the top).

Smoke-test numbers come from a jittered 10-row sample and **have no scientific meaning**; they only prove
that every stage runs.

Other commands:

```bash
docker run --rm --entrypoint python iot-ids-bench -m pytest -q          # 7 tests
./scripts/run_local.sh                                                  # same smoke test without Docker
```

### Full experiment (Week 7)

Training a 100-tree Random Forest on ~400k rows can need more than 1 GB of RAM, so training and
measurement are separate phases: train without limits, measure under the device limits
(or copy `results/<run>/` to a Raspberry Pi and run only the measure phase there).

```bash
# 0. put the CICIoT2023 CSV files into data/raw/  (see data/README.md), then build the subset
docker run --rm -v "$(pwd)/data:/app/data" --entrypoint python iot-ids-bench -m src.prepare_data --cap 50000

# 1. train + evaluate quality + export ONNX (no resource limits)
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" \
    iot-ids-bench --config configs/config.yaml --phase train

# 2. measure latency / RAM of every saved model under the device limits
docker run --rm --cpus=1 --memory=1g -v "$(pwd)/results:/app/results" \
    iot-ids-bench --phase measure --run-dir results/<run>
```

## Measurement Protocol

1. Features are ranked by mutual information on the **training fold only**; the top-k columns are kept in their original order.
2. Each model is trained in its native library (scikit-learn, LightGBM, NumPy) with fixed hyperparameters and exported to ONNX (opset 15). Scaling for neural models is part of the ONNX graph.
3. MLP and 1D-CNN are additionally quantized to INT8 (static, QDQ, MinMax calibration on 1000 training rows).
4. macro-F1, per-class recall and benign FPR are computed from the **exported ONNX model** on the test fold, so INT8 accuracy is measured on the INT8 artefact.
5. Cost is measured in a **fresh Python process** that imports only NumPy, ONNX Runtime and psutil: 100 warm-up + 1000 timed single-record predictions (`perf_counter_ns`), throughput at batch 256, peak RSS from `/proc/self/status` (`VmHWM`).
6. Guardrails are checked per row; `analyze.py` aggregates and runs the hypothesis tests.

## Contributing

Work happens on feature branches with pull requests and Conventional Commits; nothing is
committed to `main` directly. Any change to the methodology (variables, metrics, protocol) must be reflected in both methodology cards in `docs/`.

## Citation

If you use this repository, please cite it via [`CITATION.cff`](CITATION.cff) and the dataset:

> Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. A. (2023).
> CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment.
> *Sensors, 23*(13), 5941. https://doi.org/10.3390/s23135941

## License

Code: [MIT](LICENSE). The CICIoT2023 dataset is distributed by the Canadian Institute for Cybersecurity
under its own terms and is not included in this repository.

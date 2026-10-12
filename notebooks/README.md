# Notebooks

Analysis notebooks are added in Week 7, after the full run. They only **read** the outputs of
`src.benchmark` (`results/<run>/metrics.csv`, `summary.csv`) and never retrain models, so every
figure can be regenerated from a results folder.

Planned:

| Notebook | Content |
|---|---|
| `01_pareto_front.ipynb` | macro-F1 vs p50 latency for all 28 configurations, guardrail region shaded |
| `02_feature_curve.ipynb` | latency and macro-F1 as a function of k (46 / 30 / 15 / 10) |
| `03_quantization.ipynb` | FP32 vs INT8: size, latency, per-class recall |

Clear outputs before committing (`jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`);
`.ipynb_checkpoints/` is git-ignored.

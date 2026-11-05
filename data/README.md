# Data

## Task 3 addition

The compact `raw/test_run.jsonl` contains actually measured local timings on the
synthetic sample described below. `raw/provenance.json` records origin, hashes,
configuration and environment. The allowlisted Task 3 summaries in `processed/`
are versioned, while the full CICIoT2023 dataset and large experimental runs are
still ignored. The folder table below describes the original Task 2 setup.

| Folder | Versioned | Content |
|---|---|---|
| `sample/` | yes | `ciciot2023_sample.csv` — 10 rows, 46 features + `label`, all 8 classes |
| `raw/` | no (git-ignored) | original CICIoT2023 CSV files |
| `processed/` | no (git-ignored) | `ciciot2023_capped.csv` built by `src.prepare_data` |

## Obtaining CICIoT2023

1. Request the dataset from the Canadian Institute for Cybersecurity:
   https://www.unb.ca/cic/datasets/iotdataset-2023.html
2. Download the CSV release (feature-extracted flows, one `label` column with 34 classes) and put
   all `*.csv` files into `data/raw/`.
3. Build the experiment subset (at most 50 000 rows per 8-class category, uniform random sample):

```bash
python -m src.prepare_data --raw-dir data/raw --out data/processed/ciciot2023_capped.csv --cap 50000 --seed 42
```

Labels are mapped 34 → 8 classes in `src/data.py` (`map_label`); an unknown label stops the run
with an explicit error instead of being silently dropped.

## Sample file

`sample/ciciot2023_sample.csv` is **schema-conformant synthetic data**: exact CICIoT2023 column
names and order, value ranges typical for each traffic type, one or two rows per class. It exists
only to smoke-test the pipeline (`configs/smoke.yaml`) and must not be used for any conclusion.
It can be replaced by 10 real rows from the dataset without changing any code.

Citation: Neto, E. C. P. et al. (2023). CICIoT2023: A real-time dataset and benchmark for
large-scale attacks in IoT environment. *Sensors, 23*(13), 5941. https://doi.org/10.3390/s23135941

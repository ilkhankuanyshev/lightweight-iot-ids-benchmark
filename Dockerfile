# syntax=docker/dockerfile:1
# Reproducible benchmark image. Build:  docker build -t iot-ids-bench .
FROM python:3.12.11-slim-bookworm

# Single-threaded math libraries = controlled variable for latency measurements.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# libgomp1: OpenMP runtime required by the LightGBM wheel
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
# --no-deps + full transitive pin list = nothing is resolved at build time; pip check verifies it.
COPY requirements.txt .
RUN pip install pip==25.2 \
    && pip install --no-deps -r requirements.txt \
    && pip check

COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/
COPY data/sample/ data/sample/
RUN mkdir -p results data/raw data/processed

ENTRYPOINT ["python", "-m", "src.benchmark"]
CMD ["--config", "configs/smoke.yaml"]

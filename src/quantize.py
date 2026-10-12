"""Post-training static INT8 quantization (ONNX Runtime, QDQ format)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)


class _ArrayReader(CalibrationDataReader):
    def __init__(self, X: np.ndarray, input_name: str, batch: int = 100):
        self._batches = iter([{input_name: X[i:i + batch]} for i in range(0, len(X), batch)])

    def get_next(self):
        return next(self._batches, None)


def quantize_int8(fp32_path: Path, int8_path: Path, X_calib: np.ndarray,
                  qcfg: dict[str, Any]) -> None:
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)   # ORT prints a 'consider pre-processing' hint for every call
    try:
        _quantize(fp32_path, int8_path, X_calib, qcfg)
    finally:
        root.setLevel(level)


def _quantize(fp32_path: Path, int8_path: Path, X_calib: np.ndarray,
              qcfg: dict[str, Any]) -> None:
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=_ArrayReader(X_calib.astype(np.float32), "input"),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=bool(qcfg.get("per_channel", False)),
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )

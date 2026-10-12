"""Model factory. Every model is trained in its native library and exported to ONNX, so all
five families are timed in the SAME runtime (ONNX Runtime) -- a controlled variable."""
from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from lightgbm import LGBMClassifier
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType as MLFloatTensorType
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .cnn1d import CNN1DClassifier

MODEL_NAMES = ["decision_tree", "random_forest", "lightgbm", "mlp", "cnn1d"]
NEURAL = {"mlp", "cnn1d"}           # models eligible for INT8 quantization
OPSET = {"": 15, "ai.onnx.ml": 3}   # 15 = highest opset the LightGBM converter supports


def build_model(name: str, params: dict[str, Any], seed: int, n_classes: int):
    params = dict(params or {})
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if name == "lightgbm":
        return LGBMClassifier(random_state=seed, **params)
    if name == "mlp":
        if "hidden_layer_sizes" in params:
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        # trees are scale-invariant; neural nets get a scaler fitted on the training fold only
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", MLPClassifier(random_state=seed, **params))])
    if name == "cnn1d":
        return CNN1DClassifier(n_classes=n_classes, random_state=seed, **params)
    raise ValueError(f"Unknown model '{name}'. Known: {MODEL_NAMES}")


def to_onnx(name: str, model, n_features: int) -> onnx.ModelProto:
    """Export a fitted model; every graph has input 'input' [N, k] and output 'label' [N]."""
    initial_types = [("input", FloatTensorType([None, n_features]))]
    if name == "cnn1d":
        return model.to_onnx(opset=OPSET[""])
    if name == "lightgbm":
        return convert_lightgbm(model, initial_types=[("input", MLFloatTensorType([None, n_features]))],
                                zipmap=False,
                                target_opset=OPSET[""])
    clf = model.steps[-1][1] if isinstance(model, Pipeline) else model
    return convert_sklearn(model, initial_types=initial_types,
                           options={id(clf): {"zipmap": False}}, target_opset=OPSET)


def onnx_label_output(model: onnx.ModelProto) -> str:
    names = [o.name for o in model.graph.output]
    for candidate in ("label", "output_label"):
        if candidate in names:
            return candidate
    return names[0]


def check_classes(model, name: str, n_classes: int) -> None:
    """ONNX label = class id only if the model saw every class in training."""
    if name == "cnn1d":
        return
    clf = model.steps[-1][1] if isinstance(model, Pipeline) else model
    classes = np.asarray(clf.classes_)
    if not np.array_equal(classes, np.arange(n_classes)):
        raise ValueError(f"{name}: training fold is missing classes, got {classes.tolist()}")

"""Compact 1D-CNN for tabular flow features, implemented in NumPy and exported to ONNX.

Architecture (input = k standardized features treated as a 1-channel sequence):
    Conv1d(1 -> F, kernel K, 'same' padding) -> ReLU -> Flatten
    -> Dense(F*k -> H) -> ReLU -> Dense(H -> C) -> softmax

Why NumPy instead of PyTorch/TensorFlow: the network is tiny, and avoiding a 1-2 GB
framework keeps the Docker image small and the dependency set fully pinned. The exported
ONNX graph uses only standard operators (Conv, Gemm, Relu), and tests/test_cnn1d.py checks
(a) the analytic gradients against finite differences and (b) that ONNX Runtime reproduces
the NumPy forward pass.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from numpy.lib.stride_tricks import sliding_window_view


class CNN1DClassifier:
    def __init__(self, n_classes: int, filters: int = 16, kernel_size: int = 3,
                 hidden: int = 32, epochs: int = 10, batch_size: int = 512,
                 lr: float = 1e-3, random_state: int = 42):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd ('same' padding)")
        self.n_classes, self.filters, self.kernel_size = n_classes, filters, kernel_size
        self.hidden, self.epochs, self.batch_size = hidden, epochs, batch_size
        self.lr, self.random_state = lr, random_state

    # ---------------------------------------------------------------- parameters
    def _init_params(self, length: int, rng: np.random.Generator) -> None:
        F, K, H, C = self.filters, self.kernel_size, self.hidden, self.n_classes
        he = lambda fan_in, shape: rng.normal(0, np.sqrt(2.0 / fan_in), shape)  # noqa: E731
        self.params = {
            "Wc": he(K, (F, K)), "bc": np.zeros(F),
            "W1": he(F * length, (F * length, H)), "b1": np.zeros(H),
            "W2": he(H, (H, C)), "b2": np.zeros(C),
        }
        self.params = {k: v.astype(np.float64) for k, v in self.params.items()}

    # ---------------------------------------------------------------- forward / backward
    def _forward(self, x: np.ndarray):
        p, pad = self.params, self.kernel_size // 2
        xp = np.pad(x, ((0, 0), (pad, pad)))
        cols = sliding_window_view(xp, self.kernel_size, axis=1)            # (B, L, K)
        z1 = cols @ p["Wc"].T + p["bc"]                                      # (B, L, F)
        a1 = np.maximum(z1, 0)
        B, L, F = a1.shape
        h0 = a1.transpose(0, 2, 1).reshape(B, F * L)                         # ONNX Flatten order
        z2 = h0 @ p["W1"] + p["b1"]
        a2 = np.maximum(z2, 0)
        logits = a2 @ p["W2"] + p["b2"]
        return logits, (cols, z1, h0, z2, a2)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _loss_and_grads(self, x: np.ndarray, y: np.ndarray):
        p = self.params
        logits, (cols, z1, h0, z2, a2) = self._forward(x)
        probs = self._softmax(logits)
        B = len(y)
        loss = -np.log(probs[np.arange(B), y] + 1e-12).mean()
        dz3 = probs.copy()
        dz3[np.arange(B), y] -= 1
        dz3 /= B
        g = {"W2": a2.T @ dz3, "b2": dz3.sum(0)}
        dz2 = (dz3 @ p["W2"].T) * (z2 > 0)
        g["W1"], g["b1"] = h0.T @ dz2, dz2.sum(0)
        dh0 = dz2 @ p["W1"].T
        L, F = z1.shape[1], z1.shape[2]
        dz1 = dh0.reshape(B, F, L).transpose(0, 2, 1) * (z1 > 0)
        g["Wc"] = np.einsum("blf,blk->fk", dz1, cols)
        g["bc"] = dz1.sum((0, 1))
        return loss, g

    # ---------------------------------------------------------------- sklearn-like API
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNN1DClassifier":
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std > 1e-12, std, 1.0)
        Xs = (X - self.mean_) / self.scale_
        self.n_features_in_ = X.shape[1]
        self._init_params(self.n_features_in_, rng)
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        self.loss_curve_ = []
        for _ in range(self.epochs):
            order = rng.permutation(len(y))
            epoch_loss = 0.0
            for start in range(0, len(y), self.batch_size):
                idx = order[start:start + self.batch_size]
                loss, grads = self._loss_and_grads(Xs[idx], y[idx])
                epoch_loss += loss * len(idx)
                t += 1
                for k in self.params:
                    m[k] = b1 * m[k] + (1 - b1) * grads[k]
                    v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                    mh, vh = m[k] / (1 - b1 ** t), v[k] / (1 - b2 ** t)
                    self.params[k] -= self.lr * mh / (np.sqrt(vh) + eps)
            self.loss_curve_.append(epoch_loss / len(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return self._softmax(self._forward(Xs)[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ---------------------------------------------------------------- ONNX export
    def to_onnx(self, opset: int = 15) -> onnx.ModelProto:
        p, L, F, K = self.params, self.n_features_in_, self.filters, self.kernel_size
        f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
        inits = [
            numpy_helper.from_array(f32(self.mean_), "mean"),
            numpy_helper.from_array(f32(self.scale_), "scale"),
            numpy_helper.from_array(np.array([-1, 1, L], dtype=np.int64), "shape3d"),
            numpy_helper.from_array(f32(p["Wc"]).reshape(F, 1, K), "Wc"),
            numpy_helper.from_array(f32(p["bc"]), "bc"),
            numpy_helper.from_array(f32(p["W1"]), "W1"),
            numpy_helper.from_array(f32(p["b1"]), "b1"),
            numpy_helper.from_array(f32(p["W2"]), "W2"),
            numpy_helper.from_array(f32(p["b2"]), "b2"),
        ]
        pad = K // 2
        nodes = [
            helper.make_node("Sub", ["input", "mean"], ["centered"]),
            helper.make_node("Div", ["centered", "scale"], ["scaled"]),
            helper.make_node("Reshape", ["scaled", "shape3d"], ["seq"]),
            helper.make_node("Conv", ["seq", "Wc", "bc"], ["conv"],
                             kernel_shape=[K], pads=[pad, pad]),
            helper.make_node("Relu", ["conv"], ["conv_relu"]),
            helper.make_node("Flatten", ["conv_relu"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "W1", "b1"], ["fc1"]),
            helper.make_node("Relu", ["fc1"], ["fc1_relu"]),
            helper.make_node("Gemm", ["fc1_relu", "W2", "b2"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
            helper.make_node("ArgMax", ["logits"], ["label"], axis=1, keepdims=0),
        ]
        graph = helper.make_graph(
            nodes, "cnn1d",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, L])],
            [helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
             helper.make_tensor_value_info("probabilities", TensorProto.FLOAT,
                                           [None, self.n_classes])],
            inits,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                                  producer_name="lightweight-iot-ids-benchmark")
        model.ir_version = 8
        onnx.checker.check_model(model)
        return model

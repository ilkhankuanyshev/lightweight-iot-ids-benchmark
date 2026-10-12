"""Run with:  python -m pytest -q"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from src.analyze import holm
from src.benchmark import run
from src.cnn1d import CNN1DClassifier
from src.data import CLASSES, FEATURES, map_label
from src.utils import REPO_ROOT, load_config

ALL_34_LABELS = {
    "DDoS": ["DDoS-RSTFINFlood", "DDoS-PSHACK_Flood", "DDoS-SYN_Flood", "DDoS-UDP_Flood",
             "DDoS-TCP_Flood", "DDoS-ICMP_Flood", "DDoS-SynonymousIP_Flood",
             "DDoS-ACK_Fragmentation", "DDoS-UDP_Fragmentation", "DDoS-ICMP_Fragmentation",
             "DDoS-SlowLoris", "DDoS-HTTP_Flood"],
    "DoS": ["DoS-UDP_Flood", "DoS-SYN_Flood", "DoS-TCP_Flood", "DoS-HTTP_Flood"],
    "Mirai": ["Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain"],
    "Recon": ["Recon-HostDiscovery", "Recon-OSScan", "Recon-PortScan", "Recon-PingSweep",
              "VulnerabilityScan"],
    "Spoofing": ["MITM-ArpSpoofing", "DNS_Spoofing"],
    "Web": ["BrowserHijacking", "Backdoor_Malware", "XSS", "Uploading_Attack",
            "SqlInjection", "CommandInjection"],
    "BruteForce": ["DictionaryBruteForce"],
    "Benign": ["BenignTraffic"],
}


def test_label_mapping_covers_34_labels():
    labels = [lab for group in ALL_34_LABELS.values() for lab in group]
    assert len(labels) == 34
    for category, group in ALL_34_LABELS.items():
        for label in group:
            assert map_label(label) == category
    assert set(ALL_34_LABELS) == set(CLASSES)
    with pytest.raises(KeyError):
        map_label("NotARealAttack")


def test_sample_matches_schema():
    df = pd.read_csv(REPO_ROOT / "data/sample/ciciot2023_sample.csv")
    assert 5 <= len(df) <= 10
    assert list(df.columns) == FEATURES + ["label"]
    assert {map_label(v) for v in df["label"]} == set(CLASSES)


def _toy(n=60, k=7, c=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, k)), rng.integers(0, c, size=n)


def test_cnn_gradients_match_finite_differences():
    X, y = _toy()
    net = CNN1DClassifier(n_classes=3, filters=4, kernel_size=3, hidden=5, random_state=1)
    net.mean_, net.scale_, net.n_features_in_ = 0.0, 1.0, X.shape[1]
    net._init_params(X.shape[1], np.random.default_rng(1))
    _, grads = net._loss_and_grads(X, y)
    rng, eps = np.random.default_rng(2), 1e-6
    for name, param in net.params.items():
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in param.shape)
            old = param[idx]
            param[idx] = old + eps
            lp, _ = net._loss_and_grads(X, y)
            param[idx] = old - eps
            lm, _ = net._loss_and_grads(X, y)
            param[idx] = old
            assert np.isclose((lp - lm) / (2 * eps), grads[name][idx], rtol=1e-4, atol=1e-7), name


def test_cnn_onnx_matches_numpy():
    X, y = _toy(n=200)
    net = CNN1DClassifier(n_classes=3, filters=4, hidden=8, epochs=2, batch_size=32).fit(X, y)
    sess = ort.InferenceSession(net.to_onnx().SerializeToString(),
                                providers=["CPUExecutionProvider"])
    label, proba = sess.run(["label", "probabilities"], {"input": X.astype(np.float32)})
    np.testing.assert_allclose(proba, net.predict_proba(X), atol=1e-4)
    assert (label == net.predict(X)).mean() > 0.99


def test_holm_adjustment():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_smoke_run_end_to_end(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    cfg["experiment"]["output_dir"] = str(tmp_path)
    run_dir = run(cfg, max_splits=1)
    df = pd.read_csv(run_dir / "metrics.csv")
    assert (df["status"] == "ok").all(), df.loc[df["status"] != "ok", "error"].tolist()
    assert {"decision_tree", "random_forest", "lightgbm", "mlp", "cnn1d", "dummy"} <= set(
        df["model"])
    assert set(df.loc[df["model"].isin(["mlp", "cnn1d"]), "precision"]) == {"fp32", "int8"}
    ok = df[df["model"] != "dummy"]
    assert (ok["latency_p50_ms"] > 0).all() and (ok["model_size_kb"] > 0).all()
    for name in ("summary.md", "stats.md", "env.json", "run.log", "config_snapshot.yaml"):
        assert (run_dir / name).exists()


def test_two_phase_train_then_measure(tmp_path):
    from src.benchmark import measure_run
    cfg = load_config("configs/smoke.yaml")
    cfg["experiment"]["output_dir"] = str(tmp_path)
    run_dir = run(cfg, max_splits=1, phase="train")
    trained = pd.read_csv(run_dir / "metrics.csv")
    assert trained.loc[trained["model"] != "dummy", "latency_p50_ms"].isna().all()
    measure_run(run_dir)
    measured = pd.read_csv(run_dir / "metrics.csv")
    assert (measured.loc[measured["model"] != "dummy", "latency_p50_ms"] > 0).all()
    assert (run_dir / "stats.md").exists() and (run_dir / "env_measure.json").exists()

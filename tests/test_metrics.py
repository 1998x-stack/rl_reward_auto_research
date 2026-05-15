import numpy as np
import pytest
from metrics import (
    compute_performance,
    compute_sample_efficiency,
    compute_stability,
    compute_all_metrics,
    normalize_metric,
)


def test_performance_positive_returns():
    seed_returns = [[100, 150, 200, 250, 300, 350, 400, 420, 440, 460, 470, 480, 490, 495, 498]]
    perf = compute_performance(seed_returns, baseline_return=400.0, direction="positive")
    assert perf > 1.0
    assert perf < 1.3


def test_performance_negative_returns():
    seed_returns = [[-200, -180, -160, -150, -140, -130, -120, -110, -100, -95, -90, -85, -80]]
    perf = compute_performance(seed_returns, baseline_return=-100.0, direction="negative")
    assert perf > 0.0


def test_performance_multi_seed_average():
    seed_returns = [
        [100, 200, 300, 400, 500],
        [100, 200, 300, 400, 400],
        [100, 200, 300, 400, 300],
    ]
    perf = compute_performance(seed_returns, baseline_return=400.0, direction="positive")
    assert perf == pytest.approx(0.7, abs=0.01)


def test_sample_efficiency_positive():
    seed_returns = [[10, 20, 40, 80, 160, 300, 400, 450, 470, 480]]
    eff = compute_sample_efficiency(seed_returns, baseline_returns=[10]*10, direction="positive")
    assert eff > 0.0


def test_stability_perfect():
    seed_returns = [[1, 2, 3, 100, 100], [1, 2, 3, 100, 100], [1, 2, 3, 100, 100]]
    stab = compute_stability(seed_returns)
    assert stab == pytest.approx(1.0, abs=0.01)


def test_stability_variable():
    seed_returns = [[1, 2, 3, 50, 50], [1, 2, 3, 100, 100], [1, 2, 3, 150, 150]]
    stab = compute_stability(seed_returns)
    assert stab < 1.0
    assert stab > 0.0


def test_compute_all_metrics_positive():
    seed_returns = [
        [10, 50, 200, 400, 450],
        [10, 50, 180, 380, 430],
        [10, 50, 220, 420, 470],
    ]
    task_metrics, reward_metrics = compute_all_metrics(
        task_seed_returns=seed_returns,
        reward_seed_returns=seed_returns,
        baseline_return=400.0,
        baseline_returns=[10, 50, 200, 400, 450],
        direction="positive",
    )
    assert "perf" in task_metrics
    assert "eff" in task_metrics
    assert "stab" in task_metrics
    assert task_metrics["perf"] > 0
    assert task_metrics["stab"] > 0


def test_normalize_metric():
    assert normalize_metric(actual=500.0, baseline=400.0, direction="positive") == 1.25
    neg = normalize_metric(actual=-80.0, baseline=-100.0, direction="negative")
    assert neg > 1.0

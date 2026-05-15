import json
import tempfile
import os
from pathlib import Path
import pytest
from pareto import dominates, pareto_decision, load_archive, update_archive

METRICS = ["perf", "eff", "stab"]


def test_dominates_a_beats_b():
    a = {"perf": 1.2, "eff": 1.1, "stab": 0.9}
    b = {"perf": 1.0, "eff": 1.0, "stab": 0.8}
    assert dominates(a, b) is True


def test_dominates_not_all_ge():
    a = {"perf": 1.2, "eff": 1.1, "stab": 0.7}
    b = {"perf": 1.0, "eff": 1.0, "stab": 0.9}
    assert dominates(a, b) is False


def test_dominates_equal():
    a = {"perf": 1.0, "eff": 1.0, "stab": 0.5}
    b = {"perf": 1.0, "eff": 1.0, "stab": 0.5}
    assert dominates(a, b) is False


def test_pareto_decision_keep_new():
    frontier = [{"name": "old1", "perf": 1.0, "eff": 1.0, "stab": 0.8}]
    archive = {"metrics": METRICS, "environments": {"CartPole-v1": {"frontier": frontier, "baseline_return": 400.0, "baseline_returns": []}}, "dominated_count": 0}
    new_metrics = {"perf": 1.2, "eff": 1.1, "stab": 0.85}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(archive, f)
        tmp_path = f.name
    try:
        status, dominates_list, dominated_by = pareto_decision("new_reward", new_metrics, "CartPole-v1", tmp_path)
        assert status == "keep"
        assert "old1" in dominates_list
    finally:
        os.unlink(tmp_path)


def test_pareto_decision_discard():
    frontier = [{"name": "best", "perf": 2.0, "eff": 2.0, "stab": 0.95}]
    archive = {"metrics": METRICS, "environments": {"CartPole-v1": {"frontier": frontier, "baseline_return": 400.0, "baseline_returns": []}}, "dominated_count": 0}
    new_metrics = {"perf": 0.5, "eff": 0.5, "stab": 0.5}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(archive, f)
        tmp_path = f.name
    try:
        status, dominates_list, dominated_by = pareto_decision("loser", new_metrics, "CartPole-v1", tmp_path)
        assert status == "discard"
        assert "best" in dominated_by
    finally:
        os.unlink(tmp_path)


def test_pareto_decision_nan():
    new_metrics = {"perf": float("nan"), "eff": 0.5, "stab": 0.5}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"metrics": METRICS, "environments": {}, "dominated_count": 0}, f)
        tmp_path = f.name
    try:
        status, _, _ = pareto_decision("bad", new_metrics, "CartPole-v1", tmp_path)
        assert status == "crash"
    finally:
        os.unlink(tmp_path)


def test_load_archive_empty():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
    os.unlink(tmp_path)
    archive = load_archive(tmp_path)
    assert archive["metrics"] == ["perf", "eff", "stab"]
    assert archive["total_experiments"] == 0


def test_update_archive_adds_to_frontier():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"metrics": METRICS, "environments": {"CartPole-v1": {"frontier": [], "baseline_return": 400.0, "baseline_returns": []}}, "dominated_count": 0, "total_experiments": 0}, f)
        tmp_path = f.name
    try:
        update_archive(
            factor_info={"name": "new", "perf": 1.0, "eff": 1.0, "stab": 0.9},
            env_id="CartPole-v1",
            str_path=tmp_path,
        )
        archive = load_archive(tmp_path)
        assert len(archive["environments"]["CartPole-v1"]["frontier"]) == 1
        assert archive["total_experiments"] == 1
    finally:
        os.unlink(tmp_path)

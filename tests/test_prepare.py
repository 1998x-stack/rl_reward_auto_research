import json
import os
import tempfile
from pathlib import Path
import importlib
import importlib.util

import gymnasium as gym
from env_info import ENV_REGISTRY
from reward import Reward


def test_reward_discovery_with_defaults():
    spec = importlib.util.spec_from_file_location(
        "rewards_default",
        Path(__file__).resolve().parent.parent / "rewards_default.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rewards = []
    for name in dir(module):
        if name.startswith("Default"):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Reward) and obj is not Reward:
                rewards.append(obj())

    assert len(rewards) == 4
    env_ids = {r.env_id for r in rewards}
    assert "CartPole-v1" in env_ids
    assert "Pendulum-v1" in env_ids


def test_baseline_reward_produces_identity():
    from rewards_default import DefaultCartPole
    import numpy as np

    r = DefaultCartPole()
    info = ENV_REGISTRY["CartPole-v1"]
    result = r.compute(np.array([0.0, 0.0, 0.01, 0.0]), 1, 1.0, False, False, {}, info)
    assert result == 1.0


def test_env_info_consistency():
    for env_id in ENV_REGISTRY:
        env = gym.make(env_id)
        obs_space = env.observation_space
        info = ENV_REGISTRY[env_id]
        if hasattr(obs_space, 'shape'):
            assert obs_space.shape[0] == info.obs_dim, f"{env_id}: obs dim mismatch"
        env.close()


def test_pareto_archive_roundtrip():
    from pareto import load_archive, update_archive

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "metrics": ["perf", "eff", "stab"],
            "environments": {},
            "dominated_count": 0,
            "total_experiments": 0,
        }, f)
        tmp_path = f.name
    try:
        update_archive(
            factor_info={"name": "test", "perf": 1.5, "eff": 1.2, "stab": 0.9},
            env_id="CartPole-v1",
            str_path=tmp_path,
        )
        archive = load_archive(tmp_path)
        assert len(archive["environments"]["CartPole-v1"]["frontier"]) == 1
        assert archive["environments"]["CartPole-v1"]["frontier"][0]["name"] == "test"
    finally:
        os.unlink(tmp_path)

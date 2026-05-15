# tests/test_env_info.py
import numpy as np
import pytest
from env_info import EnvInfo, ENV_REGISTRY


def test_env_info_creation():
    info = EnvInfo(
        env_id="CartPole-v1",
        obs_dim=4,
        obs_low=np.array([-4.8, -np.inf, -0.418, -np.inf]),
        obs_high=np.array([4.8, np.inf, 0.418, np.inf]),
        obs_names=["cart_pos", "cart_vel", "pole_angle", "pole_angular_vel"],
        action_type="discrete",
        action_dim=2,
        max_return=500.0,
        reward_direction="positive",
    )
    assert info.env_id == "CartPole-v1"
    assert info.obs_dim == 4
    assert info.action_type == "discrete"
    assert info.max_return == 500.0


def test_registry_has_four_envs():
    assert "CartPole-v1" in ENV_REGISTRY
    assert "Acrobot-v1" in ENV_REGISTRY
    assert "MountainCar-v0" in ENV_REGISTRY
    assert "Pendulum-v1" in ENV_REGISTRY


def test_registry_obs_names_match_dim():
    for env_id, info in ENV_REGISTRY.items():
        assert len(info.obs_names) == info.obs_dim, f"{env_id}: obs_names length mismatch"


def test_registry_action_types():
    assert ENV_REGISTRY["CartPole-v1"].action_type == "discrete"
    assert ENV_REGISTRY["Pendulum-v1"].action_type == "continuous"


def test_registry_return_directions():
    assert ENV_REGISTRY["CartPole-v1"].reward_direction == "positive"
    assert ENV_REGISTRY["Acrobot-v1"].reward_direction == "negative"
    assert ENV_REGISTRY["MountainCar-v0"].reward_direction == "negative"
    assert ENV_REGISTRY["Pendulum-v1"].reward_direction == "negative"

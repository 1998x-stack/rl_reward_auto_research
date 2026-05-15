import numpy as np
import pytest
from reward import Reward, discover_rewards
from env_info import ENV_REGISTRY


class TestReward(Reward):
    name = "test_reward"
    env_id = "CartPole-v1"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        return env_reward + 0.5


def test_reward_base_class_compute_raises():
    class IncompleteReward(Reward):
        name = "incomplete"
        env_id = "Test-v0"

    r = IncompleteReward()
    with pytest.raises(NotImplementedError):
        r.compute(np.array([0.0]), 0, 1.0, False, False, {}, None)


def test_reward_compute():
    info = ENV_REGISTRY["CartPole-v1"]
    r = TestReward()
    result = r.compute(np.array([0.0, 0.0, 0.01, 0.0]), 1, 1.0, False, False, {}, info)
    assert result == 1.5


def test_reward_with_termination():
    info = ENV_REGISTRY["CartPole-v1"]
    r = TestReward()
    result = r.compute(np.array([2.4, 1.0, 0.2, 0.5]), 0, 1.0, True, False, {}, info)
    assert result == 1.5


def test_discover_rewards_empty():
    rewards = discover_rewards("env_info")
    assert len(rewards) == 0


def test_reward_requires_name():
    with pytest.raises(TypeError):
        class BadReward(Reward):
            env_id = "Test-v0"
    # Should fail: no name defined


def test_reward_requires_env_id():
    with pytest.raises(TypeError):
        class BadReward2(Reward):
            name = "bad"
    # Should fail: no env_id defined

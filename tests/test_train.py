import numpy as np
import pytest
import gymnasium as gym

from train import train_ppo, RewardEnvWrapper
from env_info import ENV_REGISTRY, EnvInfo
from reward import Reward


class TestCartPoleReward(Reward):
    name = "test_shaping"
    env_id = "CartPole-v1"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        return env_reward - 0.1 * abs(obs[2])


def test_reward_env_wrapper():
    env = gym.make("CartPole-v1", max_episode_steps=10)
    reward_fn = TestCartPoleReward()
    env_info = ENV_REGISTRY["CartPole-v1"]
    wrapped = RewardEnvWrapper(env, reward_fn, env_info)

    obs, _ = wrapped.reset()
    obs, reward, terminated, truncated, info = wrapped.step(0)

    assert isinstance(reward, (float, np.floating))
    assert reward is not None
    assert "_env_reward" in info
    assert "_agent_reward" in info


def test_train_ppo_short_run():
    env_info = ENV_REGISTRY["CartPole-v1"]
    reward_fn = TestCartPoleReward()

    def env_fn():
        return RewardEnvWrapper(
            gym.make("CartPole-v1", max_episode_steps=100), reward_fn, env_info
        )

    task_returns, reward_returns = train_ppo(
        env_fn=env_fn,
        total_steps=5000,
        seeds=[42],
        env_info=env_info,
    )

    assert len(task_returns) == 1
    assert len(task_returns[0]) > 0
    assert len(reward_returns) == 1
    assert len(reward_returns[0]) == len(task_returns[0])
    assert all(r >= 0 for r in task_returns[0])

from reward import Reward
import numpy as np


class DefaultCartPole(Reward):
    name = "baseline"
    env_id = "CartPole-v1"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        return env_reward


class DefaultAcrobot(Reward):
    name = "baseline"
    env_id = "Acrobot-v1"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        return env_reward


class DefaultMountainCar(Reward):
    name = "baseline"
    env_id = "MountainCar-v0"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        return env_reward


class DefaultPendulum(Reward):
    name = "baseline"
    env_id = "Pendulum-v1"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        return env_reward

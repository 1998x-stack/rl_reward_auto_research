from typing import List

import numpy as np

from env_info import EnvInfo


class Reward:
    name: str = "UnnamedReward"
    env_id: str = ""

    def compute(
        self,
        obs: np.ndarray,
        action,
        env_reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        env_info: EnvInfo,
    ) -> float:
        raise NotImplementedError("Subclasses must implement compute()")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.name or cls.name == "UnnamedReward":
            raise TypeError(f"{cls.__name__} must define a 'name' class attribute")
        if not cls.env_id:
            raise TypeError(f"{cls.__name__} must define an 'env_id' class attribute")


def discover_rewards(module_name: str = "reward") -> List[Reward]:
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return []

    discovered = []
    for attr_name in dir(module):
        if not attr_name.startswith("Reward") or attr_name == "Reward":
            continue
        obj = getattr(module, attr_name)
        if isinstance(obj, type) and issubclass(obj, Reward) and obj is not Reward:
            try:
                instance = obj()
                discovered.append(instance)
            except Exception:
                continue

    return discovered


class RewardBaselineCartPole(Reward):
    name = "baseline"
    env_id = "CartPole-v1"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        return env_reward


class RewardAnglePenalty(Reward):
    name = "angle_penalty"
    env_id = "CartPole-v1"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        pole_angle = abs(obs[2])
        return env_reward - 0.5 * pole_angle


class RewardBalancedPenalty(Reward):
    name = "balanced_penalty"
    env_id = "CartPole-v1"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        pole_angle = abs(obs[2])
        cart_pos = abs(obs[0]) / 2.4
        return env_reward - 0.3 * pole_angle - 0.1 * cart_pos


class RewardPotentialShaping(Reward):
    name = "potential_shaping"
    env_id = "CartPole-v1"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        pole_angle = obs[2]
        potential = -abs(pole_angle)
        return env_reward + 0.5 * potential


class RewardBaselineAcrobot(Reward):
    name = "baseline"
    env_id = "Acrobot-v1"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        return env_reward


class RewardBaselineMountainCar(Reward):
    name = "baseline"
    env_id = "MountainCar-v0"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        return env_reward


class RewardBaselinePendulum(Reward):
    name = "baseline"
    env_id = "Pendulum-v1"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        return env_reward

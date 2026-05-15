from dataclasses import dataclass
from typing import List, Literal

import numpy as np


@dataclass(frozen=True)
class EnvInfo:
    env_id: str
    obs_dim: int
    obs_low: np.ndarray
    obs_high: np.ndarray
    obs_names: List[str]
    action_type: Literal["discrete", "continuous"]
    action_dim: int
    max_return: float
    reward_direction: Literal["positive", "negative"]

    def __post_init__(self):
        if len(self.obs_names) != self.obs_dim:
            raise ValueError(
                f"{self.env_id}: obs_names length ({len(self.obs_names)}) "
                f"!= obs_dim ({self.obs_dim})"
            )


ENV_REGISTRY: dict[str, EnvInfo] = {
    "CartPole-v1": EnvInfo(
        env_id="CartPole-v1",
        obs_dim=4,
        obs_low=np.array([-4.8, -np.inf, -0.418, -np.inf]),
        obs_high=np.array([4.8, np.inf, 0.418, np.inf]),
        obs_names=["cart_pos", "cart_vel", "pole_angle", "pole_angular_vel"],
        action_type="discrete",
        action_dim=2,
        max_return=500.0,
        reward_direction="positive",
    ),
    "Acrobot-v1": EnvInfo(
        env_id="Acrobot-v1",
        obs_dim=6,
        obs_low=np.array([-1.0, -1.0, -1.0, -1.0, -12.57, -28.27]),
        obs_high=np.array([1.0, 1.0, 1.0, 1.0, 12.57, 28.27]),
        obs_names=[
            "cos_theta1", "sin_theta1", "cos_theta2", "sin_theta2",
            "theta1_dot", "theta2_dot",
        ],
        action_type="discrete",
        action_dim=3,
        max_return=0.0,
        reward_direction="negative",
    ),
    "MountainCar-v0": EnvInfo(
        env_id="MountainCar-v0",
        obs_dim=2,
        obs_low=np.array([-1.2, -0.07]),
        obs_high=np.array([0.6, 0.07]),
        obs_names=["position", "velocity"],
        action_type="discrete",
        action_dim=3,
        max_return=0.0,
        reward_direction="negative",
    ),
    "Pendulum-v1": EnvInfo(
        env_id="Pendulum-v1",
        obs_dim=3,
        obs_low=np.array([-1.0, -1.0, -8.0]),
        obs_high=np.array([1.0, 1.0, 8.0]),
        obs_names=["cos_theta", "sin_theta", "theta_dot"],
        action_type="continuous",
        action_dim=1,
        max_return=0.0,
        reward_direction="negative",
    ),
}

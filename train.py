from typing import Callable, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env_info import EnvInfo


class PPONetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, discrete: bool):
        super().__init__()
        self.discrete = discrete
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.shared:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)
        if self.discrete:
            return F.log_softmax(logits, dim=-1), value
        return logits, value

    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, value = self.forward(x)
            if self.discrete:
                probs = logits.exp()
                if deterministic:
                    action = probs.argmax(dim=-1).item()
                else:
                    action = torch.multinomial(probs, 1).item()
                log_prob = logits[0, action].item()
            else:
                mean = logits[0]
                std = torch.ones_like(mean) * 0.5
                dist = torch.distributions.Normal(mean, std)
                if deterministic:
                    action = mean.numpy()
                else:
                    action = dist.sample().numpy()
                log_prob = dist.log_prob(torch.as_tensor(action)).sum().item()
            return action, log_prob, value.item()


class RewardEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_fn, env_info: EnvInfo):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.env_info = env_info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        modified_reward = self.reward_fn.compute(
            obs=obs,
            action=action,
            env_reward=env_reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            env_info=self.env_info,
        )
        info["_env_reward"] = float(env_reward)
        info["_agent_reward"] = float(modified_reward)
        return obs, float(modified_reward), terminated, truncated, info


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    rets = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
        rets[t] = advantages[t] + values[t]
        next_value = values[t]
    return advantages, rets


def train_ppo(
    env_fn: Callable[[], gym.Env],
    total_steps: int,
    seeds: List[int],
    env_info: EnvInfo,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    update_epochs: int = 4,
    batch_size: int = 64,
) -> Tuple[List[List[float]], List[List[float]]]:
    all_task_returns = []
    all_reward_returns = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = env_fn()
        obs_dim = env_info.obs_dim
        action_dim = env_info.action_dim
        discrete = (env_info.action_type == "discrete")

        network = PPONetwork(obs_dim, action_dim, discrete)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)

        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf = [], []
        task_ep_returns, reward_ep_returns = [], []
        task_ep_acc, reward_ep_acc = 0.0, 0.0
        step_count = 0

        obs, _ = env.reset()

        while step_count < total_steps:
            action, log_prob, value = network.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            env_reward = info.get("_env_reward", reward)
            agent_reward = reward

            task_ep_acc += env_reward
            reward_ep_acc += agent_reward

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(log_prob)
            val_buf.append(value)
            rew_buf.append(agent_reward)
            done_buf.append(terminated or truncated)

            step_count += 1
            obs = next_obs

            if terminated or truncated:
                task_ep_returns.append(task_ep_acc)
                reward_ep_returns.append(reward_ep_acc)
                task_ep_acc, reward_ep_acc = 0.0, 0.0
                obs, _ = env.reset()

            if len(obs_buf) >= batch_size:
                obs_t = torch.as_tensor(np.array(obs_buf), dtype=torch.float32)
                logp_old = torch.as_tensor(np.array(logp_buf), dtype=torch.float32)
                val_arr = np.array(val_buf)
                rew_arr = np.array(rew_buf)
                don_arr = np.array(done_buf, dtype=np.float32)

                advantages, returns = compute_gae(rew_arr, val_arr, don_arr, gamma, lam)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                adv_t = torch.as_tensor(advantages, dtype=torch.float32)
                ret_t = torch.as_tensor(returns, dtype=torch.float32)

                for _ in range(update_epochs):
                    _, values = network.forward(obs_t)
                    values = values.squeeze(-1)

                    if discrete:
                        logits, _ = network.forward(obs_t)
                        act_idx = torch.as_tensor(act_buf, dtype=torch.long)
                        new_logp = logits[torch.arange(len(act_idx)), act_idx]
                    else:
                        mean, _ = network.forward(obs_t)
                        std = torch.ones_like(mean) * 0.5
                        dist = torch.distributions.Normal(mean, std)
                        act_pt = torch.as_tensor(np.array(act_buf), dtype=torch.float32)
                        new_logp = dist.log_prob(act_pt).sum(dim=-1)

                    ratio = (new_logp - logp_old).exp()
                    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                    actor_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()
                    critic_loss = F.mse_loss(values, ret_t)
                    loss = actor_loss + 0.5 * critic_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                    optimizer.step()

                obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
                rew_buf, done_buf = [], []

        if task_ep_acc != 0 or reward_ep_acc != 0:
            task_ep_returns.append(task_ep_acc)
            reward_ep_returns.append(reward_ep_acc)

        all_task_returns.append(task_ep_returns)
        all_reward_returns.append(reward_ep_returns)
        env.close()

    return all_task_returns, all_reward_returns

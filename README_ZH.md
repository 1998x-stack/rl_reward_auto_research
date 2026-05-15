# RL Reward Autoresearch 🔬

<p align="center">
  <b>AI 自主强化学习奖励函数研究</b><br>
  <i>Agent 在你睡觉时发明、迭代、优化 RL 奖励函数。</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/tests-32_passed-10B981" alt="Tests">
  <img src="https://img.shields.io/badge/rewards-7_evaluated-7C3AED" alt="Rewards">
  <img src="https://img.shields.io/badge/frontier-3_non--dominated-F59E0B" alt="Frontier">
  <img src="https://img.shields.io/badge/license-MIT-00A858" alt="License">
  <img src="https://img.shields.io/badge/framework-gymnasium-ff6b35" alt="Gymnasium">
  <img src="https://img.shields.io/badge/algorithm-PPO-8B5CF6" alt="PPO">
</p>

---

## 💡 这是什么？

**灵感源自 [Karpathy 的 autoresearch](https://github.com/karpathy/autoresearch)**——应用于强化学习领域。

一个 AI agent 在夜间自主运行实验循环：
1. 修改 `reward.py`——发明新的奖励函数
2. 在 4 个 gymnasium 经典控制环境上训练 PPO
3. 计算 3 个第一性原理的 Pareto 指标——任务性能、样本效率、训练稳定性
4. 只保留非支配奖励函数，持续扩展 Pareto 前沿

**~4-7 次实验/小时。~30-50 次/晚。零人工干预。**

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  reward.py   │────▶│  train.py    │────▶│  prepare.py      │
│ Agent 编辑   │     │  PPO 训练    │     │  3 指标          │──▶ pareto_frontier.json
│ 1-5 Rewards  │     │  只读        │     │  Pareto 判定     │    (非支配前沿)
└──────────────┘     └──────────────┘     └──────────────────┘
```

---

## 🚀 快速开始

```bash
git clone https://github.com/1998x-stack/rl-reward-autoresearch.git
cd rl_reward_autoresearch
uv sync                          # 安装依赖
uv run python prepare.py         # 首次运行：建立基线 + 评估奖励函数
```

> **开箱即用。** 无需外部数据——gymnasium 环境内置。首次运行自动建立基线。

---

## 📊 三个第一性原理指标

奖励函数只有在**预测任务完成度、持续稳定、训练高效**时才有价值。

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| **task_perf** | `mean(最终回报) / baseline回报` | 更强的任务完成能力 |
| **task_eff** | `AUC(回报曲线) / baseline_AUC` | 更快的样本学习效率 |
| **task_stab** | `1 − CV(跨seed最终回报)` | 更一致的训练稳定性 |

三者形成 **Pareto 前沿**——无法同时最大化。Agent 在探索中揭示 trade-off 曲面。

> 🔒 **防作弊设计：** 指标基于**原始环境奖励**（`env_reward`）计算，而非 agent 自己设计的奖励。agent 无法通过写 `return 999.0` 来 hack 指标。详见 [ADR 0001](docs/adr/0001-dual-track-metrics.md)。

---

## 🏗️ 架构

| 文件 | 角色 | 谁修改 |
|------|------|--------|
| `reward.py` | 奖励函数定义——每次实验 1-5 个 `Reward*` 子类 | **AI agent** |
| `train.py` | PPO 训练循环（CleanRL 风格，约 190 行） | **只读** |
| `prepare.py` | 评估框架——环境管理、指标计算、Pareto 逻辑、前沿存档 | **只读** |
| `program.md` | Agent 指令——6 条迭代原则、实验循环协议、策略选择 | **人类** |
| `env_info.py` | 环境元数据注册表——4 个经典控制环境的观测语义 | **人类** |
| `pareto_frontier.json` | 研究存档——所有非支配奖励函数 | **Git 跟踪** |

---

## 🔬 首次实验结果

CartPole-v1 首次运行：3 个奖励函数，发现 **+38.2% 性能提升**。

| 奖励函数 | task_perf | task_stab | 设计思路 |
|----------|-----------|-----------|---------|
| baseline | 1.000 | 1.000 | 环境默认 +1/步 |
| angle_penalty | 1.176 | 0.718 | 惩罚杆子角度偏移 |
| **balanced_penalty** | **1.382** | 0.831 | 角度 + 位置双重惩罚 |

📖 **[详细实验报告 (中文)](docs/cartpole-experiment-report-zh.md)**

---

## ✍️ 编写奖励函数

```python
from reward import Reward
import numpy as np

class RewardAnglePenalty(Reward):
    name = "angle_penalty"
    env_id = "CartPole-v1"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        pole_angle = abs(obs[2])
        return env_reward - 0.5 * pole_angle
```

仅此而已。下次 `uv run python prepare.py` 自动发现并评估。

---

## 🧪 测试

```bash
uv run pytest tests/ -v     # 32 tests: 8 metrics + 8 pareto + 6 reward + 5 env + 4 integration + 1 train
```

---

## 🎯 支持的环境

| 环境 | 动作空间 | 观测维度 | 关键挑战 |
|------|---------|---------|---------|
| CartPole-v1 | 离散 (2) | 4 | 稠密奖励，塑形空间小 |
| Acrobot-v1 | 离散 (3) | 6 | 稀疏奖励，塑形收益大 |
| MountainCar-v0 | 离散 (3) | 2 | 经典稀疏奖励，塑形决定性 |
| Pendulum-v1 | 连续 (1) | 3 | 连续动作，塑形空间丰富 |

---

## ⚡ 设计原则

- **单文件编辑面**——agent 只碰 `reward.py`
- **不可变评估**——`train.py`、`prepare.py` 永不修改
- **Pareto 优化**——多目标而非单一数字
- **双轨指标**——任务指标（防作弊）+ 奖励指标（诊断反馈）
- **简洁优先**——3 行 reward 达到 perf=1.1 > 30 行达到 perf=1.11
- **首次自动基线**——第一次运行自动建立 identity reward 基线

---

## 🧭 迭代策略

| 策略 | 触发条件 | 动作 |
|------|---------|------|
| **Exploit** | 默认 | 对前沿最优 reward 做局部变异 |
| **Explore** | 连续 3+ 次全丢弃 | 发明全新奖励结构 |
| **Combine** | 每 10 次实验 | 合并两个前沿 reward |

---

## 📚 文档

| | 语言 | 内容 |
|---|------|------|
| [README.md](README.md) | EN | 完整项目文档（英文） |
| [CONTEXT.md](CONTEXT.md) | EN | 领域术语表 |
| [program.md](program.md) | EN | Agent 指令文件 |
| [实验报告](docs/cartpole-experiment-report-zh.md) | 中文 | CartPole 首次实验分析 |
| [ADR 0001](docs/adr/0001-dual-track-metrics.md) | EN | 双轨指标架构决策 |

---

## 📄 协议

MIT

---

<p align="center">
  <sub>灵感源自 <a href="https://github.com/karpathy/autoresearch">@karpathy/autoresearch</a> 和 <a href="https://github.com/1998x-stack/alpha-autoresearch">alpha_autoresearch</a></sub>
</p>

# RL Reward Autoresearch 🔬

<p align="center">
  <b>Autonomous RL Reward Function Research</b><br>
  <i>AI agents invent, iterate, and optimize reinforcement learning reward functions — while you sleep.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/tests-32_passed-10B981" alt="Tests">
  <img src="https://img.shields.io/badge/rewards-7_evaluated-7C3AED" alt="Rewards">
  <img src="https://img.shields.io/badge/frontier-3_non--dominated-F59E0B" alt="Frontier">
  <img src="https://img.shields.io/badge/license-MIT-00A858" alt="License">
  <img src="https://img.shields.io/badge/framework-gymnasium-ff6b35" alt="Gymnasium">
  <img src="https://img.shields.io/badge/algorithm-PPO-8B5CF6" alt="PPO">
  <a href="https://1998x-stack.github.io/rl_reward_auto_research/"><img src="https://img.shields.io/badge/%F0%9F%8C%90-GitHub_Pages-6366f1" alt="GitHub Pages"></a>
</p>

---

## 💡 What is this?

**Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)** — applied to reinforcement learning.

An AI agent autonomously runs an experiment loop overnight:
1. Modifies `reward.py` — inventing new reward functions
2. Trains PPO on 4 gymnasium classic-control environments
3. Checks 3 Pareto metrics — task performance, sample efficiency, training stability
4. Keeps only non-dominated reward functions, expanding the frontier

**~4-7 experiments/hour. ~30-50 overnight. Zero human intervention.**

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  reward.py   │────▶│  train.py    │────▶│  prepare.py      │
│ Agent edits  │     │  PPO loop    │     │  3 metrics       │──▶ pareto_frontier.json
│ 1-5 Rewards  │     │  Read-only   │     │  Pareto decision │    (non-dominated)
└──────────────┘     └──────────────┘     └──────────────────┘
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/1998x-stack/rl_reward_auto_research.git
cd rl_reward_auto_research
uv sync                          # install deps
uv run python prepare.py         # first run: establish baselines + evaluate rewards
```

> **Out of the box.** No external data needed — gymnasium environments are bundled. First run auto-establishes baselines.

---

## 📊 Three First-Principles Metrics

A reward function is only useful if it produces strong policies, learns fast, and trains consistently.

| Metric | Computation | Means |
|--------|------------|-------|
| **task_perf** | `mean(final_returns) / baseline_return` | Stronger task completion |
| **task_eff** | `AUC(return_curve) / baseline_AUC` | Faster sample learning |
| **task_stab** | `1 − CV(final_returns across seeds)` | More consistent training |

These form a **Pareto frontier** — you can't maximize all three simultaneously. The agent discovers the tradeoff surface.

> 🔒 **Anti-cheat:** All Pareto metrics are computed from the **original environment reward**, not the agent's designed reward. The agent cannot hack metrics by writing `return 999.0`. Reward metrics (from agent reward) are also reported as diagnostic feedback. See [ADR 0001](docs/adr/0001-dual-track-metrics.md).

---

## 🏗️ Architecture

| File | Role | Modified by |
|------|------|-------------|
| `reward.py` | Reward function definitions — 1–5 `Reward*` subclasses per experiment | **AI agent** |
| `train.py` | PPO training loop (CleanRL style, ~190 lines). Dual-reward tracking. | **Read-only** |
| `prepare.py` | Evaluation harness — environment management, metrics, Pareto, archive | **Read-only** |
| `env_info.py` | Static environment registry — observation semantics for 4 envs | **Human** |
| `program.md` | Agent instructions — 6 iteration principles, experiment loop, strategy selection | **Human** |

### 4 Supported Environments

| Environment | Action Space | Obs Dim | Reward Type | Key Challenge |
|-------------|-------------|---------|-------------|---------------|
| CartPole-v1 | Discrete (2) | 4 | Dense (+1/step) | Shaping space small, efficiency focus |
| Acrobot-v1 | Discrete (3) | 6 | Sparse (-1/step) | Sparse→dense conversion high payoff |
| MountainCar-v0 | Discrete (3) | 2 | Sparse (-1/step) | Classic hard case, shaping decisive |
| Pendulum-v1 | Continuous (1) | 3 | Dense (negative cost) | Continuous action, rich shaping space |

---

## 🔬 Experiment Results

First experiment on CartPole-v1: 3 reward functions, **+38.2% performance improvement**.

| Reward Function | task_perf | task_stab | Design |
|-----------------|-----------|-----------|--------|
| baseline | 1.000 | 1.000 | Identity (env default +1/step) |
| angle_penalty | 1.176 | 0.718 | Penalize pole angle deviation |
| **balanced_penalty** | **1.382** | 0.831 | Angle + position dual penalty |

📖 **[Full Experiment Report (Chinese)](docs/cartpole-experiment-report-zh.md)**

---

## ✍️ Writing a Reward Function

```python
from reward import Reward
import numpy as np

class RewardMyDesign(Reward):
    name = "my_design"
    env_id = "CartPole-v1"

    def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
        pole_angle = abs(obs[2])
        return env_reward - 0.3 * pole_angle
```

That's it. Auto-discovered on next `uv run python prepare.py`.

---

## 🧪 Tests

```bash
uv run pytest tests/ -v     # 32 tests: 8 metrics + 8 pareto + 6 reward + 5 env + 4 integration + 1 train
```

---

## 🧭 Iteration Strategies

| Strategy | When | Action |
|----------|------|--------|
| **Exploit** | Default | Modify best frontier reward slightly (adjust coefficient, change shaping form) |
| **Explore** | 3+ consecutive discards | Invent completely new reward structure |
| **Combine** | Every 10th experiment | Merge two frontier rewards (weighted average, product) |

---

## 📚 Documentation

| | Language | Content |
|---|----------|---------|
| [README_ZH.md](README_ZH.md) | 中文 | Full project documentation |
| [Experiment Report](docs/cartpole-experiment-report-zh.md) | 中文 | CartPole first experiment analysis |
| [CONTEXT.md](CONTEXT.md) | EN | Domain glossary |
| [program.md](program.md) | EN | Agent instruction file |
| [ADR 0001](docs/adr/0001-dual-track-metrics.md) | EN | Dual-track metrics architecture decision |
| [GitHub Pages](https://1998x-stack.github.io/rl_reward_auto_research/) | — | Project landing page |

---

## ⚡ Design Principles

- **Single edit surface** — agent only touches `reward.py`
- **Immutable evaluation** — `train.py`, `prepare.py` never change
- **Pareto optimization** — multi-objective, not single-number
- **Dual-track metrics** — task metrics (anti-cheat) + reward metrics (diagnostic)
- **Simplicity bias** — 3-line reward at perf=1.1 > 30-line at perf=1.11
- **Auto-baseline** — first experiment establishes identity-reward baselines

---

## 📄 License

MIT

---

<p align="center">
  <sub>Inspired by <a href="https://github.com/karpathy/autoresearch">@karpathy/autoresearch</a> and <a href="https://github.com/1998x-stack/alpha-autoresearch">alpha_autoresearch</a></sub>
</p>

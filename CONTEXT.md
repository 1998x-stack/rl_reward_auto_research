# rl_reward_autoresearch — Domain Context

## Core Concepts

### Agent Reward Function
The code the AI agent writes in `reward.py`. A `Reward*` subclass whose `compute()` method maps `(obs, action, env_reward, ...)` → `float`. This value is used for PPO gradient updates during training. The agent can freely modify, replace, or augment the environment's built-in reward. Contrast with **Environment Reward**.

### Environment Reward (env_reward)
The built-in reward signal from `gymnasium.Env.step()`. Never modified by the agent. Used exclusively for computing evaluation **Task Metrics**. This separation prevents the agent from "hacking" its own evaluation by designing degenerate reward functions.

### Task Metrics
Three metrics computed from the original **Environment Reward** that form the Pareto frontier:
- `task_perf` (Performance): How well the trained policy performs the underlying task
- `task_eff` (Sample Efficiency): How quickly the policy learns (AUC of return curve)
- `task_stab` (Stability): Consistency of final performance across random seeds

**These metrics define the Pareto frontier** — they measure genuine task competence.

### Reward Metrics
Three metrics computed from the **Agent Reward Function** that provide diagnostic feedback:
- `reward_perf`: Terminal value of the agent's own reward signal
- `reward_eff`: AUC of the agent's reward curve
- `reward_stab`: Cross-seed stability of agent reward

**These are diagnostic only** — they help the agent understand which reward designs correlate with strong task performance, but they do NOT determine Pareto status.

### Pareto Frontier
The set of non-dominated reward functions within each environment. Reward A **dominates** Reward B if A ≥ B on all three **Task Metrics** AND A > B on at least one. The frontier is maintained in `pareto_frontier.json` and organized per environment.

### Experiment
One cycle of the agent loop: modify `reward.py` (write 1-5 `Reward*` classes) → git commit → `uv run prepare.py` → PPO training (100K steps × 3 seeds) → compute dual-track metrics → Pareto check → keep/discard → log to `results.tsv`.

### Baseline
The identity reward function (`return env_reward`) run once during the first experiment on all 4 environments. Its Task Metrics serve as normalization denominators for all subsequent experiments (performance = actual / baseline). Stored as reference constants in `pareto_frontier.json`.

### Edit Surface
The single file (`reward.py`) that the agent modifies. Contains `Reward*` subclasses using `numpy` operations and accessing `EnvInfo` for environment metadata. Analogous to `factors.py` in `alpha_autoresearch`.

### Partial KEEP
When an experiment produces mixed results (some rewards KEEP, some DISCARD), only KEEP rewards are retained in `reward.py`. DISCARD classes are removed and the commit is amended. This preserves valuable discoveries while discarding failures.

## Environment Knowledge (`env_info.py`)

Observation semantics are stored in `env_info.py` — a Python module containing the `EnvInfo` dataclass and `ENV_REGISTRY` dictionary. Each environment entry includes:
- Observation dimension names and meanings
- Action space type and semantics
- Built-in reward structure
- Theoretical performance bounds
- Known reward shaping patterns

New environments are added to `env_info.md` by humans, not the agent.

## Strategy Terminology

| Term | Meaning |
|------|---------|
| **KEEP** | Reward is non-dominated (or dominates an existing frontier member). Amend commit to retain, update `pareto_frontier.json`. |
| **DISCARD** | Reward is dominated by ALL frontier members. Remove from `reward.py`, do not archive. |
| **CRASH** | Training failed (NaN loss, divergence, exception). Log to `results.tsv` with zero metrics, skip. |
| **Exploit** | Strategy: modify the best frontier reward (adjust coefficient, change shaping form, add/remove one term) |
| **Explore** | Strategy: invent a completely new reward structure or use a different shaping philosophy |
| **Combine** | Strategy: merge two frontier rewards (weighted average, hierarchical, conditional switch) |

## Relationship to Existing Systems

| Concept | alpha_autoresearch | rl_reward_autoresearch |
|---------|-------------------|----------------------|
| Agent's artifact | Factor | Agent Reward Function |
| Ground truth signal | Forward returns | Environment Reward |
| Evaluation metric | rank_ic | task_perf |
| Stability metric | ic_ir | task_eff (sample efficiency) |
| Tradeability metric | turnover_stability | task_stab (training stability) |
| Edit surface | factors.py | reward.py |
| Immutable harness | prepare.py (ops + metrics) | prepare.py (envs + metrics) |
| Archive | pareto_frontier.json | pareto_frontier.json |
| Strategy triad | exploit/explore/combine | exploit/explore/combine |

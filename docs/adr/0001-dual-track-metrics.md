# ADR 0001: Dual-Track Metrics — Task vs Reward

**Status:** Accepted | **Date:** 2026-05-15

## Context

The agent writes reward functions (in `reward.py`) that modify the reward signal used for PPO training. We need to evaluate whether a reward function is "good." There are two fundamentally different ways to measure this:

1. **Task performance:** Does the trained policy actually solve the environment's objective (e.g., balance the pole, reach the flag)?
2. **Reward quality:** Is the agent's designed reward signal well-correlated with good learning behavior?

A naive single-track approach — evaluating only the agent's own reward — creates a reward hacking vulnerability: the agent could write `return 999.0` and claim perfect metrics even though the pole is falling.

## Decision

We use **dual-track metrics**:

| Track | Source | Purpose |
|-------|--------|---------|
| **Task metrics** | Original `env_reward` from `gymnasium.Env.step()` | Pareto frontier — measures genuine task competence |
| **Reward metrics** | Agent's modified reward (return value of `compute()`) | Diagnostic feedback — helps agent learn which designs work |

The Pareto frontier uses **task metrics only**. The reward metrics are purely diagnostic.

## Alternatives Considered

### A. Single-track: agent reward only
- **Rejected.** Trivially hackable. Agent can inflate metrics by designing degenerate rewards. Fails to measure actual task completion.

### B. Single-track: environment reward only
- **Rejected.** Agent gets no feedback on its reward design quality. Cannot distinguish between "good reward that leads to strong task performance" and "lucky initialization that happened to work."

### C. Dual-track (selected)
- Both metric sets computed. Pareto on task metrics prevents hacking. Reward metrics provide learning signal for the agent.

## Consequences

- **Positive:** Agent cannot hack the Pareto frontier. Task metrics always reflect actual environment performance.
- **Positive:** Reward metrics give agent rich diagnostic feedback — it can observe whether reward design improvements correlate with task improvements.
- **Negative:** Output format is more verbose (6 metrics instead of 3).
- **Negative:** For the baseline (identity reward), task metrics and reward metrics are identical — this is a special case, not the norm.
- **Design constraint:** `train.py` must track both reward signals during rollout and return separate episode return curves for each.

## Cross-Reference

- Analogous to `alpha_autoresearch` where the agent writes factor formulas (`factors.py`) but `prepare.py` independently computes `rank_ic` using forward returns — the ground truth signal the agent cannot manipulate.
- Analogous to `autoresearch` where the agent modifies `train.py` but `evaluate_bpb` in `prepare.py` is the immutable evaluation function.

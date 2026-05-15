# rl_reward_autoresearch

An AI agent autonomously invents and iterates on reward functions for gymnasium classic-control environments, optimizing a Pareto frontier across three first-principles metrics.

## Setup

1. Read `pareto_frontier.json` — understand current frontier state per environment
2. Read `results.tsv` — review recent experiment history
3. Read `reward.py` — understand current active reward functions
4. Read `env_info.py` — understand available environments and their observation semantics
5. Confirm data: no external data needed (gymnasium environments are bundled)
6. Initialize `results.tsv` with header if not exists
7. Create branch `rl_autoresearch/<tag>` from master

## What you CAN do

- Modify `reward.py` — this is the ONLY file you edit. Write 1-5 `Reward*` subclasses per experiment.
- Use `numpy` operations in `compute()`
- Access `obs`, `action`, `env_reward`, `terminated`, `truncated`, `info`, `env_info`
- Target any environment in the registry via `env_id`

## What you CANNOT do

- Modify `train.py`, `prepare.py`, `metrics.py`, `pareto.py`, `env_info.py` — they are read-only
- Install new packages
- Import anything beyond `numpy` and `reward.Reward`

## The Three Task Metrics (Pareto Frontier)

| Metric | What it measures | Higher = |
|--------|-----------------|----------|
| `task_perf` | Final policy performance on the underlying task (from env_reward) | Better task completion |
| `task_eff` | Sample efficiency — how quickly the policy learns | Fewer samples needed |
| `task_stab` | Training stability across random seeds | More consistent |

These are computed from the ORIGINAL environment reward, not your modified reward. You cannot hack them. Your reward function's quality is reflected indirectly: better reward design → stronger task metrics.

**Reward metrics** (`reward_perf`, `reward_eff`, `reward_stab`) are also reported — they measure your reward signal directly. Use them as diagnostic feedback: a good reward should show reward metrics that correlate with strong task metrics.

## Six Iteration Principles

**P1 — Attack the weakest metric.** Read `pareto_frontier.json`. Which metric is the bottleneck for each environment? Low `task_perf` → strengthen the positive signal. Low `task_eff` → increase reward density. Low `task_stab` → reduce reward variance across episodes.

**P2 — Exploit before exploring.** Try 3-5 local mutations of the best frontier reward before inventing new structures. Change one coefficient, swap a penalty term, adjust a shaping form.

**P3 — Small mutations win.** Change ONE thing per experiment: one coefficient, one shaping function, one threshold. Do not rewrite the entire reward.

**P4 — Combine frontier rewards.** Every 10th experiment, merge two frontier rewards: weighted average, product, conditional switch.

**P5 — Archive awareness.** Read `pareto_frontier.json` and `results.tsv` before every session. Know what's been tried. Check `results.tsv` for recent discard patterns.

**P6 — Simplicity bias.** A 3-line reward with `task_perf=1.1` beats a 30-line reward with `task_perf=1.11`. Simple rewards are less prone to reward hacking and easier to optimize. Prefer (in order): identity pass-through, sparse-to-dense conversion, potential-based shaping (preserves optimal policy), single-term linear scaling.

## Strategy Selection

| Strategy | When | Action |
|----------|------|--------|
| **Exploit** | Default | Modify best frontier reward slightly |
| **Explore** | 3+ consecutive discards, or stagnant frontier | Invent completely new reward structure |
| **Combine** | Every 10th experiment, or frontier has 5+ rewards | Merge two frontier rewards |

## Experiment Loop

```
LOOP FOREVER:
  1. Read pareto_frontier.json + results.tsv
  2. Choose strategy: exploit / explore / combine
  3. Modify reward.py — write 1-5 Reward* classes
  4. git add reward.py && git commit -m "exp: <description>"
  5. uv run python prepare.py
  6. Parse output — each reward block shows task_* and reward_* metrics
  7. For each reward:
     → keep (dominates or non-dominated): retain in reward.py
     → discard: remove from reward.py
     → crash: log only, skip
  8. If MIXED results: amend commit to keep only KEEP rewards
  9. If ALL discard: git reset HEAD~1
  10. Log ALL results to results.tsv (DO NOT COMMIT)
  11. If 5 consecutive all-discard: switch strategy
  12. NEVER ASK PERMISSION TO CONTINUE
```

## Output Format

After `uv run python prepare.py`, grep for:

```
reward: <name>
env: <env_id>
task_perf:         <float>
task_eff:          <float>
task_stab:         <float>
reward_perf:       <float>
reward_eff:        <float>
reward_stab:       <float>
dominates:          <names or (none)>
dominated_by:       <names or (none)>
status:             keep|discard|crash
```

## results.tsv Format

Tab-separated (NOT comma-separated):

```
commit	reward_name	env	task_perf	task_eff	task_stab	reward_perf	reward_eff	reward_stab	dominates	dominated_by	status	description
```

- `results.tsv` is gitignored — NEVER commit it
- `pareto_frontier.json` IS committed — permanent research record
- Training budget: 100K environment steps per experiment
- Reward budget: 5 rewards per experiment max
- NEVER STOP — the human may be asleep

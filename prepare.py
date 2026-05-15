import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import gymnasium as gym
import numpy as np
from loguru import logger

from env_info import ENV_REGISTRY, EnvInfo
from reward import discover_rewards, Reward
from train import train_ppo, RewardEnvWrapper
from metrics import compute_all_metrics
from pareto import (
    METRIC_NAMES,
    pareto_decision,
    update_archive,
    load_archive,
)

TOTAL_STEPS = 100_000
NUM_SEEDS = 3
MAX_REWARDS_PER_EXPERIMENT = 5
PROJECT_ROOT = Path(__file__).resolve().parent
ARCHIVE_PATH = PROJECT_ROOT / "pareto_frontier.json"
DEFAULT_SEEDS = [42, 123, 777]


def make_env_with_reward(reward_fn: Reward, env_info: EnvInfo) -> gym.Env:
    env = gym.make(env_info.env_id)
    return RewardEnvWrapper(env, reward_fn, env_info)


def establish_baselines(output_results: List[Dict]) -> Dict[str, Dict]:
    archive = load_archive(str(ARCHIVE_PATH))
    baselines = {}

    for env_id, env_info in ENV_REGISTRY.items():
        existing = (
            archive.get("environments", {})
            .get(env_id, {})
            .get("baseline_return", 0.0)
        )
        if abs(existing) > 1e-8:
            logger.info(f"Baseline already exists for {env_id}: {existing:.2f}")
            baselines[env_id] = {
                "baseline_return": existing,
                "baseline_returns": (
                    archive["environments"][env_id].get("baseline_returns", [])
                ),
            }
            continue

        logger.info(f"Establishing baseline for {env_id}...")

        _env_id = env_id
        class BaselineReward(Reward):
            name = "baseline"
            env_id = _env_id

            def compute(self, obs, action, env_reward, terminated, truncated, info, env_info):
                return env_reward

        def env_fn():
            return make_env_with_reward(BaselineReward(), env_info)

        task_returns, _ = train_ppo(
            env_fn=env_fn,
            total_steps=TOTAL_STEPS,
            seeds=DEFAULT_SEEDS[:NUM_SEEDS],
            env_info=env_info,
        )

        # Average final return across all seeds for robust baseline
        seed_finals = []
        for returns in task_returns:
            n = min(10, len(returns))
            seed_finals.append(float(np.mean(returns[-n:])))
        baseline_return = float(np.mean(seed_finals))

        # Use first seed's episode returns as reference curve for AUC normalization
        primary_returns = task_returns[0]

        baselines[env_id] = {
            "baseline_return": baseline_return,
            "baseline_returns": primary_returns,
        }

        if "environments" not in archive:
            archive["environments"] = {}
        if env_id not in archive["environments"]:
            archive["environments"][env_id] = {
                "frontier": [],
                "baseline_return": 0.0,
                "baseline_returns": [],
            }
        archive["environments"][env_id]["baseline_return"] = baseline_return
        archive["environments"][env_id]["baseline_returns"] = primary_returns

        baseline_info = {
            "name": "baseline",
            "perf": 1.0,
            "eff": 1.0,
            "stab": 1.0,
            "description": "Identity reward (environment default)",
            "commit": "",
            "added": datetime.now().isoformat(),
        }
        archive["environments"][env_id]["frontier"].append(baseline_info)

        output_results.append({
            "reward_name": "baseline",
            "env": env_id,
            "task_metrics": {"perf": 1.0, "eff": 1.0, "stab": 1.0},
            "reward_metrics": {"perf": 1.0, "eff": 1.0, "stab": 1.0},
            "status": "keep",
            "dominates": [],
            "dominated_by": [],
        })

        logger.info(f"  Baseline for {env_id}: return={baseline_return:.2f}")

    with open(ARCHIVE_PATH, "w") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)

    return baselines


def evaluate_all_rewards() -> Dict[str, Dict]:
    archive = load_archive(str(ARCHIVE_PATH))
    baselines = {
        env_id: {
            "baseline_return": (
                archive.get("environments", {})
                .get(env_id, {})
                .get("baseline_return", 0.0)
            ),
            "baseline_returns": (
                archive.get("environments", {})
                .get(env_id, {})
                .get("baseline_returns", [])
            ),
        }
        for env_id in ENV_REGISTRY
    }

    need_baseline = any(
        abs(b["baseline_return"]) < 1e-8 for b in baselines.values()
    )
    output_results = []

    if need_baseline:
        logger.info("First experiment detected — establishing baselines...")
        baselines = establish_baselines(output_results)

    rewards = discover_rewards("reward")
    if not rewards:
        logger.warning("No Reward* classes found in reward.py")

    rewards = rewards[:MAX_REWARDS_PER_EXPERIMENT]

    for i, reward_fn in enumerate(rewards):
        env_id = reward_fn.env_id
        if env_id not in ENV_REGISTRY:
            logger.error(f"Unknown env_id '{env_id}' in {reward_fn.name}, skipping")
            output_results.append({
                "reward_name": reward_fn.name,
                "env": env_id,
                "task_metrics": {"perf": 0.0, "eff": 0.0, "stab": 0.0},
                "reward_metrics": {"perf": 0.0, "eff": 0.0, "stab": 0.0},
                "status": "crash",
                "dominates": [],
                "dominated_by": [],
            })
            continue

        env_info = ENV_REGISTRY[env_id]
        bl = baselines.get(env_id, {})
        baseline_return = bl.get("baseline_return", 1.0)
        baseline_returns = bl.get("baseline_returns", [])

        if abs(baseline_return) < 1e-8:
            logger.error(f"No baseline for {env_id}, skipping {reward_fn.name}")
            continue

        logger.info(f"Evaluating reward {i+1}/{len(rewards)}: {reward_fn.name} on {env_id}...")

        try:
            def env_fn():
                return make_env_with_reward(reward_fn, env_info)

            t0 = time.time()
            task_returns, reward_returns = train_ppo(
                env_fn=env_fn,
                total_steps=TOTAL_STEPS,
                seeds=DEFAULT_SEEDS[:NUM_SEEDS],
                env_info=env_info,
            )
            elapsed = time.time() - t0

            task_metrics, reward_metrics = compute_all_metrics(
                task_seed_returns=task_returns,
                reward_seed_returns=reward_returns,
                baseline_return=baseline_return,
                baseline_returns=baseline_returns,
                direction=env_info.reward_direction,
            )

            status, dominates_list, dominated_by = pareto_decision(
                reward_fn.name,
                task_metrics,
                env_id,
                str(ARCHIVE_PATH),
            )

            result = {
                "reward_name": reward_fn.name,
                "env": env_id,
                "task_metrics": task_metrics,
                "reward_metrics": reward_metrics,
                "status": status,
                "dominates": dominates_list,
                "dominated_by": dominated_by,
            }
            output_results.append(result)

            logger.info(
                f"  {status.upper():6s} | perf={task_metrics['perf']:.4f} "
                f"eff={task_metrics['eff']:.4f} stab={task_metrics['stab']:.4f} "
                f"({elapsed:.1f}s)"
            )

            if status == "keep":
                from subprocess import check_output

                try:
                    commit = check_output(
                        ["git", "rev-parse", "--short=7", "HEAD"],
                        cwd=PROJECT_ROOT, text=True,
                    ).strip()[:7]
                except Exception:
                    commit = "unknown"

                factor_info = {
                    "name": reward_fn.name,
                    "perf": task_metrics["perf"],
                    "eff": task_metrics["eff"],
                    "stab": task_metrics["stab"],
                    "description": "",
                    "commit": commit,
                    "added": datetime.now().isoformat(),
                }
                update_archive(
                    factor_info,
                    env_id=env_id,
                    dominates_list=dominates_list,
                    str_path=str(ARCHIVE_PATH),
                )

        except Exception as e:
            logger.error(f"  CRASH  | {reward_fn.name}: {e}")
            output_results.append({
                "reward_name": reward_fn.name,
                "env": env_id,
                "task_metrics": {"perf": 0.0, "eff": 0.0, "stab": 0.0},
                "reward_metrics": {"perf": 0.0, "eff": 0.0, "stab": 0.0},
                "status": "crash",
                "dominates": [],
                "dominated_by": [],
                "error": str(e),
            })

    for r in output_results:
        print("---")
        print(f"reward:            {r['reward_name']}")
        print(f"env:               {r['env']}")
        tm = r.get("task_metrics", {})
        print(f"task_perf:         {tm.get('perf', 0):.4f}")
        print(f"task_eff:          {tm.get('eff', 0):.4f}")
        print(f"task_stab:         {tm.get('stab', 0):.4f}")
        rm = r.get("reward_metrics", {})
        print(f"reward_perf:       {rm.get('perf', 0):.4f}")
        print(f"reward_eff:        {rm.get('eff', 0):.4f}")
        print(f"reward_stab:       {rm.get('stab', 0):.4f}")
        dominates_str = ", ".join(r.get("dominates", [])) or "(none)"
        dominated_str = ", ".join(r.get("dominated_by", [])) or "(none)"
        print(f"dominates:          {dominates_str}")
        print(f"dominated_by:       {dominated_str}")
        print(f"status:             {r['status']}")
        if r.get("error"):
            print(f"error:              {r['error']}")

    return baselines


def main():
    evaluate_all_rewards()


if __name__ == "__main__":
    main()

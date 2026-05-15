from typing import Dict, List

import numpy as np


def compute_performance(returns: List[float], baseline_return: float, direction: str) -> float:
    if len(returns) == 0:
        return 0.0
    n = min(10, len(returns))
    final_mean = float(np.mean(returns[-n:]))
    return normalize_metric(final_mean, baseline_return, direction)


def compute_sample_efficiency(
    returns: List[float],
    baseline_returns: List[float],
    direction: str,
) -> float:
    if len(returns) == 0:
        return 0.0
    auc = float(np.trapezoid(np.array(returns, dtype=float)))
    baseline_auc = float(np.trapezoid(np.array(baseline_returns, dtype=float)))
    if abs(baseline_auc) < 1e-8:
        return 1.0
    return normalize_metric(auc, baseline_auc, direction)


def compute_stability(seed_returns: List[List[float]]) -> float:
    if len(seed_returns) < 2:
        return 1.0
    final_returns = []
    for returns in seed_returns:
        if len(returns) == 0:
            final_returns.append(0.0)
        else:
            n = min(10, len(returns))
            final_returns.append(float(np.mean(returns[-n:])))
    finals = np.array(final_returns)
    mean_val = finals.mean()
    std_val = finals.std()
    if abs(mean_val) < 1e-8:
        return 1.0 if std_val < 1e-8 else 0.0
    cv = std_val / abs(mean_val)
    return float(max(0.0, 1.0 - min(1.0, cv)))


def normalize_metric(actual: float, baseline: float, direction: str) -> float:
    if abs(baseline) < 1e-8:
        return 1.0
    if direction == "positive":
        return actual / baseline
    if abs(actual) < 1e-8:
        return 0.0
    ratio = baseline / actual
    return max(0.0, ratio)


def compute_all_metrics(
    task_seed_returns: List[List[float]],
    reward_seed_returns: List[List[float]],
    baseline_return: float,
    baseline_returns: List[float],
    direction: str,
) -> tuple:
    primary_seed = task_seed_returns[0] if task_seed_returns else []
    task_metrics = {
        "perf": compute_performance(primary_seed, baseline_return, direction),
        "eff": compute_sample_efficiency(primary_seed, baseline_returns, direction),
        "stab": compute_stability(task_seed_returns),
    }
    reward_primary = reward_seed_returns[0] if reward_seed_returns else []
    reward_metrics = {
        "perf": compute_performance(reward_primary, baseline_return, direction),
        "eff": compute_sample_efficiency(reward_primary, baseline_returns, direction),
        "stab": compute_stability(reward_seed_returns),
    }
    return task_metrics, reward_metrics

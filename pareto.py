import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

METRIC_NAMES = ["perf", "eff", "stab"]
DEFAULT_ARCHIVE_PATH = Path(__file__).resolve().parent / "pareto_frontier.json"


def dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    all_ge = all(a[m] >= b[m] for m in METRIC_NAMES)
    any_gt = any(a[m] > b[m] for m in METRIC_NAMES)
    return all_ge and any_gt


def load_archive(path: str = None):
    p = Path(path) if path else DEFAULT_ARCHIVE_PATH
    if not p.exists():
        return {
            "metrics": METRIC_NAMES,
            "environments": {},
            "dominated_count": 0,
            "total_experiments": 0,
        }
    with open(p) as f:
        return json.load(f)


def pareto_decision(
    name: str,
    metrics: Dict[str, float],
    env_id: str,
    archive_path: str = None,
) -> Tuple[str, List[str], List[str]]:
    archive = load_archive(archive_path)
    env_data = archive.get("environments", {}).get(env_id, {})
    frontier = env_data.get("frontier", [])

    if any(np.isnan(metrics.get(m, np.nan)) for m in METRIC_NAMES):
        return ("crash", [], [])

    dominated_by = []
    dominates_list = []

    for f in frontier:
        f_metrics = {m: f.get(m, 0.0) for m in METRIC_NAMES}
        if dominates(f_metrics, metrics):
            dominated_by.append(f["name"])
        if dominates(metrics, f_metrics):
            dominates_list.append(f["name"])

    if dominates_list:
        return ("keep", dominates_list, dominated_by)
    elif dominated_by and not dominates_list:
        if len(dominated_by) == len(frontier) and len(frontier) > 0:
            return ("discard", [], dominated_by)
        else:
            return ("keep", [], dominated_by)
    else:
        return ("keep", [], dominated_by)


def update_archive(
    factor_info: Dict,
    env_id: str,
    dominates_list: List[str] = None,
    str_path: str = None,
) -> None:
    archive = load_archive(str_path)
    p = Path(str_path) if str_path else DEFAULT_ARCHIVE_PATH

    if "environments" not in archive:
        archive["environments"] = {}
    if env_id not in archive["environments"]:
        archive["environments"][env_id] = {
            "frontier": [],
            "baseline_return": 0.0,
            "baseline_returns": [],
        }

    env_data = archive["environments"][env_id]

    if dominates_list:
        dominated_set = set(dominates_list)
        env_data["frontier"] = [
            f for f in env_data["frontier"] if f["name"] not in dominated_set
        ]
        archive["dominated_count"] = archive.get("dominated_count", 0) + len(dominated_set)

    env_data["frontier"].append(factor_info)
    archive["total_experiments"] = archive.get("total_experiments", 0) + 1

    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(archive, f, indent=2, ensure_ascii=False)

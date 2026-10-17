import numpy as np
from .fcfs import fcfs


def timetable_policy(observation: np.ndarray, mask: np.ndarray, max_platforms: int = 8) -> int:
    """Prefer the plan, then use the first safe alternative; no privileged timetable."""
    planned = round(float(observation[10 * max_platforms + 9]) * max_platforms)
    if 0 < planned < len(mask) and mask[planned]:
        return planned
    return fcfs(observation, mask, max_platforms)

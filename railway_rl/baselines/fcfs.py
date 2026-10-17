import numpy as np


def fcfs(observation: np.ndarray, mask: np.ndarray, max_platforms: int = 8) -> int:
    """Common FCFS queue head; choose the lowest-numbered feasible platform."""
    assignments = np.flatnonzero(mask[1:]) + 1
    return int(assignments[0]) if len(assignments) else 0

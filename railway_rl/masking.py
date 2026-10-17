"""Shared feasibility contract for learning agents and all baselines."""
import numpy as np
from .simulator import Simulator


def action_mask(sim: Simulator, max_platforms: int, hold_limit: int) -> np.ndarray:
    mask = np.zeros(max_platforms + 1, dtype=bool)
    mask[0] = True
    train = sim.current
    if train is not None:
        for p in sim.station.platforms:
            mask[p.id] = sim.feasible(train, p.id)
        if sim.states[train.id].waiting >= hold_limit - 1e-9 and mask[1:].any():
            mask[0] = False
    return mask

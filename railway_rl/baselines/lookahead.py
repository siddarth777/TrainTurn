import numpy as np


def lookahead(observation: np.ndarray, mask: np.ndarray, max_platforms: int = 8) -> int:
    """One-action projection with greedy forecast scheduling; no simulator cloning."""
    n = max_platforms
    platforms = observation[:10 * n].reshape(n, 10)
    current = observation[10 * n:11 * n + 12]
    forecasts = observation[11 * n + 12:-9].reshape(-1, 12 + n)
    valid = np.flatnonzero(mask)
    def cost(action):
        releases = platforms[:, 2].astype(float) * 120
        releases[(platforms[:, 0] == 0) | (platforms[:, 4] == 1)] = np.inf
        value = 0.0
        if action:
            releases[action - 1] = current[4] * 60 + 2 + platforms[action - 1, 9] * 10
            value += 2.0 * (action != round(float(current[9]) * n))
        else:
            # Include the current train after one minute; holding cannot make it disappear.
            candidates = np.flatnonzero(current[12:] > 0)
            if not len(candidates) or not np.isfinite(releases[candidates]).any():
                return float("inf")
            p = min(candidates, key=lambda p: (releases[p], p))
            start = max(1.0, releases[p])
            value += 5 * current[1] * start + 0.2
            releases[p] = start + current[4] * 60 + 2 + platforms[p, 9] * 10
        for future in forecasts:
            if not future[0]:
                continue
            candidates = np.flatnonzero(future[12:] > 0)
            if not len(candidates) or not np.isfinite(releases[candidates]).any():
                value += 5 * future[3] * 120
                continue
            p = min(candidates, key=lambda p: (releases[p], p))
            arrival = future[1] * 120
            start = max(arrival, releases[p])
            value += 5 * future[3] * (start - arrival)
            releases[p] = start + future[4] * 60 + 2 + platforms[p, 9] * 10
        return value
    # At equal cost, prefer an assignment over a hold.
    return int(min(valid, key=lambda action: (cost(action), action == 0, action)))

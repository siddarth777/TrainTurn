import numpy as np


def priority_policy(observation: np.ndarray, mask: np.ndarray, max_platforms: int = 8) -> int:
    """Preserve scarce platforms for high-priority forecasts; cannot reorder FCFS."""
    valid = np.flatnonzero(mask[1:]) + 1
    if not len(valid):
        return 0
    n = max_platforms
    platforms = observation[:10 * n].reshape(n, 10)
    current = observation[10 * n:11 * n + 12]
    forecasts = observation[11 * n + 12:-9].reshape(-1, 12 + n)
    def score(p):
        reservation_cost = sum(float(f[3]) / max(1, np.count_nonzero(f[12:]))
                               for f in forecasts if f[0] and f[12 + p - 1]
                               and f[1] * 120 < current[4] * 60 + 4)
        excess_length = max(0, platforms[p - 1, 6] - current[3])
        return (reservation_cost + 0.1 * excess_length, p)
    return int(min(valid, key=score))

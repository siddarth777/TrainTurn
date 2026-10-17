"""Per-minute costs: no incoming lateness is charged to the allocator."""

def minute_reward(costs: dict[str, float], coefficients: dict, *,
                  platform_change: bool = False, unnecessary_hold: bool = False) -> float:
    cost = sum(coefficients[key] * costs[key] for key in ("weighted_wait", "outer", "upstream"))
    cost += coefficients["platform_change"] * platform_change
    cost += coefficients["unnecessary_hold"] * unnecessary_hold
    return -cost / coefficients["scale"]

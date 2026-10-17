"""Observable-only controller view and its fixed-dimensional vector encoding."""
import math
import numpy as np
from .simulator import Simulator


def observation_size(max_platforms: int, forecast_count: int) -> int:
    return 11 * max_platforms + 21 + forecast_count * (12 + max_platforms)


def controller_view(sim: Simulator, config: dict) -> dict:
    n, k = config["max_platforms"], config["forecast_count"]
    corridors = sorted({c for p in sim.station.platforms for c in p.approaches + p.departures})
    if len(corridors) > 4:
        raise ValueError("Observation encoding supports at most four named corridors")
    codes = {name: (i + 1) / 4 for i, name in enumerate(corridors)}

    def train_view(train):
        state = sim.states[train.id]
        return {"id": train.id, "priority": train.priority, "dwell": train.dwell,
                "length": train.length, "electric": train.electric, "through": train.kind == "through",
                "arrival_corridor": codes[train.arrival_corridor],
                "departure_corridor": codes[train.departure_corridor],
                "planned": train.planned_platform, "rake": train.rake_from is not None,
                "waiting": state.waiting, "signal_delay": max(0, (state.signal_arrival or 0) - train.scheduled_arrival),
                "compatibility": [p.compatible(train) for p in sim.station.platforms] + [False] * (n - len(sim.station.platforms))}

    platforms = []
    for p in sim.station.platforms:
        occupant = sim.occupants.get(p.id)
        release, priority = max(sim.time, sim.release[p.id]), 0.0
        if occupant is not None:
            state, train = sim.states[occupant], sim.trains[occupant]
            # Do not read state.ready: it includes unrevealed dwell perturbations.
            nominal = max(train.scheduled_departure,
                          state.entry + sim.station.entry_minutes + train.dwell + sim.station.departure_minutes)
            release = max(sim.time + 1, nominal) + p.clearance
            if state.departure_start is not None:
                release = state.departure_start + sim.station.departure_minutes + p.clearance
            priority = train.priority
        route_available = sim.current is not None and sim.route_free(
            sim.station.route(sim.current, p.id, "in"), sim.time, sim.station.entry_minutes)
        platforms.append({"id": p.id, "occupied": occupant is not None, "release": release - sim.time,
                          "priority": priority, "closed": sim.closed(p.id), "route_available": route_available,
                          "length": p.length, "electric": p.electrified, "through": p.kind == "through",
                          "clearance": p.clearance})
    forecasts = []
    for train in sim.scenario.trains:
        if sim.states[train.id].status != "future":
            continue
        noise = sim.noise[train.id]
        horizon = max(0, train.scheduled_arrival - sim.time)
        # Frozen noisy estimate + a persistent error whose amplitude decreases with horizon.
        uncertainty = 2.0 + 0.1 * min(horizon, 240)
        eta = train.scheduled_arrival + max(0, noise.estimated_delay + noise.forecast_error * uncertainty)
        if train.rake_from is not None:
            previous = sim.states[train.rake_from]
            if previous.departure is not None:
                eta = max(eta, previous.departure + train.turnaround)
        eta = max(sim.time + 1, eta)  # A train not yet observed cannot already have arrived.
        view = train_view(train)
        view.update({"eta": eta - sim.time, "estimated_delay": eta - train.scheduled_arrival,
                     "uncertainty": uncertainty})
        forecasts.append(view)
    forecasts.sort(key=lambda t: (t["eta"], t["id"]))
    return {"time": sim.time, "platforms": platforms, "current": train_view(sim.current) if sim.current else None,
            "forecasts": forecasts[:k], "outer": len(sim.outer), "upstream": len(sim.upstream),
            "remaining": sum(s.status != "departed" for s in sim.states.values()),
            "total": len(sim.states)}


def encode(view: dict, config: dict) -> np.ndarray:
    n, k = config["max_platforms"], config["forecast_count"]
    values = []
    for p in view["platforms"]:
        values.extend([1, p["occupied"], p["release"] / 120, p["priority"] / 5,
                       p["closed"], p["route_available"], p["length"] / 30,
                       p["electric"], p["through"], p["clearance"] / 10])
    values.extend([0] * (10 * (n - len(view["platforms"]))))
    t = view["current"]
    if t:
        values.extend([1, t["priority"] / 5, t["waiting"] / 120, t["length"] / 30,
                       t["dwell"] / 60, t["electric"], t["through"], t["arrival_corridor"],
                       t["departure_corridor"], t["planned"] / n, t["signal_delay"] / 180, t["rake"]])
        values.extend(t["compatibility"])
    else:
        values.extend([0] * (12 + n))
    for t in view["forecasts"]:
        values.extend([1, t["eta"] / 120, t["estimated_delay"] / 180, t["priority"] / 5,
                       t["dwell"] / 60, t["length"] / 30, t["electric"], t["through"],
                       t["arrival_corridor"], t["departure_corridor"], t["planned"] / n, t["uncertainty"] / 60])
        values.extend(t["compatibility"])
    values.extend([0] * ((k - len(view["forecasts"])) * (12 + n)))
    occupied = sum(p["occupied"] for p in view["platforms"])
    available = sum(not p["occupied"] and not p["closed"] and p["release"] <= 0 for p in view["platforms"])
    phase = 2 * math.pi * (view["time"] % 1440) / 1440
    values.extend([math.sin(phase), math.cos(phase), view["outer"] / 3, view["upstream"] / 150,
                   available / n, view["remaining"] / 150, view["time"] / config["horizon"],
                   occupied / n, (view["outer"] + view["upstream"]) / max(1, view["total"])])
    result = np.clip(np.asarray(values, dtype=np.float32), -1, 1)
    if result.shape != (observation_size(n, k),):
        raise RuntimeError("Observation layout mismatch")
    return result

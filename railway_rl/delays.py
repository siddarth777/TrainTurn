"""Immutable, assumption-based exogenous randomness, independent of policy actions."""
import numpy as np
from .station import Scenario, Disturbance, Closure
from .timetable import nominal_timetable


def generate_scenario(config: dict, seed: int) -> Scenario:
    station, trains = nominal_timetable(config, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    forecasts = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    c, d = config["scenario"], config["delays"]
    severity = c["severity"] * {"normal": 1, "fog": 2, "monsoon": 1.5, "peak": 1.2, "closure": 1, "corridor": 1}[c["mode"]]
    shocks = {}
    disturbances = []
    category_factor = {"premium": 0.7, "superfast": 0.85, "express": 1, "passenger": 1.15, "freight": 1.25}
    for train in trains:
        key = (train.arrival_corridor, train.scheduled_arrival // 120)
        if key not in shocks:
            shocks[key] = float(rng.lognormal(d["corridor_logmean"], d["corridor_logsigma"])) if rng.random() < d["corridor_probability"] else 0
        u = rng.random()
        if u < d["on_time_probability"]:
            individual = 0.0
        elif u < d["on_time_probability"] + d["short_probability"]:
            individual = rng.lognormal(d["short_logmean"], d["short_logsigma"])
        else:
            individual = rng.lognormal(d["severe_logmean"], d["severe_logsigma"])
        peak_factor = 1.2 if 420 <= train.scheduled_arrival % 1440 < 600 else 1.0
        corridor_extra = 60.0 if c["mode"] == "corridor" and train.arrival_corridor == "A" else 0.0
        delay = round(severity * (category_factor[train.category] * individual * peak_factor + shocks[key] + corridor_extra))
        dwell = round(severity * rng.lognormal(0.8, 0.5)) if rng.random() < d["dwell_probability"] else 0
        estimate = max(0.0, delay + forecasts.normal(0, d["forecast_sigma"]))
        disturbances.append(Disturbance(delay, dwell, float(estimate), float(forecasts.normal())))
    closures = ()
    if c["mode"] == "closure":
        start = c["start_minute"] + 20
        closures = (Closure(1, start, start + 60),)
    return Scenario(station, trains, tuple(disturbances), closures, seed)

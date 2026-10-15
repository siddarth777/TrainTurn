"""Validated YAML settings, shared by all command-line entry points."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

DEFAULTS = {
    "station": {"platform_count": 3, "outer_capacity": 2, "platforms": [],
                "entry_minutes": 3, "departure_minutes": 3, "route_clearance": 1,
                "extra_conflicts": []},
    "scenario": {"trains": 15, "start_minute": 0, "peak_fraction": 0.7,
                 "severity": 1.0, "mode": "normal", "rake_links": False,
                 "priorities": {"premium": 5.0, "superfast": 4.0, "express": 3.0,
                                "passenger": 1.0, "freight": 1.5}},
    "delays": {"on_time_probability": 0.65, "short_probability": 0.30,
               "short_logmean": 1.8, "short_logsigma": 0.6, "severe_logmean": 3.8,
               "severe_logsigma": 0.9, "corridor_probability": 0.3, "corridor_logmean": 2.3,
               "corridor_logsigma": 0.7, "dwell_probability": 0.15, "forecast_sigma": 2.0},
    "environment": {"backend": "sumo", "max_platforms": 8, "forecast_count": 6, "horizon": 360,
                    "hold_limit": 30, "gamma": 0.99},
    "reward": {"weighted_wait": 1.0, "outer": 0.05, "upstream": 0.2,
               "platform_change": 2.0, "unnecessary_hold": 0.2, "scale": 10.0},
    "agent": {"hidden": [128, 128], "learning_rate": 0.0003, "batch_size": 128,
              "replay_capacity": 100000, "warmup": 1000, "target_update": 1000,
              "gradient_clip": 10.0, "epsilon_start": 1.0, "epsilon_end": 0.05,
              "epsilon_steps": 50000, "train_every": 1, "device": "cuda", "threads": 1,
              "vram_limit_gib": 5.5},
    "training": {"episodes": 20, "validate_every": 5, "validation_episodes": 2,
                 "validation_seed": 100000, "test_seed": 200000},
    "curriculum": [],
}


def merge(base: dict, updates: dict) -> dict:
    result = deepcopy(base)
    for key, value in updates.items():
        if key not in result:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected mapping for {key}")
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate(config: dict) -> dict:
    s, e, a, t = (config[k] for k in ("scenario", "environment", "agent", "training"))
    if e["backend"] not in ("sumo", "reference"):
        raise ValueError("Unknown simulator backend")
    if a["device"] != "cuda":
        raise ValueError("This project is configured for CUDA-only neural execution")
    n = len(config["station"]["platforms"]) or config["station"]["platform_count"]
    if not 3 <= n <= e["max_platforms"] <= 8 or not 1 <= s["trains"] <= 150:
        raise ValueError("Require 3–8 platforms, sufficient padding, and 1–150 trains")
    if not 0 < e["gamma"] <= 1 or min(e["forecast_count"], e["horizon"], e["hold_limit"]) < 1:
        raise ValueError("Invalid environment horizon, forecasts, hold limit, or discount")
    d = config["delays"]
    for key in ("on_time_probability", "short_probability", "corridor_probability", "dwell_probability"):
        if not 0 <= d[key] <= 1:
            raise ValueError("Delay probabilities must be in [0,1]")
    if d["on_time_probability"] + d["short_probability"] > 1:
        raise ValueError("Delay mixture probabilities exceed one")
    if min(d["short_logsigma"], d["severe_logsigma"], d["corridor_logsigma"], d["forecast_sigma"]) < 0:
        raise ValueError("Delay distribution scales must be nonnegative")
    if not 0 <= s["peak_fraction"] <= 1 or s["severity"] < 0:
        raise ValueError("Invalid scenario severity or peak fraction")
    if s["mode"] not in ("normal", "fog", "monsoon", "peak", "closure", "corridor"):
        raise ValueError("Unknown disruption mode")
    if min(s["priorities"].values()) <= 0:
        raise ValueError("Priorities must be positive")
    if min(config["reward"].values()) < 0 or config["reward"]["scale"] <= 0:
        raise ValueError("Invalid reward coefficients")
    if not a["hidden"] or min(a["hidden"]) < 1:
        raise ValueError("Network requires positive hidden widths")
    for name in ("batch_size", "replay_capacity", "warmup", "target_update", "epsilon_steps", "train_every", "threads"):
        if not isinstance(a[name], int) or a[name] < 1:
            raise ValueError(f"Invalid agent {name}")
    if a["batch_size"] > a["replay_capacity"] or a["warmup"] > a["replay_capacity"]:
        raise ValueError("Replay capacity must cover warmup and batch size")
    if not 0 <= a["epsilon_end"] <= a["epsilon_start"] <= 1:
        raise ValueError("Invalid exploration schedule")
    if a["learning_rate"] <= 0 or a["gradient_clip"] <= 0:
        raise ValueError("Invalid optimizer settings")
    if a["vram_limit_gib"] <= 0:
        raise ValueError("VRAM limit must be positive")
    if min(t["episodes"], t["validate_every"], t["validation_episodes"]) < 1:
        raise ValueError("Training budgets must be positive")
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return not isinstance(value, (int, float)) or math.isfinite(value)
    if not finite(config):
        raise ValueError("Configuration contains nonfinite values")
    for stage in config["curriculum"]:
        if set(stage) != {"episodes", "platforms", "trains", "severity"} or stage["episodes"] < 1:
            raise ValueError("Curriculum stages require episodes/platforms/trains/severity")
        if not 3 <= stage["platforms"] <= e["max_platforms"] or not 1 <= stage["trains"] <= 150 or stage["severity"] < 0:
            raise ValueError("Invalid curriculum stage")
    return config


def load_config(path: str | Path | None = None) -> dict:
    updates = {} if path is None else yaml.safe_load(Path(path).read_text())
    if not isinstance(updates, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return validate(merge(DEFAULTS, updates))

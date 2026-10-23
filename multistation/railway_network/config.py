"""Network YAML loading; the junction section reuses the original validated config."""
from copy import deepcopy
from pathlib import Path
import yaml
from railway_rl.config import load_config, merge, validate

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "network.yaml"
REQUIRED = ("line_speed", "acceleration", "deceleration", "block_length", "run_allowance",
            "station_entry_length", "station_track_length", "station_exit_length", "drain_minutes",
            "west", "east", "stations", "stops", "minor_dwell_minutes",
            "dwell_extension_probability", "dwell_extension_max_minutes", "westbound_share")


def load_network_config(path: str | Path | None = None) -> dict:
    """Return ``{"junction": <railway_rl config>, "network": {...}, "evaluation": {...}}``."""
    path = Path(path or DEFAULT_PATH).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "network" not in raw or "junction_config" not in raw:
        raise ValueError("Network config needs 'junction_config' and 'network' sections")
    junction = load_config((path.parent / raw["junction_config"]).resolve())
    junction = validate(merge(junction, raw.get("junction_overrides") or {}))
    if junction["environment"]["backend"] != "sumo":
        raise ValueError("The network adapter requires the SUMO junction backend")
    network = deepcopy(raw["network"])
    missing = [k for k in REQUIRED if k not in network]
    if missing:
        raise ValueError(f"Network config is missing: {', '.join(missing)}")
    classes = set(junction["scenario"]["priorities"])
    if set(network["stops"]) != classes or set(network["minor_dwell_minutes"]) != classes:
        raise ValueError("Stopping patterns and minor dwell must cover every train class")
    if not 0 <= network["run_allowance"] < 1 or not 0 <= network["dwell_extension_probability"] <= 1:
        raise ValueError("Allowance must be in [0,1) and the extension probability in [0,1]")
    if not 0 <= network["westbound_share"] <= 1:
        raise ValueError("westbound_share must be in [0,1]")
    if min(network["line_speed"], network["acceleration"], network["deceleration"]) <= 0:
        raise ValueError("Speeds and rates must be positive")
    evaluation = {"test_seed": 200000, "episodes": 3, **(raw.get("evaluation") or {})}
    return {"junction": junction, "network": network, "evaluation": evaluation, "source": str(path)}

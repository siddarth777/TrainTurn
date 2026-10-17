import numpy as np
from railway_rl.baselines import POLICIES
from railway_rl.config import load_config as _load_config

def load_config():
    config = _load_config()
    config["environment"]["backend"] = "reference"
    config["station"]["entry_minutes"] = 1
    config["station"]["departure_minutes"] = 1
    return config

from railway_rl.environment import TrainTurnEnv
from railway_rl.evaluation.metrics import episode_metrics


def test_baselines_safe_and_timetable_zero_nominal_delay():
    config = load_config()
    config["scenario"]["trains"] = 15
    config["scenario"]["severity"] = 0
    for name, policy in POLICIES.items():
        env = TrainTurnEnv(config)
        obs, info = env.reset(seed=8)
        for _ in range(2000):
            action = policy(obs, info["action_mask"], 8)
            assert info["action_mask"][action]
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        assert term and not trunc
        metrics, rows = episode_metrics(env.sim, config["environment"]["horizon"])
        assert metrics["safety_violations"] == 0 and metrics["processed_fraction"] == 1
        assert len(rows) == 15
        if name == "timetable":
            assert metrics["weighted_station_delay"] == 0


def test_all_policies_only_hold_when_no_assignments():
    obs = np.zeros(229, dtype=np.float32)
    mask = np.zeros(9, dtype=bool)
    mask[0] = True
    for policy in POLICIES.values():
        assert policy(obs, mask, 8) == 0

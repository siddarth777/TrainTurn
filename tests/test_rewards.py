import pytest
from railway_rl.config import load_config as _load_config

def load_config():
    config = _load_config()
    config["environment"]["backend"] = "reference"
    config["station"]["entry_minutes"] = 1
    config["station"]["departure_minutes"] = 1
    return config

from railway_rl.environment import TrainTurnEnv
from railway_rl.rewards import minute_reward
from test_simulator import scenario


def test_hand_calculated_reward():
    coefficients = load_config()["reward"]
    assert minute_reward({"weighted_wait": 5, "outer": 1, "upstream": 0}, coefficients) == -0.505
    env = TrainTurnEnv(config=load_config(), scenario=scenario())
    env.reset()
    _, reward, _, _, _ = env.step(0)
    assert reward == pytest.approx(-(3 + 0.05 + 0.2) / 10)
    assert env.sim.states["T1"].waiting == 1

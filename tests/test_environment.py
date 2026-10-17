from dataclasses import replace
import numpy as np
import pytest
from railway_rl.config import load_config as _load_config

def load_config():
    config = _load_config()
    config["environment"]["backend"] = "reference"
    config["station"]["entry_minutes"] = 1
    config["station"]["departure_minutes"] = 1
    return config

from railway_rl.environment import TrainTurnEnv
from railway_rl.station import Disturbance, Closure
from railway_rl.observations import observation_size
from test_simulator import scenario


def test_observation_and_hold_time():
    env = TrainTurnEnv(config=load_config(), scenario=scenario())
    obs, info = env.reset(seed=4)
    assert obs.shape == (229,) and env.observation_space.contains(obs)
    assert observation_size(8, 6) == 229
    _, reward, term, trunc, info = env.step(0)
    assert info["elapsed_minutes"] == 1 and env.sim.time == 1
    assert reward < 0 and not term and not trunc
    assert not info["action_mask"][4:].any()


def test_terminal_and_truncation_conservation():
    env = TrainTurnEnv(config=load_config(), scenario=scenario())
    env.reset()
    _, _, terminal, truncated, info = env.step(1)
    assert terminal and not truncated and not info["unfinished_trains"]
    assert env.sim.complete
    with pytest.raises(RuntimeError):
        env.step(0)
    config = load_config()
    config["environment"]["horizon"] = 3
    env = TrainTurnEnv(config, scenario(closures=(Closure(1, 0, 10), Closure(2, 0, 10), Closure(3, 0, 10))))
    env.reset()
    for _ in range(3):
        _, _, terminal, truncated, info = env.step(0)
    assert truncated and not terminal and info["unfinished_trains"] == ("T1",)
    assert env.sim.states["T1"].waiting == 3


def test_hold_limit_preserves_safety():
    config = load_config()
    config["environment"]["hold_limit"] = 1
    env = TrainTurnEnv(config, scenario())
    env.reset(); env.step(0)
    assert not env.action_masks()[0]
    with pytest.raises(ValueError):
        env.step(0)
    env = TrainTurnEnv(config, scenario(closures=tuple(Closure(p, 0, 10) for p in range(1, 4))))
    env.reset(); env.step(0)
    assert env.action_masks().tolist() == [True] + [False] * 8
    env.step(0)
    assert not env.sim.occupants


def test_hidden_dwell_does_not_leak_into_release_estimate():
    a = scenario()
    b = replace(a, disturbances=(Disturbance(dwell_extension=20),))
    config = load_config()
    config["environment"]["horizon"] = 2
    envs = [TrainTurnEnv(config, x) for x in (a, b)]
    observations = []
    for env in envs:
        env.reset(); observations.append(env.step(1)[0])
    np.testing.assert_array_equal(*observations)


def test_future_arrival_truth_does_not_leak():
    base = scenario()
    future = replace(base.trains[0], id="future", scheduled_arrival=50, scheduled_departure=57)
    a = replace(base, trains=base.trains + (future,), disturbances=(Disturbance(), Disturbance(10, 0, 5, 0.4)))
    b = replace(a, disturbances=(Disturbance(), Disturbance(80, 0, 5, 0.4)))
    envs = [TrainTurnEnv(config=load_config(), scenario=x) for x in (a, b)]
    np.testing.assert_array_equal(envs[0].reset(seed=1)[0], envs[1].reset(seed=1)[0])


def test_random_masked_episode_determinism():
    config = load_config()
    config["station"]["platform_count"] = 3
    config["scenario"]["trains"] = 15
    histories = []
    for _ in range(2):
        env = TrainTurnEnv(config)
        obs, info = env.reset(seed=42)
        rng = np.random.default_rng(21)
        history = []
        while True:
            action = int(rng.choice(np.flatnonzero(info["action_mask"])))
            obs, reward, term, trunc, info = env.step(action)
            assert env.observation_space.contains(obs)
            history.append((action, reward, info["elapsed_minutes"]))
            if term or trunc:
                break
        assert term and not trunc
        env.sim.assert_safety()
        histories.append(history)
    assert histories[0] == histories[1]

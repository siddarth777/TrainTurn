import pytest
from railway_rl import training


@pytest.mark.parametrize("fraction, expected", [(1.0, 3.0), (0.5, None)])
def test_validation_preserves_incomplete_status(monkeypatch, fraction, expected):
    rows = iter([
        {"processed_fraction": 1.0, "weighted_station_delay": 2.0,
         "episode_reward": -2.0, "discretionary_holds": 0},
        {"processed_fraction": fraction, "weighted_station_delay": 4.0,
         "episode_reward": -4.0, "discretionary_holds": 2},
    ])
    monkeypatch.setattr(training, "run_episode", lambda *args: (next(rows), [], []))
    result = training.validation_metrics({}, [1, 2], lambda o, m: 0)
    assert result["mean_weighted_delay"] == expected
    assert result["complete"] == (fraction == 1.0)
    assert result["mean_discretionary_holds"] == 1.0
    assert result["mean_reward"] == -3.0

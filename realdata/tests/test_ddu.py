import csv
from pathlib import Path
import pandas as pd
import pytest
from railway_rl.baselines import POLICIES
from railway_rl.evaluation.evaluator import run_episode
from railway_rl.timetable import load_timetable
from railway_real.ddu import build_runs, classify, complete_days, day_scenario, load_real_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "ddu.yaml"
cfg = load_real_config(CONFIG)
pytestmark = pytest.mark.skipif(not (Path(cfg["raw"]["rstgcn"]) / "train_routes_delays_Sep2024.csv").exists(),
                                reason="raw data not downloaded (realdata/scripts/fetch_data.sh)")
DAY = "2024-09-15"


@pytest.fixture(scope="module")
def runs():
    return build_runs(cfg)[0]


def test_class_rules():
    rules = cfg["classes"]
    assert classify("Rajdhani Express", None, "12301", rules) == "premium"
    assert classify("Rnc Ndls Raj Express", None, "20407", rules) == "premium"
    assert classify("Varanasi Memu", None, "63210", rules) == "passenger"
    assert classify("Neelachal Sf Express", None, "12875", rules) == "superfast"
    assert classify("Doon Express", "Exp", "13009", rules) == "express"
    assert classify("Netaji Express", None, "12312", rules) == "superfast"  # 12xxx numbering


def test_runs_cover_complete_days_with_real_delays(runs):
    days = complete_days(runs)
    assert days[0] == "2024-09-03" and days[-1] == "2024-09-30" and len(days) == 28
    df = pd.DataFrame([r.__dict__ for r in runs])
    assert df.incoming_delay.notna().all() and (df.incoming_delay >= 0).all()
    assert set(df.arrival_corridor) == set(df.departure_corridor) == {"A", "B"}
    # Calendar day = start date + day offset along the route.
    assert ((pd.to_datetime(df.calendar_date) - pd.to_datetime(df.start_date)).dt.days.between(0, 3)).all()


def test_day_scenarios_are_valid_and_inside_the_window(runs):
    for mode in ("planned", "actual"):
        case, info = day_scenario(cfg, runs, DAY, mode)
        assert info["trains"] == len(case.trains) > 50
        assert len({t.id for t in case.trains}) == len(case.trains)
        assert len(case.station.platforms) == 8
        for train, noise in zip(case.trains, case.disturbances):
            assert 0 <= train.scheduled_arrival + noise.incoming_delay < cfg["day"]["last_arrival_minute"]
            assert noise.dwell_extension == 0 and noise.incoming_delay >= 0
            if mode == "planned":
                assert noise.incoming_delay == 0 and noise.estimated_delay == 0
        # Long trains only fit the four long platforms.
        assert all(case.station.platforms[t.planned_platform - 1].compatible(t) for t in case.trains)


def test_planned_timetable_round_trips_through_the_original_csv_loader(runs, tmp_path):
    case, _ = day_scenario(cfg, runs, DAY, "planned")
    fields = ["id", "category", "scheduled_arrival", "scheduled_departure", "planned_platform", "length",
              "electric", "kind", "arrival_corridor", "departure_corridor", "dwell", "priority"]
    path = tmp_path / "day.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: getattr(t, k) for k in fields} for t in case.trains)
    assert load_timetable(path) == case.trains


def test_forecast_uses_an_earlier_observation_than_the_incoming_delay(runs):
    case, _ = day_scenario(cfg, runs, DAY, "actual")
    by_id = {}
    for r in runs:
        by_id.setdefault(r.number, []).append(r)
    for train, noise in zip(case.trains, case.disturbances):
        candidates = by_id[train.id.split("-")[0]]
        assert any((r.forecast_delay or 0.0) == pytest.approx(noise.estimated_delay) for r in candidates)


def test_real_day_replays_safely():
    case, _ = day_scenario(cfg, build_runs(cfg)[0], DAY, "actual")
    metrics, rows, _ = run_episode(cfg["junction"], case, lambda o, m: POLICIES["fcfs"](o, m, 8))
    assert metrics["safety_violations"] == 0 and metrics["processed_fraction"] == 1
    assert len(rows) == len(case.trains)


def test_finetune_split_is_chronological_and_disjoint(runs):
    from railway_real.finetune import days_between
    f, days = cfg["finetune"], complete_days(runs)
    train, val, test = (days_between(days, f[k]) for k in ("train_days", "validation_days", "test_days"))
    assert train and val and test
    assert max(train) < min(val) and max(val) < min(test)
    assert sorted(train + val + test) == days  # every complete day is used exactly once
